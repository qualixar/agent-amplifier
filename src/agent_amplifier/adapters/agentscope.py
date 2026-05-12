# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""AgentScopeAdapter — Memory binding for AgentScope (Alibaba Tongyi Lab).

.1 row 6; research:

Design notes:
    * The user supplies a working-memory instance (``InMemoryMemory``,
      ``RedisMemory``, ``AsyncSQLAlchemyMemory``, ``TablestoreMemory``) OR
      a long-term-memory instance (``Mem0LongTermMemory``,
      ``ReMePersonalLongTermMemory``, etc.). We duck-type rather than
      ``isinstance``-check.
    * NOTE: we target the **working-memory** shape stable in AgentScope
      late-2026: ``get_memory()`` returning ``list[Msg]`` where each ``Msg``
      has ``.content`` (str) + ``.name`` (sender) + ``.role``. AgentScope's
      working memory is non-semantic by default — we filter by substring.
    * AgentScope ``Msg`` constructor signature varies between subclasses
      and across versions. Our remember path tries the documented
      ``(name=..., content=..., role=...)`` keyword shape; if that fails
      we fall back to positional ``("amplifier", text, "system")``; if
      THAT fails we log WARNING and return.
    * Lazy import: ``agentscope`` is NEVER imported at module top — only
      inside ``detect()`` via ``importlib.util.find_spec``.
"""
from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.types import RecalledPattern

if TYPE_CHECKING:  # pragma: no cover
    from agent_amplifier.types import Outcome

LOG = logging.getLogger("agent_amplifier.adapters.agentscope")

_PER_CHUNK_BYTES: int = 4096


class AgentScopeAdapter(AdapterBase):
    """Adapter for AgentScope working-memory or long-term-memory instances."""

    framework_name: ClassVar[str] = "agentscope"
    HOST_NAME: ClassVar[str] = "agentscope"
    version: ClassVar[str] = "1.0.0"

    def __init__(
        self,
        memory: Any,
        *,
        kernel: Any = None,
    ) -> None:
        """Bind a user-supplied AgentScope memory instance.

        Args:
            memory: Any object with a ``get_memory()`` method returning
                ``list[Msg]``. Typically an ``InMemoryMemory`` or any
                ``MemoryBase`` subclass; tests pass mocks with the same
                shape.
            kernel: Optional kernel reference for ``AdapterBase.__init__``.
        """
        super().__init__(kernel=kernel)
        self._memory = memory

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls) -> bool:
        """True iff ``agentscope`` is importable. Lazy: never imports it."""
        try:
            import importlib.util

            return importlib.util.find_spec("agentscope") is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False
        except Exception:  # pragma: no cover
            return False

    # ------------------------------------------------------------------
    # required AdapterBase abstract methods
    # ------------------------------------------------------------------

    def install(self) -> None:  # pragma: no cover - exercised by tests
        self._mark_installed()

    def uninstall(self) -> None:  # pragma: no cover - exercised by tests
        self._mark_uninstalled()

    def on_before_step(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover - identity
        return context

    def on_after_step(
        self,
        context: dict[str, Any],
        result: dict[str, Any] | str,
    ) -> dict[str, Any]:  # pragma: no cover - identity
        return {"action": "continue"}

    # ------------------------------------------------------------------
    # memory plane (.5.1)
    # ------------------------------------------------------------------

    def default_memory_recall(
        self, query: str, limit: int = 3
    ) -> list[RecalledPattern]:
        """Call ``memory.get_memory()`` and filter by ``query`` substring.

        AgentScope's working memory is non-semantic by default — recall is
        a recency-ordered slice with optional substring filter.

        Errors swallowed: any exception logs WARNING and returns ``[]``.
        """
        try:
            messages = self._memory.get_memory()
        except Exception as exc:
            LOG.warning("agentscope recall: get_memory failed: %r", exc)
            return []

        if messages is None:
            return []

        # bound iteration with itertools.islice so a
        # generator-backed memory cannot be fully materialized.  We multiply
        # ``limit`` by a small fan-in factor because the keyword filter
        # below may discard items, but we still cap at a hard ceiling.
        try:
            msg_iter = iter(messages)
        except TypeError as exc:
            LOG.warning(
                "agentscope recall: get_memory returned non-iterable: %r", exc
            )
            return []
        # Hard ceiling: filter pre-cap ratio of 8 then truncate again later.
        msg_list = list(itertools.islice(msg_iter, limit * 8))

        q_lower = (query or "").lower()
        out: list[RecalledPattern] = []
        for msg in msg_list:
            text = self._stringify(msg)
            if not text:
                continue
            if q_lower and q_lower not in text.lower():
                continue
            name = self._get_name(msg)
            role = self._get_role(msg)
            tags: tuple[str, ...] = (role,) if role else ()
            out.append(
                RecalledPattern(
                    text=text[:_PER_CHUNK_BYTES],
                    tags=tags,
                    source=f"{self.HOST_NAME}:{name or 'memory'}",
                )
            )
        # Recency: take the last `limit` matches (AgentScope returns
        # chronological order, oldest first).
        if len(out) > limit:
            return out[-limit:]
        return out

    def default_memory_remember(self, outcome: Outcome) -> None:
        """Append an amplifier ``Msg`` to the working memory.

        Tries multiple ``Msg`` constructor shapes because AgentScope's
        ``Msg`` has varied across releases:
            1. ``Msg(name="amplifier", content=..., role="system")`` (kwargs)
            2. ``Msg("amplifier", content, "system")`` (positional)
            3. dict fallback if ``Msg`` import fails
        Fire-and-forget: any failure logs WARNING and returns.
        """
        if not hasattr(self._memory, "add"):
            LOG.warning(
                "agentscope remember: memory has no .add method (got %r)",
                type(self._memory).__name__,
            )
            return

        text = (outcome.query or "")[:_PER_CHUNK_BYTES]
        msg_obj = self._build_msg(text)
        if msg_obj is None:
            LOG.warning(
                "agentscope remember: could not construct Msg; falling back to dict"
            )
            msg_obj = {
                "name": "amplifier",
                "content": text,
                "role": "system",
            }

        try:
            self._memory.add(msg_obj)
        except Exception as exc:
            LOG.warning("agentscope remember: memory.add failed: %r", exc)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _stringify(msg: Any) -> str:
        """Extract text content from an AgentScope Msg (or compatible dict)."""
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for blk in content:
                if isinstance(blk, str):
                    parts.append(blk)
                elif isinstance(blk, dict):
                    txt = blk.get("text") or blk.get("content")
                    if isinstance(txt, str):
                        parts.append(txt)
            if parts:
                return "\n".join(parts)
        if isinstance(msg, dict):
            ctn = msg.get("content")
            if isinstance(ctn, str):
                return ctn
        if isinstance(msg, str):
            return msg
        return ""

    @staticmethod
    def _get_name(msg: Any) -> str:
        name = getattr(msg, "name", None)
        if isinstance(name, str):
            return name
        if isinstance(msg, dict):
            n = msg.get("name")
            if isinstance(n, str):
                return n
        return ""

    @staticmethod
    def _get_role(msg: Any) -> str:
        role = getattr(msg, "role", None)
        if isinstance(role, str):
            return role
        if isinstance(msg, dict):
            r = msg.get("role")
            if isinstance(r, str):
                return r
        return ""

    @staticmethod
    def _build_msg(text: str) -> Any:
        """Build an AgentScope ``Msg`` instance, tolerating signature drift.

        Returns ``None`` if every attempt fails — caller falls back to dict.
        """
        try:
            from agentscope.message import Msg  # type: ignore[import-not-found]
        except ImportError:
            return None
        except Exception:  # pragma: no cover - extremely defensive
            return None

        # Attempt 1: kwargs shape (current docs).
        try:
            return Msg(name="amplifier", content=text, role="system")
        except TypeError:
            pass
        except Exception:  # pragma: no cover - upstream constructor bug
            return None

        # Attempt 2: positional fallback.
        try:
            return Msg("amplifier", text, "system")
        except TypeError:
            return None
        except Exception:  # pragma: no cover - upstream constructor bug
            return None
