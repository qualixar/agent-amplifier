# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""LangChainAdapter — memory-based binding for LangChain.

Targets the ``BaseMemory`` and ``BaseChatMessageHistory`` APIs — NOT
``BaseCheckpointSaver`` (that's LangGraph's; see ``langgraph.py``).

Design notes:
    * The user supplies EITHER a ``BaseMemory`` instance (wraps
      ``load_memory_variables`` / ``save_context``) OR a
      ``BaseChatMessageHistory`` instance (wraps ``messages`` property /
      ``add_message``). Both are duck-typed — no isinstance checks — so
      users on any LangChain version that keeps the same shape work, AND
      tests inject plain mocks.
    * Lazy import: ``langchain`` is NEVER imported at module top — only
      inside ``detect()`` via ``importlib.util.find_spec``.
    * Remember writes via ``save_context`` (BaseMemory) or ``add_message``
      (BaseChatMessageHistory). Both are fire-and-forget; errors log at
      WARNING and return.

Thread-safety contract ():
    The user MUST supply a thread-safe memory object. The adapter does NOT
    serialize access. ``ConversationBufferMemory`` and other built-in
    LangChain memory classes are NOT thread-safe by default — callers using
    them in concurrent settings should wrap with a lock or use a
    thread-safe backing store (Redis, Postgres).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.types import RecalledPattern

if TYPE_CHECKING:  # pragma: no cover
    from agent_amplifier.types import Outcome

LOG = logging.getLogger("agent_amplifier.adapters.langchain")

_PER_CHUNK_BYTES: int = 4096


class LangChainAdapter(AdapterBase):
    """Adapter for LangChain (BaseMemory / BaseChatMessageHistory)."""

    framework_name: ClassVar[str] = "langchain"
    HOST_NAME: ClassVar[str] = "langchain"
    version: ClassVar[str] = "1.0.0"

    def __init__(
        self,
        memory: Any,
        *,
        memory_key: str = "history",
        input_key: str = "input",
        kernel: Any = None,
    ) -> None:
        """Bind a user-supplied LangChain memory object.

        Args:
            memory: A ``BaseMemory`` instance (has ``load_memory_variables``
                and ``save_context``) OR a ``BaseChatMessageHistory``
                instance (has ``messages`` property and ``add_message``).
                Duck-typed for version resilience.
            memory_key: The key used to extract text from the dict returned
                by ``load_memory_variables``. Defaults to ``"history"``.
            input_key: The key name for the input dict passed to
                ``save_context``. Defaults to ``"input"``.
            kernel: Optional kernel reference for ``AdapterBase.__init__``.
        """
        super().__init__(kernel=kernel)
        self._memory = memory
        self._memory_key = memory_key
        self._input_key = input_key
        self._mode = self._detect_mode(memory)

    # ------------------------------------------------------------------
    # mode detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_mode(memory: Any) -> str:
        """Determine whether ``memory`` is BaseMemory or BaseChatMessageHistory.

        Returns ``"base_memory"`` if ``load_memory_variables`` is callable,
        ``"chat_history"`` if ``messages`` exists and ``add_message`` is
        callable, or ``"unknown"`` otherwise.
        """
        if callable(getattr(memory, "load_memory_variables", None)):
            return "base_memory"
        if hasattr(memory, "messages") and callable(
            getattr(memory, "add_message", None)
        ):
            return "chat_history"
        return "unknown"

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls) -> bool:
        """True iff ``langchain`` or ``langchain_core`` is importable."""
        try:
            import importlib.util

            for pkg in ("langchain", "langchain_core"):
                if importlib.util.find_spec(pkg) is not None:
                    return True
            return False
        except (ImportError, ValueError, ModuleNotFoundError):
            return False
        except Exception:  # pragma: no cover - defensive
            return False

    # ------------------------------------------------------------------
    # lifecycle (framework adapter = no persistent hooks)
    # ------------------------------------------------------------------

    def install(self) -> None:  # pragma: no cover
        self._mark_installed()

    def uninstall(self) -> None:  # pragma: no cover
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
    # memory plane
    # ------------------------------------------------------------------

    def default_memory_recall(
        self, query: str, limit: int = 3
    ) -> list[RecalledPattern]:
        """Read LangChain memory and return matching patterns.

        BaseMemory path:
            Calls ``load_memory_variables({"input": query})`` and extracts
            the string at ``memory_key``. Splits on double-newline into
            chunks. Naive substring match against ``query``.

        BaseChatMessageHistory path:
            Reads ``messages`` property. Extracts ``.content`` from each
            message object. Naive substring match against ``query``.

        Errors are swallowed: any exception logs at WARNING and returns [].
        """
        tags: tuple[str, ...] = ("langchain",)
        source = f"{self.HOST_NAME}:memory"

        try:
            if self._mode == "base_memory":
                return self._recall_from_base_memory(query, limit, tags, source)
            if self._mode == "chat_history":
                return self._recall_from_chat_history(query, limit, tags, source)
            LOG.warning("langchain recall: unknown memory mode %r", self._mode)
            return []
        except Exception as exc:
            LOG.warning("langchain recall failed: %r", exc)
            return []

    def default_memory_remember(self, outcome: Outcome) -> None:
        """Persist the amplifier outcome to LangChain memory.

        BaseMemory path:
            Calls ``save_context({"input": summary}, {"output": ""})``.

        BaseChatMessageHistory path:
            Calls ``add_message(HumanMessage(content=summary))``.
            Uses a lazy import for ``HumanMessage`` to avoid hard dep.
            Falls back to ``add_user_message`` if ``add_message`` fails.

        Errors are swallowed: any exception logs at WARNING and returns.
        """
        summary = getattr(outcome, "summary", None) or str(outcome)
        try:
            if self._mode == "base_memory":
                self._memory.save_context(
                    {self._input_key: summary}, {"output": ""}
                )
            elif self._mode == "chat_history":
                self._remember_to_chat_history(summary)
            else:
                LOG.warning(
                    "langchain remember: unknown memory mode %r", self._mode
                )
        except Exception as exc:
            LOG.warning("langchain remember failed: %r", exc)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _recall_from_base_memory(
        self,
        query: str,
        limit: int,
        tags: tuple[str, ...],
        source: str,
    ) -> list[RecalledPattern]:
        """Extract patterns from BaseMemory.load_memory_variables."""
        variables = self._memory.load_memory_variables(
            {self._input_key: query}
        )
        raw = variables.get(self._memory_key, "")
        text = raw if isinstance(raw, str) else str(raw)
        if not text.strip():
            return []

        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        q_lower = (query or "").lower()
        results: list[RecalledPattern] = []
        for chunk in reversed(chunks):
            if q_lower and q_lower not in chunk.lower():
                continue
            results.append(
                RecalledPattern(
                    text=chunk[:_PER_CHUNK_BYTES],
                    tags=tags,
                    source=source,
                )
            )
            if len(results) >= limit:
                break
        results.reverse()
        return results

    def _recall_from_chat_history(
        self,
        query: str,
        limit: int,
        tags: tuple[str, ...],
        source: str,
    ) -> list[RecalledPattern]:
        """Extract patterns from BaseChatMessageHistory.messages."""
        messages = self._memory.messages
        if not messages:
            return []

        q_lower = (query or "").lower()
        results: list[RecalledPattern] = []
        for msg in reversed(messages):
            text = self._stringify_message(msg)
            if not text:
                continue
            if q_lower and q_lower not in text.lower():
                continue
            results.append(
                RecalledPattern(
                    text=text[:_PER_CHUNK_BYTES],
                    tags=tags,
                    source=source,
                )
            )
            if len(results) >= limit:
                break
        results.reverse()
        return results

    def _remember_to_chat_history(self, summary: str) -> None:
        """Write to BaseChatMessageHistory via add_message or add_user_message."""
        try:
            import importlib.util

            if importlib.util.find_spec("langchain_core") is not None:  # pragma: no cover
                from langchain_core.messages import HumanMessage  # type: ignore[import-not-found]

                self._memory.add_message(HumanMessage(content=summary))
                return
        except (ImportError, ModuleNotFoundError):
            pass
        if callable(getattr(self._memory, "add_user_message", None)):
            self._memory.add_user_message(summary)
        elif callable(getattr(self._memory, "add_message", None)):
            self._memory.add_message(summary)
        else:
            LOG.warning("langchain remember: no add_message method found")

    @staticmethod
    def _stringify_message(msg: Any) -> str:
        """Convert a message object to string (same logic as LangGraphAdapter)."""
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
