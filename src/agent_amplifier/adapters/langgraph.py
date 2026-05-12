# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""LangGraphAdapter — checkpointer-based memory binding for LangGraph.

.1 row 4; research:

Design notes:
    * The user supplies a ``BaseCheckpointSaver`` instance and a
      ``thread_id``. We do NOT instantiate a checkpointer — the user owns
      it and its lifecycle (`MemorySaver()`, `SqliteSaver.from_conn_string`,
      `PostgresSaver.from_conn_string`, etc.).
    * NOTE: we target the `BaseCheckpointSaver` shape stable since
      LangGraph 0.2 — `get_tuple(config) -> CheckpointTuple | None` and the
      `checkpoint["channel_values"]["messages"]` shape. If LangGraph
      changes either, only ``_extract_messages`` and the call to
      ``get_tuple`` need updating.
    * Lazy import: ``langgraph`` is NEVER imported at module top — only
      inside ``detect()`` via ``importlib.util.find_spec`` so users without
      the framework keep ``import agent_amplifier`` cheap and side-effect-free.
    * Remember is intentionally a no-op for V1: writing to a checkpointer
      directly is risky because the graph runtime owns checkpoint lifecycle
      (versioning, parent_config, channel_values invariants). Outcomes
      naturally land in the next checkpoint via the user's graph state
      updates. A V1.1 enhancement could write into an `InMemoryStore` /
      vector store layer if one is also supplied.

Thread-safety contract (H11):
    The user MUST supply a thread-safe ``BaseCheckpointSaver``. The adapter
    does NOT serialize access to the checkpointer — concurrent
    ``before_step`` / ``after_step`` calls on the same kernel will call
    ``get_tuple`` (and any future ``put`` paths) concurrently. In-process
    ``MemorySaver`` is thread-safe per LangGraph's own guarantees;
    ``SqliteSaver`` and ``PostgresSaver`` rely on the underlying connection's
    thread-safety mode (``SqliteSaver.from_conn_string`` defaults to
    check_same_thread=False). If you wrap a non-thread-safe saver, wrap it
    yourself with a lock, or use an async-native saver and the
    ``AsyncAgentAmplifier`` facade.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.types import RecalledPattern

if TYPE_CHECKING:  # pragma: no cover
    from agent_amplifier.types import Outcome

LOG = logging.getLogger("agent_amplifier.adapters.langgraph")

# Per-chunk soft cap. Half the kernel's MAX_RECALLED_TEXT_BYTES so a long
# checkpoint message doesn't dominate one recall batch.
_PER_CHUNK_BYTES: int = 4096


class LangGraphAdapter(AdapterBase):
    """Adapter for LangGraph (BaseCheckpointSaver-backed memory)."""

    framework_name: ClassVar[str] = "langgraph"
    HOST_NAME: ClassVar[str] = "langgraph"
    version: ClassVar[str] = "1.0.0"

    def __init__(
        self,
        checkpointer: Any,
        thread_id: str = "default",
        *,
        kernel: Any = None,
    ) -> None:
        """Bind a user-supplied LangGraph checkpointer + thread.

        Args:
            checkpointer: A ``BaseCheckpointSaver`` instance (e.g.
                ``MemorySaver``, ``SqliteSaver``, ``PostgresSaver``). We
                duck-type rather than ``isinstance``-check so users on a
                future LangGraph version that ships a new saver class still
                work, AND so tests can inject mocks without importing
                LangGraph.
            thread_id: The ``configurable.thread_id`` used to scope the
                checkpoint lookup. Defaults to ``"default"``.
            kernel: Optional kernel reference for ``AdapterBase.__init__``.
        """
        super().__init__(kernel=kernel)
        self._checkpointer = checkpointer
        self._thread_id = thread_id

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls) -> bool:
        """True iff ``langgraph`` is importable. Lazy: never imports it."""
        try:
            import importlib.util

            return importlib.util.find_spec("langgraph") is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False
        except Exception:  # pragma: no cover - extremely defensive
            return False

    # ------------------------------------------------------------------
    # required AdapterBase abstract methods (framework adapter = no hooks
    # at the in-process callback layer; LangGraph composes via graphs, not
    # via mutable callbacks the way Claude Code's CLI does).
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
        """Read the latest checkpoint for ``thread_id`` and return matching messages.

        Recall semantics (V1):
            * Look up the most recent checkpoint via ``get_tuple``.
            * Extract messages from ``checkpoint["channel_values"]["messages"]``.
            * Naive substring match (case-insensitive) against ``query``.
              Empty query returns the most recent ``limit`` messages.
            * Each message becomes one ``RecalledPattern`` with
              ``source = f"langgraph:thread-{self._thread_id}"`` and
              ``tags = ("checkpoint", f"thread:{self._thread_id}")``.

        Errors are swallowed: any exception from the checkpointer logs at
        WARNING and returns ``[]``.
        """
        config = {"configurable": {"thread_id": self._thread_id}}
        try:
            tup = self._checkpointer.get_tuple(config)
        except Exception as exc:
            LOG.warning(
                "langgraph recall: get_tuple failed for thread %s: %r",
                self._thread_id,
                exc,
            )
            return []

        if tup is None:
            return []

        messages = self._extract_messages(tup)
        if not messages:
            return []

        q_lower = (query or "").lower()
        tags: tuple[str, ...] = (
            "checkpoint",
            f"thread:{self._thread_id}",
        )
        source = f"{self.HOST_NAME}:thread-{self._thread_id}"
        # scan in reverse and stop early once ``limit``
        # matches are collected.  A long-lived graph thread used to scan
        # all N messages even when only the last 3 mattered; this makes
        # recall O(matches) on the recent tail rather than O(N) on the
        # full history.
        recent: list[RecalledPattern] = []
        for msg_text in reversed(messages):
            if q_lower and q_lower not in msg_text.lower():
                continue
            recent.append(
                RecalledPattern(
                    text=msg_text[:_PER_CHUNK_BYTES],
                    tags=tags,
                    source=source,
                )
            )
            if len(recent) >= limit:
                break
        # Restore chronological order (oldest first within the recent slice).
        recent.reverse()
        return recent

    def default_memory_remember(
        self, outcome: Outcome
    ) -> None:
        """V1: no-op.

        Direct ``checkpointer.put()`` calls are risky because the graph
        runtime owns checkpoint lifecycle (versioning, parent_config,
        channel_values invariants). Outcomes naturally land in the next
        checkpoint via the user's graph state updates. Documented in module
        docstring.
        """
        return None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_messages(tup: Any) -> list[str]:
        """Extract message text strings from a ``CheckpointTuple``.

        Tolerates several shapes:
            * ``tup.checkpoint["channel_values"]["messages"]`` (canonical)
            * ``tup.checkpoint`` is a Mapping or has ``__getitem__``
            * messages are objects with ``.content`` OR plain strings
              OR dicts with ``"content"`` key.

        Any structural mismatch returns ``[]`` (caller treats as "no recall").
        """
        try:
            checkpoint = getattr(tup, "checkpoint", None)
            if checkpoint is None:
                return []
            channel_values = checkpoint["channel_values"]
            messages = channel_values.get("messages", [])
        except (KeyError, TypeError, AttributeError):
            return []

        out: list[str] = []
        for msg in messages:
            text = LangGraphAdapter._stringify_message(msg)
            if text:
                out.append(text)
        return out

    @staticmethod
    def _stringify_message(msg: Any) -> str:
        """Convert a single message-ish object into a string.

        Order: ``.content`` attr → ``["content"]`` key → ``str(msg)``.
        """
        # Prefer the LangChain BaseMessage-style ``.content`` attribute.
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # LangChain content blocks (tool calls, multi-part). Join text bits.
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

        # Dict-shaped messages.
        if isinstance(msg, dict):
            ctn = msg.get("content")
            if isinstance(ctn, str):
                return ctn

        # Plain string.
        if isinstance(msg, str):
            return msg

        return ""
