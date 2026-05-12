# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""CrewAIAdapter — unified Memory binding for CrewAI.

.1 row 5; research:

Design notes:
    * The user supplies a ``Crew`` instance (or any object exposing
      ``.memory`` with ``.search()`` / ``.save()`` methods). We do NOT
      instantiate a Crew — the user owns its lifecycle.
    * NOTE: we target the **unified Memory** API documented in CrewAI
      late-2026 (``crew.memory.search(query, limit)`` returning a list of
      dicts with ``memory``/``content`` + ``score`` + ``metadata`` keys).
      If the upstream API renames the score key or restructures the dicts,
      the only place to update is ``_pluck_text`` / ``_pluck_score``.
    * Lazy import: ``crewai`` is NEVER imported at module top — only inside
      ``detect()`` via ``importlib.util.find_spec``.
    * Remember writes ``crew.memory.save(value=..., metadata={...})``.
      Fire-and-forget — any exception logs WARNING and returns.
"""
from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.types import RecalledPattern

if TYPE_CHECKING:  # pragma: no cover
    from agent_amplifier.types import Outcome

LOG = logging.getLogger("agent_amplifier.adapters.crewai")

_PER_CHUNK_BYTES: int = 4096


class CrewAIAdapter(AdapterBase):
    """Adapter for CrewAI (unified ``Memory`` class)."""

    framework_name: ClassVar[str] = "crewai"
    HOST_NAME: ClassVar[str] = "crewai"
    version: ClassVar[str] = "1.0.0"

    def __init__(
        self,
        crew: Any,
        *,
        kernel: Any = None,
    ) -> None:
        """Bind a user-supplied Crew (or memory-like object).

        Args:
            crew: A ``Crew`` instance with a ``.memory`` attribute. Duck-typed
                — any object whose ``.memory.search(query, limit)`` returns
                an iterable of dicts works. Tests pass mock objects directly.
            kernel: Optional kernel reference for ``AdapterBase.__init__``.
        """
        super().__init__(kernel=kernel)
        self._crew = crew

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls) -> bool:
        """True iff ``crewai`` is importable. Lazy: never imports it."""
        try:
            import importlib.util

            return importlib.util.find_spec("crewai") is not None
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
        """Call ``crew.memory.search(query, limit)`` and adapt to RecalledPattern.

        On any of these conditions, log WARNING and return ``[]``:
            * ``self._crew`` has no ``.memory`` attribute
            * ``crew.memory.search`` raises
            * ``crew.memory.search`` returns a non-iterable
        """
        memory = getattr(self._crew, "memory", None)
        if memory is None:
            LOG.warning(
                "crewai recall: crew has no .memory attribute (got %r)",
                type(self._crew).__name__,
            )
            return []

        try:
            raw = memory.search(query=query, limit=limit)
        except Exception as exc:
            LOG.warning("crewai recall: memory.search failed: %r", exc)
            return []

        # bound with itertools.islice — even though
        # CrewAI memory.search SHOULD respect ``limit`` server-side, a
        # buggy or adversarial provider can return a larger generator.
        try:
            raw_iter = iter(raw)
        except TypeError as exc:
            LOG.warning(
                "crewai recall: memory.search returned non-iterable %r: %r",
                type(raw).__name__,
                exc,
            )
            return []
        items = list(itertools.islice(raw_iter, limit))

        out: list[RecalledPattern] = []
        for item in items:
            text = self._pluck_text(item)
            if not text:
                continue
            score = self._pluck_score(item)
            metadata = self._pluck_metadata(item)
            out.append(
                RecalledPattern(
                    text=text[:_PER_CHUNK_BYTES],
                    score=score,
                    tags=("crew-memory",),
                    source=f"{self.HOST_NAME}:memory",
                    metadata=metadata,
                )
            )
        return out

    def default_memory_remember(self, outcome: Outcome) -> None:
        """Persist outcome via ``crew.memory.save(value=..., metadata=...)``.

        Fire-and-forget. Any exception logs WARNING and returns.
        """
        memory = getattr(self._crew, "memory", None)
        if memory is None:
            LOG.warning(
                "crewai remember: crew has no .memory attribute (got %r)",
                type(self._crew).__name__,
            )
            return
        try:
            memory.save(
                value=outcome.query,
                metadata={
                    "quality": outcome.quality,
                    "iterations": outcome.iterations,
                    "effort": outcome.effort.value,
                    "converged": outcome.converged,
                    "tokens_used": outcome.tokens_used,
                },
            )
        except Exception as exc:
            LOG.warning("crewai remember: memory.save failed: %r", exc)

    # ------------------------------------------------------------------
    # internals — duck-typed extractors for the CrewAI memory dict shape
    # ------------------------------------------------------------------

    @staticmethod
    def _pluck_text(item: Any) -> str:
        """Extract text from a CrewAI memory item.

        Tolerates: dict with ``"memory"``, ``"content"``, ``"value"``, or
        ``"text"`` keys; objects with ``.content`` / ``.text`` / ``.memory``
        attrs; raw strings.
        """
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("memory", "content", "value", "text"):
                v = item.get(key)
                if isinstance(v, str) and v:
                    return v
            return ""
        for attr in ("content", "text", "memory"):
            v = getattr(item, attr, None)
            if isinstance(v, str) and v:
                return v
        return ""

    @staticmethod
    def _pluck_score(item: Any) -> float:
        """Extract a relevance score 0..1.

        Default 0.0 if absent or non-numeric. Clamps to [0,1] so a misbehaving
        backend cannot violate the ``RecalledPattern.score`` invariant.
        """
        raw: Any = (
            item.get("score")
            if isinstance(item, dict)
            else getattr(item, "score", None)
        )
        if raw is None:
            return 0.0
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val

    @staticmethod
    def _pluck_metadata(item: Any) -> dict[str, Any]:
        """Extract metadata dict; default empty."""
        if isinstance(item, dict):
            md = item.get("metadata")
            if isinstance(md, dict):
                return dict(md)
        else:
            md = getattr(item, "metadata", None)
            if isinstance(md, dict):
                return dict(md)
        return {}
