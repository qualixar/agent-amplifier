# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Agent Amplifier — shared immutable types.

Stdlib-only. All types are frozen dataclasses with ``__slots__`` for
immutability and memory efficiency. Enums are used for closed value sets
that other modules exhaustively match against.


    * ``AmplifierEvent.ON_BUDGET_LOW``
    * ``ObservabilityCallback`` type alias
    * ``AmplifierConfig.observability_callback``
    * ``AmplifierConfig.escalate_low_confidence``


    * ``RecalledPattern`` — rich return type for the memory plane recall
      callback (replaces the SLM-coupled version that lived in slm_bridge).
    * ``Outcome`` — frozen aggregate of a session's amplification result;
      consumed by ``memory_remember`` callbacks.


    * ``AmplifierConfig.recall_limit`` — bounds the kernel's per-call recall
      batch size (.6.M ``_resolve_recall(query, limit=...)``).
      Default ``3``; validated to ``[1, 100]``. Without this field the kernel
      previously hardcoded ``3`` (see audit FINDING H6) — making the cap
      configurable closes the gap without changing default behavior.

Invariants enforced in ``__post_init__`` are belt-and-suspenders alongside
type checks; runtime input may come from TOML or external adapters and we
do NOT trust it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any

# ---------------------------------------------------------------------------
# 0. Public type aliases (V2.0)
# ---------------------------------------------------------------------------

#: An observability sink. The kernel and TokenBudgetController call this on
#: every event. Implementations MUST complete in <5 ms or push to a queue
#: (e.g. ``queue.Queue``). Exceptions raised by the callback are swallowed
#: by the caller — see .
ObservabilityCallback = Callable[["AmplifierEvent", dict[str, Any]], None]


# ---------------------------------------------------------------------------
# 1. Enumerations (closed value sets)
# ---------------------------------------------------------------------------


class AmplifierEvent(str, Enum):
    """Lifecycle events emitted by the kernel.

    Mirrors MASTER-PLAN.md §3 lifecycle events. The ``str`` mixin makes the
    member directly JSON-serializable as its string value.

    V2.0 added: ``ON_BUDGET_LOW`` (per .11).
    """

    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_ITERATION = "on_iteration"
    ON_CONVERGE = "on_converge"
    ON_DRIFT = "on_drift"
    # NEW V2.0 — fired at 70/80/90 % budget crossings.
    ON_BUDGET_LOW = "on_budget_low"
    # Fired at the 100 % budget crossing.
    ON_BUDGET_HIT = "on_budget_hit"


class EffortLevel(str, Enum):
    """Coarse complexity tier produced by the effort router (IP-2).

    Ordering is conceptual (minimal < low < ... < max) but we deliberately
    do NOT make it comparable — comparisons live in ``effort_router`` via
    an explicit rank table to keep ordering changes localized.
    """

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


class PhaseIndex(IntEnum):
    """Iteration phase. ``IntEnum`` so ``current + 1`` is legal arithmetic.

    Values are stable: do not renumber without a migration. Any persisted
    state (e.g. SLM records) keys on the integer.
    """

    EXPLORE = 0
    EVALUATE = 1
    EXECUTE = 2
    VERIFY = 3
    REFINE = 4


class ConvergenceState(str, Enum):
    """Output stability classification from the convergence detector (IP-4)."""

    IMPROVING = "improving"
    STAGNANT = "stagnant"
    OSCILLATING = "oscillating"
    CONVERGED = "converged"


class BudgetMode(str, Enum):
    """Token-budget posture (IP-10)."""

    AUTO = "auto"
    MINIMAL = "minimal"
    BALANCED = "balanced"
    UNLIMITED = "unlimited"


# ---------------------------------------------------------------------------
# 2. Frozen dataclasses (data carriers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskClassification:
    """Output of the effort router (IP-2).

    Includes ``confidence: float`` and ``matched_signals: tuple[str, ...]``
    for explainability.

    Invariants:
        * ``estimated_tokens >= 0``
        * ``domain`` is non-empty (router returns ``"general"`` if unknown)
        * ``0.0 <= confidence <= 1.0``
    """

    complexity: EffortLevel
    domain: str
    estimated_tokens: int
    confidence: float = 0.5
    matched_signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.estimated_tokens < 0:
            raise ValueError(
                f"estimated_tokens must be >= 0, got {self.estimated_tokens}"
            )
        if not self.domain:
            raise ValueError("domain must be a non-empty string")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0,1], got {self.confidence}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "complexity": self.complexity.value,
            "domain": self.domain,
            "estimated_tokens": self.estimated_tokens,
            "confidence": self.confidence,
            "matched_signals": list(self.matched_signals),
        }


@dataclass(frozen=True, slots=True)
class QualityScore:
    """One iteration's quality measurement.

    Invariants:
        * ``0.0 <= score <= 1.0``
        * ``-1.0 <= delta_from_previous <= 1.0``
        * ``iteration >= 0``
    """

    score: float
    delta_from_previous: float
    iteration: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0,1], got {self.score}")
        if not -1.0 <= self.delta_from_previous <= 1.0:
            raise ValueError(
                "delta_from_previous must be in [-1,1], got "
                f"{self.delta_from_previous}"
            )
        if self.iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {self.iteration}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecalledPattern:
    """One unit of recalled memory, normalized to amplifier semantics (V2.1).

    Replaces the SLM-coupled ``RecalledPattern`` that lived in
    ``slm_bridge.py`` V2.0. Adapters MUST construct this dataclass when
    implementing ``default_memory_recall`` (see .5.5).

    Authors who don't have score / tags / source / metadata simply pass
    defaults — the cost is one constructor call vs one string append.
    Authors who DO have richer signal (LangGraph checkpointer, Semantic
    Kernel VectorStoreCollection, AgentScope memory) populate the fields,
    enabling kernel-level ranking and observability.

    The kernel applies ``recall_safety.apply_recall_safety()`` to ``text``
    once it receives the pattern. Adapters return raw text; the kernel
    handles capping + neutralization + smuggling-signal logging.

    H2: ``metadata`` is wrapped in ``MappingProxyType`` in
    ``__post_init__`` so callers cannot mutate the underlying dict after
    construction. Without this, ``dataclasses.replace(pat, text=safe)``
    would share the same dict reference across kernel-internal copies and
    user code, allowing one site to surprise another.
    """

    text: str
    """The recalled content. Capped to ``MAX_RECALLED_TEXT_BYTES`` and
    neutralized by ``recall_safety.apply_recall_safety()`` once the kernel
    receives it."""

    score: float = 0.0
    """Relevance score, 0.0–1.0. Default 0.0 means "adapter doesn't know".
    Adapters that return ranked results (vector search, BM25) populate this."""

    tags: tuple[str, ...] = ()
    """Adapter-specific tags. Examples: ``("project-rule",)`` for Cursor MDC
    with ``alwaysApply=true``; ``("checkpoint", "thread:42")`` for LangGraph;
    ``("user-rule",)`` for Cursor user-scoped rules."""

    source: str = ""
    """Provenance string for debugging + observability. Format:
    ``"<adapter-name>:<sub-source>"``. Examples:
    ``"claude-code:CLAUDE.md"``, ``"cursor:.cursor/rules/python.mdc"``,
    ``"langgraph:thread-42"``, ``"agentscope:TemporaryMemory"``."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Adapter-specific extensibility escape hatch. NOT used by the kernel
    today. Reserved for adapter-specific data that doesn't fit other fields.

    H2: frozen via ``MappingProxyType`` in ``__post_init__``;
    attempts to mutate raise ``TypeError``.
    """

    def __post_init__(self) -> None:
        # validate field types so a malformed adapter
        # callback cannot persist garbage values.  We accept ``int`` for
        # ``score`` because Python literals like ``0`` / ``1`` are common
        # for "no rank" / "exact match" — coerced to float on the way out.
        if not isinstance(self.text, str):
            raise TypeError(
                f"RecalledPattern.text must be str, got {type(self.text).__name__}"
            )
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError(
                f"RecalledPattern.score must be a number, got {type(self.score).__name__}"
            )
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError(
                f"RecalledPattern.score must be in [0,1], got {self.score}"
            )
        if not isinstance(self.tags, tuple) or any(
            not isinstance(t, str) for t in self.tags
        ):
            raise TypeError(
                "RecalledPattern.tags must be a tuple of str"
            )
        if not isinstance(self.source, str):
            raise TypeError(
                f"RecalledPattern.source must be str, got {type(self.source).__name__}"
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                f"RecalledPattern.metadata must be a Mapping, got {type(self.metadata).__name__}"
            )
        # Freeze metadata regardless of what was passed in. The dataclass
        # is frozen, so we use object.__setattr__ to bypass the
        # dataclass-level setattr block. Defensive copy via dict() so the
        # caller cannot mutate the source dict and see it reflected here.
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(
                self,
                "metadata",
                MappingProxyType(dict(self.metadata)),
            )


@dataclass(frozen=True, slots=True)
class Outcome:
    """Frozen aggregate of one amplification session's result (V2.1).

    Passed to ``memory_remember`` callbacks (and ``AdapterBase.default_memory_remember``)
    so the host's native memory can persist a learning record. The kernel
    builds this in ``finalize()`` from per-session state; it is NEVER mutated.

    Fields (all required):
        * ``query`` — the goal anchor text the session was working from.
        * ``effort`` — the classified effort tier (MINIMAL..MAX).
        * ``iterations`` — total iterations completed (>=0).
        * ``quality`` — final quality score, 0.0..1.0.
        * ``converged`` — whether convergence was reached.
        * ``tokens_used`` — total tokens consumed across the session.
    """

    query: str
    effort: EffortLevel
    iterations: int
    quality: float
    converged: bool = False
    tokens_used: int = 0

    def __post_init__(self) -> None:
        # validate runtime types.  ``bool`` is an int
        # subclass in Python, so we explicitly reject it for ``int`` fields
        # to avoid the surprise that ``Outcome(iterations=True)`` would
        # otherwise accept silently.
        if not isinstance(self.query, str):
            raise TypeError(
                f"Outcome.query must be str, got {type(self.query).__name__}"
            )
        if not isinstance(self.effort, EffortLevel):
            raise TypeError(
                f"Outcome.effort must be EffortLevel, got {type(self.effort).__name__}"
            )
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise TypeError(
                f"Outcome.iterations must be int, got {type(self.iterations).__name__}"
            )
        if isinstance(self.quality, bool) or not isinstance(self.quality, (int, float)):
            raise TypeError(
                f"Outcome.quality must be a number, got {type(self.quality).__name__}"
            )
        if not isinstance(self.converged, bool):
            raise TypeError(
                f"Outcome.converged must be bool, got {type(self.converged).__name__}"
            )
        if isinstance(self.tokens_used, bool) or not isinstance(self.tokens_used, int):
            raise TypeError(
                f"Outcome.tokens_used must be int, got {type(self.tokens_used).__name__}"
            )
        if self.iterations < 0:
            raise ValueError(
                f"iterations must be >= 0, got {self.iterations}"
            )
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(
                f"quality must be in [0,1], got {self.quality}"
            )
        if self.tokens_used < 0:
            raise ValueError(
                f"tokens_used must be >= 0, got {self.tokens_used}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "effort": self.effort.value,
            "iterations": self.iterations,
            "quality": self.quality,
            "converged": self.converged,
            "tokens_used": self.tokens_used,
        }


@dataclass(frozen=True, slots=True)
class AmplifierConfig:
    """User-facing configuration. Loaded by ``config.load_config()``.

    Defaults match PHASE-0-KERNEL.md Module 2.

    Invariants enforced in ``__post_init__``:
        * ``1 <= max_iterations <= 10`` (hard cap; LTI safety per IP-4)
        * ``0.0 <= convergence_threshold <= 1.0``
        * ``goal_reinjection_interval >= 1``
        * ``effort_router in _ALLOWED_ROUTERS``
        * ``tool_selector in _ALLOWED_SELECTORS``
        * ``budget_mode`` is a ``BudgetMode`` enum (no string coercion here
          — A-4 strict)
        * ``observability_callback`` is callable or ``None``
        * ``escalate_low_confidence`` is exactly ``bool``


        * ``observability_callback`` — Optional callable.
          NEVER serialized in ``to_dict`` — emits ``"<callable>"`` sentinel.
          Implementations MUST complete in <5 ms or push to ``queue.Queue``.
        * ``escalate_low_confidence`` —
          default ``False`` (safe). Power users opt in to escalate HIGH+
          tiers even when classifier confidence < 0.6.
    """

    max_iterations: int = 4
    convergence_threshold: float = 0.95
    budget_mode: BudgetMode = BudgetMode.AUTO
    goal_reinjection_interval: int = 5
    effort_router: str = "heuristic"
    tool_selector: str = "heuristic"
    # --- V2.0 additions ---
    observability_callback: ObservabilityCallback | None = None
    escalate_low_confidence: bool = False
    # --- addition (H6) ---
    # Bound on the kernel's per-call recall batch (.6.M).
    # Default 3 matches the prior hardcoded value in `kernel._resolve_recall`.
    # Validated to [1, 100] — see __post_init__.
    recall_limit: int = 3
    # --- Dashboard backend additions ---
    # Persisted by the FastAPI dashboard. These are intentionally shallow
    # TOML arrays so the existing config loader remains the only schema gate.
    disabled_ips: tuple[str, ...] = ()
    ip_order: tuple[str, ...] = ()
    # --- IP-8 v2: user-selected persona flavor (slug) ---
    # Resolved at kernel init via ``persona_docs.resolve_flavor()``.
    # Defaults to ``senior-engineer`` — identical to the prior behavior where
    # LEVEL_0 was always used for iteration 0.
    persona: str = "senior-engineer"

    def __post_init__(self) -> None:
        if not 1 <= self.max_iterations <= 10:
            raise ValueError(
                f"max_iterations must be in [1,10], got {self.max_iterations}"
            )
        if not 0.0 <= self.convergence_threshold <= 1.0:
            raise ValueError(
                "convergence_threshold must be in [0,1], got "
                f"{self.convergence_threshold}"
            )
        if self.goal_reinjection_interval < 1:
            raise ValueError(
                "goal_reinjection_interval must be >= 1, got "
                f"{self.goal_reinjection_interval}"
            )
        if not isinstance(self.budget_mode, BudgetMode):
            raise TypeError(
                "budget_mode must be a BudgetMode enum, got "
                f"{type(self.budget_mode).__name__}"
            )
        if self.effort_router not in _ALLOWED_ROUTERS:
            raise ValueError(
                f"effort_router must be one of {_ALLOWED_ROUTERS}, "
                f"got {self.effort_router!r}"
            )
        if self.tool_selector not in _ALLOWED_SELECTORS:
            raise ValueError(
                f"tool_selector must be one of {_ALLOWED_SELECTORS}, "
                f"got {self.tool_selector!r}"
            )

        # of classes that define ``__call__``. ``None`` is the default.
        if self.observability_callback is not None and not callable(
            self.observability_callback
        ):
            raise TypeError(
                "observability_callback must be callable or None, got "
                f"{type(self.observability_callback).__name__}"
            )
        # ``bool`` is a subclass of ``int``; we want the strict check.
        if not isinstance(self.escalate_low_confidence, bool):
            raise TypeError(
                "escalate_low_confidence must be bool, got "
                f"{type(self.escalate_low_confidence).__name__}"
            )
        # — H6 fix: bound the kernel's recall batch size. Strict
        # int check refuses ``True`` (bool is an int subclass) so TOML
        # quirks can't silently coerce booleans into 1.
        if isinstance(self.recall_limit, bool) or not isinstance(
            self.recall_limit, int
        ):
            raise TypeError(
                "recall_limit must be int, got "
                f"{type(self.recall_limit).__name__}"
            )
        if not 1 <= self.recall_limit <= 100:
            raise ValueError(
                f"recall_limit must be in [1, 100], got {self.recall_limit}"
            )
        if not isinstance(self.disabled_ips, tuple) or any(
            not isinstance(ip_id, str) or not ip_id for ip_id in self.disabled_ips
        ):
            raise TypeError("disabled_ips must be a tuple of non-empty str")
        if not isinstance(self.ip_order, tuple) or any(
            not isinstance(ip_id, str) or not ip_id for ip_id in self.ip_order
        ):
            raise TypeError("ip_order must be a tuple of non-empty str")
        # IP-8 v2: validate persona slug.
        # Same regex as custom_personas._NAME_RE — enforces [a-z][a-z0-9_-]{0,63}.
        if not re.match(r"^[a-z][a-z0-9_-]{0,63}$", self.persona):
            raise ValueError(
                f"persona must be a valid slug matching [a-z][a-z0-9_-]{{0,63}}, "
                f"got {self.persona!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization.

        callables are NEVER serialized. We emit the sentinel string
        ``"<callable>"`` if a callback is set, else ``None``. This keeps
        ``to_dict()`` round-trippable with the SLM CLI argv layer ()
        without leaking function objects or their reprs (which can carry
        sensitive lambda closures).
        """
        return {
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "budget_mode": self.budget_mode.value,
            "goal_reinjection_interval": self.goal_reinjection_interval,
            "effort_router": self.effort_router,
            "tool_selector": self.tool_selector,
            "observability_callback": (
                "<callable>" if self.observability_callback is not None else None
            ),
            "escalate_low_confidence": self.escalate_low_confidence,
            "recall_limit": self.recall_limit,
            "disabled_ips": list(self.disabled_ips),
            "ip_order": list(self.ip_order),
            "persona": self.persona,
        }


# ---------------------------------------------------------------------------
# 3. Module-level constants (immutable;  anti-drift)
# ---------------------------------------------------------------------------

# Tuples — already immutable. No ``MappingProxyType`` needed.
_ALLOWED_ROUTERS: tuple[str, ...] = ("heuristic",)
_ALLOWED_SELECTORS: tuple[str, ...] = ("heuristic",)

# A read-only mapping of allowed ``AmplifierConfig`` field names — used by
# ``config.validate_config`` to detect unknown keys (A-3 strict reject).
# ``MappingProxyType`` ensures direct assignment raises ``TypeError``.
# (Backing-dict mutation is footgunned in §0.4 — never expose the backing.)
_ALLOWED_CONFIG_FIELDS: Mapping[str, type] = MappingProxyType(
    {
        "max_iterations": int,
        "convergence_threshold": float,
        "budget_mode": BudgetMode,
        "goal_reinjection_interval": int,
        "effort_router": str,
        "tool_selector": str,
        # ``object`` because we accept callables; the strict callable check
        # lives in ``AmplifierConfig.__post_init__``.
        "observability_callback": object,
        "escalate_low_confidence": bool,
        # — H6 fix. Validated to [1, 100] in __post_init__.
        "recall_limit": int,
        "disabled_ips": tuple,
        "ip_order": tuple,
        # IP-8 v2: user-selected persona slug.
        "persona": str,
    }
)


__all__ = [
    "AmplifierConfig",
    "AmplifierEvent",
    "BudgetMode",
    "ConvergenceState",
    "EffortLevel",
    "ObservabilityCallback",
    "Outcome",
    "PhaseIndex",
    "QualityScore",
    "RecalledPattern",
    "TaskClassification",
    # ``_ALLOWED_*`` are private — intentionally absent from ``__all__``.
]
