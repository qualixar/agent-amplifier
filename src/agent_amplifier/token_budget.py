# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""IP-10 Cost-Bounded Amplification.

Per . Allocate, track, and enforce a token budget. Four
budget modes (MINIMAL / AUTO / BALANCED / UNLIMITED). Graceful exhaustion
(DECISIONS-LOCKED C-4): the kernel finalizes the current iteration and
returns the convergence detector's leader output.

Token counting:
    * V1.0 default:  ``estimated = max(1, len(text) // 4)``
    * V1.0 opt-in:   ``tiktoken`` ``cl100k_base`` (DECISIONS-LOCKED C-3)

V2.0 thread-safety contract:
    All public methods (``track``, ``track_text``,
    ``track_iteration_with_prefix``, ``allocate``, ``mark_iteration``,
    ``remaining``, ``should_stop_for_budget``, ``is_exhausted``, ``report``,
    ``reset``) are thread-safe under ``self._lock``.

V2.0 verifications inlined (Gemini-grounded, 2026-04-26):
    * ``threading.Lock`` is NOT reentrant. We never call public methods
      from inside the locked critical section. The only nested call is
      ``track()`` inside ``track_text()`` — and the design ensures
      ``track_text`` does NOT hold the lock while delegating.
    * ``functools.lru_cache`` on instance methods leaks ``self`` and never
      hits for variable iteration outputs (B-09 / ). We do NOT use
      it. Prefix amortization is via the ``injected_prefix_tokens``
      constructor parameter — counted ONCE at construction by the kernel.

Cross-LLD wiring (NOTE-04-D):
    The kernel passes ``observability_callback`` (from
    ``AmplifierConfig.observability_callback``) and
    ``injected_prefix_tokens`` (from the  fixed prompt prefix's
    pre-counted token count) into ``__init__``.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from agent_amplifier.types import (
    AmplifierEvent,
    BudgetMode,
    EffortLevel,
)

LOG = logging.getLogger("agent_amplifier.token_budget")


# ---------------------------------------------------------------------------
# Module constants (immutable)
# ---------------------------------------------------------------------------


_BASE_BUDGET_BY_EFFORT: Final[dict[EffortLevel, int]] = {
    EffortLevel.MINIMAL: 1_000,
    EffortLevel.LOW: 3_000,
    EffortLevel.MEDIUM: 8_000,
    EffortLevel.HIGH: 20_000,
    EffortLevel.MAX: 50_000,
}

_EFFORT_MULTIPLIER: Final[dict[EffortLevel, float]] = {
    EffortLevel.MINIMAL: 0.0,
    EffortLevel.LOW: 1.0,
    EffortLevel.MEDIUM: 2.0,
    EffortLevel.HIGH: 3.0,
    EffortLevel.MAX: 4.0,
}

#: emit a warning at every 10 % crossing past 70 %. Each level
#: fires exactly once per session via :class:`TokenBudgetController`.
_WARN_RATIO_THRESHOLDS: Final[tuple[float, ...]] = (0.70, 0.80, 0.90, 1.00)


#: Type alias for documentation; the canonical alias lives in
#: :mod:`agent_amplifier.types` (``ObservabilityCallback``). We re-state the
#: shape here without importing the alias to avoid a circular import.
_ObservabilityCallback = Callable[[AmplifierEvent, dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BudgetReport:
    """Snapshot of controller state at one moment.

    Immutable. Returned by :meth:`TokenBudgetController.report`.
    """

    mode: BudgetMode
    allocated: int
    used: int
    remaining: int
    iterations_completed: int
    exhausted: bool
    warnings_emitted: int
    use_tiktoken: bool
    encoding_name: str | None = None


def _select_encoding_name(model: str | None) -> str:
    """Pick the right tiktoken encoding name for a given model identifier.

    Returns ``"o200k_base"`` for modern frontier models (GPT-4o, GPT-5,
    Claude 3.5+, Claude 4.x / Opus 4.x) and ``"cl100k_base"`` for legacy
    GPT-3.5 / GPT-4 (non-Turbo) families. ``None`` or unknown input
    defaults to ``"cl100k_base"`` for backward compatibility with the
    pre-V1.0 hardcoded behaviour.

    Anthropic does not publish a tokenizer; ``o200k_base`` is the closest
    public proxy and is what most cost-estimation tooling adopts. The
    counts are approximate but materially better than ``len(text) // 4``
    for any modern model.
    """
    if not model:
        return "cl100k_base"
    m = model.lower()
    legacy_markers = ("gpt-3.5", "text-davinci", "davinci-002")
    if any(x in m for x in legacy_markers):
        return "cl100k_base"
    # gpt-4 (without -o or -turbo) is the legacy GPT-4 family.
    if "gpt-4" in m and "gpt-4o" not in m and "gpt-4-turbo" not in m:
        return "cl100k_base"
    return "o200k_base"


@dataclass(frozen=True, slots=True)
class BudgetExhaustedSignal:
    """Payload sent to ``observability_callback`` on every threshold
    crossing.

    Replaces V1's ``warnings.warn(BudgetExhaustedWarning, ...)``.
    """

    mode: BudgetMode
    allocated: int
    used: int
    ratio_used: float
    iterations_completed: int
    actionable_hint: str


# ---------------------------------------------------------------------------
# TokenBudgetController
# ---------------------------------------------------------------------------


class TokenBudgetController:
    """Stateful per-session controller. Thread-safe.

    See module docstring for token-counting modes and budget-mode formulas.
    """

    def __init__(
        self,
        mode: BudgetMode = BudgetMode.AUTO,
        *,
        max_tokens_override: int | None = None,
        use_tiktoken: bool = False,
        model: str | None = None,
        observability_callback: _ObservabilityCallback | None = None,
        injected_prefix_tokens: int = 0,
    ) -> None:
        self._mode = mode
        self._allocated: int = 0
        self._used: int = 0
        self._iterations_completed: int = 0
        self._exhausted: bool = False
        self._warnings_emitted: int = 0
        self._allocation_locked: bool = False
        self._max_override = max_tokens_override
        self._use_tiktoken = bool(use_tiktoken)
        self._tiktoken_encoder: Any = None
        self._encoding_name: str | None = None
        self._observability_callback = observability_callback
        self._injected_prefix_tokens = max(0, int(injected_prefix_tokens))
        self._warning_levels_fired: set[float] = set()

        self._lock = threading.Lock()

        if self._use_tiktoken:
            try:
                import tiktoken  # type: ignore[import-not-found]

                encoding_name = _select_encoding_name(model)
                self._tiktoken_encoder = tiktoken.get_encoding(encoding_name)
                self._encoding_name = encoding_name
            except Exception:
                self._use_tiktoken = False

                LOG.warning(
                    "tiktoken not available; falling back to char/4 estimation"
                )

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(self, effort: EffortLevel) -> int:
        """Compute total budget. Idempotent within a session."""
        with self._lock:
            return self._allocate_locked(effort)

    def _allocate_locked(self, effort: EffortLevel) -> int:
        """Caller holds lock."""
        if self._allocation_locked:
            return self._allocated

        if self._max_override is not None and self._max_override > 0:
            self._allocated = int(self._max_override)
        elif self._mode == BudgetMode.UNLIMITED:
            self._allocated = sys.maxsize
        elif self._mode == BudgetMode.MINIMAL:
            self._allocated = int(
                _BASE_BUDGET_BY_EFFORT.get(
                    effort, _BASE_BUDGET_BY_EFFORT[EffortLevel.MEDIUM]
                )
            )
        elif self._mode == BudgetMode.BALANCED:
            self._allocated = int(
                2
                * _BASE_BUDGET_BY_EFFORT.get(
                    effort, _BASE_BUDGET_BY_EFFORT[EffortLevel.MEDIUM]
                )
            )
        else:  # AUTO (default)
            base = _BASE_BUDGET_BY_EFFORT.get(
                effort, _BASE_BUDGET_BY_EFFORT[EffortLevel.MEDIUM]
            )
            mult = _EFFORT_MULTIPLIER.get(effort, 2.0)
            self._allocated = int(base * (1.0 + 0.5 * mult))

        self._allocation_locked = True
        return self._allocated

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def track(self, tokens_used: int) -> None:
        """Accumulate token usage. Negative values rejected."""
        if tokens_used < 0:
            raise ValueError(
                f"tokens_used must be >= 0, got {tokens_used}"
            )
        # Snapshot the deferred-callback payloads inside the lock; fire them
        # OUTSIDE the lock so a callback that re-enters the controller does
        # NOT deadlock (the V2.0 CRIT noted reentrancy as a documented user
        # constraint; we strengthen it here by releasing the lock before
        # firing).
        deferred: list[tuple[AmplifierEvent, dict[str, Any]]] = []
        with self._lock:
            if not self._allocation_locked:
                # Tracking before allocate(): default to AUTO+MEDIUM.
                self._allocated = int(
                    _BASE_BUDGET_BY_EFFORT[EffortLevel.MEDIUM] * 2.0
                )
                self._allocation_locked = True
            self._used += int(tokens_used)
            deferred.extend(
                self._collect_pending_threshold_events_locked()
            )
            if self._used >= self._allocated and not self._exhausted:
                self._exhausted = True
        for event, payload in deferred:
            self._fire_callback(event, payload)

    def track_text(self, text: str) -> int:
        """Estimate token count from text and track. Returns the estimate.

         (B-09): NO ``@lru_cache``. Each call counts uncached at the
        correct cost. Prefix amortization is via
        :meth:`track_iteration_with_prefix` — see that method's docstring.

        Tokenizer encode runs WITHOUT the controller lock. ``tiktoken``
        encoders are read-only after construction (Gemini-verified during
        Cluster A pre-flight), so two threads sharing one encoder is safe.
        """
        count = self._estimate_tokens(text)
        if count > 0:
            self.track(count)
        return count

    def track_iteration_with_prefix(self, iteration_text: str) -> int:
        """account for the fixed prompt prefix exactly once per
        iteration.

        The kernel pre-counts the  prompt prefix's tokens at session
        construction and threads that count into ``injected_prefix_tokens``.
        This method charges (prefix + variable text) ATOMICALLY in a
        single ``track`` call so threshold-crossing observability is
        attributed to one logical iteration, not split between two
        ``track`` invocations.

        CRIT-1 fix: pre-count the variable text WITHOUT recording it,
        then call ``track(variable + prefix)`` once. A concurrent reader
        therefore observes either the before-iteration state or the
        after-iteration state, never a half-charged intermediate.

        Returns the total tokens charged.
        """
        variable = self._estimate_tokens(iteration_text)
        total = variable + self._injected_prefix_tokens
        if total > 0:
            self.track(total)
        return total

    def _estimate_tokens(self, text: str) -> int:
        """Tokenizer encode without recording. Used by
        :meth:`track_iteration_with_prefix` to keep the iteration's
        accounting atomic.

        ``tiktoken`` encoders are read-only after construction; concurrent
        callers therefore do not require a lock here.
        """
        if not text:
            return 0
        if self._tiktoken_encoder is not None:
            try:
                return len(self._tiktoken_encoder.encode(text))
            except Exception:
                return max(1, len(text) // 4)
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Stop conditions / introspection
    # ------------------------------------------------------------------

    def remaining(self) -> int:
        with self._lock:
            if self._mode == BudgetMode.UNLIMITED:
                return sys.maxsize
            return max(0, self._allocated - self._used)

    def should_stop_for_budget(self) -> bool:
        with self._lock:
            if self._mode == BudgetMode.UNLIMITED:
                return False
            return self._used >= self._allocated

    def is_exhausted(self) -> bool:
        with self._lock:
            return self._exhausted

    def mark_iteration(self) -> None:
        with self._lock:
            self._iterations_completed += 1

    def report(self) -> BudgetReport:
        with self._lock:
            return BudgetReport(
                mode=self._mode,
                allocated=self._allocated,
                used=self._used,
                remaining=(
                    sys.maxsize
                    if self._mode == BudgetMode.UNLIMITED
                    else max(0, self._allocated - self._used)
                ),
                iterations_completed=self._iterations_completed,
                exhausted=self._exhausted,
                warnings_emitted=self._warnings_emitted,
                use_tiktoken=self._use_tiktoken,
                encoding_name=self._encoding_name,
            )

    def reset(self) -> None:
        """Wipe all state. Use only between sessions."""
        with self._lock:
            self._allocated = 0
            self._used = 0
            self._iterations_completed = 0
            self._exhausted = False
            self._warnings_emitted = 0
            self._allocation_locked = False
            self._warning_levels_fired.clear()

    # ------------------------------------------------------------------
    # Threshold-crossing event helpers
    # ------------------------------------------------------------------

    def _collect_pending_threshold_events_locked(
        self,
    ) -> list[tuple[AmplifierEvent, dict[str, Any]]]:
        """Caller holds lock. Returns events to fire OUTSIDE the lock.

         + . Emits at every 10 % crossing past 70 %
        (70/80/90/100). Per-level dedup via ``_warning_levels_fired``.
        """
        events: list[tuple[AmplifierEvent, dict[str, Any]]] = []
        if self._mode == BudgetMode.UNLIMITED or self._allocated <= 0:
            return events
        ratio_used = self._used / self._allocated
        for level in _WARN_RATIO_THRESHOLDS:
            if (
                ratio_used >= level
                and level not in self._warning_levels_fired
            ):
                self._warning_levels_fired.add(level)
                self._warnings_emitted += 1
                event, payload = self._build_event_locked(level, ratio_used)
                events.append((event, payload))
        return events

    def _build_event_locked(
        self, level: float, ratio_used: float
    ) -> tuple[AmplifierEvent, dict[str, Any]]:
        """Caller holds lock. Constructs the (event, payload) tuple."""
        hint = (
            "To raise: set AGENT_AMP_BUDGET=unlimited in the environment, "
            "or AmplifierConfig(budget_mode=BudgetMode.UNLIMITED)."
        )
        msg = (
            f"Token budget at {ratio_used:.1%}: {self._used}/"
            f"{self._allocated} used. Iterations: "
            f"{self._iterations_completed}. {hint}"
        )
        # Log inside the lock — Python's stdlib logging is thread-safe and
        # cheap; it also avoids any reorder against the per-level dedup.
        LOG.warning(msg)
        event = (
            AmplifierEvent.ON_BUDGET_HIT
            if level >= 1.0
            else AmplifierEvent.ON_BUDGET_LOW
        )
        payload = {
            "signal": BudgetExhaustedSignal(
                mode=self._mode,
                allocated=self._allocated,
                used=self._used,
                ratio_used=ratio_used,
                iterations_completed=self._iterations_completed,
                actionable_hint=hint,
            )
        }
        return event, payload

    def _fire_callback(
        self, event: AmplifierEvent, payload: dict[str, Any]
    ) -> None:
        """Run the user callback OUTSIDE the lock with try/except."""
        cb = self._observability_callback
        if cb is None:
            return
        try:
            cb(event, payload)
        except Exception as exc:
            # Per E-6: callbacks must never crash the kernel.
            LOG.warning("observability_callback raised: %s", exc)


__all__ = [
    "_BASE_BUDGET_BY_EFFORT",
    "_EFFORT_MULTIPLIER",
    "_WARN_RATIO_THRESHOLDS",
    "BudgetExhaustedSignal",
    "BudgetReport",
    "TokenBudgetController",
    "_select_encoding_name",
]
