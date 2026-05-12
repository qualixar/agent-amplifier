# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""IP-4 LTI Convergence Detector.

Per  V2.0 §1. Detect when iterative refinement stabilizes; expose
LTI damping; cap iterations at ``max_iterations``.

Algorithm V1.0 (DECISIONS-LOCKED C-1, C-2):
    Jaccard similarity over normalized keyword sets (zero deps). Constructor
    accepts an ``embedder`` injection point reserved for V1.1 cosine, but
    the V1.0 cosine code path is **disabled** in this cluster.

Stability:
    Damping factor uses the Gompertz ``exp(-exp(x))`` parameterization
    (a smooth, strictly-monotone-decreasing function on ``(0, 1)``). The
    parameterization gives a closed-form, allocation-free per-iteration
    factor with provable bounds — see ``damping_factor`` below for the
    floor/ceiling clamp.

Thread-safety (:
    All public methods are thread-safe under a single internal
    ``threading.Lock``. The lock is leaf-only — nothing under it calls back
    into the kernel and nothing under it performs I/O.

V2.0 verifications inlined (Gemini-grounded, 2026-04-26):
    * ``threading.Lock`` is NOT reentrant (Python ``threading`` docs). We
      therefore never re-acquire ``self._lock`` inside its own critical
      section.
    * PEP 703 free-threading per-object micro-locks make ``deque.append``
      atomic for the data-race surface, BUT compound check-then-act is
      still unsafe — that is exactly what ``update()`` does. Hence the
      explicit ``threading.Lock``.
    * ``functools.lru_cache`` on instance methods leaks ``self`` (Ruff B019,
      Python ``functools`` docs note). We do not use it anywhere in this
      module.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from agent_amplifier._internal.keyword_set import (
    keyword_set as _keyword_set,
)
from agent_amplifier.types import ConvergenceState

LOG = logging.getLogger("agent_amplifier.convergence")


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """One iteration's compact record.

    ``timestamp_monotonic`` has ``compare=False`` so equality of
    records with otherwise-identical content is preserved across clock
    skews — important for golden-test stability.

    ``output`` is ``str | None``. Non-leader records have ``output``
    nulled by :meth:`ConvergenceDetector._evict_non_leader_outputs` to bound
    memory. ``output_hash`` and ``keyword_set`` are always retained — they
    suffice for similarity, equality, and quality comparisons.
    """

    iteration: int
    output: str | None
    output_hash: str
    keyword_set: frozenset[str]
    similarity_to_prev: float
    quality_proxy: float
    timestamp_monotonic: float = field(compare=False, default=0.0)


# ---------------------------------------------------------------------------
# Custom warning class (kept for back-compat; V2.0 emits via LOG)
# ---------------------------------------------------------------------------


class ConvergenceDetectorWarning(UserWarning):
    """Reserved for legacy callers. V2.0 emits via ``LOG.warning`` instead
    of ``warnings.warn`` (). Kept so tests inspecting the exception
    hierarchy continue to work.
    """


# ---------------------------------------------------------------------------
# ConvergenceDetector
# ---------------------------------------------------------------------------


class ConvergenceDetector:
    """Detect convergence/oscillation/stagnation across iterative refinement.

    See module docstring for algorithm + thread-safety notes.

    All public methods (``update``, ``should_stop``, ``best_output``,
    ``reset``, ``state``, ``history``, ``iteration_count``,
    ``last_keyword_set``) are thread-safe under ``self._lock``.
    """

    DEFAULT_CONVERGED_THRESHOLD: float = 0.95
    DEFAULT_OSCILLATION_NEAR_DUP: float = 0.90
    DEFAULT_OSCILLATION_DISSIM: float = 0.70
    DEFAULT_STAGNATION_BAND: float = 0.05
    DEFAULT_MAX_ITERATIONS: int = 4
    DEFAULT_HISTORY_KEEP: int = 5  #  (was 8 in V1)
    DEFAULT_DAMPING_LOG_DT: float = 0.0
    DEFAULT_DAMPING_LOG_A0: float = 0.0

    def __init__(
        self,
        *,
        converged_threshold: float = DEFAULT_CONVERGED_THRESHOLD,
        oscillation_near_dup: float = DEFAULT_OSCILLATION_NEAR_DUP,
        oscillation_dissim: float = DEFAULT_OSCILLATION_DISSIM,
        stagnation_band: float = DEFAULT_STAGNATION_BAND,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        history_keep: int = DEFAULT_HISTORY_KEEP,
        damping_log_dt: float = DEFAULT_DAMPING_LOG_DT,
        damping_log_a0: float = DEFAULT_DAMPING_LOG_A0,
        clock: Callable[[], float] = time.monotonic,
        embedder: Callable[[str], list[float]] | None = None,
    ) -> None:
        if not (0.0 < converged_threshold <= 1.0):
            raise ValueError(
                "converged_threshold must be in (0,1], got "
                f"{converged_threshold}"
            )
        if max_iterations < 1 or max_iterations > 32:
            raise ValueError(
                f"max_iterations must be in [1,32], got {max_iterations}"
            )
        if history_keep < 2 or history_keep > 32:
            raise ValueError(
                f"history_keep must be in [2,32], got {history_keep}"
            )

        self._converged_threshold = float(converged_threshold)
        self._osc_near = float(oscillation_near_dup)
        self._osc_dis = float(oscillation_dissim)
        self._stag_band = float(stagnation_band)
        self._max_iterations = int(max_iterations)
        self._damping_log_dt = float(damping_log_dt)
        self._damping_log_a0 = float(damping_log_a0)
        self._clock = clock
        self._embedder = embedder  # V1.1 reserved; UNUSED in V1.0

        self._lock = threading.Lock()  #
        self._history: deque[IterationRecord] = deque(maxlen=history_keep)
        self._last_state: ConvergenceState = ConvergenceState.IMPROVING
        self._iteration_counter: int = 0
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(
        self, iteration_output: str | None, iteration: int
    ) -> ConvergenceState:
        """Ingest one iteration's output. Thread-safe.

        Pre-computes keyword set, hash, and quality proxy WITHOUT the lock
        (pure functions). Then takes the lock once for the compound
        check-then-act on history + state classification.

        lock wraps every mutation of the four pieces of mutable state.
        time read via injected ``self._clock`` (default
        ``time.monotonic``).
        """
        if iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {iteration}")
        text = "" if iteration_output is None else iteration_output

        # --- pure pre-compute (no lock) ---
        # Tokenization, hashing, and quality proxy are pure functions of
        # ``text``; they do NOT read shared state and can run outside the
        # lock. Similarity is computed INSIDE the lock against the same
        # ``_history`` snapshot that this method mutates — this closes the
        # read-then-write race that would otherwise let two concurrent
        # ``update()`` calls compute ``similarity_to_prev`` against a stale
        # tail.
        kw = _keyword_set(text)
        h = self._hash(text)
        quality = self._quality_proxy(text, kw)
        now = self._clock()

        with self._lock:
            sim = 0.0
            if self._history:
                prev = self._history[-1]
                sim = self._jaccard(prev.keyword_set, kw)

            record = IterationRecord(
                iteration=iteration,
                output=text,
                output_hash=h,
                keyword_set=kw,
                similarity_to_prev=sim,
                quality_proxy=quality,
                timestamp_monotonic=now,
            )
            self._history.append(record)
            self._iteration_counter = max(
                self._iteration_counter, iteration + 1
            )
            self._evict_non_leader_outputs_locked()
            self._last_state = self._classify_locked()
            if self._last_state in (
                ConvergenceState.CONVERGED,
                ConvergenceState.OSCILLATING,
            ):
                self._stopped = True
            return self._last_state

    # ------------------------------------------------------------------
    # Classification (private; caller holds lock)
    # ------------------------------------------------------------------

    def _classify_locked(self) -> ConvergenceState:
        """Caller MUST hold ``self._lock``. Pure read of ``self._history``."""
        n = len(self._history)
        if n <= 1:
            return ConvergenceState.IMPROVING

        last = self._history[-1]
        prev = self._history[-2]
        sim_last = last.similarity_to_prev

        # CONVERGED first — highest priority.
        if sim_last >= self._converged_threshold:
            return ConvergenceState.CONVERGED

        # OSCILLATING (need t-2)
        if n >= 3:
            prev_prev = self._history[-3]
            sim_to_pp = self._jaccard(last.keyword_set, prev_prev.keyword_set)
            if (
                sim_to_pp >= self._osc_near
                and sim_last <= self._osc_dis
            ):
                return ConvergenceState.OSCILLATING

        # STAGNANT
        if n >= 3:
            sim_prev = prev.similarity_to_prev
            if (
                abs(sim_last - sim_prev) <= self._stag_band
                and sim_last < self._converged_threshold
            ):
                return ConvergenceState.STAGNANT

        return ConvergenceState.IMPROVING

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
        if not a and not b:
            return 1.0
        union = a | b
        if not union:  # pragma: no cover - both-empty handled above
            return 1.0
        inter = a & b
        return len(inter) / len(union)

    # ------------------------------------------------------------------
    # Stop conditions
    # ------------------------------------------------------------------

    def should_stop(self) -> bool:
        """Thread-safe. ``True`` if CONVERGED/OSCILLATING or hit
        ``max_iterations``."""
        with self._lock:
            if self._stopped:
                return True
            return self._iteration_counter >= self._max_iterations

    # ------------------------------------------------------------------
    # Damping (LTI)
    # ------------------------------------------------------------------

    def damping_factor(self, iteration: int) -> float:
        """LTI damping factor (Gompertz-style parameterization).

        ``alpha = exp(-exp(clamp(log_dt + log_a0 + ln(1+t), -20, 20)))``

        Mathematical guarantee: result strictly in ``(0, 1)``.
        Strictly decreasing in ``t`` within the clamp range — verified by the
        hypothesis property in tests/test_convergence_properties.py.

        Numerical floor: clamp to ``[eps, 1-eps]`` with ``eps = 1e-12`` so
        callers never see 0 or 1 exactly.

        Pure function — no shared mutable state read; therefore not under
        ``self._lock``.
        """
        if iteration < 0:
            iteration = 0
        x = (
            self._damping_log_dt
            + self._damping_log_a0
            + math.log1p(iteration)
        )
        x = max(-20.0, min(20.0, x))
        inner = math.exp(x)
        alpha = math.exp(-inner)
        eps = 1e-12
        return max(eps, min(1.0 - eps, alpha))

    # ------------------------------------------------------------------
    # Output retrieval
    # ------------------------------------------------------------------

    def best_output(self) -> str | None:
        """Return the best surviving iteration output.

        Strategy:
            * empty history             → ``None``
            * CONVERGED / IMPROVING      → leader is the latest record
            * OSCILLATING                → leader is the highest-quality
                                          record whose ``output`` survived
                                          eviction
            * STAGNANT                   → leader is the latest record;
                                          ``LOG.warning`` advises re-anchor

        Because eviction nulls only records older than position ``-2``, the
        leader's ``output`` is always present. (See
        :meth:`_evict_non_leader_outputs_locked`.)
        """
        with self._lock:
            if not self._history:
                return None

            leader = self._current_leader_locked()
            if self._last_state == ConvergenceState.STAGNANT:
                LOG.warning(
                    "Returning leader output despite stagnation; consider "
                    "re-anchoring goal."
                )
            if leader.output is None:  # pragma: no cover - invariant guard
                raise RuntimeError(
                    "Internal invariant violation: leader.output is None. "
                    "Report this with the iteration history."
                )
            return leader.output

    def _current_leader_locked(self) -> IterationRecord:
        """Caller holds lock. Returns the leader record (output non-null)."""
        if self._last_state == ConvergenceState.OSCILLATING:
            with_text = [r for r in self._history if r.output is not None]
            if with_text:
                return max(with_text, key=lambda r: r.quality_proxy)
        return self._history[-1]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha1(
            text.encode("utf-8", errors="replace"), usedforsecurity=False
        ).hexdigest()

    @staticmethod
    def _quality_proxy(text: str, kw: frozenset[str]) -> float:
        """Heuristic informativeness — bounded, deterministic.

        caps text length at 256_000 chars before computing density,
        matching :data:`agent_amplifier._internal.keyword_set.MAX_OUTPUT_CHARS_FOR_ANALYSIS`.
        """
        if not text:
            return 0.0
        capped = text if len(text) <= 256_000 else text[:256_000]
        n_words = max(1, len(capped.split()))
        density = len(kw) / n_words
        return math.log1p(len(capped)) * density

    def _evict_non_leader_outputs_locked(self) -> None:
        """Caller holds lock. keep ``output`` only on the two newest
        records.

        Eviction policy: only records older than position ``-2`` are nulled.
        This guarantees that the OSCILLATING-by-quality leader (which lives
        in the newest two records under ``DEFAULT_HISTORY_KEEP=5`` + an
        A/B/A pattern) always retains its ``output``. Verified by
        ``test_oscillation_leader_output_is_retained_after_eviction``.
        """
        if len(self._history) < 3:
            return
        for i in range(len(self._history) - 2):
            rec = self._history[i]
            if rec.output is not None:
                self._history[i] = IterationRecord(
                    iteration=rec.iteration,
                    output=None,
                    output_hash=rec.output_hash,
                    keyword_set=rec.keyword_set,
                    similarity_to_prev=rec.similarity_to_prev,
                    quality_proxy=rec.quality_proxy,
                    timestamp_monotonic=rec.timestamp_monotonic,
                )

    # ------------------------------------------------------------------
    # Public state inspection (V2.0 — all thread-safe)
    # ------------------------------------------------------------------

    @property
    def history(self) -> tuple[IterationRecord, ...]:
        """Snapshot of the current history. Thread-safe."""
        with self._lock:
            return tuple(self._history)

    @property
    def state(self) -> ConvergenceState:
        with self._lock:
            return self._last_state

    @property
    def iteration_count(self) -> int:
        with self._lock:
            return self._iteration_counter

    def last_keyword_set(self) -> frozenset[str]:
        """Published so the kernel can pass it as ``precomputed_kw`` to
        :meth:`agent_amplifier.goal_anchor.GoalAnchorService.measure_drift`,
        avoiding double tokenization ().
        """
        with self._lock:
            if not self._history:
                return frozenset()
            return self._history[-1].keyword_set

    def reset(self) -> None:
        """Clear all state. Use between sessions, never mid-loop."""
        with self._lock:
            self._history.clear()
            self._last_state = ConvergenceState.IMPROVING
            self._iteration_counter = 0
            self._stopped = False


__all__ = [
    "ConvergenceDetector",
    "ConvergenceDetectorWarning",
    "IterationRecord",
]
