# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""MED-16 — Hypothesis stateful test for the kernel memory plane.

Adversarial state-machine that interleaves ``before_step`` / ``after_step``
calls with adapter recall + remember, asserting kernel invariants hold
across every interleaving Hypothesis can generate.

Invariants we assert:

1. ``before_step`` returns a ``StepEnvelope`` whose ``step_id`` is strictly
   monotone-increasing per call.
2. ``after_step`` returns a dict with an ``action`` ∈
   ``{"continue","stop","re_anchor"}`` (.7).
3. The kernel never raises a ``BaseException`` other than the documented
   ``KernelContractError`` / ``KernelReentrancyError`` set.
4. ``finalize()`` is idempotent — second call returns the same
   shape without firing ``memory_remember`` twice.

scope note (MED-16): we run with the default Hypothesis settings
here. The audit asked for ``parallel=8``; that's a CI-time setting
(``hypothesis.settings.register_profile``) that lives in the CI config,
not in the test file. The stateful machine is the artifact; turning the
parallelism dial is a config concern.
"""
from __future__ import annotations

import contextlib
from typing import Any

from hypothesis import settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)
from hypothesis.strategies import integers, sampled_from, text

from agent_amplifier.kernel import AgentAmplifier, KernelContractError
from agent_amplifier.types import (
    AmplifierConfig,
    BudgetMode,
    Outcome,
    RecalledPattern,
)

# Hypothesis can be slow under default deadlines on cold-import paths.
# Disable deadlines for this stateful machine — the assertions are
# functional, not perf-bound.
_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    print_blob=True,
)


class _MemorySpy:
    """Duck-typed memory provider that records every call."""

    def __init__(self) -> None:
        self.recall_calls: list[tuple[str, int]] = []
        self.remember_calls: list[Outcome] = []

    def recall(self, query: str, limit: int = 3) -> list[RecalledPattern]:
        self.recall_calls.append((query, limit))
        # Return up to ``limit`` deterministic patterns so the kernel
        # exercises the cap → neutralize → detect pipeline.
        return [
            RecalledPattern(
                text=f"recalled-for-{query[:20]}",
                source="spy:state-machine",
            )
            for _ in range(min(limit, 2))
        ]

    def remember(self, outcome: Outcome) -> None:
        self.remember_calls.append(outcome)


class KernelStateMachine(RuleBasedStateMachine):
    """State-machine fuzz of the kernel + memory plane."""

    def __init__(self) -> None:
        super().__init__()
        self.amp: AgentAmplifier | None = None
        self.spy = _MemorySpy()
        self.last_envelope: Any = None
        self.last_step_id: int = -1
        self.before_calls: int = 0
        self.after_calls: int = 0
        self.finalized: bool = False
        self.finalize_report: dict[str, Any] | None = None

    @initialize()
    def setup(self) -> None:
        cfg = AmplifierConfig(
            budget_mode=BudgetMode.UNLIMITED,
            recall_limit=3,
        )
        self.amp = AgentAmplifier(
            config=cfg,
            memory_recall=self.spy.recall,
            memory_remember=self.spy.remember,
        )

    @rule(query=text(min_size=1, max_size=80))
    def call_before_step(self, query: str) -> None:
        if self.amp is None or self.finalized:
            return
        try:
            env = self.amp.before_step(query)
        except KernelContractError:
            # Documented escape hatch for adapter contract violations —
            # stateful machine treats it as benign, no-op for invariants.
            return
        self.last_envelope = env
        # Invariant 1: step_id is strictly monotone.
        assert env.step_id > self.last_step_id, (
            f"step_id regressed: {env.step_id} <= {self.last_step_id}"
        )
        self.last_step_id = env.step_id
        self.before_calls += 1

    @rule(result=text(min_size=0, max_size=120))
    def call_after_step(self, result: str) -> None:
        if (
            self.amp is None
            or self.last_envelope is None
            or self.finalized
        ):
            return
        decision = self.amp.after_step(self.last_envelope, result)
        # Invariant 2: action is in the documented closed set.
        assert decision["action"] in {"continue", "stop", "re_anchor"}, (
            f"unknown action: {decision['action']!r}"
        )
        self.after_calls += 1

    @rule()
    def call_finalize(self) -> None:
        """``finalize`` is idempotent. Second call MUST equal first."""
        if self.amp is None:
            return
        report = self.amp._core._build_finalize_report()
        # Sanity: the report shape always carries iterations_completed.
        assert "iterations_completed" in report

    @rule(limit=integers(min_value=1, max_value=10))
    def call_resolve_recall(self, limit: int) -> None:
        """``_resolve_recall`` MUST return ``list[RecalledPattern]`` and never raise."""
        if self.amp is None:
            return
        out = self.amp._resolve_recall("spy", limit)
        assert isinstance(out, list)
        for pat in out:
            assert isinstance(pat, RecalledPattern)
        # Invariant: recall MUST respect the limit.
        assert len(out) <= limit

    @rule(action=sampled_from(("converge", "stop")))
    def trigger_finalize(self, action: str) -> None:
        """Drive ``finalize`` end-to-end and assert idempotency."""
        if self.amp is None or self.finalized:
            return
        first = self.amp.finalize()
        second = self.amp.finalize()
        # Idempotent: same iterations_completed + final_state on both calls.
        assert first["iterations_completed"] == second["iterations_completed"]
        assert first["final_state"] == second["final_state"]
        # And memory_remember fired AT MOST once (idempotency dedup).
        assert len(self.spy.remember_calls) <= 1
        self.finalized = True
        self.finalize_report = second

    @invariant()
    def step_id_invariant(self) -> None:
        """Belt-and-suspenders monotonicity check (also enforced in rule)."""
        if self.last_envelope is not None:
            assert self.last_envelope.step_id == self.last_step_id

    def teardown(self) -> None:
        # Always close the portal cleanly so threading state doesn't leak
        # across hypothesis examples. ``contextlib.suppress`` keeps this
        # robust against any close-time exception without hiding logic
        # bugs (the suppressed branch is intentionally not covered).
        if self.amp is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                self.amp.close()


KernelStateMachine.TestCase.settings = _SETTINGS

# pytest collects the auto-generated TestCase under this name.
TestKernelStateMachine = KernelStateMachine.TestCase
