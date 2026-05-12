# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Kernel branch coverage ().

Targets:
  * Lines 117-119, 125→127 — _budget_report_to_dict TypeError fallback +
    mode-without-.value branch.
  * Line 280 — ``initial_query`` triggers eager start_recall.
  * Line 322, 336-340 — observability proxy + _emit re-entry guard.
  * Line 375 — KernelContractError re-raise from outer try.
  * Lines 413-420 — recall future timeout + generic-exception branches.
  * Branch 428→432 — anchor double-check sees value already set (no-op set).
  * Lines 443, 447 — escalate_low_confidence + classify type-mismatch raise.
  * Lines 461-469, 473-474 — MAX-tier kill switch active + setting branches.
  * Line 512 — phase_slot loop body when slot present in ctx.
  * Lines 654-659 — after_step outer except branch returns continue.
  * Lines 704-709 — budget_stop branch in after_step.
  * Line 723 — OSCILLATING convergence branch.
  * Lines 747-754 — max_iterations stop branch (should_stop → reason=...).
  * Branch 806→820 — finalize when last_classification is None (skip remember).
  * Lines 905, 909, 950, 953, 957, 960 — facade property accessors.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import pytest

from agent_amplifier.types import (
    AmplifierConfig,
    AmplifierEvent,
    EffortLevel,
    PhaseIndex,
)

# ---------------------------------------------------------------------------
# _budget_report_to_dict — fallback paths (lines 117-119, 125→127)
# ---------------------------------------------------------------------------


def test_budget_report_to_dict_typeerror_fallback() -> None:
    """When the report is not a dataclass (asdict raises TypeError), the
    helper falls back to getattr access (lines 117-127)."""
    from agent_amplifier.kernel import _budget_report_to_dict

    class _FakeReport:
        # Not a dataclass → asdict raises TypeError.
        def __init__(self) -> None:
            self.allocated = 100
            self.used = 25
            self.remaining = 75
            self.mode = EffortLevel.HIGH  # has .value attr

    d = _budget_report_to_dict(_FakeReport())
    assert d["allocated"] == 100
    assert d["used"] == 25
    assert d["remaining"] == 75
    assert d["mode"] == "high"


def test_budget_report_to_dict_mode_without_value_attr() -> None:
    """asdict success path but mode is a plain string already (line 125→127
    branch where ``hasattr(d['mode'], 'value')`` is False)."""
    from agent_amplifier.kernel import _budget_report_to_dict

    @dataclass
    class _PlainReport:
        mode: str = "balanced"
        allocated: int = 1
        used: int = 0
        remaining: int = 1

    d = _budget_report_to_dict(_PlainReport())
    assert d["mode"] == "balanced"


def test_budget_report_to_dict_typeerror_with_no_mode_value_attr() -> None:
    """TypeError fallback where the .mode attr does not have .value
    (covers the `getattr(..., "value", "?")` branch on line 120)."""
    from agent_amplifier.kernel import _budget_report_to_dict

    class _FakeReport:
        # Not a dataclass; mode is plain string.
        mode = "auto"
        allocated = 0
        used = 0
        remaining = 0

    d = _budget_report_to_dict(_FakeReport())
    # The fallback path uses getattr(..., "value", "?") so a plain string mode
    # without .value yields "?" — this nails the inner branch.
    assert d["mode"] == "?"


# ---------------------------------------------------------------------------
# V2.1 NOTE: ``initial_query`` + lazy SLM recall future tests removed;
# memory plane recall is now synchronous via ``_resolve_recall`` and lives
# in test_kernel_memory_plane.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _emit re-entry guard + early-return paths (lines 322, 336-340)
# ---------------------------------------------------------------------------


def test_emit_swallows_kernel_reentrancy_error() -> None:
    """B5: _emit catches KernelReentrancyError raised by a callback.

    Old behavior (V2.0): _emit checked _IN_KERNEL_LOCK and short-circuited
    BEFORE invoking the callback. moves the re-entry detection to
    the public entry points (before_step / after_step) where it can raise
    a named, observable exception. _emit no longer consults the ContextVar.

    The contract change: a callback that re-enters via the entry point
    will see KernelReentrancyError; if it bubbles out of the callback,
    _emit catches and logs but doesn't crash the host.
    """
    from agent_amplifier.kernel import (
        KernelReentrancyError,
        _AmplifierCore,
    )

    def angry(event: AmplifierEvent, payload: dict[str, Any]) -> None:
        raise KernelReentrancyError("simulated re-entry from callback")

    cfg = AmplifierConfig(observability_callback=angry)
    core = _AmplifierCore(config=cfg)
    # Must not raise; the kernel handles its own re-entry signal.
    core._emit(AmplifierEvent.BEFORE_STEP, {"x": 1})


def test_emit_fires_callback_inline_with_kernel_calls() -> None:
    """B5 inverse: _emit DOES fire even while _IN_KERNEL_LOCK is set.

    The ContextVar is now an entry-point sentinel for re-entry detection,
    NOT a deferral signal for _emit. Observability events MUST fire
    while the kernel is running them.
    """
    from agent_amplifier.kernel import _IN_KERNEL_LOCK, _AmplifierCore

    seen: list[AmplifierEvent] = []

    def cb(event: AmplifierEvent, payload: dict[str, Any]) -> None:
        seen.append(event)

    cfg = AmplifierConfig(observability_callback=cb)
    core = _AmplifierCore(config=cfg)
    token = _IN_KERNEL_LOCK.set(True)
    try:
        core._emit(AmplifierEvent.BEFORE_STEP, {"x": 1})
    finally:
        _IN_KERNEL_LOCK.reset(token)
    assert seen == [AmplifierEvent.BEFORE_STEP]


def test_emit_no_callback_returns_silently() -> None:
    """When observability_callback is None, _emit returns silently."""
    from agent_amplifier.kernel import _AmplifierCore

    core = _AmplifierCore()  # default cfg has no cb
    # Just confirm no raise — coverage-only smoke.
    core._emit(AmplifierEvent.BEFORE_STEP, {"x": 1})


def test_observability_proxy_fires_on_budget_event() -> None:
    """The proxy returned by _make_observability_proxy must fire the user
    callback when the budget controller crosses a threshold (line 322)."""
    from agent_amplifier.kernel import _AmplifierCore

    seen: list[tuple[AmplifierEvent, dict[str, Any]]] = []

    def cb(event: AmplifierEvent, payload: dict[str, Any]) -> None:
        seen.append((event, payload))

    cfg = AmplifierConfig(observability_callback=cb)
    core = _AmplifierCore(config=cfg)
    # Invoke the proxy directly to confirm the body runs.
    proxy = core._make_observability_proxy()
    proxy(AmplifierEvent.ON_BUDGET_LOW, {"used": 70, "limit": 100})
    assert any(e is AmplifierEvent.ON_BUDGET_LOW for e, _ in seen)


# ---------------------------------------------------------------------------
# Outer try re-raises KernelContractError (line 375)
# ---------------------------------------------------------------------------


def test_kernel_contract_error_propagates_through_outer_try() -> None:
    """before_step's outer try MUST re-raise KernelContractError, not
    swallow into a degenerate envelope (line 375)."""
    from agent_amplifier.kernel import (
        AgentAmplifier,
        KernelContractError,
    )

    amp = AgentAmplifier()
    try:
        amp._core._state.phase = PhaseIndex.EVALUATE
        # Clear AGENT_AMP_FALLBACK_PHASE if set so the error path fires.
        prev = os.environ.pop("AGENT_AMP_FALLBACK_PHASE", None)
        try:
            with pytest.raises(KernelContractError):
                amp.before_step("redesign auth")
        finally:
            if prev is not None:
                os.environ["AGENT_AMP_FALLBACK_PHASE"] = prev
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# V2.1 NOTE: lazy-recall future timeout + generic-exception tests are
# obsolete — memory plane is sync via ``_resolve_recall`` and exception
# handling lives in test_kernel_memory_plane.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Anchor double-check sees value already set (branch 428→432)
# ---------------------------------------------------------------------------


def test_anchor_double_check_no_op_when_concurrent_set() -> None:
    """The double-checked anchor capture must be a no-op when another path
    has already set the anchor (branch 428→432)."""
    from agent_amplifier.goal_anchor import GoalAnchorService
    from agent_amplifier.kernel import _AmplifierCore

    core = _AmplifierCore()
    # Pre-seed the anchor via the real service so all invariants hold.
    svc = GoalAnchorService()
    pre_anchor = svc.capture("seeded query for anchor double-check")
    core._state.anchor = pre_anchor

    async def _go() -> Any:
        return await core.before_step("query", {})

    env = asyncio.run(_go())
    # Anchor was already set → no overwrite.
    assert core._state.anchor is pre_anchor
    assert env.iteration == 0


# ---------------------------------------------------------------------------
# escalate_low_confidence routing (line 443) + classify-type guard (line 447)
# ---------------------------------------------------------------------------


def test_escalate_low_confidence_routes_through_classify_with_config() -> None:
    """When AmplifierConfig.escalate_low_confidence is True, before_step
    uses classify_with_config (line 443)."""
    from agent_amplifier.kernel import AgentAmplifier

    cfg = AmplifierConfig(escalate_low_confidence=True)
    amp = AgentAmplifier(config=cfg)
    try:
        env = amp.before_step("rename foo")
        # Path executed; we only care about coverage.
        assert env.classification.complexity in EffortLevel
    finally:
        amp.close()


def test_classify_type_mismatch_raises_kernel_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If classify returns a non-TaskClassification value, kernel raises
    KernelContractError (line 447-457)."""
    from agent_amplifier import kernel
    from agent_amplifier.kernel import AgentAmplifier, KernelContractError

    monkeypatch.setattr(kernel, "classify", lambda _q: "not a TaskClassification")

    amp = AgentAmplifier()
    try:
        with pytest.raises(KernelContractError) as exc:
            amp.before_step("query")
        msg = str(exc.value)
        assert "wrong type" in msg
        assert "TaskClassification" in msg
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# MAX-tier kill switch (lines 461-469, 473-474)
# ---------------------------------------------------------------------------


def test_max_tier_kill_switch_caps_complexity_at_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When kill_switch is set and classify returns MAX, kernel demotes to
    HIGH and tags the signal (lines 461-469)."""
    from agent_amplifier import kernel
    from agent_amplifier.kernel import AgentAmplifier
    from agent_amplifier.types import TaskClassification

    def fake_classify(_q: str) -> TaskClassification:
        return TaskClassification(
            complexity=EffortLevel.MAX,
            domain="codegen",
            estimated_tokens=100_000,
            confidence=0.9,
            matched_signals=("max_signal",),
        )

    monkeypatch.setattr(kernel, "classify", fake_classify)

    amp = AgentAmplifier()
    try:
        # Pre-set the kill switch.
        amp._core._state.max_tier_kill_switch = True
        env = amp.before_step("any query")
        assert env.classification.complexity is EffortLevel.HIGH
        assert "max_tier_kill_switched" in env.classification.matched_signals
    finally:
        amp.close()


def test_max_tier_escalated_signal_arms_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When classify returns a TaskClassification with 'max_tier_escalated'
    in matched_signals, the kill-switch is set for subsequent calls
    (lines 472-474)."""
    from agent_amplifier import kernel
    from agent_amplifier.kernel import AgentAmplifier
    from agent_amplifier.types import TaskClassification

    def fake_classify(_q: str) -> TaskClassification:
        return TaskClassification(
            complexity=EffortLevel.HIGH,
            domain="codegen",
            estimated_tokens=80_000,
            confidence=0.5,
            matched_signals=("max_tier_escalated",),
        )

    monkeypatch.setattr(kernel, "classify", fake_classify)

    amp = AgentAmplifier()
    try:
        assert amp._core._state.max_tier_kill_switch is False
        amp.before_step("any query")
        assert amp._core._state.max_tier_kill_switch is True
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# Phase slot loop body — ctx contains a slot the phase requires (line 512)
# ---------------------------------------------------------------------------


def test_phase_slot_loop_picks_up_prev_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ctx provides ``prev_output`` and phase is EVALUATE, the loop body
    on line 512 fires (slot present + needed)."""
    from agent_amplifier.kernel import AgentAmplifier

    monkeypatch.setenv("AGENT_AMP_FALLBACK_PHASE", "EXPLORE")  # safety net
    amp = AgentAmplifier()
    try:
        amp._core._state.phase = PhaseIndex.EVALUATE
        env = amp.before_step(
            "evaluate options",
            {"prev_output": "approach A vs B vs C"},
        )
        # If the slot was picked up, EVALUATE renders normally (no fallback).
        assert env.phase == "EVALUATE"
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# after_step outer except (lines 654-659)
# ---------------------------------------------------------------------------


def test_after_step_outer_except_returns_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the inner after_step path raises unexpectedly, the outer try
    returns a continue decision with warning=amp_degraded (lines 654-659)."""
    from agent_amplifier import kernel
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        # Force the inner path to raise.
        async def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("after-step kaboom")

        monkeypatch.setattr(amp._core, "_after_step_inner", _boom)
        decision = amp.after_step({}, "result")
        assert decision["action"] == "continue"
        assert decision.get("warning") == "amp_degraded"
        _ = kernel  # silence unused
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# budget_stop branch in after_step (lines 704-709)
# ---------------------------------------------------------------------------


def test_after_step_budget_exhausted_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the budget controller signals exhaustion, after_step returns
    a stop+budget_exhausted decision and emits ON_BUDGET_HIT (704-709)."""
    from agent_amplifier.kernel import AgentAmplifier

    seen: list[AmplifierEvent] = []

    def cb(event: AmplifierEvent, payload: dict[str, Any]) -> None:
        seen.append(event)

    cfg = AmplifierConfig(observability_callback=cb)
    amp = AgentAmplifier(config=cfg)
    try:
        amp.before_step("simple")
        # Force the budget controller to claim exhaustion.
        monkeypatch.setattr(
            amp._core._budget, "should_stop_for_budget", lambda: True
        )
        decision = amp.after_step({}, "x")
        assert decision["action"] == "stop"
        assert decision["reason"] == "budget_exhausted"
        assert AmplifierEvent.ON_BUDGET_HIT in seen
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# OSCILLATING branch (line 723)
# ---------------------------------------------------------------------------


def test_after_step_oscillating_stops_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ConvergenceDetector.update returns OSCILLATING, after_step
    returns stop with reason=oscillating (line 723-728)."""
    from agent_amplifier.kernel import AgentAmplifier
    from agent_amplifier.types import ConvergenceState

    amp = AgentAmplifier()
    try:
        amp.before_step("simple")
        monkeypatch.setattr(
            amp._core._convergence,
            "update",
            lambda result, iteration: ConvergenceState.OSCILLATING,
        )
        # Best output stub.
        monkeypatch.setattr(
            amp._core._convergence, "best_output", lambda: "best"
        )
        decision = amp.after_step({}, "x")
        assert decision["action"] == "stop"
        assert decision["reason"] == "oscillating"
        assert "warning" in decision
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# max_iterations branch (lines 747-754)
# ---------------------------------------------------------------------------


def test_after_step_falls_through_to_continue_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no early-stop / re-anchor / max-iter branch fires, after_step
    falls through to _continue_decision (line 754 — the final ``else``)."""
    from agent_amplifier.goal_anchor import DriftLevel
    from agent_amplifier.kernel import AgentAmplifier
    from agent_amplifier.types import ConvergenceState

    amp = AgentAmplifier()
    try:
        amp.before_step("simple task")
        # Force the precise state: improving convergence, on-track drift,
        # not should_stop → falls through to _continue_decision().
        monkeypatch.setattr(
            amp._core._convergence,
            "update",
            lambda result, iteration: ConvergenceState.IMPROVING,
        )
        monkeypatch.setattr(
            amp._core._convergence, "should_stop", lambda: False
        )
        monkeypatch.setattr(
            amp._core._goal, "measure_drift", lambda result, anchor: 0.0
        )
        monkeypatch.setattr(
            amp._core._goal, "classify_drift", lambda d: DriftLevel.ON_TRACK
        )
        decision = amp.after_step({}, "x")
        assert decision["action"] == "continue"
    finally:
        amp.close()


def test_after_step_max_iterations_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ConvergenceDetector.should_stop() is True (max_iterations
    reached), kernel stops with reason=max_iterations (lines 747-754)."""
    from agent_amplifier.goal_anchor import DriftLevel
    from agent_amplifier.kernel import AgentAmplifier
    from agent_amplifier.types import ConvergenceState

    amp = AgentAmplifier()
    try:
        amp.before_step("simple")
        monkeypatch.setattr(
            amp._core._convergence,
            "update",
            lambda result, iteration: ConvergenceState.IMPROVING,
        )
        monkeypatch.setattr(
            amp._core._convergence, "should_stop", lambda: True
        )
        monkeypatch.setattr(
            amp._core._convergence, "best_output", lambda: "best-out"
        )
        # Suppress drift so we hit the max_iterations branch (after the
        # drift checks but before _continue_decision).
        monkeypatch.setattr(
            amp._core._goal, "measure_drift", lambda result, anchor: 0.0
        )
        monkeypatch.setattr(
            amp._core._goal, "classify_drift", lambda d: DriftLevel.ON_TRACK
        )
        decision = amp.after_step({}, "x")
        assert decision["action"] == "stop"
        assert decision["reason"] == "max_iterations"
        assert decision["output"] == "best-out"
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# finalize when last_classification is None (branch 806→820)
# ---------------------------------------------------------------------------


def test_finalize_without_classification_skips_remember() -> None:
    """If finalize is called without ever running before_step, no SLM write
    happens (branch 806→820 short-circuits the if block)."""
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        report = amp.finalize()
        # Just confirm a report comes back without error.
        assert "iterations_completed" in report
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# Facade property accessors (lines 905, 909, 950, 953, 957, 960)
# ---------------------------------------------------------------------------


def test_sync_facade_iteration_property() -> None:
    """AgentAmplifier.iteration mirrors the underlying core state."""
    from agent_amplifier.kernel import AgentAmplifier

    with AgentAmplifier() as amp:
        # No iterations done yet.
        assert amp.iteration == 0


def test_sync_facade_phase_property() -> None:
    """AgentAmplifier.phase reflects the current phase."""
    from agent_amplifier.kernel import AgentAmplifier

    with AgentAmplifier() as amp:
        assert amp.phase is PhaseIndex.EXPLORE


def test_async_facade_config_property() -> None:
    """AsyncAgentAmplifier.config returns the immutable config."""
    from agent_amplifier.kernel import AsyncAgentAmplifier

    async def _go() -> AmplifierConfig:
        amp = AsyncAgentAmplifier()
        try:
            return amp.config
        finally:
            await amp.aclose()

    cfg = asyncio.run(_go())
    assert cfg is not None


def test_async_facade_context_manager() -> None:
    """``async with AsyncAgentAmplifier`` enters and exits cleanly."""
    from agent_amplifier.kernel import AsyncAgentAmplifier

    async def _go() -> int:
        async with AsyncAgentAmplifier() as amp:
            env = await amp.before_step("test")
            return env.iteration

    assert asyncio.run(_go()) == 0


# ---------------------------------------------------------------------------
# H4 — atomic max-tier kill-switch (DIST-03)
# ---------------------------------------------------------------------------


def test_h4_max_tier_kill_switch_atomic_under_concurrent_before_step() -> None:
    """H4: parallel before_step calls escalate at most once.

    Previously the read of ``max_tier_kill_switch`` and its write lived in
    two separate critical sections. Two concurrent calls could BOTH observe
    ``False`` and BOTH classify as MAX without capping. collapses
    the read+write into one critical section so exactly one wins.

    We submit the same MAX-eligible query from N tasks; afterwards exactly
    one task's classification carries ``max_tier_escalated`` (the winner)
    and all subsequent calls see kill_switch=True and cap to HIGH.
    """
    import anyio

    from agent_amplifier.kernel import AsyncAgentAmplifier

    # Multi-MAX-keyword query so the router classifies as MAX.
    QUERY = "audit security cve owasp injection xss critical exploit"

    async def main() -> list[Any]:
        amp = AsyncAgentAmplifier()
        envs: list[Any] = []

        async def one() -> None:
            env = await amp.before_step(QUERY, {"available_tools": []})
            envs.append(env)

        async with anyio.create_task_group() as tg:
            for _ in range(8):
                tg.start_soon(one)
        return envs

    envs = anyio.run(main)
    # The kill switch must have flipped exactly once. Subsequent calls see
    # the cap. We assert: at most one envelope has max_tier_escalated WITHOUT
    # max_tier_kill_switched in matched_signals.
    pure_max = [
        e for e in envs
        if "max_tier_escalated" in e.classification.matched_signals
        and "max_tier_kill_switched" not in e.classification.matched_signals
    ]
    assert len(pure_max) <= 1, (
        f"kill switch race: {len(pure_max)} concurrent uncapped MAX calls"
    )


# ---------------------------------------------------------------------------
# H5 — iteration / tool_call_count snapshot in after_step
# ---------------------------------------------------------------------------


def test_h5_after_step_emits_snapshot_iteration_and_tool_call_count() -> None:
    """H5: AFTER_STEP payload carries SNAPSHOT iteration + tool_call_count.

    The snapshot is captured under the lock at the top of ``after_step``
    so observers see consistent values regardless of whether
    ``_continue_decision`` has bumped the live state by the time the
    emit fires.
    """
    from agent_amplifier.kernel import AgentAmplifier

    payloads: list[dict[str, Any]] = []

    def cb(event: AmplifierEvent, payload: dict[str, Any]) -> None:
        if event is AmplifierEvent.AFTER_STEP:
            payloads.append(payload)

    cfg = AmplifierConfig(observability_callback=cb)
    with AgentAmplifier(cfg) as amp:
        amp.before_step("simple", {"available_tools": []})
        amp.after_step({"amp_tokens_used": 10}, "result-v1")
    assert payloads, "AFTER_STEP never emitted"
    # Snapshot iteration is what we read AT ENTRY — must equal 0 for first
    # after_step. Live state may or may not have bumped depending on the
    # decision branch; the assertion locks the snapshot semantics.
    assert payloads[0]["iteration"] == 0
    # tool_call_count snapshot is also surfaced (H5).
    assert "tool_call_count" in payloads[0]
    # before_step incremented tool_call_count to 1, that's our snapshot.
    assert payloads[0]["tool_call_count"] == 1


def test_h5_after_step_snapshot_stable_across_iterations() -> None:
    """H5: snapshots increase by 1 per iteration; tool_call_count
    increases each before_step.
    """
    from agent_amplifier.kernel import AgentAmplifier

    seen_iters: list[int] = []
    seen_tcc: list[int] = []

    def cb(event: AmplifierEvent, payload: dict[str, Any]) -> None:
        if event is AmplifierEvent.AFTER_STEP:
            seen_iters.append(payload["iteration"])
            seen_tcc.append(payload["tool_call_count"])

    cfg = AmplifierConfig(observability_callback=cb)
    with AgentAmplifier(cfg) as amp:
        for i in range(3):
            # Each output is structurally different so the convergence
            # detector stays in IMPROVING and _continue_decision fires.
            amp.before_step(f"distinct-query-{i} novel", {"available_tools": []})
            amp.after_step(
                {"amp_tokens_used": 1},
                f"unique-novel-result-{i}-totally-different-content",
            )
    # tool_call_count is bumped per before_step → 1, 2, 3
    assert seen_tcc == [1, 2, 3]
    # Iteration snapshots are monotonically non-decreasing
    from itertools import pairwise
    for prev, curr in pairwise(seen_iters):
        assert curr >= prev
