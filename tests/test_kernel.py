# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.kernel`` (, /2/3/7/8/13/18/19/21).

Coverage map:
    A. Construction (cases 1-5)
    B. before_step (cases 6-13)
    C. after_step (cases 14-21)
    D. finalize (cases 22-24)
    E. Observability (cases 25-28)
    F. Async facade (cases 29-30)
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from agent_amplifier.types import (
    AmplifierConfig,
    AmplifierEvent,
    EffortLevel,
    PhaseIndex,
)

# ---------------------------------------------------------------------------
# A. Construction (cases 1-5) — V2.1: SLM coupling removed ().
# ---------------------------------------------------------------------------


def test_a1_default_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        assert amp.config is not None
    finally:
        amp.close()


def test_a2_explicit_config_honored() -> None:
    from agent_amplifier.kernel import AgentAmplifier

    cfg = AmplifierConfig(max_iterations=5, goal_reinjection_interval=7)
    amp = AgentAmplifier(config=cfg)
    try:
        assert amp.config.max_iterations == 5
        assert amp.config.goal_reinjection_interval == 7
    finally:
        amp.close()


def test_a3_default_memory_plane_yields_empty_recall() -> None:
    """V2.1 contract: with no adapter and no callbacks, recall returns []."""
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        assert amp._resolve_recall("anything", 3) == []
    finally:
        amp.close()


def test_a4_default_memory_plane_remember_is_no_op() -> None:
    """V2.1 contract: with no adapter and no callbacks, remember is a no-op."""
    from agent_amplifier.kernel import AgentAmplifier
    from agent_amplifier.types import EffortLevel, Outcome

    amp = AgentAmplifier()
    try:
        # Must not raise.
        amp._resolve_remember(
            Outcome(
                query="x",
                effort=EffortLevel.LOW,
                iterations=1,
                quality=0.5,
            )
        )
    finally:
        amp.close()


def test_a5_config_immutability(monkeypatch: pytest.MonkeyPatch) -> None:
    """mutating ``amp.config`` raises AttributeError."""
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        with pytest.raises(AttributeError):
            amp._core._config = AmplifierConfig(max_iterations=10)  # type: ignore[misc]
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# B. before_step (cases 6-13)
# ---------------------------------------------------------------------------


def test_b6_before_step_minimal_query() -> None:
    """Effort router walk-order: 'rename foo' is purely MINIMAL ('rename')."""
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        env = amp.before_step("rename foo to bar")
        assert env.classification.complexity == EffortLevel.MINIMAL
    finally:
        amp.close()


def test_b7_before_step_high_query() -> None:
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        env = amp.before_step("redesign authentication subsystem with OWASP review")
        assert env.classification.complexity in (EffortLevel.HIGH, EffortLevel.MAX)
    finally:
        amp.close()


def test_b8_envelope_contains_phase_prompt_and_query() -> None:
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        env = amp.before_step("audit security of payments service")
        # phase prompt is anchored in the envelope (EXPLORE phase first)
        assert "EXPLORE" in env.envelope
        assert "payments service" in env.envelope
    finally:
        amp.close()


def test_b9_system_reminder_block_present_when_modifiers_chosen() -> None:
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        env = amp.before_step("audit security of payments service")
        assert "<system-reminder" in env.envelope
        assert "</system-reminder" in env.envelope
    finally:
        amp.close()


def test_b10_recommended_tools_intersect_with_available() -> None:
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        env = amp.before_step(
            "research best practice for OWASP",
            {"available_tools": ["Read", "WebSearch", "Bash"]},
        )
        assert set(env.recommended_tools).issubset(
            {"Read", "WebSearch", "Bash"}
        )
    finally:
        amp.close()


def test_b13_top_level_try_catch_returns_degenerate_envelope() -> None:
    """bad context shape returns degenerate envelope, no raise."""
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        env = amp.before_step("ok query", {"available_tools": 42})  # type: ignore[arg-type]
        # Did not raise: degenerate envelope present.
        assert env.classification.domain == "degraded"
        assert env.recommended_tools == ()
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# C. after_step (cases 14-21)
# ---------------------------------------------------------------------------


def test_c14_converged_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_amplifier.kernel import AgentAmplifier

    # Use AGENT_AMP_FALLBACK_PHASE so phase advancement past EXPLORE doesn't
    # demand prev_output from this minimal test harness.
    monkeypatch.setenv("AGENT_AMP_FALLBACK_PHASE", "EXPLORE")
    amp = AgentAmplifier()
    try:
        decision = None
        # Run two before/after to allow convergence detection.
        for _ in range(2):
            env = amp.before_step("identical task")
            decision = amp.after_step(env, "identical output")
        # Identical outputs → CONVERGED on the second pass.
        assert decision is not None
        assert decision["action"] == "stop"
        assert decision["reason"] == "converged"
    finally:
        amp.close()


def test_c20_step_id_increments_per_before_step() -> None:
    """tool_call_count + step_id each increment by exactly 1 per call."""
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        e1 = amp.before_step("q1")
        e2 = amp.before_step("q1")
        e3 = amp.before_step("q1")
        assert e1.step_id == 1 and e2.step_id == 2 and e3.step_id == 3
    finally:
        amp.close()


def test_c19_continue_advances_phase() -> None:
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        env = amp.before_step("design a payments arch")
        assert env.phase == "EXPLORE"
        decision = amp.after_step(env, "first approach: A vs B vs C - AWAITING-EVALUATION")
        # If not converged, phase advances on continue.
        if decision["action"] == "continue":
            assert decision["phase"] != "EXPLORE"
    finally:
        amp.close()


def test_c11_phase_slot_missing_raises_kernel_contract_error() -> None:
    """missing phase slot raises ``KernelContractError`` with 3-field message
    (when AGENT_AMP_FALLBACK_PHASE is unset).
    """
    from agent_amplifier.kernel import AgentAmplifier, KernelContractError

    # Force phase to EVALUATE (which requires prev_output) without supplying it.
    amp = AgentAmplifier()
    try:
        # Manually set phase past EXPLORE without giving prev_output.
        amp._core._state.phase = PhaseIndex.EVALUATE  # type: ignore[attr-defined]

        # Save and clear env var to ensure raise path.
        prev = os.environ.pop("AGENT_AMP_FALLBACK_PHASE", None)
        try:
            with pytest.raises(KernelContractError) as exc:
                # Internal path so the outer try/except does not swallow.
                asyncio.run(
                    amp._core._before_step_inner("redesign auth", {})  # type: ignore[attr-defined]
                )
            msg = str(exc.value)
            assert "why" in msg and "fix" in msg
            assert "AGENT_AMP_FALLBACK_PHASE" in msg
        finally:
            if prev is not None:
                os.environ["AGENT_AMP_FALLBACK_PHASE"] = prev
    finally:
        amp.close()


def test_c12_fallback_phase_resets_on_slot_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_amplifier.kernel import AgentAmplifier

    monkeypatch.setenv("AGENT_AMP_FALLBACK_PHASE", "EXPLORE")
    amp = AgentAmplifier()
    try:
        amp._core._state.phase = PhaseIndex.EVALUATE  # type: ignore[attr-defined]
        env = amp.before_step("redesign auth")
        # Reset to EXPLORE because EVALUATE's slots were missing.
        assert env.phase == "EXPLORE"
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# D. finalize (cases 22-24) —
# ---------------------------------------------------------------------------


def test_d22_finalize_returns_report() -> None:
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        amp.before_step("simple task")
        amp.after_step({}, "result text")
        report = amp.finalize()
        assert "iterations_completed" in report
        assert "final_state" in report
    finally:
        amp.close()


def test_d24_finalize_idempotent() -> None:
    """second finalize is a no-op returning the same report shape."""
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        amp.before_step("task")
        amp.after_step({}, "result")
        r1 = amp.finalize()
        r2 = amp.finalize()
        assert r1["iterations_completed"] == r2["iterations_completed"]
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# E. Observability (cases 25-28) — ,
# ---------------------------------------------------------------------------


def test_e25_observability_callback_fires_on_before_step() -> None:
    """callback receives BEFORE_STEP and AFTER_STEP events."""
    from agent_amplifier.kernel import AgentAmplifier

    seen: list[tuple[AmplifierEvent, dict[str, Any]]] = []

    def cb(event, payload):  # type: ignore[no-untyped-def]
        seen.append((event, payload))

    cfg = AmplifierConfig(observability_callback=cb)
    amp = AgentAmplifier(config=cfg)
    try:
        amp.before_step("simple")
        amp.after_step({}, "x")
    finally:
        amp.close()

    events_seen = [e for e, _ in seen]
    assert AmplifierEvent.BEFORE_STEP in events_seen
    assert AmplifierEvent.AFTER_STEP in events_seen


def test_e26_callback_exception_does_not_crash_kernel() -> None:
    """NOTE- a raising callback is caught, not bubbled up."""
    from agent_amplifier.kernel import AgentAmplifier

    def cb(event, payload):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    cfg = AmplifierConfig(observability_callback=cb)
    amp = AgentAmplifier(config=cfg)
    try:
        env = amp.before_step("simple")
        # Did not raise; we got an envelope back.
        assert env.iteration == 0
    finally:
        amp.close()


# ---------------------------------------------------------------------------
# F. Async facade (cases 29-30)
# ---------------------------------------------------------------------------


def test_f29_async_facade_under_asyncio() -> None:
    from agent_amplifier.kernel import AsyncAgentAmplifier

    async def _go() -> None:
        amp = AsyncAgentAmplifier()
        try:
            env = await amp.before_step("simple async task")
            assert env.iteration == 0
            decision = await amp.after_step(env, "result")
            assert "action" in decision
            r = await amp.finalize()
            assert "iterations_completed" in r
        finally:
            await amp.aclose()

    asyncio.run(_go())


def test_f30_async_facade_concurrent_step_id_distinct() -> None:
    """concurrent before_step calls each get distinct, monotone step_ids."""
    from agent_amplifier.kernel import AsyncAgentAmplifier

    async def _go() -> list[int]:
        amp = AsyncAgentAmplifier()
        try:
            envs = await asyncio.gather(
                amp.before_step("a"),
                amp.before_step("b"),
                amp.before_step("c"),
                amp.before_step("d"),
            )
            return [e.step_id for e in envs]
        finally:
            await amp.aclose()

    ids = asyncio.run(_go())
    assert sorted(ids) == [1, 2, 3, 4]
    assert len(set(ids)) == 4


# ---------------------------------------------------------------------------
# Additional kernel-specific behaviors
# ---------------------------------------------------------------------------


def test_amplify_one_shot_helper() -> None:
    """``amplify`` one-shot returns a dict that mirrors StepEnvelope.to_dict."""
    from agent_amplifier.kernel import amplify

    out = amplify("simple task")
    assert "amp_iteration" in out
    assert "amp_envelope" in out


def test_step_envelope_to_dict_round_trip() -> None:
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        env = amp.before_step("hello")
        d = env.to_dict()
        assert d["amp_iteration"] == 0
        assert "amp_envelope" in d
    finally:
        amp.close()


def test_context_manager_protocol() -> None:
    from agent_amplifier.kernel import AgentAmplifier

    with AgentAmplifier() as amp:
        env = amp.before_step("hello")
        assert env.iteration == 0


def test_step_envelope_budget_is_dict_not_dataclass() -> None:
    """CRIT-1 lock: ``StepEnvelope.budget`` must be a Mapping (asdict-safe)
    even though ``BudgetReport`` is slotted+frozen and has no ``__dict__``.
    Earlier, `dict(report.__dict__)` raised AttributeError silently and the
    outer try/except masked the bug as a degenerate envelope.

    budget is now a ``MappingProxyType`` so consumers
    cannot mutate the kernel's snapshot.  We assert ``Mapping`` (covers both
    dict and MappingProxy) and verify the keys + value types.
    """
    from collections.abc import Mapping

    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        env = amp.before_step("simple")
        assert isinstance(env.budget, Mapping)
        # Required fields from BudgetReport.
        for k in ("mode", "allocated", "used", "remaining"):
            assert k in env.budget, f"missing {k} in envelope.budget"
        # mode must be a JSON-safe string, NOT a BudgetMode enum object.
        assert isinstance(env.budget["mode"], str)
    finally:
        amp.close()


def test_finalize_report_carries_budget_keys() -> None:
    """CRIT-1 lock: finalize() must produce a dict with all budget keys."""
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        amp.before_step("simple")
        amp.after_step({}, "x")
        report = amp.finalize()
        for k in (
            "mode",
            "allocated",
            "used",
            "remaining",
            "iterations_completed",
            "final_state",
            "drift_at_end",
            "max_tier_kill_switched",
        ):
            assert k in report, f"missing {k} in finalize report"
    finally:
        amp.close()
