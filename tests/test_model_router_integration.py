# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""TDD tests for model router integration into StepEnvelope + report."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from agent_amplifier.kernel import StepEnvelope
from agent_amplifier.types import EffortLevel, TaskClassification


def _make_envelope(**overrides: Any) -> StepEnvelope:
    defaults: dict[str, Any] = {
        "classification": TaskClassification(
            complexity=EffortLevel.HIGH,
            domain="general",
            estimated_tokens=100,
            confidence=0.9,
        ),
        "thinking_trigger": "think hard",
        "phase": "EXPLORE",
        "persona": "Senior Engineer",
        "recommended_tools": (),
        "recommended_groups": (),
        "iteration": 0,
        "step_id": 1,
        "envelope": "test envelope",
        "budget": MappingProxyType({}),
        "recalled_patterns": (),
        "suggested_model": "opus",
        "extras": MappingProxyType({}),
    }
    defaults.update(overrides)
    return StepEnvelope(**defaults)


class TestStepEnvelopeHasSuggestedModel:
    def test_suggested_model_field_exists(self) -> None:
        env = _make_envelope(suggested_model="opus")
        assert env.suggested_model == "opus"

    def test_suggested_model_none(self) -> None:
        env = _make_envelope(suggested_model=None)
        assert env.suggested_model is None

    def test_suggested_model_in_to_dict(self) -> None:
        env = _make_envelope(suggested_model="sonnet")
        d = env.to_dict()
        assert "amp_suggested_model" in d
        assert d["amp_suggested_model"] == "sonnet"

    def test_suggested_model_none_in_to_dict(self) -> None:
        env = _make_envelope(suggested_model=None)
        d = env.to_dict()
        assert d["amp_suggested_model"] is None

    def test_suggested_model_frozen(self) -> None:
        env = _make_envelope(suggested_model="opus")
        try:
            env.suggested_model = "haiku"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


class TestKernelPopulatesSuggestedModel:
    def test_before_step_populates_suggested_model(self) -> None:
        from agent_amplifier import AgentAmplifier

        amp = AgentAmplifier()
        env = amp.before_step("Refactor the auth module to use JWT tokens")
        assert env.suggested_model is not None
        assert env.suggested_model in ("haiku", "sonnet", "opus")

    def test_high_complexity_gets_opus(self) -> None:
        from agent_amplifier import AgentAmplifier

        amp = AgentAmplifier()
        env = amp.before_step(
            "Redesign the entire distributed caching layer with "
            "consistent hashing, circuit breakers, and automatic "
            "failover across three data centers"
        )
        if env.classification.complexity in (EffortLevel.HIGH, EffortLevel.MAX):
            assert env.suggested_model == "opus"

    def test_minimal_complexity_gets_haiku(self) -> None:
        from agent_amplifier import AgentAmplifier

        amp = AgentAmplifier()
        env = amp.before_step("yes")
        if env.classification.complexity == EffortLevel.MINIMAL:
            assert env.suggested_model == "haiku"

    def test_degraded_envelope_has_none_model(self) -> None:
        env = _make_envelope(suggested_model=None)
        assert env.suggested_model is None


class TestPhaseGatingByComplexity:
    def test_minimal_prompt_no_explore_scaffold(self) -> None:
        from agent_amplifier import AgentAmplifier

        amp = AgentAmplifier()
        env = amp.before_step("yes")
        assert "AWAITING-EVALUATION" not in env.envelope
        assert "Cast a wide net" not in env.envelope
        assert "AT LEAST 3" not in env.envelope

    def test_low_prompt_no_explore_scaffold(self) -> None:
        from agent_amplifier import AgentAmplifier

        amp = AgentAmplifier()
        env = amp.before_step("fix the typo on line 12")
        if env.classification.complexity == EffortLevel.LOW:
            assert "AWAITING-EVALUATION" not in env.envelope

    def test_high_prompt_keeps_explore_scaffold(self) -> None:
        from agent_amplifier import AgentAmplifier

        amp = AgentAmplifier()
        env = amp.before_step(
            "Redesign the entire distributed caching layer with "
            "consistent hashing, circuit breakers, and automatic "
            "failover across three data centers with hot standby"
        )
        if env.classification.complexity in (EffortLevel.HIGH, EffortLevel.MAX):
            assert "PHASE:" in env.envelope
