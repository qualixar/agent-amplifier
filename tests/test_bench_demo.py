# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.bench_demo``.

Coverage targets: 100% line + 100% branch on bench_demo.py.

Strategy: most tests run the real kernel with synthetic prompts so the
default rendering path is exercised end-to-end. Branch coverage for
``thinking_trigger=None`` / empty persona / empty recommended_groups
uses a fake envelope produced via a stubbed ``AgentAmplifier`` so we
hit those falsy paths deterministically.
"""
from __future__ import annotations

from types import MappingProxyType

import pytest

from agent_amplifier import bench_demo as _demo
from agent_amplifier.kernel import StepEnvelope
from agent_amplifier.types import EffortLevel, TaskClassification

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_envelope(
    *,
    complexity: EffortLevel = EffortLevel.HIGH,
    domain: str = "security",
    thinking_trigger: str | None = "think harder",
    persona: str = "PERSONA: Senior security engineer",
    recommended_groups: tuple[str, ...] = ("auth",),
    envelope_text: str = "[envelope text body]",
) -> StepEnvelope:
    """Build a controlled StepEnvelope for branch-coverage tests."""
    return StepEnvelope(
        classification=TaskClassification(
            complexity=complexity,
            domain=domain,
            estimated_tokens=100,
            confidence=0.9,
        ),
        thinking_trigger=thinking_trigger,
        phase="EXPLORE",
        persona=persona,
        recommended_tools=(),
        recommended_groups=recommended_groups,
        iteration=0,
        step_id=1,
        envelope=envelope_text,
        budget=MappingProxyType({}),
        recalled_patterns=(),
        suggested_model=None,
        extras=MappingProxyType({}),
    )


class _StubAmp:
    """Minimal AgentAmplifier stand-in: returns a controlled envelope."""

    def __init__(self, env: StepEnvelope) -> None:
        self._env = env

    def before_step(self, _query: str) -> StepEnvelope:
        return self._env

    def close(self) -> None:
        return None


@pytest.fixture
def stub_amp(monkeypatch: pytest.MonkeyPatch):
    """Patch ``AgentAmplifier`` in bench_demo with a stub that returns the
    envelope passed to the fixture function."""

    def _install(env: StepEnvelope) -> None:
        monkeypatch.setattr(
            _demo, "AgentAmplifier", lambda **_: _StubAmp(env)
        )

    return _install


# ---------------------------------------------------------------------------
# Empty / whitespace prompts → exit 1
# ---------------------------------------------------------------------------


def test_run_demo_empty_prompt_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _demo.run_demo("") == 1
    err = capsys.readouterr().err
    assert "must be non-empty" in err


def test_run_demo_whitespace_prompt_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _demo.run_demo("   \n\t  ") == 1
    err = capsys.readouterr().err
    assert "must be non-empty" in err


# ---------------------------------------------------------------------------
# Baseline-only path → no envelope built, no delta line
# ---------------------------------------------------------------------------


def test_run_demo_baseline_only_skips_kernel(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """show_amplified=False MUST NOT instantiate AgentAmplifier — the
    kernel can be expensive on cold paths and we want the baseline-only
    output to be a fast pass-through."""

    def _no_kernel_allowed(**_: object) -> object:  # pragma: no cover
        raise AssertionError("AgentAmplifier should not be built when show_amplified=False")

    monkeypatch.setattr(_demo, "AgentAmplifier", _no_kernel_allowed)
    rc = _demo.run_demo(
        "do a thing", show_baseline=True, show_amplified=False
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Baseline:" in out
    assert "do a thing" in out
    assert "Amplified:" not in out
    assert "Delta:" not in out


# ---------------------------------------------------------------------------
# Amplified-only path → no baseline section, no delta
# ---------------------------------------------------------------------------


def test_run_demo_amplified_only_skips_baseline_and_delta(
    stub_amp,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub_amp(_make_envelope())
    rc = _demo.run_demo(
        "x", show_baseline=False, show_amplified=True
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Amplified:" in out
    assert "Baseline:" not in out
    assert "Delta:" not in out


# ---------------------------------------------------------------------------
# Both halves → baseline + amplified + delta
# ---------------------------------------------------------------------------


def test_run_demo_both_halves_shows_delta(
    stub_amp,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub_amp(_make_envelope())
    rc = _demo.run_demo("x", show_baseline=True, show_amplified=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Baseline:" in out
    assert "Amplified:" in out
    assert "Delta:" in out
    assert "x prompt size" in out
    # classification + phase render
    assert "complexity=high" in out
    assert "domain=security" in out
    assert "phase: EXPLORE" in out
    # optional fields render when populated
    assert "thinking trigger: think harder" in out
    assert "persona: PERSONA: Senior security engineer" in out
    assert "tool groups: auth" in out


# ---------------------------------------------------------------------------
# Falsy-fields branch: thinking_trigger=None, persona="", groups=()
# ---------------------------------------------------------------------------


def test_run_demo_amplified_with_falsy_optional_fields(
    stub_amp,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub_amp(_make_envelope(
        thinking_trigger=None,
        persona="",
        recommended_groups=(),
    ))
    rc = _demo.run_demo("x", show_baseline=False, show_amplified=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Amplified:" in out
    # Optional sections elided when falsy.
    assert "thinking trigger" not in out
    assert "persona:" not in out
    assert "tool groups" not in out


# ---------------------------------------------------------------------------
# Long persona truncation: len > 80 → ellipsis
# ---------------------------------------------------------------------------


def test_run_demo_truncates_long_persona(
    stub_amp,
    capsys: pytest.CaptureFixture[str],
) -> None:
    long = "PERSONA: " + ("a" * 200)
    stub_amp(_make_envelope(persona=long))
    rc = _demo.run_demo("x", show_baseline=False, show_amplified=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "..." in out
    # Truncated to 80 chars + "..."; the full long persona is not present.
    assert long not in out


# ---------------------------------------------------------------------------
# Neither half → returns 0 with empty stdout
# ---------------------------------------------------------------------------


def test_run_demo_neither_half_returns_zero_with_no_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _no_kernel_allowed(**_: object) -> object:  # pragma: no cover
        raise AssertionError("AgentAmplifier should not build when both flags False")

    monkeypatch.setattr(_demo, "AgentAmplifier", _no_kernel_allowed)
    rc = _demo.run_demo(
        "x", show_baseline=False, show_amplified=False
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Baseline:" not in out
    assert "Amplified:" not in out


# ---------------------------------------------------------------------------
# End-to-end with the real kernel — proves no integration drift
# ---------------------------------------------------------------------------


def test_run_demo_end_to_end_with_real_kernel(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = _demo.run_demo("Refactor the auth middleware to use JWT")
    assert rc == 0
    out = capsys.readouterr().out
    assert "Baseline:" in out
    assert "Amplified:" in out
    assert "Delta:" in out
