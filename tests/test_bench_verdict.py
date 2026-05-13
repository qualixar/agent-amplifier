# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for F1E — AgentAssay verdict wrapper.

These tests REQUIRE AgentAssay to be importable (the Phase 4 benchmark
extra). The wrapper is thin — most testing focuses on the input contract,
the summary-line formatter, and the ImportError fallback when AgentAssay
is missing.
"""
from __future__ import annotations

import builtins
import sys
from typing import Any

import pytest

from agent_amplifier import bench_verdict as _bv

# AgentAssay must be installed in this venv; skip cleanly otherwise so
# CI without the bench extra still passes the rest of the suite.
pytest.importorskip("agentassay.verdicts.verdict")


# ---------------------------------------------------------------------------
# evaluate_completion_regression
# ---------------------------------------------------------------------------


def _booleans(n_pass: int, n_fail: int) -> list[bool]:
    return [True] * n_pass + [False] * n_fail


def test_completion_regression_identical_arms_passes_or_inconclusive() -> None:
    """Two identical arms cannot be a regression — verdict is PASS or
    INCONCLUSIVE (when sample size is small)."""
    baseline = _booleans(27, 3)  # 90% pass, n=30
    current = _booleans(27, 3)
    v = _bv.evaluate_completion_regression(baseline, current)
    assert v.status.value in ("pass", "inconclusive")
    assert v.regression_detected is False


def test_completion_regression_detects_obvious_regression() -> None:
    """High baseline pass rate vs low current → FAIL."""
    baseline = _booleans(95, 5)  # 95% pass
    current = _booleans(40, 60)  # 40% pass
    v = _bv.evaluate_completion_regression(baseline, current)
    assert v.status.value == "fail"
    assert v.regression_detected is True
    assert v.p_value is not None
    assert v.p_value < 0.05


def test_completion_regression_inconclusive_on_tiny_samples() -> None:
    """n < min_trials (=30 default) → INCONCLUSIVE regardless of difference."""
    v = _bv.evaluate_completion_regression([True, True], [False, False])
    assert v.status.value == "inconclusive"


def test_completion_regression_type_error_on_wrong_input() -> None:
    with pytest.raises(TypeError):
        _bv.evaluate_completion_regression("oops", [True])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _bv.evaluate_completion_regression([True], 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# evaluate_quality_distributions
# ---------------------------------------------------------------------------


def test_quality_distributions_identical_arms_passes() -> None:
    baseline = [0.5, 0.6, 0.55, 0.62, 0.58] * 7
    current = [0.5, 0.6, 0.55, 0.62, 0.58] * 7
    v = _bv.evaluate_quality_distributions(baseline, current)
    # Identical distributions → cannot be a regression
    assert v.status.value in ("pass", "inconclusive")
    assert v.regression_detected is False


def test_quality_distributions_detects_regression() -> None:
    baseline = [0.7, 0.8, 0.75, 0.82, 0.78] * 7  # mean ~0.77
    current = [0.3, 0.4, 0.35, 0.42, 0.38] * 7  # mean ~0.37
    v = _bv.evaluate_quality_distributions(baseline, current)
    assert v.status.value == "fail"
    assert v.regression_detected is True
    assert v.p_value is not None
    assert v.p_value < 0.05


def test_quality_distributions_type_error_on_wrong_input() -> None:
    with pytest.raises(TypeError):
        _bv.evaluate_quality_distributions("oops", [0.5])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _bv.evaluate_quality_distributions([0.5], 7)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# verdict_summary_line
# ---------------------------------------------------------------------------


def test_verdict_summary_line_from_real_verdict() -> None:
    baseline = _booleans(95, 5)
    current = _booleans(40, 60)
    v = _bv.evaluate_completion_regression(baseline, current)
    line = _bv.verdict_summary_line(v)
    assert "FAIL" in line
    assert "pass_rate=" in line
    assert "CI=[" in line
    assert "p=" in line
    assert "effect=" in line


def test_verdict_summary_line_handles_minimal_duck_object() -> None:
    """Defensive: accepts any object with the right attribute shape."""

    class _Fake:
        status = "pass"
        pass_rate = 0.85
        pass_rate_ci = (0.78, 0.92)
        p_value = None
        effect_size = None
        effect_size_interpretation = None

    line = _bv.verdict_summary_line(_Fake())
    assert "PASS" in line
    assert "0.8500" in line


def test_verdict_summary_line_with_status_enum_value() -> None:
    """Real StochasticVerdict.status is a StrEnum — exercise the .value path."""

    class _Status:
        value = "fail"

    class _Fake:
        status = _Status()
        pass_rate = 0.3
        pass_rate_ci = (0.2, 0.4)
        p_value = 0.001
        effect_size = 0.42
        effect_size_interpretation = "medium"

    line = _bv.verdict_summary_line(_Fake())
    assert "FAIL" in line
    assert "medium" in line


def test_verdict_summary_line_missing_status_attr() -> None:
    """If status is missing entirely, the formatter falls back to ``?``."""

    class _Fake: ...

    line = _bv.verdict_summary_line(_Fake())
    assert "?" in line


def test_verdict_summary_line_formats_non_float_pass_rate() -> None:
    """Defensive: non-float pass_rate is stringified via str() (line 205)."""

    class _Fake:
        status = "pass"
        pass_rate = "n/a"  # not a float — exercises str() path
        pass_rate_ci = ("low", "high")  # also non-float
        p_value = None
        effect_size = None
        effect_size_interpretation = None

    line = _bv.verdict_summary_line(_Fake())
    assert "n/a" in line  # value passed through str()
    assert "low" in line and "high" in line


# ---------------------------------------------------------------------------
# ImportError fallback when AgentAssay is missing
# ---------------------------------------------------------------------------


def test_import_error_when_agentassay_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate AgentAssay not installed → clear ImportError pointing at bench extra."""
    real_import = builtins.__import__

    def _block_agentassay(
        name: str, globals: Any = None, locals: Any = None,
        fromlist: tuple[str, ...] = (), level: int = 0,
    ) -> Any:
        if name.startswith("agentassay"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    # Drop any cached import too
    for mod in list(sys.modules):
        if mod.startswith("agentassay"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    monkeypatch.setattr(builtins, "__import__", _block_agentassay)

    with pytest.raises(ImportError) as ei:
        _bv.evaluate_completion_regression([True] * 30, [True] * 30)
    msg = str(ei.value)
    assert "AgentAssay" in msg
    assert "[bench]" in msg


def test_evaluate_scores_also_surfaces_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _block(name: str, *a: Any, **kw: Any) -> Any:
        if name.startswith("agentassay"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *a, **kw)

    for mod in list(sys.modules):
        if mod.startswith("agentassay"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", _block)

    with pytest.raises(ImportError):
        _bv.evaluate_quality_distributions([0.5] * 30, [0.5] * 30)
