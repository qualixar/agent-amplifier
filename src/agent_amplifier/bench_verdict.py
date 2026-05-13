# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""F1E — Statistical verdict wrapper for Phase 4 A/B benchmarks.

Delegates to AgentAssay's published statistical framework (Definition 3.2
in the AgentAssay paper, ``agentassay.verdicts.VerdictFunction``). The
benchmark harness in Phase 4 uses this module to convert raw per-task
arms — ``raw Sonnet`` vs ``Sonnet + Agent Amplifier`` vs ``raw Opus`` —
into a defensible ``StochasticVerdict`` with Wilson confidence intervals,
Fisher exact tests on completion rates, and Mann-Whitney U on
quality_score distributions.

Why this lives here and not in AgentAssay:
  * AgentAssay is the general-purpose stochastic test framework
    (regression / threshold verdicts for any agent).
  * Agent Amplifier wraps it with the *specific* benchmark-arm shape
    (completion as binary, quality_score as continuous) so the Phase 4
    pipeline calls a one-line API:
        ``evaluate_completion_regression(baseline, current)``
        ``evaluate_quality_distributions(baseline, current)``

Optional dependency: AgentAssay is installed only when users install the
``[bench]`` extra. If it is missing, the wrappers raise a clear
``ImportError`` pointing at ``pip install 'agent-amplifier[bench]'``.

Cross-product credit: every public-facing release of Agent Amplifier
that cites these verdicts MUST link to AgentAssay
(https://github.com/qualixar/agentassay) per the Qualixar
AI Reliability Engineering category policy.
"""
from __future__ import annotations

import logging
from typing import Any

LOG = logging.getLogger("agent_amplifier.bench_verdict")

# Default verdict parameters — match AgentAssay's published defaults so
# the numbers in the Agent Amplifier paper / blog are directly comparable
# to AgentAssay benchmarks.
_DEFAULT_ALPHA: float = 0.05
_DEFAULT_BETA: float = 0.20
_DEFAULT_MIN_TRIALS: int = 30
_DEFAULT_CONFIDENCE_METHOD: str = "wilson"
_DEFAULT_REGRESSION_TEST: str = "fisher"
_DEFAULT_SCORE_TEST: str = "mann_whitney"


_IMPORT_HINT: str = (
    "AgentAssay is required for Phase 4 benchmark verdicts. Install with: "
    "pip install 'agent-amplifier[bench]'  (or 'pip install agentassay' "
    "directly). See https://github.com/qualixar/agentassay for source."
)


def _build_verdict_function(
    *,
    alpha: float,
    beta: float,
    min_trials: int,
    confidence_method: str,
    regression_test: str,
) -> Any:
    """Construct an AgentAssay ``VerdictFunction``. Lazy import so the
    module is loadable even when AgentAssay is not installed (users who
    never call Phase 4 should not pay for the dep).
    """
    try:
        from agentassay.verdicts.verdict import VerdictFunction
    except ImportError as exc:
        raise ImportError(_IMPORT_HINT) from exc
    return VerdictFunction(
        alpha=alpha,
        beta=beta,
        min_trials=min_trials,
        confidence_method=confidence_method,
        regression_test=regression_test,
    )


def evaluate_completion_regression(
    baseline_results: list[bool],
    current_results: list[bool],
    *,
    alpha: float = _DEFAULT_ALPHA,
    beta: float = _DEFAULT_BETA,
    min_trials: int = _DEFAULT_MIN_TRIALS,
    confidence_method: str = _DEFAULT_CONFIDENCE_METHOD,
    regression_test: str = _DEFAULT_REGRESSION_TEST,
) -> Any:
    """Statistically compare two arms' boolean completion outcomes.

    Returns an AgentAssay ``StochasticVerdict`` with status ∈
    ``{PASS, FAIL, INCONCLUSIVE}``. PASS means the current arm has not
    regressed against the baseline at the specified ``alpha`` /
    ``beta``; FAIL means a significant regression was detected;
    INCONCLUSIVE typically means the sample is too small to call.

    Use case in Phase 4: ``baseline_results = raw Sonnet completion
    flags; current_results = Sonnet + AA completion flags`` (or any
    other arm pair you want to A/B).
    """
    if not isinstance(baseline_results, list) or not isinstance(
        current_results, list
    ):
        raise TypeError(
            "baseline_results and current_results must both be lists of bool"
        )
    vf = _build_verdict_function(
        alpha=alpha,
        beta=beta,
        min_trials=min_trials,
        confidence_method=confidence_method,
        regression_test=regression_test,
    )
    return vf.evaluate_regression(baseline_results, current_results)


def evaluate_quality_distributions(
    baseline_scores: list[float],
    current_scores: list[float],
    *,
    alpha: float = _DEFAULT_ALPHA,
    beta: float = _DEFAULT_BETA,
    min_trials: int = _DEFAULT_MIN_TRIALS,
    confidence_method: str = _DEFAULT_CONFIDENCE_METHOD,
    score_test: str = _DEFAULT_SCORE_TEST,
) -> Any:
    """Mann-Whitney U / KS / Welch's-t verdict on continuous score arms.

    Use case in Phase 4: feed the per-task ``quality_score`` distribution
    from one arm vs another. Returns an AgentAssay ``StochasticVerdict``
    where the ``pass_rate`` field is the synthetic "fraction of current
    scores at or above baseline median" and ``p_value`` /
    ``effect_size`` carry the non-parametric test results.
    """
    if not isinstance(baseline_scores, list) or not isinstance(
        current_scores, list
    ):
        raise TypeError(
            "baseline_scores and current_scores must both be lists of float"
        )
    vf = _build_verdict_function(
        alpha=alpha,
        beta=beta,
        min_trials=min_trials,
        confidence_method=confidence_method,
        regression_test=_DEFAULT_REGRESSION_TEST,  # ignored for scores
    )
    return vf.evaluate_scores(
        baseline_scores, current_scores, score_test=score_test
    )


def verdict_summary_line(verdict: Any) -> str:
    """Render a single-line summary of an AgentAssay ``StochasticVerdict``
    suitable for the Phase 4 console output and the viral-post payload.

    Format::

        <STATUS>  pass_rate=<rate> CI=[<lo>, <hi>]  p=<p>  effect=<eff> (<label>)

    Defensive: accepts any object that quacks like ``StochasticVerdict``
    (status, pass_rate, pass_rate_ci, p_value, effect_size attributes).
    """
    status = getattr(getattr(verdict, "status", None), "value", None) or str(
        getattr(verdict, "status", "?")
    )
    pass_rate = getattr(verdict, "pass_rate", None)
    ci = getattr(verdict, "pass_rate_ci", (None, None))
    p_value = getattr(verdict, "p_value", None)
    effect = getattr(verdict, "effect_size", None)
    label = getattr(verdict, "effect_size_interpretation", None) or "n/a"

    def _fmt(v: object) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    return (
        f"{str(status).upper()}  "
        f"pass_rate={_fmt(pass_rate)} "
        f"CI=[{_fmt(ci[0])}, {_fmt(ci[1])}]  "
        f"p={_fmt(p_value)}  "
        f"effect={_fmt(effect)} ({label})"
    )


__all__ = [
    "evaluate_completion_regression",
    "evaluate_quality_distributions",
    "verdict_summary_line",
]
