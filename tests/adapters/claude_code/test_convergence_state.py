# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for F1C — per-session convergence_state classifier.

The classifier looks at the rolling history of ``quality_score`` per
session and labels the trajectory:
  - ``converged``    : current score ≥ 0.85
  - ``oscillating``  : at least one sign change among consecutive deltas
  - ``stagnant``     : every delta within ±0.05
  - ``improving``    : otherwise (and the default for a session's first turn)
  - ``None``         : current score is None (we cannot judge)

State is reconstructed from DB rows per Stop hook call — no in-memory pool
needed (each hook fires in a fresh subprocess).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_amplifier.adapters.claude_code.state import StateStore
from agent_amplifier.adapters.claude_code.stop_hook import (
    _compute_convergence_state,
)


def _new_store(tmp_path: Path) -> StateStore:
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    return s


def _seed_score(s: StateStore, turn: int, score: float | None) -> None:
    s.write_outcome(
        "sid",
        turn,
        iterations_completed=1,
        converged=True,
        quality_score=score,
    )


# ---------------------------------------------------------------------------
# state.recent_quality_scores_for_session
# ---------------------------------------------------------------------------


def test_recent_quality_scores_returns_oldest_first(tmp_path: Path) -> None:
    s = _new_store(tmp_path)
    _seed_score(s, 1, 0.20)
    _seed_score(s, 2, 0.40)
    _seed_score(s, 3, 0.60)
    assert s.recent_quality_scores_for_session("sid") == [0.20, 0.40, 0.60]


def test_recent_quality_scores_limit_truncates_oldest(tmp_path: Path) -> None:
    s = _new_store(tmp_path)
    for i, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.5], start=1):
        _seed_score(s, i, v)
    # Keep the 3 newest — order is oldest→newest within the window.
    assert s.recent_quality_scores_for_session("sid", limit=3) == [0.3, 0.4, 0.5]


def test_recent_quality_scores_rejects_zero_limit(tmp_path: Path) -> None:
    s = _new_store(tmp_path)
    with pytest.raises(ValueError):
        s.recent_quality_scores_for_session("sid", limit=0)


def test_recent_quality_scores_includes_nulls(tmp_path: Path) -> None:
    s = _new_store(tmp_path)
    _seed_score(s, 1, None)
    _seed_score(s, 2, 0.4)
    assert s.recent_quality_scores_for_session("sid") == [None, 0.4]


# ---------------------------------------------------------------------------
# _compute_convergence_state — classifier
# ---------------------------------------------------------------------------


def test_convergence_state_none_current_returns_none(tmp_path: Path) -> None:
    s = _new_store(tmp_path)
    assert _compute_convergence_state(s, "sid", None) is None


def test_convergence_state_first_turn_returns_improving(tmp_path: Path) -> None:
    """No prior history → default to improving so dashboard renders."""
    s = _new_store(tmp_path)
    assert _compute_convergence_state(s, "sid", 0.5) == "improving"


def test_convergence_state_high_score_returns_converged(tmp_path: Path) -> None:
    s = _new_store(tmp_path)
    _seed_score(s, 1, 0.4)
    assert _compute_convergence_state(s, "sid", 0.86) == "converged"


def test_convergence_state_threshold_exactly_at_boundary(tmp_path: Path) -> None:
    """0.85 is the inclusive threshold for converged."""
    s = _new_store(tmp_path)
    _seed_score(s, 1, 0.5)
    assert _compute_convergence_state(s, "sid", 0.85) == "converged"


def test_convergence_state_steady_improvement_returns_improving(
    tmp_path: Path,
) -> None:
    s = _new_store(tmp_path)
    _seed_score(s, 1, 0.30)
    _seed_score(s, 2, 0.50)
    assert _compute_convergence_state(s, "sid", 0.70) == "improving"


def test_convergence_state_flat_returns_stagnant(tmp_path: Path) -> None:
    s = _new_store(tmp_path)
    _seed_score(s, 1, 0.40)
    _seed_score(s, 2, 0.41)
    assert _compute_convergence_state(s, "sid", 0.42) == "stagnant"


def test_convergence_state_alternation_returns_oscillating(
    tmp_path: Path,
) -> None:
    """High-low-high pattern across 3 deltas → at least one sign flip."""
    s = _new_store(tmp_path)
    _seed_score(s, 1, 0.30)
    _seed_score(s, 2, 0.70)
    # Delta1: +0.4, Delta2: -0.3, Delta3: +0.2 → sign flips → oscillating
    assert _compute_convergence_state(s, "sid", 0.60) == "oscillating"


def test_convergence_state_nulls_in_history_are_skipped(tmp_path: Path) -> None:
    s = _new_store(tmp_path)
    _seed_score(s, 1, None)
    _seed_score(s, 2, 0.30)
    _seed_score(s, 3, None)
    # Effective history is [0.30]; current 0.50 → delta +0.20 (>band) → improving
    assert _compute_convergence_state(s, "sid", 0.50) == "improving"


def test_convergence_state_all_nulls_then_first_real_returns_improving(
    tmp_path: Path,
) -> None:
    s = _new_store(tmp_path)
    _seed_score(s, 1, None)
    _seed_score(s, 2, None)
    assert _compute_convergence_state(s, "sid", 0.40) == "improving"


def test_convergence_state_zero_delta_is_stagnant(tmp_path: Path) -> None:
    """Identical scores yield zero delta — within stagnant band."""
    s = _new_store(tmp_path)
    _seed_score(s, 1, 0.50)
    _seed_score(s, 2, 0.50)
    assert _compute_convergence_state(s, "sid", 0.50) == "stagnant"


def test_convergence_state_descent_is_improving_not_oscillating(
    tmp_path: Path,
) -> None:
    """Pure descent (no sign flip) is classified as improving (movement),
    not stagnant. The convergence-state metric tracks trajectory shape;
    a separate quality_score is the actual quality signal.
    """
    s = _new_store(tmp_path)
    _seed_score(s, 1, 0.80)
    _seed_score(s, 2, 0.60)
    assert _compute_convergence_state(s, "sid", 0.40) == "improving"
