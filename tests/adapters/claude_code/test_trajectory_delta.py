# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for F1B — Tier 3 trajectory-delta quality penalty.

The trajectory delta subtracts up to ``_TRAJECTORY_DELTA_FLOOR`` (= -0.25
in v1.1) from the Tier 1 Jaccard quality score for deterministic agent
anti-patterns observable from the tool-call event log:

  * Loop: 3+ consecutive PreToolUse events with the same (tool_name,
    payload) tuple.
  * Missing reconnaissance: Edit/Write/MultiEdit on a file_path that was
    never Read earlier in the same turn.

Fail-open: any exception during computation returns 0.0 (no penalty),
preserving the v1.0 hook fail-open contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_amplifier.adapters.claude_code.state import StateStore
from agent_amplifier.adapters.claude_code.stop_hook import (
    _compute_trajectory_delta,
    _extract_file_path,
)

# ---------------------------------------------------------------------------
# state.list_events_for_turn — F1B read API
# ---------------------------------------------------------------------------


def test_list_events_for_turn_returns_ordered_rows(tmp_path: Path) -> None:
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    for i, kind in enumerate(["PreToolUse", "PostToolUse", "PreToolUse"], start=1):
        s.record_event(
            "sid", event_type=kind, turn_id=1, tool_name=f"T{i}", payload={"i": i}
        )
    events = s.list_events_for_turn("sid", 1)
    assert [e["event_type"] for e in events] == [
        "PreToolUse",
        "PostToolUse",
        "PreToolUse",
    ]
    assert [e["tool_name"] for e in events] == ["T1", "T2", "T3"]


def test_list_events_for_turn_empty_when_no_match(tmp_path: Path) -> None:
    s = StateStore(tmp_path / "state.db")
    assert s.list_events_for_turn("absent", 1) == []


# ---------------------------------------------------------------------------
# _extract_file_path — robust string parser
# ---------------------------------------------------------------------------


def test_extract_file_path_from_typical_payload() -> None:
    payload = json.dumps(
        {"input_summary": "{'file_path': '/a/b.py', 'limit': 5}"}
    )
    assert _extract_file_path(payload) == "/a/b.py"


def test_extract_file_path_returns_none_for_no_match() -> None:
    payload = json.dumps({"input_summary": "{'command': 'ls -la'}"})
    assert _extract_file_path(payload) is None


def test_extract_file_path_returns_none_for_invalid_json() -> None:
    assert _extract_file_path("not json") is None


def test_extract_file_path_returns_none_for_empty_payload() -> None:
    assert _extract_file_path("") is None


def test_extract_file_path_returns_none_when_input_summary_missing() -> None:
    payload = json.dumps({"other": "field"})
    assert _extract_file_path(payload) is None


def test_extract_file_path_returns_none_when_payload_is_not_dict() -> None:
    payload = json.dumps(["just", "an", "array"])
    assert _extract_file_path(payload) is None


def test_extract_file_path_returns_none_when_summary_is_not_string() -> None:
    payload = json.dumps({"input_summary": {"nested": "obj"}})
    assert _extract_file_path(payload) is None


# ---------------------------------------------------------------------------
# _compute_trajectory_delta — penalty composition
# ---------------------------------------------------------------------------


def _seed_pretool(
    s: StateStore,
    session: str,
    turn: int,
    tool: str,
    payload: dict | None = None,
) -> None:
    s.record_event(
        session,
        event_type="PreToolUse",
        turn_id=turn,
        tool_name=tool,
        payload=payload or {"input_summary": "{}"},
    )


def test_trajectory_delta_clean_returns_zero(tmp_path: Path) -> None:
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    _seed_pretool(
        s, "sid", 1, "Read", {"input_summary": "{'file_path': '/a.py'}"}
    )
    _seed_pretool(
        s, "sid", 1, "Edit", {"input_summary": "{'file_path': '/a.py'}"}
    )
    assert _compute_trajectory_delta(s, "sid", 1) == 0.0


def test_trajectory_delta_loop_applies_loop_penalty(tmp_path: Path) -> None:
    """3 consecutive identical PreToolUse events trigger -0.10."""
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    same_payload = {"input_summary": "{'command': 'ls'}"}
    for _ in range(3):
        _seed_pretool(s, "sid", 1, "Bash", same_payload)
    assert _compute_trajectory_delta(s, "sid", 1) == pytest.approx(-0.10)


def test_trajectory_delta_loop_at_four_still_only_applies_once(
    tmp_path: Path,
) -> None:
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    same_payload = {"input_summary": "{'command': 'ls'}"}
    for _ in range(4):
        _seed_pretool(s, "sid", 1, "Bash", same_payload)
    assert _compute_trajectory_delta(s, "sid", 1) == pytest.approx(-0.10)


def test_trajectory_delta_two_identical_does_not_trigger_loop(
    tmp_path: Path,
) -> None:
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    same_payload = {"input_summary": "{'command': 'ls'}"}
    for _ in range(2):
        _seed_pretool(s, "sid", 1, "Bash", same_payload)
    assert _compute_trajectory_delta(s, "sid", 1) == 0.0


def test_trajectory_delta_missing_recon_applies_recon_penalty(
    tmp_path: Path,
) -> None:
    """Edit without prior Read of the same file → -0.10."""
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    _seed_pretool(
        s, "sid", 1, "Edit", {"input_summary": "{'file_path': '/never_read.py'}"}
    )
    assert _compute_trajectory_delta(s, "sid", 1) == pytest.approx(-0.10)


def test_trajectory_delta_recon_satisfied_no_penalty(tmp_path: Path) -> None:
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    _seed_pretool(
        s, "sid", 1, "Read", {"input_summary": "{'file_path': '/x.py'}"}
    )
    _seed_pretool(
        s, "sid", 1, "Edit", {"input_summary": "{'file_path': '/x.py'}"}
    )
    _seed_pretool(
        s, "sid", 1, "Write", {"input_summary": "{'file_path': '/x.py'}"}
    )
    assert _compute_trajectory_delta(s, "sid", 1) == 0.0


def test_trajectory_delta_recon_penalty_only_once_per_turn(
    tmp_path: Path,
) -> None:
    """Two unrelated unread mutations still cap at single -0.10."""
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    _seed_pretool(
        s, "sid", 1, "Edit", {"input_summary": "{'file_path': '/a.py'}"}
    )
    _seed_pretool(
        s, "sid", 1, "Write", {"input_summary": "{'file_path': '/b.py'}"}
    )
    assert _compute_trajectory_delta(s, "sid", 1) == pytest.approx(-0.10)


def test_trajectory_delta_both_penalties_sum(tmp_path: Path) -> None:
    """Loop + missing recon → -0.20 total."""
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    # First the missing-recon: Edit a file we never Read
    _seed_pretool(
        s, "sid", 1, "Edit", {"input_summary": "{'file_path': '/x.py'}"}
    )
    # Then a loop on Bash
    same = {"input_summary": "{'command': 'ls'}"}
    for _ in range(3):
        _seed_pretool(s, "sid", 1, "Bash", same)
    assert _compute_trajectory_delta(s, "sid", 1) == pytest.approx(-0.20)


def test_trajectory_delta_ignores_post_tool_use_events(tmp_path: Path) -> None:
    """Loop detection only counts PreToolUse, not PostToolUse echoes."""
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    same = {"input_summary": "{'command': 'ls'}"}
    for _ in range(2):
        _seed_pretool(s, "sid", 1, "Bash", same)
        s.record_event(
            "sid",
            event_type="PostToolUse",
            turn_id=1,
            tool_name="Bash",
            payload=same,
        )
    # 2 Pre + 2 Post — only 2 consecutive Pre → no loop penalty
    assert _compute_trajectory_delta(s, "sid", 1) == 0.0


def test_trajectory_delta_payload_without_file_path_skipped(
    tmp_path: Path,
) -> None:
    """Edit with no extractable file_path → no recon penalty (can't judge)."""
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    _seed_pretool(s, "sid", 1, "Edit", {"input_summary": "no path here"})
    assert _compute_trajectory_delta(s, "sid", 1) == 0.0


def test_trajectory_delta_floor_clamps_pathological_input(
    tmp_path: Path,
) -> None:
    """Both penalties at full strength: -0.20, above the -0.25 floor."""
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    # Unread mutation
    _seed_pretool(
        s, "sid", 1, "Edit", {"input_summary": "{'file_path': '/x.py'}"}
    )
    # 3-streak loop
    same = {"input_summary": "{'command': 'ls'}"}
    for _ in range(5):
        _seed_pretool(s, "sid", 1, "Bash", same)
    delta = _compute_trajectory_delta(s, "sid", 1)
    assert delta >= -0.25  # honors floor
    assert delta == pytest.approx(-0.20)


# ---------------------------------------------------------------------------
# Loop streak resets on different tool/payload (covers the else branch)
# ---------------------------------------------------------------------------


def test_trajectory_delta_loop_streak_resets_on_different_payload(
    tmp_path: Path,
) -> None:
    """ABA pattern should NOT trigger loop (streak resets)."""
    s = StateStore(tmp_path / "state.db")
    s.upsert_session("sid", str(tmp_path))
    _seed_pretool(s, "sid", 1, "Bash", {"input_summary": "{'command': 'ls'}"})
    _seed_pretool(s, "sid", 1, "Bash", {"input_summary": "{'command': 'pwd'}"})
    _seed_pretool(s, "sid", 1, "Bash", {"input_summary": "{'command': 'ls'}"})
    assert _compute_trajectory_delta(s, "sid", 1) == 0.0


# ---------------------------------------------------------------------------
# End-to-end stop_hook integration — quality_score arrives in outcomes row
# ---------------------------------------------------------------------------


def test_on_stop_writes_real_quality_score_with_trajectory_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end stop hook with transcript + events → outcomes row has
    a real quality_score that reflects Tier 1 + Tier 3 composition.
    """
    import io
    import sys

    from agent_amplifier.adapters.claude_code import state as _state
    from agent_amplifier.adapters.claude_code import stop_hook as _sh
    from agent_amplifier.adapters.claude_code.transcript import (
        encoded_project_dir,
    )

    # Isolate state.db to tmp
    db_dir = tmp_path / "amp"
    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", db_dir)
    db_path = db_dir / "state.db"

    # Isolate transcript dir
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(transcripts))

    # Seed session + envelope with overlapping goal/output words
    s = _state.StateStore(db_path)
    s.upsert_session("sid-e2e", "/cwd")
    s.record_envelope(
        "sid-e2e",
        1,
        user_prompt_redacted="fix the auth bug in login flow",
        classification_complexity="medium",
        classification_domain="auth",
        thinking_trigger=None,
        persona=None,
        phase="EXECUTE",
        envelope_text="",
    )
    # Two PreToolUse + two PostToolUse → in_flight=0 → converged=True
    for _ in range(2):
        s.record_event(
            "sid-e2e",
            "PreToolUse",
            turn_id=1,
            tool_name="Read",
            payload={"input_summary": "{'file_path': '/auth.py'}"},
        )
    for _ in range(2):
        s.record_event(
            "sid-e2e",
            "PostToolUse",
            turn_id=1,
            tool_name="Read",
            payload={"input_summary": "{'file_path': '/auth.py'}"},
        )

    # Seed transcript with matching text
    target = transcripts / encoded_project_dir(Path("/cwd")) / "sid-e2e.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "fixed the auth bug in the login flow",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    # Feed Stop event
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"session_id": "sid-e2e", "cwd": "/cwd"})),
    )
    rc = _sh.on_stop()
    assert rc == 0

    # Read back the outcome
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT completed, quality_score, convergence_state "
            "FROM outcomes WHERE session_id='sid-e2e'"
        )
        row = dict(cur.fetchone())
    assert row["completed"] == 1
    # Tier 1 should be > 0 (overlapping keywords) and there's no trajectory
    # penalty (Read-only with consistent paths), so quality_score > 0.
    assert row["quality_score"] is not None
    assert row["quality_score"] > 0.0
    # F1C: first turn of a session with a real quality_score defaults to
    # "improving" (no prior history to classify against).
    assert row["convergence_state"] == "improving"
