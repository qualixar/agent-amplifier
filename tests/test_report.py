# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.report``.

Coverage targets: 100% line + 100% branch on report.py.

Test isolation:
    * Every test uses ``tmp_path`` for the SQLite file.
    * No test ever writes to ``~/.claude/agent-amp/state.db``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_amplifier.adapters.claude_code.state import StateStore
from agent_amplifier.report import render_report

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_minimal(db: Path) -> None:
    """Populate the DB with one session, one envelope, one outcome."""
    s = StateStore(db)
    sid = "sess-aaaaaaaa-bbbb-cccc-dddd"
    s.upsert_session(sid, "/proj")
    s.next_turn_id(sid)
    s.record_envelope(
        sid,
        1,
        user_prompt_redacted="<redacted>",
        classification_complexity="medium",
        classification_domain="api",
        thinking_trigger="megathink",
        persona="default",
        phase="EXPLORE",
        envelope_text="<env>",
    )
    s.write_outcome(
        sid,
        1,
        iterations_completed=1,
        converged=True,
        duration_ms=120,
        finalize_report={"stop_reason": "real", "tool_calls": 2},
    )


# ---------------------------------------------------------------------------
# Missing DB → exit 1, hint on stderr
# ---------------------------------------------------------------------------


def test_render_report_missing_db_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.sqlite"
    rc = render_report(db_path=missing)
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err
    assert "install hooks" in err.lower()


# ---------------------------------------------------------------------------
# Empty DB (schema only) → exit 0, every section reports its empty state
# ---------------------------------------------------------------------------


def test_render_report_empty_db_renders_empty_sections(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "state.db"
    StateStore(db)  # creates schema, no rows
    rc = render_report(db_path=db)
    assert rc == 0
    out = capsys.readouterr().out
    # All section headers present
    assert "Agent Amplifier Report" in out
    assert "## Health" in out
    assert "## Last 10 turns" in out
    assert "## Classification distribution" in out
    assert "## Convergence" in out
    assert "## Sweep efficacy" in out
    # Empty-state messages
    assert "(no turns recorded yet)" in out
    assert "(no envelopes)" in out
    assert "(no outcomes recorded yet)" in out
    assert "(no outcomes)" in out
    # Empty health: no Coverage line because envelopes==0
    assert "Coverage:" not in out


# ---------------------------------------------------------------------------
# Populated DB → exit 0, all sections show real numbers
# ---------------------------------------------------------------------------


def test_render_report_with_data_shows_all_sections(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "state.db"
    _seed_minimal(db)
    rc = render_report(db_path=db)
    assert rc == 0
    out = capsys.readouterr().out
    # Health section populated
    assert "Sessions:" in out
    assert "Coverage:" in out
    assert "(1/1 turns)" in out
    # Last turns section
    assert "sess-aaa" in out  # session_id truncated to 8 chars
    assert "medium" in out
    assert "api" in out
    assert "megathink" in out
    assert "EXPLORE" in out
    assert "yes" in out  # converged
    assert "real" in out  # stop_reason
    # Classification populated
    assert "complexity" in out
    # Convergence populated
    assert "Converged: 1/1 (100.0%)" in out
    # Sweep populated
    assert "Real Stop:" in out
    assert "Abandoned:" in out


# ---------------------------------------------------------------------------
# Mixed real + abandoned outcomes → sweep section breakdown
# ---------------------------------------------------------------------------


def test_render_report_sweep_section_counts_abandoned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "state.db"
    s = StateStore(db)
    s.upsert_session("sess-A", "/proj")
    s.next_turn_id("sess-A")
    s.record_envelope(
        "sess-A", 1,
        user_prompt_redacted="x",
        classification_complexity="low",
        classification_domain="general",
        thinking_trigger=None,
        persona=None,
        phase="EXPLORE",
        envelope_text="<env>",
    )
    s.write_outcome(
        "sess-A", 1,
        iterations_completed=1, converged=True,
        finalize_report={"stop_reason": "real"},
    )
    s.upsert_session("sess-B", "/proj")
    s.next_turn_id("sess-B")
    s.record_envelope(
        "sess-B", 1,
        user_prompt_redacted="y",
        classification_complexity="high",
        classification_domain="api",
        thinking_trigger=None,
        persona=None,
        phase="EXPLOIT",
        envelope_text="<env>",
    )
    s.write_outcome(
        "sess-B", 1,
        iterations_completed=1, converged=False,
        finalize_report={"stop_reason": "abandoned"},
    )
    rc = render_report(db_path=db)
    assert rc == 0
    out = capsys.readouterr().out
    # Two outcomes: 1 real + 1 abandoned = 50%
    assert "Real Stop:" in out
    assert "Abandoned:" in out
    assert "(50.0%)" in out
    # Convergence: 1/2 = 50.0%
    assert "Converged: 1/2 (50.0%)" in out


# ---------------------------------------------------------------------------
# Default db_path resolution → uses module-level _DEFAULT_STATE_DIR
# ---------------------------------------------------------------------------


def test_render_report_default_db_path_resolves_without_creating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When db_path=None and the default DB does not exist, render_report
    MUST return 1 with the install hint — it must NOT instantiate StateStore
    (which would create the schema and mask the not-found UX path)."""
    from agent_amplifier.adapters.claude_code import state as _state

    redirect = tmp_path / "no-amp-yet" / "agent-amp"
    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", redirect)
    rc = render_report()
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err


def test_render_report_default_db_path_used_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When db_path=None and the default DB exists, render_report uses it."""
    from agent_amplifier.adapters.claude_code import state as _state

    redirect = tmp_path / "amp"
    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", redirect)
    db = redirect / _state._STATE_DB_FILENAME
    StateStore(db)  # creates schema at the redirected default
    rc = render_report()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Agent Amplifier Report" in out


# ---------------------------------------------------------------------------
# --last truncates the turn table to N rows
# ---------------------------------------------------------------------------


def test_render_report_last_arg_limits_turn_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import contextlib as _ctx
    import sqlite3 as _sql

    db = tmp_path / "state.db"
    s = StateStore(db)
    s.upsert_session("sess-X", "/proj")
    for _ in range(5):
        tid = s.next_turn_id("sess-X")
        s.record_envelope(
            "sess-X", tid,
            user_prompt_redacted="x",
            classification_complexity="medium",
            classification_domain=f"domain-{tid}",
            thinking_trigger=None,
            persona=None,
            phase="EXPLORE",
            envelope_text="<env>",
        )
    # Backdate created_at deterministically so ORDER BY is unambiguous —
    # record_envelope uses ``time.time()`` which can tie to the microsecond
    # in tight loops, leaving SQLite free to return any tied rows.
    with _ctx.closing(_sql.connect(str(db))) as conn:
        conn.executemany(
            "UPDATE envelopes SET created_at = ? WHERE turn_id = ?",
            [(1000.0 + tid, tid) for tid in range(1, 6)],
        )
        conn.commit()
    rc = render_report(db_path=db, last=2)
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Last 2 turns" in out
    # Slice to just the turn-table section — domain-1 still appears in the
    # classification distribution (which lists all 5 envelopes), so a global
    # ``not in out`` assertion is too strong.
    turn_section = out.split("## Last 2 turns", 1)[1].split("##", 1)[0]
    assert "domain-5" in turn_section
    assert "domain-4" in turn_section
    assert "domain-1" not in turn_section


# ---------------------------------------------------------------------------
# Long values truncate (session_id → 8 chars, domain → 20 chars)
# ---------------------------------------------------------------------------


def test_render_report_shows_model_routing_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "state.db"
    _seed_minimal(db)
    rc = render_report(db_path=db)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Model Routing" in out
    assert "opus" in out.lower() or "sonnet" in out.lower() or "haiku" in out.lower()


def test_render_report_model_routing_shows_complexity_map(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "state.db"
    s = StateStore(db)
    s.upsert_session("sess-MR", "/proj")
    s.next_turn_id("sess-MR")
    s.record_envelope(
        "sess-MR", 1,
        user_prompt_redacted="complex task",
        classification_complexity="high",
        classification_domain="backend",
        thinking_trigger="ultrathink",
        persona=None,
        phase="EXPLORE",
        envelope_text="<env>",
    )
    s.next_turn_id("sess-MR")
    s.record_envelope(
        "sess-MR", 2,
        user_prompt_redacted="quick fix",
        classification_complexity="minimal",
        classification_domain="general",
        thinking_trigger=None,
        persona=None,
        phase="EXPLORE",
        envelope_text="<env>",
    )
    rc = render_report(db_path=db)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Model Routing" in out
    assert "HIGH" in out or "high" in out
    assert "MINIMAL" in out or "minimal" in out


def test_render_report_truncates_long_domain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "state.db"
    s = StateStore(db)
    s.upsert_session("sess-Y", "/proj")
    s.next_turn_id("sess-Y")
    long_domain = "this-is-a-really-very-long-domain-string"
    s.record_envelope(
        "sess-Y", 1,
        user_prompt_redacted="x",
        classification_complexity="medium",
        classification_domain=long_domain,
        thinking_trigger="ultrathink-extra-long",
        persona=None,
        phase="EXPLORE",
        envelope_text="<env>",
    )
    rc = render_report(db_path=db)
    assert rc == 0
    out = capsys.readouterr().out
    # Domain truncated to 20 chars in classification AND turn table.
    assert long_domain[:20] in out
    # Trigger truncated to 11 chars in turn table.
    assert "ultrathink-"[:11] in out
