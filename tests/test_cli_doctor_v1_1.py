# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for F3 — v1.1 `agent-amp doctor` enhancements.

Covers:
  * ``_telemetry_health()`` — both with and without a state.db, and with
    pre-F2 / pre-F1A schemas where the new columns don't exist yet.
  * ``_slm_daemon_probe()`` — alive (mocked socket) and dead paths.
  * ``_cmd_doctor(as_json=True)`` — JSON output schema.
  * ``_cmd_doctor()`` text output includes the new telemetry section.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from agent_amplifier import cli as _cli

# ---------------------------------------------------------------------------
# _telemetry_health
# ---------------------------------------------------------------------------


def test_telemetry_health_no_state_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_amplifier.adapters.claude_code import state as _state

    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path / "nowhere")
    health = _cli._telemetry_health()
    assert health["state_db_exists"] is False
    assert health["sessions"] is None


def test_telemetry_health_full_v1_1_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real v1.1 state.db with F2 + F1A columns gets all fields populated."""
    from agent_amplifier.adapters.claude_code import state as _state

    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path / "amp")
    db_path = tmp_path / "amp" / "state.db"
    store = _state.StateStore(db_path)
    store.upsert_session("real", str(tmp_path))
    store.upsert_session("synth", "/nonexistent")  # auto-synthetic by cwd
    store.record_envelope(
        "real",
        1,
        user_prompt_redacted="p",
        classification_complexity="high",
        classification_domain="general",
        thinking_trigger=None,
        persona=None,
        phase="EXPLORE",
        envelope_text="",
    )
    store.write_outcome(
        "real",
        1,
        iterations_completed=1,
        converged=True,
        quality_score=0.7,
    )
    store.write_outcome(
        "real",
        2,
        iterations_completed=1,
        converged=True,
        quality_score=None,  # null = unscored
    )

    health = _cli._telemetry_health()
    assert health["state_db_exists"] is True
    assert health["sessions"] == 2
    assert health["real_sessions"] == 1
    assert health["synthetic_sessions"] == 1
    # 1 scored / 2 outcomes = 50%
    assert health["quality_coverage_pct"] == 50.0
    assert health["last_activity_at"] is not None


def test_telemetry_health_pre_f2_db_skips_synthetic_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-F2 sessions table has no is_synthetic column — query falls back."""
    import sqlite3
    from contextlib import closing

    from agent_amplifier.adapters.claude_code import state as _state

    db_path = tmp_path / "state.db"
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                started_at REAL NOT NULL,
                last_seen_at REAL NOT NULL,
                turn_count INTEGER NOT NULL DEFAULT 0,
                amplification_enabled INTEGER NOT NULL DEFAULT 1,
                config_json TEXT NOT NULL DEFAULT '{}',
                closed_at REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE envelopes (session_id TEXT, turn_id INTEGER)
            """
        )
        conn.execute(
            """
            CREATE TABLE outcomes (
                session_id TEXT NOT NULL, turn_id INTEGER NOT NULL,
                iterations_completed INTEGER NOT NULL,
                converged INTEGER NOT NULL,
                drift_at_end REAL DEFAULT 0.0,
                tokens_used INTEGER DEFAULT 0,
                duration_ms INTEGER DEFAULT 0,
                amplification_enabled INTEGER DEFAULT 1,
                quality_estimate REAL,
                finalize_report_json TEXT DEFAULT '{}',
                written_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO sessions(session_id, cwd, started_at, last_seen_at) "
            "VALUES('s', '/p', 1.0, 2.0)"
        )
        conn.commit()
    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path)
    health = _cli._telemetry_health()
    assert health["state_db_exists"] is True
    # F2 columns not present → real/synthetic stay None
    assert health["real_sessions"] is None
    # F1A columns not present → quality_coverage_pct stays None
    assert health["quality_coverage_pct"] is None


def test_telemetry_health_swallows_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If an unexpected error pops mid-collection, _telemetry_health
    returns its default skeleton dict (never raises).
    """
    import sqlite3

    from agent_amplifier.adapters.claude_code import state as _state

    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path / "amp")
    # Seed a real DB so we get past the existence check.
    _state.StateStore(tmp_path / "amp" / "state.db")
    # Now sabotage sqlite3.connect so the inner block raises.
    def _boom(*a: Any, **k: Any) -> None:
        raise sqlite3.DatabaseError("simulated corruption")

    monkeypatch.setattr(sqlite3, "connect", _boom)
    health = _cli._telemetry_health()
    # Path was set, exists check passed, then exception surfaced — defaults
    # for the inner fields stay None, no exception escapes the helper.
    assert health["sessions"] is None
    assert health["envelopes"] is None


# ---------------------------------------------------------------------------
# _slm_daemon_probe
# ---------------------------------------------------------------------------


def test_slm_daemon_probe_returns_true_when_socket_connects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeSock:
        def __enter__(self) -> _FakeSock:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: _FakeSock()
    )
    assert _cli._slm_daemon_probe() is True


def test_slm_daemon_probe_returns_false_on_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*a: Any, **k: Any) -> None:
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", _boom)
    assert _cli._slm_daemon_probe() is False


# ---------------------------------------------------------------------------
# _cmd_doctor — JSON mode + telemetry text section
# ---------------------------------------------------------------------------


def test_doctor_json_mode_emits_valid_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.adapters.claude_code import state as _state

    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path / "amp")
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    rc = _cli.main(["doctor", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "agent_amp_version" in payload
    assert "adapters" in payload
    assert payload["slm"]["daemon_alive"] is False
    assert "telemetry" in payload


def test_doctor_text_includes_telemetry_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.adapters.claude_code import state as _state

    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path / "amp")
    # Seed a real DB so the telemetry block has populated fields.
    store = _state.StateStore(tmp_path / "amp" / "state.db")
    store.upsert_session("s", str(tmp_path))
    store.record_envelope(
        "s",
        1,
        user_prompt_redacted="p",
        classification_complexity="low",
        classification_domain="general",
        thinking_trigger=None,
        persona=None,
        phase="EXPLORE",
        envelope_text="",
    )
    store.write_outcome(
        "s", 1, iterations_completed=1, converged=True, quality_score=0.5
    )
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    rc = _cli.main(["doctor"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "telemetry:" in out
    assert "state.db" in out
    assert "real / synthetic" in out
    assert "quality coverage" in out
    assert "last activity" in out
    assert "slm daemon" in out


def test_doctor_text_handles_missing_state_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.adapters.claude_code import state as _state

    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path / "nowhere")
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    rc = _cli.main(["doctor"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "state.db missing" in out


def test_telemetry_health_zero_outcomes_keeps_quality_pct_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty outcomes table → ``quality_coverage_pct`` stays None (branch
    where total==0 skips the percentage assignment)."""
    from agent_amplifier.adapters.claude_code import state as _state

    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path / "amp")
    # Create the schema but don't write any outcomes.
    _state.StateStore(tmp_path / "amp" / "state.db")
    health = _cli._telemetry_health()
    assert health["state_db_exists"] is True
    assert health["outcomes"] == 0
    assert health["quality_coverage_pct"] is None


def test_telemetry_health_no_sessions_keeps_last_activity_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty sessions table → ``last_activity_at`` stays None (branch where
    ``last`` is ``(None,)`` and the inner ``last[0]`` is falsy)."""
    from agent_amplifier.adapters.claude_code import state as _state

    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path / "amp")
    _state.StateStore(tmp_path / "amp" / "state.db")
    health = _cli._telemetry_health()
    assert health["sessions"] == 0
    assert health["last_activity_at"] is None


def test_doctor_text_pre_f2_db_omits_optional_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the state.db exists but is pre-F2/F1A (no is_synthetic, no
    quality_score columns, no sessions), the doctor's text output skips
    the conditional 'real / synthetic', 'quality coverage', and 'last
    activity' lines — exercises the False branches in `_cmd_doctor`.
    """
    import sqlite3
    from contextlib import closing

    from agent_amplifier.adapters.claude_code import state as _state

    db_path = tmp_path / "amp" / "state.db"
    db_path.parent.mkdir(parents=True)
    # Build an empty, pre-F2/F1A schema by hand (no migrations run).
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT, cwd TEXT, "
            "started_at REAL, last_seen_at REAL)"
        )
        conn.execute("CREATE TABLE envelopes (session_id TEXT)")
        conn.execute(
            "CREATE TABLE outcomes (session_id TEXT, turn_id INTEGER, "
            "iterations_completed INTEGER, converged INTEGER, "
            "written_at REAL)"
        )
        conn.commit()
    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path / "amp")
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: (_ for _ in ()).throw(OSError()),
    )
    rc = _cli.main(["doctor"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "telemetry:" in out
    assert "real / synthetic" not in out  # column missing → block skipped
    assert "quality coverage" not in out  # column missing → block skipped
    assert "last activity" not in out  # no rows → branch skipped


def test_doctor_text_slm_daemon_alive_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.adapters.claude_code import state as _state

    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path / "nowhere")

    class _FakeSock:
        def __enter__(self) -> _FakeSock:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _FakeSock())
    rc = _cli.main(["doctor"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "slm daemon       up" in out
