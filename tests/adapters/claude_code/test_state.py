# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.adapters.claude_code.state``.

Coverage targets: 100% line + 100% branch on state.py, plus a multi-process
concurrent-write test that exercises SQLite WAL under realistic load.

Test isolation:
    * Every test uses ``tmp_path`` for the SQLite file.
    * No test ever writes to ``~/.claude/agent-amp/state.db``.
"""
from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import sqlite3
import time
from pathlib import Path

import pytest

from agent_amplifier.adapters.claude_code.state import (
    _DEFAULT_SESSION_TTL_SECONDS,
    StateStore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "state.db")


# ---------------------------------------------------------------------------
# Construction + schema
# ---------------------------------------------------------------------------


def test_store_creates_parent_dir_and_db(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "path" / "state.db"
    s = StateStore(nested)
    assert nested.parent.is_dir()
    assert nested.exists()
    assert s.db_path == nested


def test_store_default_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When no path is passed, db lives under module's _DEFAULT_STATE_DIR.

    The module-level constant is computed at import time from ``Path.home()``,
    so we monkeypatch the resolved constants directly to redirect into
    ``tmp_path`` for hermetic testing.
    """
    from agent_amplifier.adapters.claude_code import state as _state

    redirect_dir = tmp_path / "redir" / "agent-amp"
    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", redirect_dir)
    s = StateStore()
    assert s.db_path == redirect_dir / "state.db"
    assert s.db_path.parent.is_dir()


def test_schema_idempotent(tmp_path: Path) -> None:
    """Re-opening the same DB does not re-create or drop tables."""
    p = tmp_path / "state.db"
    s1 = StateStore(p)
    s1.upsert_session("abc", "/cwd")
    s2 = StateStore(p)
    assert s2.get_session("abc") is not None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_upsert_session_insert_then_update_preserves_started_at(
    tmp_path: Path,
) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid1", "/cwd1", model="claude-opus", model_provider="anthropic")
    row1 = s.get_session("sid1")
    assert row1 is not None
    started_at_1 = row1["started_at"]
    time.sleep(0.01)
    s.upsert_session("sid1", "/cwd1", model="claude-opus")
    row2 = s.get_session("sid1")
    assert row2 is not None
    assert row2["started_at"] == started_at_1
    assert row2["last_seen_at"] >= started_at_1


def test_upsert_session_rejects_empty(tmp_path: Path) -> None:
    s = _store(tmp_path)
    with pytest.raises(ValueError):
        s.upsert_session("", "/cwd")


def test_upsert_session_with_config_dict_persists(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd", config={"foo": "bar", "n": 42})
    row = s.get_session("sid")
    assert row is not None
    assert json.loads(row["config_json"]) == {"foo": "bar", "n": 42}


def test_upsert_session_update_keeps_existing_config_when_empty_dict_passed(
    tmp_path: Path,
) -> None:
    """The CASE expression should preserve config_json when ``{}`` passed."""
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd", config={"keep": True})
    s.upsert_session("sid", "/cwd")  # no config arg
    row = s.get_session("sid")
    assert row is not None
    assert json.loads(row["config_json"]) == {"keep": True}


def test_upsert_session_disabled_amplification(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd", amplification_enabled=False)
    row = s.get_session("sid")
    assert row is not None
    assert row["amplification_enabled"] == 0


def test_close_session_sets_closed_at(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")
    s.close_session("sid")
    row = s.get_session("sid")
    assert row is not None
    assert row["closed_at"] is not None


def test_close_session_empty_id_silent(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.close_session("")  # must not raise


def test_get_session_missing_returns_none(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.get_session("nope") is None


# ---------------------------------------------------------------------------
# Turn-id allocation
# ---------------------------------------------------------------------------


def test_next_turn_id_monotonic(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")
    assert s.next_turn_id("sid") == 1
    assert s.next_turn_id("sid") == 2
    assert s.next_turn_id("sid") == 3


def test_next_turn_id_unknown_session_raises(tmp_path: Path) -> None:
    s = _store(tmp_path)
    with pytest.raises(KeyError):
        s.next_turn_id("never-upserted")


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------


def test_record_and_latest_envelope_roundtrip(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")
    turn = s.next_turn_id("sid")
    s.record_envelope(
        "sid",
        turn,
        user_prompt_redacted="hello [REDACTED]",
        classification_complexity="medium",
        classification_domain="architecture",
        thinking_trigger="think hard",
        persona="audit",
        phase="EXPLORE",
        envelope_text="ENV BODY",
    )
    last = s.latest_envelope("sid")
    assert last is not None
    assert last["turn_id"] == turn
    assert last["envelope_text"] == "ENV BODY"
    assert last["thinking_trigger"] == "think hard"


def test_latest_envelope_returns_highest_turn(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")
    for i in range(5):
        s.record_envelope(
            "sid",
            i + 1,
            user_prompt_redacted="x",
            classification_complexity="low",
            classification_domain="generic",
            thinking_trigger=None,
            persona=None,
            phase="EXPLORE",
            envelope_text=f"env-{i}",
        )
    last = s.latest_envelope("sid")
    assert last is not None
    assert last["turn_id"] == 5
    assert last["envelope_text"] == "env-4"


def test_latest_envelope_missing_returns_none(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")
    assert s.latest_envelope("sid") is None


def test_record_envelope_replaces_on_collision(tmp_path: Path) -> None:
    """INSERT OR REPLACE — same (session, turn) overwrites."""
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")
    s.record_envelope(
        "sid",
        1,
        user_prompt_redacted="a",
        classification_complexity="low",
        classification_domain="d",
        thinking_trigger=None,
        persona=None,
        phase="P",
        envelope_text="first",
    )
    s.record_envelope(
        "sid",
        1,
        user_prompt_redacted="b",
        classification_complexity="low",
        classification_domain="d",
        thinking_trigger=None,
        persona=None,
        phase="P",
        envelope_text="second",
    )
    last = s.latest_envelope("sid")
    assert last is not None
    assert last["envelope_text"] == "second"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_record_event_basic(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")
    s.record_event(
        "sid",
        "PreToolUse",
        turn_id=1,
        tool_name="Bash",
        payload={"input": "ls"},
    )
    assert s.count_events("sid") == 1
    assert s.count_events("sid", turn_id=1) == 1
    assert s.count_events("sid", turn_id=2) == 0


def test_record_event_empty_session_id_silent(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.record_event("", "PreToolUse")  # no-op


def test_record_event_unserializable_payload_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When ``json.dumps`` raises, payload_json defaults to ``'{}'`` and the
    event row is still written. We monkeypatch ``json.dumps`` inside the
    state module to force a TypeError on the payload encode call only —
    NOT on the post-fallback empty dict encode (which the fallback path
    does not perform; the literal ``"{}"`` is used).
    """
    from agent_amplifier.adapters.claude_code import state as _state

    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")

    real_dumps = _state.json.dumps
    sentinel = {"trip": True}

    def fake_dumps(obj: object, *a: object, **kw: object) -> str:
        if obj == sentinel:
            raise TypeError("simulated unserializable")
        return real_dumps(obj, *a, **kw)

    monkeypatch.setattr(_state.json, "dumps", fake_dumps)
    s.record_event("sid", "PreToolUse", turn_id=None, payload=sentinel)
    # Did not raise; row was written.
    assert s.count_events("sid") == 1


def test_record_event_no_payload_writes_empty_dict(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")
    s.record_event("sid", "PreToolUse")
    assert s.count_events("sid") == 1


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def test_write_outcome_basic(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")
    s.write_outcome(
        "sid",
        1,
        iterations_completed=2,
        converged=True,
        drift_at_end=0.1,
        tokens_used=1234,
        duration_ms=500,
        amplification_enabled=True,
        quality_estimate=0.85,
        finalize_report={"note": "ok"},
    )
    # Read back via raw SQL since StateStore exposes no read API for outcomes.
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM outcomes WHERE session_id = ? AND turn_id = ?",
            ("sid", 1),
        ).fetchone()
    assert row is not None
    assert row["converged"] == 1
    assert row["tokens_used"] == 1234
    assert json.loads(row["finalize_report_json"]) == {"note": "ok"}


def test_write_outcome_disabled_and_defaults(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")
    s.write_outcome(
        "sid",
        7,
        iterations_completed=1,
        converged=False,
        amplification_enabled=False,
    )
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        row = conn.execute(
            "SELECT amplification_enabled, finalize_report_json FROM outcomes "
            "WHERE session_id=? AND turn_id=?",
            ("sid", 7),
        ).fetchone()
    assert row is not None
    assert row[0] == 0
    assert row[1] == "{}"


def test_write_outcome_idempotent_replace(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")
    s.write_outcome("sid", 1, iterations_completed=1, converged=True)
    s.write_outcome("sid", 1, iterations_completed=99, converged=False)
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        row = conn.execute(
            "SELECT iterations_completed, converged FROM outcomes WHERE session_id=? AND turn_id=?",
            ("sid", 1),
        ).fetchone()
    assert row == (99, 0)


# ---------------------------------------------------------------------------
# GC
# ---------------------------------------------------------------------------


def test_gc_old_sessions_no_op_when_empty(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.gc_old_sessions() == 0


def test_gc_old_sessions_no_op_when_all_recent(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid", "/cwd")
    assert s.gc_old_sessions() == 0  # default 7-day TTL covers brand-new row


def test_gc_old_sessions_deletes_expired_with_cascade(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("sid-old", "/cwd")
    s.record_envelope(
        "sid-old", 1,
        user_prompt_redacted="x",
        classification_complexity="low",
        classification_domain="d",
        thinking_trigger=None, persona=None, phase="P",
        envelope_text="x",
    )
    s.record_event("sid-old", "PreToolUse", turn_id=1, tool_name="t", payload={})
    s.write_outcome("sid-old", 1, iterations_completed=1, converged=True)
    # Backdate last_seen_at well past the TTL.
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        conn.execute(
            "UPDATE sessions SET last_seen_at = ? WHERE session_id = ?",
            (time.time() - 10 * _DEFAULT_SESSION_TTL_SECONDS, "sid-old"),
        )
        conn.commit()
    deleted = s.gc_old_sessions()
    assert deleted == 1
    # Cascade: envelope/event/outcome rows for sid-old must be gone.
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        for table in ("envelopes", "events", "outcomes", "sessions"):
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id=?", ("sid-old",)
            ).fetchone()[0]
            assert n == 0, f"{table} still has rows after GC"


# ---------------------------------------------------------------------------
# Concurrent writers (multiprocess) — WAL mode contract
# ---------------------------------------------------------------------------


def _writer_proc(db_path: str, session_id: str, n: int) -> None:
    """Process body: open store, upsert session, record N events."""
    s = StateStore(Path(db_path))
    s.upsert_session(session_id, "/cwd")
    for i in range(n):
        s.record_event(
            session_id,
            "PreToolUse",
            turn_id=1,
            tool_name=f"tool-{i}",
            payload={"i": i},
        )


def test_multiprocess_concurrent_writes_dont_corrupt(tmp_path: Path) -> None:
    """Two writers on the same DB → both rows land, no SQLite corruption."""
    db = tmp_path / "concur.db"
    StateStore(db)  # ensure schema
    procs = [
        mp.Process(target=_writer_proc, args=(str(db), f"sess-{i}", 25))
        for i in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    s = StateStore(db)
    for i in range(2):
        assert s.get_session(f"sess-{i}") is not None
        assert s.count_events(f"sess-{i}") == 25


# ---------------------------------------------------------------------------
# find_abandoned_envelopes — flushes Cmd+Q / force-quit gaps
# ---------------------------------------------------------------------------


def _seed_envelope(s: StateStore, sid: str, turn: int, age_seconds: float) -> None:
    """Insert an envelope and backdate its created_at by ``age_seconds``."""
    s.upsert_session(sid, "/proj")
    s.next_turn_id(sid)
    s.record_envelope(
        sid,
        turn,
        user_prompt_redacted="<redacted>",
        classification_complexity="medium",
        classification_domain="general",
        thinking_trigger=None,
        persona=None,
        phase="EXPLORE",
        envelope_text="<env>",
    )
    backdated = time.time() - age_seconds
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        conn.execute(
            "UPDATE envelopes SET created_at = ? WHERE session_id=? AND turn_id=?",
            (backdated, sid, turn),
        )
        conn.commit()


def test_find_abandoned_envelopes_empty_db(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.find_abandoned_envelopes() == []


def test_find_abandoned_envelopes_excludes_recent(tmp_path: Path) -> None:
    """An envelope freshly written (age=0) is not abandoned yet."""
    s = _store(tmp_path)
    _seed_envelope(s, "sess-fresh", 1, age_seconds=0.0)
    assert s.find_abandoned_envelopes(age_seconds=30.0) == []


def test_find_abandoned_envelopes_excludes_with_outcome(tmp_path: Path) -> None:
    """An envelope that already has a matching outcome is not orphaned."""
    s = _store(tmp_path)
    _seed_envelope(s, "sess-done", 1, age_seconds=120.0)
    s.write_outcome(
        "sess-done", 1,
        iterations_completed=1, converged=True,
        finalize_report={"stop_reason": "real"},
    )
    assert s.find_abandoned_envelopes(age_seconds=30.0) == []


def test_find_abandoned_envelopes_orphan_no_events(tmp_path: Path) -> None:
    """Orphan with no events → pre/post counts coerce from None to 0
    (covers the ``or 0`` short-circuit fallback branch)."""
    s = _store(tmp_path)
    _seed_envelope(s, "sess-bare", 1, age_seconds=120.0)
    out = s.find_abandoned_envelopes(age_seconds=30.0)
    assert len(out) == 1
    row = out[0]
    assert row["session_id"] == "sess-bare"
    assert row["turn_id"] == 1
    assert row["pre_count"] == 0
    assert row["post_count"] == 0
    # No events → last_event_at == created_at
    assert row["last_event_at"] == row["created_at"]


def test_find_abandoned_envelopes_orphan_with_events(tmp_path: Path) -> None:
    """Orphan with mixed PreToolUse / PostToolUse events → counts populated
    (covers the truthy branch of the ``SUM(...) or 0`` short-circuit)."""
    s = _store(tmp_path)
    _seed_envelope(s, "sess-busy", 1, age_seconds=120.0)
    for _ in range(3):
        s.record_event(
            "sess-busy", "PreToolUse", turn_id=1, tool_name="t", payload={}
        )
    for _ in range(2):
        s.record_event(
            "sess-busy", "PostToolUse", turn_id=1, tool_name="t", payload={}
        )
    out = s.find_abandoned_envelopes(age_seconds=30.0)
    assert len(out) == 1
    row = out[0]
    assert row["pre_count"] == 3
    assert row["post_count"] == 2
    # Events were just inserted → last_event_at > created_at (which we backdated).
    assert row["last_event_at"] > row["created_at"]
