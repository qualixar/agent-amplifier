# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for F2 — synthetic-session quarantine.

A session is tagged ``is_synthetic=1`` if ANY of:
  1. Env var ``AGENT_AMP_SYNTHETIC=1`` at session creation.
  2. ``cwd`` does not exist on disk at session creation.
  3. Caller passes explicit ``is_synthetic=True`` to ``upsert_session``.

Default for normal sessions: ``is_synthetic=0``.

Migration: adding the column to a v1.0 DB is idempotent and preserves
existing rows with ``is_synthetic=0`` (safe default — real users are
assumed real unless flagged).
"""
from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

import pytest

from agent_amplifier.adapters.claude_code.state import StateStore


def _store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "state.db")


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


def test_new_db_has_is_synthetic_column(tmp_path: Path) -> None:
    """Fresh v1.1 DB exposes the column with default 0."""
    s = _store(tmp_path)
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        cur = conn.execute("PRAGMA table_info(sessions)")
        cols = {row[1] for row in cur.fetchall()}
    assert "is_synthetic" in cols


def test_v1_0_db_gets_column_added_via_migration(tmp_path: Path) -> None:
    """Open a v1.0-shaped sessions table; v1.1 migration adds the column."""
    db = tmp_path / "state.db"
    # Build a v1.0-shape DB by hand (no is_synthetic column).
    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        conn.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                model TEXT,
                model_provider TEXT,
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
            "INSERT INTO sessions(session_id, cwd, started_at, last_seen_at) "
            "VALUES('legacy', '/some/cwd', 0.0, 0.0)"
        )
        conn.commit()

    # Opening via StateStore triggers migration.
    StateStore(db)

    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        cur = conn.execute("PRAGMA table_info(sessions)")
        cols = {row[1] for row in cur.fetchall()}
        assert "is_synthetic" in cols
        # Legacy row preserved with default 0.
        (val,) = conn.execute(
            "SELECT is_synthetic FROM sessions WHERE session_id='legacy'"
        ).fetchone()
        assert val == 0


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Re-opening a v1.1 DB does NOT re-run ALTER TABLE (which would fail)."""
    db = tmp_path / "state.db"
    StateStore(db)
    StateStore(db)
    # If we got here without OperationalError, idempotency holds.
    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        cur = conn.execute("PRAGMA table_info(sessions)")
        cols = [r[1] for r in cur.fetchall()]
    assert cols.count("is_synthetic") == 1


# ---------------------------------------------------------------------------
# Auto-detect rules
# ---------------------------------------------------------------------------


def test_env_var_marks_session_synthetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENT_AMP_SYNTHETIC=1 forces synthetic regardless of cwd."""
    monkeypatch.setenv("AGENT_AMP_SYNTHETIC", "1")
    s = _store(tmp_path)
    s.upsert_session("sid-env", str(tmp_path))  # cwd EXISTS
    row = s.get_session("sid-env")
    assert row is not None
    assert row["is_synthetic"] == 1


def test_env_var_other_value_does_not_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only ``AGENT_AMP_SYNTHETIC=1`` triggers; other truthy values do not."""
    monkeypatch.setenv("AGENT_AMP_SYNTHETIC", "true")  # NOT '1'
    s = _store(tmp_path)
    s.upsert_session("sid-true", str(tmp_path))
    row = s.get_session("sid-true")
    assert row is not None
    assert row["is_synthetic"] == 0


def test_nonexistent_cwd_marks_synthetic(tmp_path: Path) -> None:
    """cwd that does not resolve = ephemeral fixture = synthetic."""
    s = _store(tmp_path)
    s.upsert_session("sid-bench", "/proj")  # /proj does not exist
    row = s.get_session("sid-bench")
    assert row is not None
    assert row["is_synthetic"] == 1


def test_existing_cwd_is_not_synthetic(tmp_path: Path) -> None:
    """Real cwd → real session."""
    s = _store(tmp_path)
    s.upsert_session("sid-real", str(tmp_path))
    row = s.get_session("sid-real")
    assert row is not None
    assert row["is_synthetic"] == 0


# ---------------------------------------------------------------------------
# Explicit override
# ---------------------------------------------------------------------------


def test_explicit_synthetic_true_overrides_auto(tmp_path: Path) -> None:
    """Caller can force synthetic=True even when cwd is real."""
    s = _store(tmp_path)
    s.upsert_session("sid-x", str(tmp_path), is_synthetic=True)
    row = s.get_session("sid-x")
    assert row is not None
    assert row["is_synthetic"] == 1


def test_explicit_synthetic_false_overrides_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caller can force synthetic=False even when env var set."""
    monkeypatch.setenv("AGENT_AMP_SYNTHETIC", "1")
    s = _store(tmp_path)
    s.upsert_session("sid-x", "/proj", is_synthetic=False)
    row = s.get_session("sid-x")
    assert row is not None
    assert row["is_synthetic"] == 0


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------


def test_list_sessions_default_excludes_synthetic(tmp_path: Path) -> None:
    """The dashboard-facing list method hides synthetic by default."""
    s = _store(tmp_path)
    s.upsert_session("real-1", str(tmp_path))
    s.upsert_session("synth-1", "/proj")
    real = s.list_sessions()  # default include_synthetic=False
    ids = {r["session_id"] for r in real}
    assert "real-1" in ids
    assert "synth-1" not in ids


def test_list_sessions_include_synthetic_returns_all(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("real-1", str(tmp_path))
    s.upsert_session("synth-1", "/proj")
    rows = s.list_sessions(include_synthetic=True)
    ids = {r["session_id"] for r in rows}
    assert {"real-1", "synth-1"}.issubset(ids)


def test_list_sessions_synthetic_only_returns_just_synthetic(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_session("real-1", str(tmp_path))
    s.upsert_session("synth-1", "/proj")
    rows = s.list_sessions(synthetic_only=True)
    ids = {r["session_id"] for r in rows}
    assert ids == {"synth-1"}


def test_list_sessions_synthetic_only_and_include_synthetic_is_rejected(
    tmp_path: Path,
) -> None:
    """The two flags are mutually exclusive."""
    s = _store(tmp_path)
    with pytest.raises(ValueError):
        s.list_sessions(include_synthetic=True, synthetic_only=True)


# ---------------------------------------------------------------------------
# Upsert preserves is_synthetic across touches
# ---------------------------------------------------------------------------


def test_upsert_does_not_flip_existing_is_synthetic_to_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeat upsert without env var must not 'demote' a synthetic row."""
    monkeypatch.setenv("AGENT_AMP_SYNTHETIC", "1")
    s = _store(tmp_path)
    s.upsert_session("sid", str(tmp_path))  # synthetic via env
    monkeypatch.delenv("AGENT_AMP_SYNTHETIC")
    s.upsert_session("sid", str(tmp_path))  # repeat WITHOUT env
    row = s.get_session("sid")
    assert row is not None
    assert row["is_synthetic"] == 1  # preserved


def test_explicit_override_on_update_path_flips_synthetic(tmp_path: Path) -> None:
    """Explicit ``is_synthetic=True/False`` on a repeat upsert flips the flag.

    Covers the UPDATE-with-override branch of ``upsert_session``.
    """
    s = _store(tmp_path)
    s.upsert_session("sid", str(tmp_path))
    row = s.get_session("sid")
    assert row is not None
    assert row["is_synthetic"] == 0

    # Now explicitly mark it synthetic on a repeat upsert.
    s.upsert_session("sid", str(tmp_path), is_synthetic=True)
    row = s.get_session("sid")
    assert row is not None
    assert row["is_synthetic"] == 1

    # And flip it back explicitly.
    s.upsert_session("sid", str(tmp_path), is_synthetic=False)
    row = s.get_session("sid")
    assert row is not None
    assert row["is_synthetic"] == 0
