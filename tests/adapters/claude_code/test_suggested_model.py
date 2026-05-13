# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for F4 — persist ``ModelRouter.suggest()`` tier per envelope.

The Claude Code hook adapter calls ``ModelRouter().suggest(complexity)`` at
envelope creation time and records the resulting tier on the envelope row.
This gives the report / dashboard real cost-routing receipts.

Backward-compatible: v1.0 envelopes have ``suggested_model=NULL``.
"""
from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

from agent_amplifier.adapters.claude_code.state import StateStore


def _store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "state.db")


def _record_minimal_envelope(
    store: StateStore,
    session_id: str = "sid",
    turn_id: int = 1,
    suggested_model: str | None = None,
) -> None:
    store.upsert_session(session_id, str(store.db_path.parent))
    store.record_envelope(
        session_id,
        turn_id,
        user_prompt_redacted="prompt",
        classification_complexity="high",
        classification_domain="general",
        thinking_trigger=None,
        persona=None,
        phase="EXPLORE",
        envelope_text="<env>",
        suggested_model=suggested_model,
    )


def _legacy_record_envelope(
    store: StateStore, session_id: str = "sid", turn_id: int = 1
) -> None:
    """Call ``record_envelope`` WITHOUT the new ``suggested_model`` kwarg.

    Exercises the backward-compatible default (None) for callers that have
    not yet upgraded.
    """
    store.upsert_session(session_id, str(store.db_path.parent))
    store.record_envelope(
        session_id,
        turn_id,
        user_prompt_redacted="prompt",
        classification_complexity="minimal",
        classification_domain="general",
        thinking_trigger=None,
        persona=None,
        phase="EXPLORE",
        envelope_text="<env>",
    )


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


def test_new_db_has_suggested_model_column(tmp_path: Path) -> None:
    s = _store(tmp_path)
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        cur = conn.execute("PRAGMA table_info(envelopes)")
        cols = {row[1] for row in cur.fetchall()}
    assert "suggested_model" in cols


def test_v1_0_db_gets_column_added_via_migration(tmp_path: Path) -> None:
    """A v1.0-shape envelopes table receives the column on first open."""
    db = tmp_path / "state.db"
    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        conn.execute(
            """
            CREATE TABLE envelopes (
                session_id TEXT NOT NULL,
                turn_id INTEGER NOT NULL,
                user_prompt_redacted TEXT NOT NULL,
                classification_complexity TEXT NOT NULL,
                classification_domain TEXT NOT NULL,
                thinking_trigger TEXT,
                persona TEXT,
                phase TEXT NOT NULL,
                envelope_text TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (session_id, turn_id)
            )
            """
        )
        conn.commit()

    StateStore(db)  # triggers migration

    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        cur = conn.execute("PRAGMA table_info(envelopes)")
        cols = {row[1] for row in cur.fetchall()}
    assert "suggested_model" in cols


def test_envelope_migration_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    StateStore(db)
    StateStore(db)
    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        cur = conn.execute("PRAGMA table_info(envelopes)")
        col_names = [r[1] for r in cur.fetchall()]
    assert col_names.count("suggested_model") == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_envelope_persists_suggested_model_when_passed(tmp_path: Path) -> None:
    s = _store(tmp_path)
    _record_minimal_envelope(s, suggested_model="sonnet")
    env = s.latest_envelope("sid")
    assert env is not None
    assert env["suggested_model"] == "sonnet"


def test_envelope_suggested_model_null_when_omitted(tmp_path: Path) -> None:
    """The default kwarg yields NULL — preserves v1.0 caller contract."""
    s = _store(tmp_path)
    _legacy_record_envelope(s)
    env = s.latest_envelope("sid")
    assert env is not None
    assert env["suggested_model"] is None


def test_envelope_suggested_model_can_be_any_known_tier(tmp_path: Path) -> None:
    s = _store(tmp_path)
    for i, tier in enumerate(["haiku", "sonnet", "opus"], start=1):
        _record_minimal_envelope(s, turn_id=i, suggested_model=tier)
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        cur = conn.execute(
            "SELECT suggested_model FROM envelopes ORDER BY turn_id"
        )
        tiers = [r[0] for r in cur.fetchall()]
    assert tiers == ["haiku", "sonnet", "opus"]


# ---------------------------------------------------------------------------
# Hooks integration smoke (router → record path)
# ---------------------------------------------------------------------------


def test_model_router_tier_string_round_trips(tmp_path: Path) -> None:
    """The tier string returned by ModelRouter survives the DB round trip.

    Production caller does: ``ModelRouter().suggest(complexity).tier`` and
    passes the result. We verify that contract with a real router call.
    """
    from agent_amplifier.model_router import ModelRouter

    router = ModelRouter()
    s = _store(tmp_path)
    s.upsert_session("sid", str(tmp_path))
    for i, complexity in enumerate(
        ["minimal", "low", "medium", "high", "max"], start=1
    ):
        tier = router.suggest(complexity).tier
        s.record_envelope(
            "sid",
            i,
            user_prompt_redacted="p",
            classification_complexity=complexity,
            classification_domain="general",
            thinking_trigger=None,
            persona=None,
            phase="EXPLORE",
            envelope_text="<env>",
            suggested_model=tier,
        )
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        cur = conn.execute(
            "SELECT classification_complexity, suggested_model "
            "FROM envelopes ORDER BY turn_id"
        )
        rows = cur.fetchall()
    # Every row has a non-null tier string.
    assert all(tier is not None and tier != "" for _, tier in rows)
    # Tier strings are from the known set.
    tiers = {tier for _, tier in rows}
    assert tiers.issubset({"haiku", "sonnet", "opus"})
