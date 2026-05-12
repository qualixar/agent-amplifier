# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""SQLite WAL state store for the Claude Code hook adapter (day-0).

Per decision H-4 (DECISIONS-LOCKED.md §H-4), the Claude Code hook adapter
maps Claude Code's hook events to kernel calls at PER-USER-TURN granularity:

  - UserPromptSubmit → kernel.before_step (amplification injection)
  - PreToolUse  → state.track_tool_call (logging only)
  - PostToolUse → state.track_tool_result (logging only)
  - Stop        → state.write_outcome + summary (no kernel.after_step in v1.0.0)

This module owns durable cross-process state. Each Claude Code session is a
row in `sessions`. Each user turn within a session writes one envelope row
(in `envelopes`) and one outcome row (in `outcomes` at Stop). Tool calls and
results write rows in `events`.

Concurrency: every Claude Code session fires hooks in its own subprocess.
Real users run 5-10 concurrent Claude sessions. SQLite WAL mode + short
transactions handle this; we do not hold open transactions across hook
invocations.

Schema is forward-compat: new columns added via `ALTER TABLE ADD COLUMN`
in `_migrate`. Never dropped, never renamed, per Apache-2.0 product
back-compat discipline.
"""
from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Final

LOG = logging.getLogger("agent_amplifier.adapters.claude_code.state")

# ---------------------------------------------------------------------------
# Storage location (XDG-respectful + Claude Code convention)
# ---------------------------------------------------------------------------

_DEFAULT_STATE_DIR: Final[Path] = Path.home() / ".claude" / "agent-amp"
_STATE_DB_FILENAME: Final[str] = "state.db"

# 7-day default TTL for session GC (decision: lazy sweep on store open)
_DEFAULT_SESSION_TTL_SECONDS: Final[int] = 7 * 24 * 3600
# Hard cap on rows held by a single session row's events query — defense
# against a runaway Claude session writing millions of tool events.
_MAX_EVENTS_PER_SESSION: Final[int] = 10_000


# ---------------------------------------------------------------------------
# Schema (DDL strings — explicit, reviewable, no ORM)
# ---------------------------------------------------------------------------

_SCHEMA_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
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

_SCHEMA_ENVELOPES = """
CREATE TABLE IF NOT EXISTS envelopes (
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

_SCHEMA_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id INTEGER,
    event_type TEXT NOT NULL,
    tool_name TEXT,
    payload_json TEXT NOT NULL,
    timestamp REAL NOT NULL
)
"""

_SCHEMA_OUTCOMES = """
CREATE TABLE IF NOT EXISTS outcomes (
    session_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    iterations_completed INTEGER NOT NULL,
    converged INTEGER NOT NULL,
    drift_at_end REAL NOT NULL DEFAULT 0.0,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    amplification_enabled INTEGER NOT NULL DEFAULT 1,
    quality_estimate REAL,
    finalize_report_json TEXT NOT NULL DEFAULT '{}',
    written_at REAL NOT NULL,
    PRIMARY KEY (session_id, turn_id)
)
"""

_SCHEMA_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_turn ON events(session_id, turn_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_last_seen ON sessions(last_seen_at)",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_session ON outcomes(session_id)",
)


# ---------------------------------------------------------------------------
# StateStore — single class, owns the connection lifecycle
# ---------------------------------------------------------------------------


class StateStore:
    """SQLite-backed state store for the Claude Code hook adapter.

    Every method is a short, atomic transaction. Connections are short-lived
    (open-write-close) so concurrent hook subprocesses never block each other
    in WAL mode. The schema is created on first open; subsequent opens are
    idempotent.

    Cost: opening a SQLite connection on local disk on macOS APFS is ~1-3 ms.
    Each hook fires once per event (PreToolUse, PostToolUse, etc.) and we
    target sub-50ms total hook latency, so a single connection-per-call is
    cheap.

    Tests: see ``tests/adapters/claude_code/test_state.py`` — covers
    multi-process concurrent writes, schema migration safety, GC behavior,
    and bounds.
    """

    __slots__ = ("_db_path",)

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            db_path = _DEFAULT_STATE_DIR / _STATE_DB_FILENAME
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ------------------------------------------------------------------
    # Schema lifecycle
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute(_SCHEMA_SESSIONS)
            conn.execute(_SCHEMA_ENVELOPES)
            conn.execute(_SCHEMA_EVENTS)
            conn.execute(_SCHEMA_OUTCOMES)
            for ddl in _SCHEMA_INDEXES:
                conn.execute(ddl)
            conn.commit()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # ``isolation_level=None`` → autocommit OFF; we manage with explicit
        # commit. ``timeout=5.0`` lets WAL writes wait 5s under contention.
        conn = sqlite3.connect(
            str(self._db_path), timeout=5.0, isolation_level="DEFERRED"
        )
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def upsert_session(
        self,
        session_id: str,
        cwd: str,
        *,
        model: str | None = None,
        model_provider: str | None = None,
        amplification_enabled: bool = True,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Create-or-touch a session row.

        Idempotent. Updates ``last_seen_at`` on every call. Preserves
        ``started_at`` if the row already exists. Does NOT clobber model
        / config if a NULL is passed and the row already has a value.
        """
        if not session_id:
            raise ValueError("session_id must be non-empty")
        now = time.time()
        cfg_json = json.dumps(dict(config) if config else {}, default=str)
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            )
            exists = cur.fetchone() is not None
            if exists:
                # Update last_seen and (only if non-null) the optional fields.
                conn.execute(
                    """
                    UPDATE sessions
                       SET last_seen_at = ?,
                           model = COALESCE(?, model),
                           model_provider = COALESCE(?, model_provider),
                           amplification_enabled = ?,
                           config_json = CASE WHEN ? = '{}' THEN config_json ELSE ? END
                     WHERE session_id = ?
                    """,
                    (
                        now,
                        model,
                        model_provider,
                        1 if amplification_enabled else 0,
                        cfg_json,
                        cfg_json,
                        session_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, cwd, model, model_provider,
                        started_at, last_seen_at, turn_count,
                        amplification_enabled, config_json
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        session_id,
                        cwd,
                        model,
                        model_provider,
                        now,
                        now,
                        1 if amplification_enabled else 0,
                        cfg_json,
                    ),
                )
            conn.commit()

    def close_session(self, session_id: str) -> None:
        """Mark a session closed (called from Stop hook)."""
        if not session_id:
            return
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET closed_at = ? WHERE session_id = ?",
                (time.time(), session_id),
            )
            conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Envelope (per-turn amplification record)
    # ------------------------------------------------------------------

    def next_turn_id(self, session_id: str) -> int:
        """Atomically allocate the next turn id for this session."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "SELECT turn_count FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                conn.commit()
                raise KeyError(
                    f"session_id {session_id!r} not found; call upsert_session first"
                )
            next_id = int(row[0]) + 1
            conn.execute(
                "UPDATE sessions SET turn_count = ? WHERE session_id = ?",
                (next_id, session_id),
            )
            conn.commit()
            return next_id

    def record_envelope(
        self,
        session_id: str,
        turn_id: int,
        *,
        user_prompt_redacted: str,
        classification_complexity: str,
        classification_domain: str,
        thinking_trigger: str | None,
        persona: str | None,
        phase: str,
        envelope_text: str,
    ) -> None:
        """Persist the envelope produced by ``kernel.before_step`` for this turn.

        Call this right after the kernel returns the envelope and BEFORE we
        emit the system-reminder injection, so a crash mid-injection still
        leaves a record of what we were trying to inject.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO envelopes (
                    session_id, turn_id, user_prompt_redacted,
                    classification_complexity, classification_domain,
                    thinking_trigger, persona, phase, envelope_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_id,
                    user_prompt_redacted,
                    classification_complexity,
                    classification_domain,
                    thinking_trigger,
                    persona,
                    phase,
                    envelope_text,
                    time.time(),
                ),
            )
            conn.commit()

    def latest_envelope(self, session_id: str) -> dict[str, Any] | None:
        """Return the most recent envelope row for this session, or None."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT * FROM envelopes
                 WHERE session_id = ?
                 ORDER BY turn_id DESC
                 LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def find_abandoned_envelopes(
        self, *, age_seconds: float = 30.0
    ) -> list[dict[str, Any]]:
        """Return envelope rows older than ``age_seconds`` that have no outcome.

        Used by the UserPromptSubmit hook to retroactively flush envelopes
        whose Stop hook never fired (Cmd+Q mid-response, force-quit, or
        "submit prompt then close before the reply"). Each returned dict has:
            - ``session_id``, ``turn_id``, ``created_at`` (envelope row)
            - ``last_event_at`` — max timestamp from ``events`` for this turn,
              or ``created_at`` if no PreToolUse/PostToolUse fired
            - ``pre_count`` / ``post_count`` — PreToolUse / PostToolUse counts

        The caller writes a synthetic outcome via ``write_outcome`` with
        ``stop_reason="abandoned"`` for each row. ``INSERT OR REPLACE`` on
        ``outcomes`` makes a later real Stop hook (extremely rare — Claude
        Code does not retroactively fire Stop) overwrite the abandonment row.
        False positives self-heal.

        Sweeps ALL sessions, not just the current one — Claude Code's
        session_id does not persist across process restarts, so yesterday's
        abandoned envelopes would never see their own UPS again.
        """
        cutoff = time.time() - age_seconds
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT
                    e.session_id AS session_id,
                    e.turn_id    AS turn_id,
                    e.created_at AS created_at,
                    COALESCE(MAX(ev.timestamp), e.created_at) AS last_event_at,
                    SUM(CASE WHEN ev.event_type='PreToolUse'  THEN 1 ELSE 0 END) AS pre_count,
                    SUM(CASE WHEN ev.event_type='PostToolUse' THEN 1 ELSE 0 END) AS post_count
                  FROM envelopes e
                  LEFT JOIN events ev
                    ON ev.session_id = e.session_id AND ev.turn_id = e.turn_id
                 WHERE e.created_at < ?
                   AND NOT EXISTS (
                       SELECT 1 FROM outcomes o
                        WHERE o.session_id = e.session_id
                          AND o.turn_id = e.turn_id
                   )
                 GROUP BY e.session_id, e.turn_id, e.created_at
                 ORDER BY e.created_at ASC
                """,
                (cutoff,),
            )
            out: list[dict[str, Any]] = []
            for row in cur.fetchall():
                d = dict(row)
                # SUM(CASE WHEN ...) returns None when the LEFT JOIN finds
                # no events; coerce to 0 so downstream arithmetic is safe.
                d["pre_count"] = int(d["pre_count"] or 0)
                d["post_count"] = int(d["post_count"] or 0)
                d["last_event_at"] = float(d["last_event_at"])
                out.append(d)
            return out

    # ------------------------------------------------------------------
    # Events (tool calls + results, log-only)
    # ------------------------------------------------------------------

    def record_event(
        self,
        session_id: str,
        event_type: str,
        *,
        turn_id: int | None = None,
        tool_name: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Append an event row. Always succeeds (or logs WARNING and returns).

        Per H-4: PreToolUse / PostToolUse hooks call this for paper data
        capture. The payload is JSON-serialized; callers MUST scrub secrets
        before passing in (use ``agent_amplifier._internal.redact``).
        """
        if not session_id:
            return
        try:
            payload_json = json.dumps(
                dict(payload) if payload else {}, default=str
            )
        except (TypeError, ValueError) as e:
            LOG.warning("event payload not JSON-serializable: %s", e)
            payload_json = "{}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    session_id, turn_id, event_type, tool_name,
                    payload_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_id,
                    event_type,
                    tool_name,
                    payload_json,
                    time.time(),
                ),
            )
            conn.commit()

    def count_events(self, session_id: str, *, turn_id: int | None = None) -> int:
        with self._connect() as conn:
            if turn_id is None:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE session_id = ?",
                    (session_id,),
                )
            else:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE session_id = ? AND turn_id = ?",
                    (session_id, turn_id),
                )
            (n,) = cur.fetchone()
            return int(n)

    # ------------------------------------------------------------------
    # Outcomes (per-turn paper-data row)
    # ------------------------------------------------------------------

    def prior_tokens_for_session(
        self, session_id: str, before_turn_id: int
    ) -> int:
        """Sum ``tokens_used`` for outcome rows of ``session_id`` with
        ``turn_id < before_turn_id``.

        Used by the Stop hook (IP-10 v2 / Option C) to compute the per-turn
        delta vs prior outcomes when reading cumulative tokens from the
        Claude Code transcript JSONL. Returns 0 if no prior rows exist.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens_used), 0) "
                "FROM outcomes WHERE session_id = ? AND turn_id < ?",
                (session_id, int(before_turn_id)),
            ).fetchone()
        return int(row[0]) if row else 0

    def write_outcome(
        self,
        session_id: str,
        turn_id: int,
        *,
        iterations_completed: int,
        converged: bool,
        drift_at_end: float = 0.0,
        tokens_used: int = 0,
        duration_ms: int = 0,
        amplification_enabled: bool = True,
        quality_estimate: float | None = None,
        finalize_report: Mapping[str, Any] | None = None,
    ) -> None:
        """Write the per-turn outcome row.

        Called from Stop hook after computing summary metrics.
        Idempotent (INSERT OR REPLACE) — re-running the Stop hook for the
        same (session_id, turn_id) overwrites the prior write.
        """
        report_json = json.dumps(
            dict(finalize_report) if finalize_report else {}, default=str
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO outcomes (
                    session_id, turn_id, iterations_completed, converged,
                    drift_at_end, tokens_used, duration_ms,
                    amplification_enabled, quality_estimate,
                    finalize_report_json, written_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_id,
                    int(iterations_completed),
                    1 if converged else 0,
                    float(drift_at_end),
                    int(tokens_used),
                    int(duration_ms),
                    1 if amplification_enabled else 0,
                    quality_estimate,
                    report_json,
                    time.time(),
                ),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Garbage collection (lazy)
    # ------------------------------------------------------------------

    def gc_old_sessions(self, *, ttl_seconds: int = _DEFAULT_SESSION_TTL_SECONDS) -> int:
        """Delete sessions whose ``last_seen_at`` is older than ttl_seconds.

        Cascades to envelopes / events / outcomes via session_id. Returns
        the number of sessions deleted.
        """
        cutoff = time.time() - ttl_seconds
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "SELECT session_id FROM sessions WHERE last_seen_at < ?",
                (cutoff,),
            )
            ids = [r[0] for r in cur.fetchall()]
            if not ids:
                conn.commit()
                return 0
            placeholders = ",".join("?" for _ in ids)

            # below, never user input, so the f-string interpolation cannot
            # introduce SQL injection. ``placeholders`` is a fixed-length
            # ``?,?,?`` string built from ``ids`` length only, never values.
            # Values flow exclusively through the parameterized ``ids`` arg.
            for table in ("envelopes", "events", "outcomes"):
                conn.execute(
                    f"DELETE FROM {table} WHERE session_id IN ({placeholders})",
                    ids,
                )
            conn.execute(
                f"DELETE FROM sessions WHERE session_id IN ({placeholders})",
                ids,
            )
            conn.commit()
            return len(ids)


__all__ = [
    "StateStore",
]
