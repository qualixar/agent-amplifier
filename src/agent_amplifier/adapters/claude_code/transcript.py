# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Claude Code transcript JSONL reader (IP-10 v2 / Option C).

Parses ``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`` to extract
real per-session token usage that the Stop hook can write to
``outcomes.tokens_used``.

**Why this exists**: the v1.0 Stop hook hardcoded ``tokens_used=0`` because
the Claude Code hook payload does not expose token counts. The transcript
JSONL, however, does — each ``type: "assistant"`` line carries a
``message.usage`` dict.

**Fail-open contract**: every public function returns 0 (or an empty path)
rather than raising. The Stop hook MUST NOT crash because of a transcript
glitch — a stale token reading is preferable to a missed Outcome row.

**Token fields summed** (all four count toward "tokens used"):
    input_tokens, output_tokens,
    cache_creation_input_tokens, cache_read_input_tokens
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Final

LOG = logging.getLogger(__name__)

_USAGE_FIELDS: Final[tuple[str, ...]] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def encoded_project_dir(project_cwd: Path) -> str:
    """Encode ``project_cwd`` the way Claude Code does for its transcript tree.

    Claude Code replaces ``/``, ``_``, and ``.`` with ``-`` (verified against
    ``~/.claude/projects/`` on disk — e.g. ``/Users/x/Agentic_official`` →
    ``-Users-x-Agentic-official``, ``~/.claude`` → ``--claude``).
    """
    encoded = str(project_cwd)
    for ch in ("/", "_", "."):
        encoded = encoded.replace(ch, "-")
    return encoded


def transcript_path(session_id: str, project_cwd: Path) -> Path:
    """Return the absolute path to the JSONL transcript for a session.

    Uses ``AGENT_AMP_TRANSCRIPT_DIR`` (test/CI override) if set, otherwise
    ``$HOME/.claude/projects/``.
    """
    override = os.environ.get("AGENT_AMP_TRANSCRIPT_DIR")
    if override:
        base = Path(override)
    else:
        base = Path(os.environ.get("HOME", "~")).expanduser() / ".claude" / "projects"
    return base / encoded_project_dir(project_cwd) / f"{session_id}.jsonl"


def _sum_usage(usage: object) -> int:
    """Sum the four token fields from a single ``message.usage`` dict.

    Non-int values are silently ignored — Claude Code occasionally returns
    nested dicts (e.g. ``cache_creation`` with ``ephemeral_*`` keys); only
    plain int fields at the top level contribute.
    """
    if not isinstance(usage, dict):
        return 0
    total = 0
    for field in _USAGE_FIELDS:
        v = usage.get(field)
        if isinstance(v, int) and not isinstance(v, bool):
            total += v
    return total


def tokens_for_session(session_id: str, project_cwd: Path) -> int:
    """Return the cumulative tokens used across all assistant messages.

    Fail-open: every error path returns 0.
    """
    path = transcript_path(session_id, project_cwd)
    if not path.is_file():
        return 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        LOG.debug("transcript read failed for %s: %s", session_id, exc)
        return 0
    total = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Malformed line (including a partial last line during live writes)
            # — skip and keep summing.
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "assistant":
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        total += _sum_usage(msg.get("usage"))
    return total


# ---------------------------------------------------------------------------
# Backfill helper — one-shot DB update for existing v1.0 outcome rows
# ---------------------------------------------------------------------------


def backfill_outcomes(db_path: Path) -> dict[str, int]:
    """Recompute ``tokens_used`` for every outcome row from its transcript.

    Strategy: for each session, find the latest turn's outcome row, write the
    cumulative transcript total to it, and zero out earlier rows. Earlier rows
    cannot recover per-turn deltas (we did not snapshot cumulative at each
    Stop), so concentrating the total at the session-close turn gives the
    most useful approximation for the "Tokens Today" widget.

    Returns ``{"sessions_processed": N, "rows_updated": M, "tokens_total": T}``.

    Fail-open per session: if a session's transcript is missing or unreadable,
    that session is skipped (its rows stay at 0). Other sessions still get
    backfilled. Never raises.
    """
    import contextlib
    import sqlite3

    summary = {"sessions_processed": 0, "rows_updated": 0, "tokens_total": 0}
    if not db_path.is_file():
        return summary

    # NOTE: ``with sqlite3.connect(...)`` only manages the transaction, not
    # the connection lifetime. ``contextlib.closing`` ensures the connection
    # is released even if the inner block raises, avoiding the unraisable
    # ResourceWarning on GC.
    with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
        sessions = conn.execute(
            "SELECT session_id, cwd FROM sessions"
        ).fetchall()
        for sid, cwd in sessions:
            if not cwd:
                continue
            try:
                cumulative = tokens_for_session(sid, Path(cwd))
            except Exception:
                continue
            summary["sessions_processed"] += 1
            if cumulative <= 0:
                continue
            # Find the latest turn for this session.
            row = conn.execute(
                "SELECT MAX(turn_id) FROM outcomes WHERE session_id = ?",
                (sid,),
            ).fetchone()
            if not row or row[0] is None:
                continue
            latest_turn = int(row[0])
            # Zero out earlier rows, set cumulative on latest.
            conn.execute(
                "UPDATE outcomes SET tokens_used = 0 "
                "WHERE session_id = ? AND turn_id < ?",
                (sid, latest_turn),
            )
            cur = conn.execute(
                "UPDATE outcomes SET tokens_used = ? "
                "WHERE session_id = ? AND turn_id = ?",
                (int(cumulative), sid, latest_turn),
            )
            summary["rows_updated"] += cur.rowcount
            summary["tokens_total"] += int(cumulative)
        conn.commit()
    return summary


__all__ = [
    "backfill_outcomes",
    "encoded_project_dir",
    "tokens_for_session",
    "transcript_path",
]
