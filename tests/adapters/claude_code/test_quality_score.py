# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for F1 Part A — outcome layered quality columns + transcript reader.

F1 Part A scope:
  * ``outcomes`` table gains three nullable / defaulted columns:
      - ``completed`` INTEGER NOT NULL DEFAULT 0
      - ``quality_score`` REAL  (nullable)
      - ``convergence_state`` TEXT (nullable)
  * ``transcript.final_assistant_message`` extracts the last text block
    from a Claude Code session's JSONL transcript.
  * ``write_outcome`` accepts the three new kwargs; ``completed`` defaults
    to the value of ``converged`` for backward compatibility.

F1 Part B (Tier 3 trajectory delta) and F1 Part C (per-session
convergence_state pool) land in subsequent commits.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path

import pytest

from agent_amplifier.adapters.claude_code.state import StateStore
from agent_amplifier.adapters.claude_code.transcript import (
    _extract_text_from_message_content,
    encoded_project_dir,
    final_assistant_message,
)

# ---------------------------------------------------------------------------
# Schema + migration for outcomes
# ---------------------------------------------------------------------------


def test_new_db_outcomes_has_three_v1_1_columns(tmp_path: Path) -> None:
    s = StateStore(tmp_path / "state.db")
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        cur = conn.execute("PRAGMA table_info(outcomes)")
        cols = {row[1] for row in cur.fetchall()}
    assert {"completed", "quality_score", "convergence_state"}.issubset(cols)


def test_v1_0_outcomes_db_gets_columns_added_via_migration(tmp_path: Path) -> None:
    """v1.0-shape outcomes table picks up the three new columns on open."""
    db = tmp_path / "state.db"
    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        conn.execute(
            """
            CREATE TABLE outcomes (
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
        )
        conn.execute(
            "INSERT INTO outcomes(session_id, turn_id, iterations_completed, "
            "converged, written_at) VALUES('legacy', 1, 1, 1, 0.0)"
        )
        conn.commit()

    StateStore(db)  # migration runs

    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        cur = conn.execute("PRAGMA table_info(outcomes)")
        cols = {row[1] for row in cur.fetchall()}
        assert {"completed", "quality_score", "convergence_state"}.issubset(cols)
        (legacy_completed,) = conn.execute(
            "SELECT completed FROM outcomes WHERE session_id='legacy'"
        ).fetchone()
        assert legacy_completed == 0  # safe default


def test_outcomes_migration_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    StateStore(db)
    StateStore(db)
    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        cur = conn.execute("PRAGMA table_info(outcomes)")
        names = [r[1] for r in cur.fetchall()]
    assert names.count("completed") == 1
    assert names.count("quality_score") == 1
    assert names.count("convergence_state") == 1


# ---------------------------------------------------------------------------
# write_outcome — new kwargs + backward-compat defaults
# ---------------------------------------------------------------------------


def _seed_session(s: StateStore, sid: str = "sid") -> None:
    s.upsert_session(sid, str(s.db_path.parent))


def test_write_outcome_persists_new_layered_fields(tmp_path: Path) -> None:
    s = StateStore(tmp_path / "state.db")
    _seed_session(s)
    s.write_outcome(
        "sid",
        1,
        iterations_completed=1,
        converged=True,
        completed=True,
        quality_score=0.71,
        convergence_state="improving",
    )
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        row = conn.execute(
            "SELECT completed, quality_score, convergence_state "
            "FROM outcomes WHERE session_id='sid'"
        ).fetchone()
    assert row == (1, 0.71, "improving")


def test_write_outcome_completed_defaults_to_converged_value(
    tmp_path: Path,
) -> None:
    """When ``completed`` is omitted, it mirrors ``converged`` for v1.0 compat."""
    s = StateStore(tmp_path / "state.db")
    _seed_session(s)
    s.write_outcome("sid", 1, iterations_completed=1, converged=True)
    s.write_outcome("sid", 2, iterations_completed=1, converged=False)
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        rows = conn.execute(
            "SELECT turn_id, converged, completed FROM outcomes "
            "WHERE session_id='sid' ORDER BY turn_id"
        ).fetchall()
    assert rows == [(1, 1, 1), (2, 0, 0)]  # completed mirrors converged


def test_write_outcome_quality_score_null_when_omitted(tmp_path: Path) -> None:
    s = StateStore(tmp_path / "state.db")
    _seed_session(s)
    s.write_outcome("sid", 1, iterations_completed=1, converged=True)
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        (qs, cs) = conn.execute(
            "SELECT quality_score, convergence_state FROM outcomes "
            "WHERE session_id='sid'"
        ).fetchone()
    assert qs is None
    assert cs is None


def test_write_outcome_quality_score_can_be_zero(tmp_path: Path) -> None:
    """Quality 0.0 is a valid (worst) score, distinct from NULL."""
    s = StateStore(tmp_path / "state.db")
    _seed_session(s)
    s.write_outcome(
        "sid",
        1,
        iterations_completed=1,
        converged=False,
        quality_score=0.0,
    )
    with contextlib.closing(sqlite3.connect(str(s.db_path))) as conn:
        (qs,) = conn.execute(
            "SELECT quality_score FROM outcomes WHERE session_id='sid'"
        ).fetchone()
    assert qs == 0.0


# ---------------------------------------------------------------------------
# transcript.final_assistant_message + content extractor
# ---------------------------------------------------------------------------


def test_extract_text_from_string_content() -> None:
    assert _extract_text_from_message_content("hello") == "hello"


def test_extract_text_from_list_of_text_blocks() -> None:
    content = [
        {"type": "text", "text": "alpha"},
        {"type": "text", "text": "beta"},
    ]
    assert _extract_text_from_message_content(content) == "alpha\nbeta"


def test_extract_text_ignores_tool_use_blocks() -> None:
    content = [
        {"type": "text", "text": "before"},
        {"type": "tool_use", "name": "Read", "input": {"path": "x"}},
        {"type": "text", "text": "after"},
    ]
    assert _extract_text_from_message_content(content) == "before\nafter"


def test_extract_text_from_non_string_non_list_returns_empty() -> None:
    assert _extract_text_from_message_content(42) == ""
    assert _extract_text_from_message_content(None) == ""


def test_extract_text_ignores_non_dict_list_items() -> None:
    content = ["raw string entry", {"type": "text", "text": "kept"}]
    assert _extract_text_from_message_content(content) == "kept"


def test_extract_text_block_without_text_field() -> None:
    content = [
        {"type": "text"},  # malformed — no text field
        {"type": "text", "text": "real"},
    ]
    assert _extract_text_from_message_content(content) == "real"


def _write_transcript(
    tmp_path: Path,
    session_id: str,
    project_cwd: Path,
    lines: list[dict],
) -> Path:
    """Write a synthetic Claude Code transcript at the override path."""
    base = tmp_path / "transcripts"
    target = base / encoded_project_dir(project_cwd) / f"{session_id}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(ln) for ln in lines), encoding="utf-8")
    return base


def test_final_assistant_message_missing_file_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(tmp_path / "empty"))
    assert final_assistant_message("nope", Path("/some/cwd")) is None


def test_final_assistant_message_returns_last_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _write_transcript(
        tmp_path,
        "sess",
        Path("/p"),
        [
            {"type": "user", "message": {"content": "hi"}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "first"}]},
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "FINAL"}]},
            },
        ],
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    assert final_assistant_message("sess", Path("/p")) == "FINAL"


def test_final_assistant_message_skips_blank_and_malformed_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "transcripts"
    target = base / encoded_project_dir(Path("/p")) / "sess.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(
            [
                "",
                "not-json{{",
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": "ok"}]
                        },
                    }
                ),
                "   ",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    assert final_assistant_message("sess", Path("/p")) == "ok"


def test_final_assistant_message_only_tool_use_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _write_transcript(
        tmp_path,
        "sess",
        Path("/p"),
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read", "input": {}}
                    ]
                },
            }
        ],
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    assert final_assistant_message("sess", Path("/p")) is None


def test_final_assistant_message_caps_at_max_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    big = "x" * 300_000
    base = _write_transcript(
        tmp_path,
        "sess",
        Path("/p"),
        [
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": big}]},
            }
        ],
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    result = final_assistant_message("sess", Path("/p"))
    assert result is not None
    assert len(result) == 256_000


def test_final_assistant_message_oserror_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OSError on read (e.g. transcript path is a directory) → None, no crash."""
    base = tmp_path / "transcripts"
    target = base / encoded_project_dir(Path("/p")) / "sess.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Make path a DIRECTORY so .is_file() is True for a dir? No — is_file
    # returns False for dirs. Instead make it unreadable via permissions.
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o000)
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    try:
        assert final_assistant_message("sess", Path("/p")) is None
    finally:
        target.chmod(0o644)  # restore so tmp_path cleanup works


def test_final_assistant_message_skips_non_assistant_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _write_transcript(
        tmp_path,
        "sess",
        Path("/p"),
        [
            {"type": "user", "message": {"content": "ignored"}},
            {"type": "system", "message": {"content": "ignored"}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "kept"}]},
            },
            {"foo": "bar"},  # malformed row, no type
            "not-a-dict",  # not a dict — JSON parses as str
        ],
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    assert final_assistant_message("sess", Path("/p")) == "kept"


def test_final_assistant_message_empty_text_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An assistant message whose text is whitespace-only counts as no content."""
    base = _write_transcript(
        tmp_path,
        "sess",
        Path("/p"),
        [
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "   "}]},
            }
        ],
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    assert final_assistant_message("sess", Path("/p")) is None


def test_final_assistant_message_non_dict_message_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _write_transcript(
        tmp_path,
        "sess",
        Path("/p"),
        [
            {"type": "assistant", "message": "this is not a dict"},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "real"}]},
            },
        ],
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    assert final_assistant_message("sess", Path("/p")) == "real"


# ---------------------------------------------------------------------------
# stop_hook._compute_quality_score_tier1 — Jaccard quality scorer
# ---------------------------------------------------------------------------


def _seed_assistant_transcript(
    tmp_path: Path,
    session_id: str,
    project_cwd: Path,
    text: str,
) -> Path:
    base = tmp_path / "transcripts"
    target = base / encoded_project_dir(project_cwd) / f"{session_id}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": text}]},
            }
        ),
        encoding="utf-8",
    )
    return base


def test_quality_score_identical_text_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_amplifier.adapters.claude_code.stop_hook import (
        _compute_quality_score_tier1,
    )

    base = _seed_assistant_transcript(
        tmp_path, "sess", Path("/p"), "alpha beta gamma"
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    envelope = {
        "user_prompt_redacted": "alpha beta gamma",
        "envelope_text": "",
    }
    score = _compute_quality_score_tier1(
        envelope=envelope, session_id="sess", project_cwd="/p"
    )
    assert score == 1.0


def test_quality_score_disjoint_text_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_amplifier.adapters.claude_code.stop_hook import (
        _compute_quality_score_tier1,
    )

    base = _seed_assistant_transcript(
        tmp_path, "sess", Path("/p"), "completely unrelated words here"
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    envelope = {
        "user_prompt_redacted": "alpha beta gamma delta",
        "envelope_text": "",
    }
    score = _compute_quality_score_tier1(
        envelope=envelope, session_id="sess", project_cwd="/p"
    )
    assert score == 0.0


def test_quality_score_partial_overlap_in_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_amplifier.adapters.claude_code.stop_hook import (
        _compute_quality_score_tier1,
    )

    base = _seed_assistant_transcript(
        tmp_path, "sess", Path("/p"), "alpha beta gamma delta"
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    envelope = {
        "user_prompt_redacted": "alpha beta",
        "envelope_text": "",
    }
    score = _compute_quality_score_tier1(
        envelope=envelope, session_id="sess", project_cwd="/p"
    )
    assert score is not None
    assert 0.0 < score < 1.0


def test_quality_score_returns_none_when_transcript_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_amplifier.adapters.claude_code.stop_hook import (
        _compute_quality_score_tier1,
    )

    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(tmp_path / "empty"))
    envelope = {
        "user_prompt_redacted": "anything",
        "envelope_text": "",
    }
    assert (
        _compute_quality_score_tier1(
            envelope=envelope, session_id="absent", project_cwd="/p"
        )
        is None
    )


def test_quality_score_returns_none_when_envelope_has_no_string_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_amplifier.adapters.claude_code.stop_hook import (
        _compute_quality_score_tier1,
    )

    base = _seed_assistant_transcript(
        tmp_path, "sess", Path("/p"), "some output"
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    envelope: dict[str, object] = {
        "user_prompt_redacted": None,
        "envelope_text": None,
    }
    assert (
        _compute_quality_score_tier1(
            envelope=envelope, session_id="sess", project_cwd="/p"
        )
        is None
    )


def test_quality_score_uses_envelope_text_when_prompt_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_amplifier.adapters.claude_code.stop_hook import (
        _compute_quality_score_tier1,
    )

    base = _seed_assistant_transcript(
        tmp_path, "sess", Path("/p"), "alpha beta"
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    envelope: dict[str, object] = {
        "user_prompt_redacted": None,
        "envelope_text": "alpha beta",
    }
    score = _compute_quality_score_tier1(
        envelope=envelope, session_id="sess", project_cwd="/p"
    )
    assert score == 1.0


def test_quality_score_returns_none_when_goal_keyword_set_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whitespace-only prompt yields empty keyword set → None, not divide-by-zero."""
    from agent_amplifier.adapters.claude_code.stop_hook import (
        _compute_quality_score_tier1,
    )

    base = _seed_assistant_transcript(
        tmp_path, "sess", Path("/p"), "real output text"
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    envelope = {
        "user_prompt_redacted": "    ",
        "envelope_text": "",
    }
    assert (
        _compute_quality_score_tier1(
            envelope=envelope, session_id="sess", project_cwd="/p"
        )
        is None
    )


def test_quality_score_returns_none_when_output_keyword_set_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whitespace-only assistant message → final_assistant_message returns
    None (strips whitespace), so the scorer short-circuits to None.
    """
    from agent_amplifier.adapters.claude_code.stop_hook import (
        _compute_quality_score_tier1,
    )

    base = _seed_assistant_transcript(tmp_path, "sess", Path("/p"), "      ")
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(base))
    envelope = {
        "user_prompt_redacted": "real prompt",
        "envelope_text": "",
    }
    assert (
        _compute_quality_score_tier1(
            envelope=envelope, session_id="sess", project_cwd="/p"
        )
        is None
    )
