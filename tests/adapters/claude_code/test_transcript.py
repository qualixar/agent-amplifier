# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""RED tests for Claude Code transcript JSONL reader (Option C, IP-10 v2).

Replaces the v1.0 ``tokens_used=0`` hardcode in stop_hook.py with real
per-session token counts parsed from the Claude Code transcript.

The transcript lives at:
    ~/.claude/projects/<encoded-cwd>/<session-id>.jsonl
Each ``type: "assistant"`` line carries a ``message.usage`` dict with:
    input_tokens, output_tokens,
    cache_creation_input_tokens, cache_read_input_tokens
Total tokens for the session = sum of all four fields across all assistant
messages.

The reader is fail-open: missing file, malformed JSON, partial last line,
or any other error MUST return 0 rather than raise. Stop hook callers
depend on this contract — a transcript glitch must never abort the hook.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# encoded_project_dir: the path-encoding scheme Claude Code uses
# ---------------------------------------------------------------------------


def test_encoded_project_dir_replaces_slashes_with_dashes() -> None:
    from agent_amplifier.adapters.claude_code.transcript import encoded_project_dir

    assert encoded_project_dir(
        Path("/Users/varunpratapbhardwaj/Documents/work/varun-world/Agentic_official")
    ) == "-Users-varunpratapbhardwaj-Documents-work-varun-world-Agentic-official"


def test_encoded_project_dir_handles_relative_path() -> None:
    from agent_amplifier.adapters.claude_code.transcript import encoded_project_dir

    enc = encoded_project_dir(Path("./project"))
    assert "/" not in enc


def test_encoded_project_dir_handles_root_path() -> None:
    from agent_amplifier.adapters.claude_code.transcript import encoded_project_dir

    assert encoded_project_dir(Path("/")) == "-"


# ---------------------------------------------------------------------------
# transcript_path: resolves the JSONL file location
# ---------------------------------------------------------------------------


def test_transcript_path_under_claude_projects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent_amplifier.adapters.claude_code.transcript import transcript_path

    monkeypatch.setenv("HOME", str(tmp_path))
    p = transcript_path("abc-123", Path("/Users/me/proj"))
    assert p == tmp_path / ".claude" / "projects" / "-Users-me-proj" / "abc-123.jsonl"


def test_transcript_path_respects_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AGENT_AMP_TRANSCRIPT_DIR lets tests/CI pin the projects root."""
    from agent_amplifier.adapters.claude_code.transcript import transcript_path

    custom = tmp_path / "custom-projects"
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(custom))
    p = transcript_path("sid", Path("/x"))
    assert p == custom / "-x" / "sid.jsonl"


# ---------------------------------------------------------------------------
# tokens_for_session: fail-open + correct summation
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Stand up an isolated ``~/.claude/projects/<enc>/`` and return its parent cwd."""
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(tmp_path / "projects"))
    return Path("/fake/cwd")


def _write_transcript(
    tmp_path: Path, project_cwd: Path, session_id: str, lines: list[dict[str, object]]
) -> Path:
    from agent_amplifier.adapters.claude_code.transcript import (
        encoded_project_dir,
        transcript_path,
    )
    p = transcript_path(session_id, project_cwd)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    _ = encoded_project_dir(project_cwd)
    return p


def test_missing_file_returns_zero(project_root: Path) -> None:
    from agent_amplifier.adapters.claude_code.transcript import tokens_for_session

    assert tokens_for_session("never-existed", project_root) == 0


def test_empty_file_returns_zero(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_amplifier.adapters.claude_code.transcript import (
        tokens_for_session,
        transcript_path,
    )

    p = transcript_path("s", project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    assert tokens_for_session("s", project_root) == 0


def test_sums_four_token_fields(project_root: Path, tmp_path: Path) -> None:
    from agent_amplifier.adapters.claude_code.transcript import tokens_for_session

    _write_transcript(
        tmp_path,
        project_root,
        "s",
        [
            {
                "type": "assistant",
                "message": {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "cache_creation_input_tokens": 100,
                        "cache_read_input_tokens": 1000,
                    }
                },
            }
        ],
    )
    assert tokens_for_session("s", project_root) == 10 + 20 + 100 + 1000


def test_multiple_assistant_messages_sum(
    project_root: Path, tmp_path: Path
) -> None:
    from agent_amplifier.adapters.claude_code.transcript import tokens_for_session

    _write_transcript(
        tmp_path,
        project_root,
        "s",
        [
            {"type": "assistant", "message": {"usage": {"input_tokens": 5}}},
            {"type": "user", "message": {}},  # non-assistant entry skipped
            {
                "type": "assistant",
                "message": {
                    "usage": {"output_tokens": 7, "cache_read_input_tokens": 13}
                },
            },
        ],
    )
    assert tokens_for_session("s", project_root) == 5 + 7 + 13


def test_skips_non_assistant_entries(project_root: Path, tmp_path: Path) -> None:
    from agent_amplifier.adapters.claude_code.transcript import tokens_for_session

    _write_transcript(
        tmp_path,
        project_root,
        "s",
        [
            {"type": "user", "message": {"usage": {"input_tokens": 999}}},
            {"type": "tool_result", "message": {"usage": {"input_tokens": 999}}},
            {"type": "assistant", "message": {"usage": {"input_tokens": 1}}},
        ],
    )
    assert tokens_for_session("s", project_root) == 1


def test_handles_missing_usage_field(project_root: Path, tmp_path: Path) -> None:
    from agent_amplifier.adapters.claude_code.transcript import tokens_for_session

    _write_transcript(
        tmp_path,
        project_root,
        "s",
        [
            {"type": "assistant", "message": {}},
            {"type": "assistant", "message": {"usage": {"input_tokens": 42}}},
        ],
    )
    assert tokens_for_session("s", project_root) == 42


def test_handles_missing_message_field(project_root: Path, tmp_path: Path) -> None:
    from agent_amplifier.adapters.claude_code.transcript import tokens_for_session

    _write_transcript(
        tmp_path,
        project_root,
        "s",
        [
            {"type": "assistant"},  # no message at all
            {"type": "assistant", "message": {"usage": {"input_tokens": 9}}},
        ],
    )
    assert tokens_for_session("s", project_root) == 9


def test_malformed_json_lines_are_skipped(
    project_root: Path, tmp_path: Path
) -> None:
    from agent_amplifier.adapters.claude_code.transcript import (
        tokens_for_session,
        transcript_path,
    )

    p = transcript_path("s", project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "not json at all\n"
        '{"type": "assistant", "message": {"usage": {"input_tokens": 50}}}\n'
        "{ broken json\n"
    )
    assert tokens_for_session("s", project_root) == 50


def test_partial_last_line_is_skipped(
    project_root: Path, tmp_path: Path
) -> None:
    """Transcript is being written live; the last partial line must not crash."""
    from agent_amplifier.adapters.claude_code.transcript import (
        tokens_for_session,
        transcript_path,
    )

    p = transcript_path("s", project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        '{"type": "assistant", "message": {"usage": {"input_tokens": 1}}}\n'
        '{"type": "assistant", "message"'  # truncated mid-write
    )
    assert tokens_for_session("s", project_root) == 1


def test_non_dict_usage_returns_zero_for_that_entry(
    project_root: Path, tmp_path: Path
) -> None:
    from agent_amplifier.adapters.claude_code.transcript import tokens_for_session

    _write_transcript(
        tmp_path,
        project_root,
        "s",
        [
            {"type": "assistant", "message": {"usage": "not a dict"}},
            {"type": "assistant", "message": {"usage": {"output_tokens": 11}}},
        ],
    )
    assert tokens_for_session("s", project_root) == 11


def test_non_int_token_value_is_skipped(
    project_root: Path, tmp_path: Path
) -> None:
    from agent_amplifier.adapters.claude_code.transcript import tokens_for_session

    _write_transcript(
        tmp_path,
        project_root,
        "s",
        [
            {
                "type": "assistant",
                "message": {
                    "usage": {
                        "input_tokens": "not-an-int",
                        "output_tokens": 8,
                    }
                },
            }
        ],
    )
    assert tokens_for_session("s", project_root) == 8


class TestBackfillOutcomes:
    def test_backfill_writes_total_to_latest_turn(
        self, project_root: Path, tmp_path: Path
    ) -> None:
        from agent_amplifier.adapters.claude_code import state as _state
        from agent_amplifier.adapters.claude_code.transcript import (
            backfill_outcomes,
            transcript_path,
        )

        db_path = tmp_path / "state.db"
        store = _state.StateStore(db_path)
        store.upsert_session("sid-a", str(project_root))
        for turn_id in (1, 2, 3):
            store.record_envelope(
                "sid-a", turn_id,
                user_prompt_redacted="r",
                classification_complexity="low",
                classification_domain="d",
                thinking_trigger=None, persona=None, phase="P",
                envelope_text="env",
            )
            store.write_outcome(
                "sid-a", turn_id,
                iterations_completed=1, converged=True, tokens_used=0,
            )
        p = transcript_path("sid-a", project_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            '{"type": "assistant", "message": {"usage": {"input_tokens": 500}}}\n'
        )
        summary = backfill_outcomes(db_path)
        assert summary["sessions_processed"] == 1
        assert summary["rows_updated"] == 1
        assert summary["tokens_total"] == 500
        # Latest turn has full total; earlier turns are zero.
        import contextlib
        import sqlite3
        with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
            rows = conn.execute(
                "SELECT turn_id, tokens_used FROM outcomes "
                "WHERE session_id=? ORDER BY turn_id",
                ("sid-a",),
            ).fetchall()
        assert rows == [(1, 0), (2, 0), (3, 500)]

    def test_backfill_skips_session_with_no_transcript(
        self, project_root: Path, tmp_path: Path
    ) -> None:
        from agent_amplifier.adapters.claude_code import state as _state
        from agent_amplifier.adapters.claude_code.transcript import backfill_outcomes

        db_path = tmp_path / "state.db"
        store = _state.StateStore(db_path)
        store.upsert_session("sid-b", str(project_root))
        store.record_envelope(
            "sid-b", 1,
            user_prompt_redacted="r",
            classification_complexity="low",
            classification_domain="d",
            thinking_trigger=None, persona=None, phase="P",
            envelope_text="env",
        )
        store.write_outcome(
            "sid-b", 1, iterations_completed=1, converged=True, tokens_used=0,
        )
        summary = backfill_outcomes(db_path)
        assert summary["sessions_processed"] == 1
        assert summary["rows_updated"] == 0
        assert summary["tokens_total"] == 0

    def test_backfill_skips_session_with_blank_cwd(
        self, tmp_path: Path
    ) -> None:
        """Defensive: session row with empty cwd is skipped."""
        import contextlib
        import sqlite3

        from agent_amplifier.adapters.claude_code import state as _state
        from agent_amplifier.adapters.claude_code.transcript import backfill_outcomes

        db_path = tmp_path / "state.db"
        _ = _state.StateStore(db_path)  # creates schema
        with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, cwd, started_at, last_seen_at) "
                "VALUES (?, ?, ?, ?)",
                ("sid-empty", "", 1.0, 1.0),
            )
            conn.commit()
        summary = backfill_outcomes(db_path)
        assert summary["sessions_processed"] == 0

    def test_backfill_skips_session_with_no_outcome_rows(
        self, project_root: Path, tmp_path: Path
    ) -> None:
        """Session has a transcript but no outcomes — nothing to update."""
        from agent_amplifier.adapters.claude_code import state as _state
        from agent_amplifier.adapters.claude_code.transcript import (
            backfill_outcomes,
            transcript_path,
        )

        db_path = tmp_path / "state.db"
        store = _state.StateStore(db_path)
        store.upsert_session("sid-c", str(project_root))
        p = transcript_path("sid-c", project_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            '{"type": "assistant", "message": {"usage": {"input_tokens": 50}}}\n'
        )
        summary = backfill_outcomes(db_path)
        assert summary["sessions_processed"] == 1
        assert summary["rows_updated"] == 0

    def test_backfill_skips_session_with_unreadable_transcript(
        self,
        project_root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agent_amplifier.adapters.claude_code import state as _state
        from agent_amplifier.adapters.claude_code import transcript as _t

        db_path = tmp_path / "state.db"
        store = _state.StateStore(db_path)
        store.upsert_session("sid-d", str(project_root))
        store.record_envelope(
            "sid-d", 1,
            user_prompt_redacted="r",
            classification_complexity="low",
            classification_domain="d",
            thinking_trigger=None, persona=None, phase="P",
            envelope_text="env",
        )
        store.write_outcome(
            "sid-d", 1, iterations_completed=1, converged=True, tokens_used=0,
        )

        def _boom(*_a: object, **_k: object) -> int:
            raise RuntimeError("simulated transcript read failure")

        monkeypatch.setattr(_t, "tokens_for_session", _boom)
        summary = _t.backfill_outcomes(db_path)
        assert summary["sessions_processed"] == 0
        assert summary["rows_updated"] == 0

    def test_backfill_missing_db_returns_empty_summary(self, tmp_path: Path) -> None:
        from agent_amplifier.adapters.claude_code.transcript import backfill_outcomes

        result = backfill_outcomes(tmp_path / "does-not-exist.db")
        assert result == {
            "sessions_processed": 0,
            "rows_updated": 0,
            "tokens_total": 0,
        }


def test_blank_lines_in_transcript_are_skipped(
    project_root: Path, tmp_path: Path
) -> None:
    """Empty/whitespace-only lines do not contribute to the total."""
    from agent_amplifier.adapters.claude_code.transcript import (
        tokens_for_session,
        transcript_path,
    )

    p = transcript_path("s", project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        '\n'
        '   \n'
        '{"type": "assistant", "message": {"usage": {"input_tokens": 7}}}\n'
        '\n'
    )
    assert tokens_for_session("s", project_root) == 7


def test_non_dict_json_line_is_skipped(
    project_root: Path, tmp_path: Path
) -> None:
    """A JSON line decoding to a list/scalar is skipped, not crashed."""
    from agent_amplifier.adapters.claude_code.transcript import (
        tokens_for_session,
        transcript_path,
    )

    p = transcript_path("s", project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        '[1, 2, 3]\n'
        '"a string"\n'
        '42\n'
        '{"type": "assistant", "message": {"usage": {"output_tokens": 4}}}\n'
    )
    assert tokens_for_session("s", project_root) == 4


def test_unreadable_file_returns_zero_does_not_raise(
    project_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """OSError on read must surface as 0, never propagate."""
    from agent_amplifier.adapters.claude_code import transcript as _t

    # Patch read_text to raise.
    def _boom(self: Path, *args: object, **kwargs: object) -> str:
        raise OSError("simulated read failure")

    monkeypatch.setattr(_t.Path, "read_text", _boom)
    # Create file so the exists() short-circuit doesn't fire.
    p = _t.transcript_path("s", project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("anything")
    assert _t.tokens_for_session("s", project_root) == 0
