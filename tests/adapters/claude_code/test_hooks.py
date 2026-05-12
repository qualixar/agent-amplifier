# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.adapters.claude_code.hooks``.

Coverage targets: 100% line + 100% branch on hooks.py.

Test isolation:
    * Every test uses ``tmp_path`` for the SQLite state file via a fixture
      that monkeypatches ``StateStore`` default location.
    * stdin / stdout / stderr are redirected per-test.
    * The kernel is exercised via ``AgentAmplifier()`` with no I/O — it
      builds an envelope from a synthetic prompt; no LLM calls.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from agent_amplifier.adapters.claude_code import hooks as _hooks
from agent_amplifier.adapters.claude_code import state as _state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect StateStore default DB into tmp_path, return the DB path."""
    db_dir = tmp_path / "amp"
    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", db_dir)
    return db_dir / "state.db"


@pytest.fixture
def stdin_event(monkeypatch: pytest.MonkeyPatch):
    """Helper: feed a dict as JSON to sys.stdin."""

    def _feed(payload: dict | str | None) -> None:
        if payload is None:
            buf = io.StringIO("")
        elif isinstance(payload, str):
            buf = io.StringIO(payload)
        else:
            buf = io.StringIO(json.dumps(payload))
        monkeypatch.setattr(sys, "stdin", buf)

    return _feed




# ---------------------------------------------------------------------------
# UserPromptSubmit
# ---------------------------------------------------------------------------


def test_user_prompt_submit_happy_path(
    isolated_state: Path,
    stdin_event,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stdin_event(
        {
            "session_id": "sess-1",
            "cwd": "/proj",
            "model": "claude-sonnet-4-6",
            "prompt": "Refactor the authentication module to use JWT.",
        }
    )
    rc = _hooks.on_user_prompt_submit()
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "additionalContext" in payload["hookSpecificOutput"]
    assert "[amp]" in captured.err
    # Envelope persisted.
    s = _state.StateStore(isolated_state)
    last = s.latest_envelope("sess-1")
    assert last is not None
    assert last["turn_id"] == 1


def test_user_prompt_submit_missing_prompt_no_op(
    isolated_state: Path,
    stdin_event,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stdin_event({"session_id": "sess-x"})
    rc = _hooks.on_user_prompt_submit()
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_user_prompt_submit_alt_prompt_field_user_prompt(
    isolated_state: Path,
    stdin_event,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stdin_event({"session_id": "s", "user_prompt": "hi"})
    rc = _hooks.on_user_prompt_submit()
    assert rc == 0
    assert capsys.readouterr().out != ""


def test_user_prompt_submit_fail_open_on_kernel_error(
    isolated_state: Path,
    stdin_event,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If kernel crashes, hook exits 0 and logs error to stderr (fail-open)."""
    from agent_amplifier import kernel as _k

    class BoomAmp:
        def before_step(self, *a: object, **kw: object) -> object:
            raise RuntimeError("kernel boom")

        def close(self) -> None:
            pass

    monkeypatch.setattr(_k, "AgentAmplifier", lambda *a, **k: BoomAmp())
    monkeypatch.setattr(_hooks, "AgentAmplifier", lambda *a, **k: BoomAmp())
    stdin_event({"session_id": "s", "prompt": "hi"})
    rc = _hooks.on_user_prompt_submit()
    assert rc == 0
    assert "fail-open" in capsys.readouterr().err


def test_user_prompt_submit_fall_back_session_id(
    isolated_state: Path,
    stdin_event,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No session_id in event AND no env var → uses ppid sentinel."""
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    stdin_event({"prompt": "x"})
    rc = _hooks.on_user_prompt_submit()
    assert rc == 0
    assert capsys.readouterr().out != ""


def test_user_prompt_submit_uses_env_session_id(
    isolated_state: Path,
    stdin_event,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_SESSION_ID", "env-sid")
    stdin_event({"prompt": "x"})
    _hooks.on_user_prompt_submit()
    s = _state.StateStore(isolated_state)
    assert s.get_session("env-sid") is not None


def test_user_prompt_submit_model_provider_anthropic(
    isolated_state: Path, stdin_event
) -> None:
    stdin_event({"session_id": "m1", "model": "claude-haiku", "prompt": "x"})
    _hooks.on_user_prompt_submit()
    row = _state.StateStore(isolated_state).get_session("m1")
    assert row is not None
    assert row["model_provider"] == "anthropic"


def test_user_prompt_submit_model_provider_unknown(
    isolated_state: Path, stdin_event
) -> None:
    stdin_event({"session_id": "m2", "model": "gpt-5", "prompt": "x"})
    _hooks.on_user_prompt_submit()
    row = _state.StateStore(isolated_state).get_session("m2")
    assert row is not None
    assert row["model_provider"] == "unknown"


def test_user_prompt_submit_no_model_no_provider(
    isolated_state: Path, stdin_event
) -> None:
    stdin_event({"session_id": "m3", "prompt": "x"})
    _hooks.on_user_prompt_submit()
    row = _state.StateStore(isolated_state).get_session("m3")
    assert row is not None
    assert row["model"] is None


def test_user_prompt_submit_uses_event_cwd(
    isolated_state: Path, stdin_event
) -> None:
    stdin_event({"session_id": "c1", "cwd": "/explicit/cwd", "prompt": "x"})
    _hooks.on_user_prompt_submit()
    row = _state.StateStore(isolated_state).get_session("c1")
    assert row is not None
    assert row["cwd"] == "/explicit/cwd"


# ---------------------------------------------------------------------------
# PreToolUse / PostToolUse — log-only paths
# ---------------------------------------------------------------------------


def _seed_session_with_envelope(db_path: Path, sid: str = "seed") -> None:
    s = _state.StateStore(db_path)
    s.upsert_session(sid, "/cwd")
    s.record_envelope(
        sid, 1,
        user_prompt_redacted="x",
        classification_complexity="low",
        classification_domain="d",
        thinking_trigger=None, persona=None, phase="P",
        envelope_text="env",
    )


def test_pre_tool_use_logs_event(
    isolated_state: Path, stdin_event
) -> None:
    _seed_session_with_envelope(isolated_state)
    stdin_event({
        "session_id": "seed",
        "tool_name": "Bash",
        "tool_input": {"cmd": "ls"},
    })
    rc = _hooks.on_pre_tool_use()
    assert rc == 0
    s = _state.StateStore(isolated_state)
    assert s.count_events("seed") == 1


def test_pre_tool_use_no_session_id_no_op(
    isolated_state: Path, stdin_event, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When _resolve_session_id returns empty, the impl should early-return."""
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    monkeypatch.setattr(_hooks, "_resolve_session_id", lambda e: "")
    stdin_event({})
    rc = _hooks.on_pre_tool_use()
    assert rc == 0


def test_pre_tool_use_caps_event_count(
    isolated_state: Path, stdin_event, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_session_with_envelope(isolated_state)
    monkeypatch.setattr(
        _state.StateStore, "count_events", lambda self, sid, **k: 10_000
    )
    stdin_event({"session_id": "seed", "tool_name": "X"})
    rc = _hooks.on_pre_tool_use()
    assert rc == 0
    # Did NOT add another row — count remains zero from our seed.
    pre_count_query = "SELECT COUNT(*) FROM events WHERE session_id='seed'"
    import contextlib as _c
    import sqlite3 as _sq
    with _c.closing(_sq.connect(str(isolated_state))) as conn:
        n = conn.execute(pre_count_query).fetchone()[0]
    assert n == 0


def test_pre_tool_use_alt_field_names(
    isolated_state: Path, stdin_event
) -> None:
    _seed_session_with_envelope(isolated_state)
    stdin_event({"session_id": "seed", "tool": "T", "input": "i"})
    _hooks.on_pre_tool_use()
    assert _state.StateStore(isolated_state).count_events("seed") == 1


def test_pre_tool_use_no_envelope_uses_none_turn(
    isolated_state: Path, stdin_event
) -> None:
    """If no envelope exists yet, turn_id falls back to None — still logs."""
    s = _state.StateStore(isolated_state)
    s.upsert_session("noenv", "/cwd")
    stdin_event({"session_id": "noenv", "tool_name": "T"})
    _hooks.on_pre_tool_use()
    assert s.count_events("noenv") == 1


def test_post_tool_use_logs_event(
    isolated_state: Path, stdin_event
) -> None:
    _seed_session_with_envelope(isolated_state)
    stdin_event({
        "session_id": "seed",
        "tool_name": "Bash",
        "tool_input": {"cmd": "ls"},
        "tool_response": "out",
    })
    rc = _hooks.on_post_tool_use()
    assert rc == 0
    assert _state.StateStore(isolated_state).count_events("seed") == 1


def test_post_tool_use_caps_event_count(
    isolated_state: Path, stdin_event, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_session_with_envelope(isolated_state)
    monkeypatch.setattr(
        _state.StateStore, "count_events", lambda self, sid, **k: 10_000
    )
    stdin_event({"session_id": "seed", "tool_name": "X"})
    rc = _hooks.on_post_tool_use()
    assert rc == 0


def test_post_tool_use_alt_field_names(
    isolated_state: Path, stdin_event
) -> None:
    _seed_session_with_envelope(isolated_state)
    stdin_event({
        "session_id": "seed", "tool": "T", "input": "i", "output": "o",
    })
    _hooks.on_post_tool_use()
    assert _state.StateStore(isolated_state).count_events("seed") == 1


def test_post_tool_use_no_session_id_no_op(
    isolated_state: Path, stdin_event, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_hooks, "_resolve_session_id", lambda e: "")
    stdin_event({})
    rc = _hooks.on_post_tool_use()
    assert rc == 0


def test_post_tool_use_no_envelope_uses_none_turn(
    isolated_state: Path, stdin_event
) -> None:
    s = _state.StateStore(isolated_state)
    s.upsert_session("noenv2", "/cwd")
    stdin_event({"session_id": "noenv2", "tool_name": "T"})
    _hooks.on_post_tool_use()
    assert s.count_events("noenv2") == 1


# ---------------------------------------------------------------------------
# Stdin parsing edge cases
# ---------------------------------------------------------------------------


def test_read_event_empty_stdin(
    isolated_state: Path, stdin_event, capsys: pytest.CaptureFixture[str]
) -> None:
    stdin_event(None)
    # No prompt → no-op (UserPromptSubmit) — silent.
    rc = _hooks.on_user_prompt_submit()
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_read_event_malformed_json(
    isolated_state: Path, stdin_event, capsys: pytest.CaptureFixture[str]
) -> None:
    stdin_event("{not valid json")
    rc = _hooks.on_user_prompt_submit()
    assert rc == 0
    # Empty event → no prompt → no output.
    assert capsys.readouterr().out == ""


def test_read_event_non_dict_root(
    isolated_state: Path, stdin_event, capsys: pytest.CaptureFixture[str]
) -> None:
    stdin_event("[1, 2, 3]")
    rc = _hooks.on_user_prompt_submit()
    assert rc == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# main() dispatcher
# ---------------------------------------------------------------------------


def test_main_user_prompt_submit(
    isolated_state: Path, stdin_event, capsys: pytest.CaptureFixture[str]
) -> None:
    stdin_event({"session_id": "s", "prompt": "hi"})
    rc = _hooks.main(["UserPromptSubmit"])
    assert rc == 0


def test_main_pre_tool_use(
    isolated_state: Path, stdin_event
) -> None:
    _seed_session_with_envelope(isolated_state)
    stdin_event({"session_id": "seed", "tool_name": "T"})
    rc = _hooks.main(["PreToolUse"])
    assert rc == 0


def test_main_post_tool_use(
    isolated_state: Path, stdin_event
) -> None:
    _seed_session_with_envelope(isolated_state)
    stdin_event({"session_id": "seed", "tool_name": "T"})
    rc = _hooks.main(["PostToolUse"])
    assert rc == 0


def test_main_no_args_returns_2(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["hooks.py"])
    rc = _hooks.main()
    assert rc == 2
    assert "usage:" in capsys.readouterr().err


def test_main_unknown_event_returns_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = _hooks.main(["Garbage"])
    assert rc == 2
    assert "unknown" in capsys.readouterr().err


def test_main_uses_sys_argv_when_argv_none(
    isolated_state: Path,
    stdin_event,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_session_with_envelope(isolated_state)
    stdin_event({"session_id": "seed", "tool_name": "T"})
    monkeypatch.setattr(sys, "argv", ["hooks.py", "PreToolUse"])
    rc = _hooks.main()
    assert rc == 0


# ---------------------------------------------------------------------------
# _maybe_build_slm_adapter — H-5 Mode 2 wiring
# ---------------------------------------------------------------------------


def test_maybe_build_slm_adapter_when_slm_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_amplifier.adapters.slm import memory as _slm_memory

    monkeypatch.setattr(_slm_memory.shutil, "which", lambda name: "/usr/local/bin/slm")
    adapter = _hooks._resolve_adapter()
    assert adapter is not None
    assert adapter.framework_name == "slm"


def test_maybe_build_slm_adapter_when_slm_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When SLM is absent, the resolver falls back to ClaudeCodeAdapter so
    every user gets IP-9 + all 3 modes regardless of SLM install status."""
    from agent_amplifier.adapters.slm import memory as _slm_memory

    monkeypatch.setattr(_slm_memory.shutil, "which", lambda name: None)
    adapter = _hooks._resolve_adapter()
    assert adapter is not None
    assert adapter.framework_name == "claude_code"


# ---------------------------------------------------------------------------
# Abandoned-envelope sweep (Cmd+Q / force-quit / no-reply gap recovery)
# ---------------------------------------------------------------------------


def _seed_orphan(db_path: Path, sid: str, turn: int, age_seconds: float) -> None:
    """Backdate-insert an orphan envelope (no outcome) for sweep tests."""
    import contextlib as _ctx
    import sqlite3 as _sql
    import time as _time

    s = _state.StateStore(db_path)
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
    with _ctx.closing(_sql.connect(str(s.db_path))) as conn:
        conn.execute(
            "UPDATE envelopes SET created_at = ? WHERE session_id=? AND turn_id=?",
            (_time.time() - age_seconds, sid, turn),
        )
        conn.commit()


def test_user_prompt_submit_sweeps_abandoned_envelope(
    isolated_state: Path,
    stdin_event,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An orphan envelope older than 30s gets a synthetic ``abandoned`` outcome
    written when the next UPS fires (could be on a different session)."""
    _seed_orphan(isolated_state, "sess-old", 1, age_seconds=120.0)
    stdin_event(
        {"session_id": "sess-new", "cwd": "/proj", "prompt": "do a thing"}
    )
    rc = _hooks.on_user_prompt_submit()
    assert rc == 0
    capsys.readouterr()  # drain
    # The orphan should now have an outcome row with stop_reason=abandoned.
    import contextlib as _ctx
    import json as _json
    import sqlite3 as _sql
    # NOTE: ``with sqlite3.connect(...) as c`` only manages the transaction
    # — NOT the connection lifecycle. Use contextlib.closing() so the conn
    # is closed deterministically; otherwise GC'd connections raise
    # ResourceWarning later, which pytest converts to spurious unrelated
    # test failures via its unraisableexception plugin.
    with _ctx.closing(_sql.connect(str(isolated_state))) as conn:
        cur = conn.execute(
            "SELECT converged, finalize_report_json FROM outcomes "
            "WHERE session_id=? AND turn_id=?",
            ("sess-old", 1),
        )
        row = cur.fetchone()
    assert row is not None, "abandoned envelope should produce an outcome row"
    assert row[0] == 0  # converged=False
    report = _json.loads(row[1])
    assert report["stop_reason"] == "abandoned"
    assert report["tool_calls"] == 0
    assert report["tool_results"] == 0
    assert "wallclock_at_stop" in report


def test_user_prompt_submit_sweep_query_failure_is_fail_open(
    isolated_state: Path,
    stdin_event,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If ``find_abandoned_envelopes`` raises, UPS still completes normally —
    the user's amplification injection MUST NOT be blocked by a sweep bug."""
    def _boom(self, *, age_seconds: float = 30.0):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated sweep query failure")

    monkeypatch.setattr(
        _state.StateStore, "find_abandoned_envelopes", _boom
    )
    stdin_event(
        {"session_id": "sess-x", "cwd": "/proj", "prompt": "still works"}
    )
    rc = _hooks.on_user_prompt_submit()
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "sweep error" in captured.err


def test_pre_compact_logs_event_and_marks_in_flight(
    isolated_state: Path,
    stdin_event,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """PreCompact with a recent envelope present logs in_flight_amp_turn=True."""
    # Pre-create a session + envelope so the pre-compact handler sees one.
    s = _state.StateStore(isolated_state)
    s.upsert_session("sess-pc", "/proj")
    s.next_turn_id("sess-pc")
    s.record_envelope(
        "sess-pc", 1,
        user_prompt_redacted="x",
        classification_complexity="medium",
        classification_domain="general",
        thinking_trigger=None,
        persona=None,
        phase="EXPLORE",
        envelope_text="<env>",
    )
    stdin_event(
        {"session_id": "sess-pc", "compaction_reason": "size-limit"}
    )
    rc = _hooks.on_pre_compact()
    assert rc == 0
    err = capsys.readouterr().err
    assert "pre-compact" in err
    assert "in_flight_amp_turn=True" in err
    # Event row written.
    import contextlib as _ctx
    import sqlite3 as _sql
    with _ctx.closing(_sql.connect(str(isolated_state))) as conn:
        cur = conn.execute(
            "SELECT event_type, payload_json FROM events "
            "WHERE session_id='sess-pc' AND event_type='PreCompact'"
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "PreCompact"
    import json as _json
    payload = _json.loads(row[1])
    assert payload["in_flight_amp_turn"] is True
    assert payload["compaction_reason"] == "size-limit"


def test_pre_compact_no_envelope_marks_not_in_flight(
    isolated_state: Path,
    stdin_event,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """PreCompact when no prior envelope exists → in_flight_amp_turn=False."""
    s = _state.StateStore(isolated_state)
    s.upsert_session("sess-empty", "/proj")
    stdin_event({"session_id": "sess-empty"})
    rc = _hooks.on_pre_compact()
    assert rc == 0
    err = capsys.readouterr().err
    assert "in_flight_amp_turn=False" in err


def test_pre_compact_event_count_capped(
    isolated_state: Path,
    stdin_event,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """At the 10K events cap, PreCompact still logs the [amp] line but does
    NOT write a new event row (matches the pre/post-tool-use guard)."""
    s = _state.StateStore(isolated_state)
    s.upsert_session("sess-cap", "/proj")
    monkeypatch.setattr(
        _state.StateStore, "count_events", lambda self, sid: 10_000
    )
    stdin_event({"session_id": "sess-cap"})
    rc = _hooks.on_pre_compact()
    assert rc == 0
    err = capsys.readouterr().err
    assert "pre-compact" in err
    # No event row written.
    import contextlib as _ctx
    import sqlite3 as _sql
    with _ctx.closing(_sql.connect(str(isolated_state))) as conn:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='PreCompact'"
        ).fetchone()
    assert n == 0


def test_pre_compact_no_session_id_returns_zero(
    isolated_state: Path,
    stdin_event,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty session_id (defensive) returns 0 without writing anything.
    Resolved session_id never empty in practice (pid fallback), but the
    early return must remain reachable."""
    stdin_event({})
    monkeypatch.setattr(_hooks, "_resolve_session_id", lambda _e: "")
    rc = _hooks.on_pre_compact()
    assert rc == 0


def test_pre_compact_dispatch_via_main(
    isolated_state: Path,
    stdin_event,
) -> None:
    """``python -m agent_amplifier.adapters.claude_code.hooks PreCompact``
    dispatches to on_pre_compact via the main() entry point."""
    stdin_event({"session_id": "sess-dispatch"})
    rc = _hooks.main(["PreCompact"])
    assert rc == 0


def test_user_prompt_submit_sweep_row_write_failure_is_fail_open(
    isolated_state: Path,
    stdin_event,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If a per-row ``write_outcome`` raises, the sweep logs and continues —
    UPS injection still completes for the new turn."""
    _seed_orphan(isolated_state, "sess-old", 1, age_seconds=120.0)

    real_write = _state.StateStore.write_outcome

    def _raise_then_real(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Raise on every call from sweep (kwargs include stop_reason=abandoned
        # under finalize_report). UPS hook itself does NOT call write_outcome
        # for the new turn (only Stop does), so all calls during this test
        # come from the sweep. Raise unconditionally and rely on fail-open.
        report = kwargs.get("finalize_report") or {}
        if isinstance(report, dict) and report.get("stop_reason") == "abandoned":
            raise RuntimeError("simulated row-write failure")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(_state.StateStore, "write_outcome", _raise_then_real)

    stdin_event(
        {"session_id": "sess-new", "cwd": "/proj", "prompt": "fail-open test"}
    )
    rc = _hooks.on_user_prompt_submit()
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "sweep row error" in captured.err
    # New session's envelope is still persisted (UPS injection did not break).
    s = _state.StateStore(isolated_state)
    assert s.latest_envelope("sess-new") is not None


# ---------------------------------------------------------------------------
# _read_prior_classification — cross-turn context awareness helper
# ---------------------------------------------------------------------------


def test_read_prior_classification_returns_none_when_no_envelope(
    isolated_state: Path,
) -> None:
    """First turn of a session: latest_envelope returns None → helper returns None."""
    store = _state.StateStore(isolated_state)
    assert _hooks._read_prior_classification(store, "fresh-session") is None


def test_read_prior_classification_builds_from_valid_row(
    isolated_state: Path,
) -> None:
    """Valid envelope row → synthetic TaskClassification with right complexity+domain."""
    from agent_amplifier.types import EffortLevel

    store = _state.StateStore(isolated_state)
    store.upsert_session("prior-sess", "/proj")
    store.record_envelope(
        "prior-sess",
        turn_id=1,
        user_prompt_redacted="refactor auth to use JWT",
        classification_complexity="high",
        classification_domain="api",
        thinking_trigger="ultrathink",
        persona="senior",
        phase="EXPLORE",
        envelope_text="<envelope>",
    )
    result = _hooks._read_prior_classification(store, "prior-sess")
    assert result is not None
    assert result.complexity == EffortLevel.HIGH
    assert result.domain == "api"
    assert "prior_envelope_row" in result.matched_signals


def test_read_prior_classification_handles_store_exception(
    isolated_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any StateStore error during read → fail-open, return None."""
    store = _state.StateStore(isolated_state)

    def _raise(self, session_id):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated db lock")

    monkeypatch.setattr(_state.StateStore, "latest_envelope", _raise)
    assert _hooks._read_prior_classification(store, "any") is None


def test_read_prior_classification_handles_invalid_complexity_value(
    isolated_state: Path,
) -> None:
    """Row with unknown complexity string (e.g. 'frobnicate') → return None."""
    store = _state.StateStore(isolated_state)
    store.upsert_session("bad-complexity", "/proj")
    store.record_envelope(
        "bad-complexity",
        turn_id=1,
        user_prompt_redacted="x",
        classification_complexity="frobnicate",
        classification_domain="general",
        thinking_trigger="think",
        persona="senior",
        phase="EXPLORE",
        envelope_text="<env>",
    )
    assert _hooks._read_prior_classification(store, "bad-complexity") is None


def test_read_prior_classification_handles_non_string_fields(
    isolated_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row whose complexity/domain is not a string → return None."""
    store = _state.StateStore(isolated_state)

    def _bad_row(self, session_id):  # type: ignore[no-untyped-def]
        return {
            "session_id": session_id,
            "turn_id": 1,
            "classification_complexity": 42,
            "classification_domain": None,
        }

    monkeypatch.setattr(_state.StateStore, "latest_envelope", _bad_row)
    assert _hooks._read_prior_classification(store, "any") is None


def test_user_prompt_submit_passes_prior_classification_to_kernel(
    isolated_state: Path,
    stdin_event,
) -> None:
    """End-to-end: prior 'high/api' turn + bare 'ok' next prompt → kernel
    inherits via context dict (kernel-side branch coverage)."""
    store = _state.StateStore(isolated_state)
    store.upsert_session("ctx-sess", "/proj")
    store.record_envelope(
        "ctx-sess",
        turn_id=1,
        user_prompt_redacted="refactor auth to use JWT",
        classification_complexity="high",
        classification_domain="api",
        thinking_trigger="ultrathink",
        persona="senior",
        phase="EXPLORE",
        envelope_text="<env>",
    )
    stdin_event({"session_id": "ctx-sess", "cwd": "/proj", "prompt": "ok"})
    assert _hooks.on_user_prompt_submit() == 0
    # Read back the new envelope via the StateStore API itself (avoids
    # rogue raw sqlite connections leaking into pytest teardown).
    rows = _state.StateStore(isolated_state).envelopes_for_session(
        "ctx-sess"
    ) if hasattr(_state.StateStore, "envelopes_for_session") else None
    if rows is None:
        # Fallback: use latest_envelope which returns the freshest row.
        latest = _state.StateStore(isolated_state).latest_envelope("ctx-sess")
        assert latest is not None
        # Inheritance produces medium/api (one-step-down from high).
        assert latest["classification_complexity"] == "medium"
        assert latest["classification_domain"] == "api"
