# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.adapters.claude_code.stop_hook``.

Coverage targets: 100% line + 100% branch on stop_hook.py.
"""
from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from agent_amplifier.adapters.claude_code import state as _state
from agent_amplifier.adapters.claude_code import stop_hook as _sh


@pytest.fixture
def isolated_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    db_dir = tmp_path / "amp"
    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", db_dir)
    return db_dir / "state.db"


@pytest.fixture
def stdin_event(monkeypatch: pytest.MonkeyPatch):
    def _feed(payload: dict | str | None) -> None:
        if payload is None:
            buf = io.StringIO("")
        elif isinstance(payload, str):
            buf = io.StringIO(payload)
        else:
            buf = io.StringIO(json.dumps(payload))
        monkeypatch.setattr(sys, "stdin", buf)
    return _feed


def _seed(db_path: Path, sid: str = "s") -> None:
    s = _state.StateStore(db_path)
    s.upsert_session(sid, "/cwd")
    s.record_envelope(
        sid, 1,
        user_prompt_redacted="r",
        classification_complexity="low",
        classification_domain="d",
        thinking_trigger=None, persona=None, phase="P",
        envelope_text="env",
    )


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


def test_on_stop_writes_outcome_and_logs(
    isolated_state: Path, stdin_event,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed(isolated_state, "s1")
    s = _state.StateStore(isolated_state)
    # Two pre-tool, two post-tool — converged (in_flight=0).
    for _ in range(2):
        s.record_event("s1", "PreToolUse", turn_id=1, tool_name="T", payload={})
    for _ in range(2):
        s.record_event("s1", "PostToolUse", turn_id=1, tool_name="T", payload={})
    stdin_event({"session_id": "s1", "stop_reason": "complete"})
    rc = _sh.on_stop()
    assert rc == 0
    err = capsys.readouterr().err
    assert "[amp] stop" in err
    assert "tools=2/2" in err
    # Outcome row written.
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        row = conn.execute(
            "SELECT converged, finalize_report_json FROM outcomes "
            "WHERE session_id=? AND turn_id=?", ("s1", 1)
        ).fetchone()
    assert row is not None
    assert row[0] == 1  # converged
    report = json.loads(row[1])
    assert report["tool_calls"] == 2
    assert report["tool_results"] == 2
    assert report["in_flight_at_stop"] == 0
    assert report["stop_reason"] == "complete"


def test_on_stop_in_flight_when_pre_exceeds_post(
    isolated_state: Path, stdin_event
) -> None:
    _seed(isolated_state, "s2")
    s = _state.StateStore(isolated_state)
    for _ in range(3):
        s.record_event("s2", "PreToolUse", turn_id=1, tool_name="T", payload={})
    s.record_event("s2", "PostToolUse", turn_id=1, tool_name="T", payload={})
    stdin_event({"session_id": "s2"})
    _sh.on_stop()
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        row = conn.execute(
            "SELECT converged, finalize_report_json FROM outcomes WHERE session_id=?",
            ("s2",),
        ).fetchone()
    assert row is not None
    assert row[0] == 0  # not converged because in-flight > 0
    assert json.loads(row[1])["in_flight_at_stop"] == 2


def test_on_stop_no_envelope_path(
    isolated_state: Path, stdin_event,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If no envelope row exists for this session, log no-envelope and close."""
    _state.StateStore(isolated_state).upsert_session("orphan", "/cwd")
    stdin_event({"session_id": "orphan"})
    rc = _sh.on_stop()
    assert rc == 0
    assert "no-envelope" in capsys.readouterr().err
    row = _state.StateStore(isolated_state).get_session("orphan")
    assert row is not None
    assert row["closed_at"] is not None


def test_on_stop_duration_zero_when_lt_two_events(
    isolated_state: Path, stdin_event
) -> None:
    _seed(isolated_state, "s3")
    s = _state.StateStore(isolated_state)
    s.record_event("s3", "PreToolUse", turn_id=1, tool_name="T", payload={})
    stdin_event({"session_id": "s3"})
    _sh.on_stop()
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        d = conn.execute(
            "SELECT duration_ms FROM outcomes WHERE session_id=?", ("s3",)
        ).fetchone()[0]
    assert d == 0


def test_on_stop_duration_calculated(
    isolated_state: Path, stdin_event
) -> None:
    _seed(isolated_state, "s4")
    s = _state.StateStore(isolated_state)
    # Manually insert two events 0.5s apart via raw SQL.
    now = time.time()
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        conn.execute(
            "INSERT INTO events (session_id, turn_id, event_type, tool_name, payload_json, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("s4", 1, "PreToolUse", "T", "{}", now),
        )
        conn.execute(
            "INSERT INTO events (session_id, turn_id, event_type, tool_name, payload_json, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("s4", 1, "PostToolUse", "T", "{}", now + 0.5),
        )
        conn.commit()
    stdin_event({"session_id": "s4"})
    _sh.on_stop()
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        d = conn.execute(
            "SELECT duration_ms FROM outcomes WHERE session_id=?", ("s4",)
        ).fetchone()[0]
    assert 400 <= d <= 600
    # Touch the unused StateStore so the variable is meaningful (lint hygiene).
    assert s.get_session("s4") is not None


# ---------------------------------------------------------------------------
# Stdin parsing edge cases
# ---------------------------------------------------------------------------


def test_on_stop_empty_stdin(
    isolated_state: Path, stdin_event
) -> None:
    stdin_event(None)
    rc = _sh.on_stop()
    assert rc == 0  # fail-open / no-envelope path


def test_on_stop_malformed_json(
    isolated_state: Path, stdin_event
) -> None:
    stdin_event("{bad")
    rc = _sh.on_stop()
    assert rc == 0


def test_on_stop_non_dict_root(
    isolated_state: Path, stdin_event
) -> None:
    stdin_event("[1,2,3]")
    rc = _sh.on_stop()
    assert rc == 0


def test_on_stop_uses_env_session_id(
    isolated_state: Path, stdin_event,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed(isolated_state, "envsid")
    monkeypatch.setenv("CLAUDE_SESSION_ID", "envsid")
    stdin_event({})
    _sh.on_stop()
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        row = conn.execute(
            "SELECT 1 FROM outcomes WHERE session_id='envsid'"
        ).fetchone()
    assert row is not None


def test_on_stop_falls_back_to_ppid_session(
    isolated_state: Path, stdin_event,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    stdin_event({})
    rc = _sh.on_stop()
    assert rc == 0  # no-envelope branch hits the ppid sentinel sid


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------


def test_on_stop_fail_open_when_state_raises(
    isolated_state: Path, stdin_event,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*a: object, **k: object) -> object:
        raise RuntimeError("simulated state failure")

    monkeypatch.setattr(_sh, "_store", boom)
    stdin_event({"session_id": "x"})
    rc = _sh.on_stop()
    assert rc == 0
    assert "fail-open" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_no_args(
    isolated_state: Path, stdin_event,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdin_event({})
    monkeypatch.setattr(sys, "argv", ["stop_hook.py"])
    rc = _sh.main()
    assert rc == 0


def test_main_with_args(
    isolated_state: Path, stdin_event,
) -> None:
    stdin_event({})
    rc = _sh.main(["--debug"])
    assert rc == 0


def test_on_stop_out_of_order_event_does_not_update_last_ts(
    isolated_state: Path, stdin_event,
) -> None:
    """Hits the false branch of `if last_ts is None or ts_f > last_ts:` — when
    a later-fetched event has an EARLIER timestamp than the running max."""
    _seed(isolated_state, "ord")
    now = time.time()
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        # Insert in (1.0, 2.0, 0.5) order — third event is OLDER than max.
        for ts in (now + 1.0, now + 2.0, now + 0.5):
            conn.execute(
                "INSERT INTO events (session_id, turn_id, event_type, "
                "tool_name, payload_json, timestamp) VALUES (?,?,?,?,?,?)",
                ("ord", 1, "PreToolUse", "T", "{}", ts),
            )
        conn.commit()
    stdin_event({"session_id": "ord"})
    _sh.on_stop()
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        d = conn.execute(
            "SELECT duration_ms FROM outcomes WHERE session_id=?", ("ord",)
        ).fetchone()[0]
    # last_ts stayed at now+2.0; first_ts ended up at now+0.5.
    assert 1400 <= d <= 1600


def test_on_stop_writeback_falls_back_to_memory_md_when_slm_absent(
    isolated_state: Path, stdin_event,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When SLM is not on PATH, Stop hook writes the outcome to MEMORY.md."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(_sh.shutil, "which", lambda name: None)
    _seed(isolated_state, "nslm")
    s = _state.StateStore(isolated_state)
    s.record_event("nslm", "PreToolUse", turn_id=1, tool_name="T", payload={})
    s.record_event("nslm", "PostToolUse", turn_id=1, tool_name="T", payload={})
    stdin_event({"session_id": "nslm"})
    rc = _sh.on_stop()
    assert rc == 0
    assert "writeback=memory.md" in capsys.readouterr().err
    assert (project_dir / "MEMORY.md").exists()
    assert "Amplifier note" in (project_dir / "MEMORY.md").read_text()


def test_on_stop_writeback_memory_md_failed_branch(
    isolated_state: Path, stdin_event,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the MEMORY.md write helper returns False, log memory.md-failed."""
    monkeypatch.setattr(_sh.shutil, "which", lambda name: None)
    monkeypatch.setattr(_sh, "_maybe_writeback_to_memory_md", lambda *a, **k: False)
    _seed(isolated_state, "f1")
    stdin_event({"session_id": "f1"})
    _sh.on_stop()
    assert "writeback=memory.md-failed" in capsys.readouterr().err


def test_on_stop_writeback_slm_failed_branch(
    isolated_state: Path, stdin_event,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When SLM is on PATH but the writeback returns False (e.g. CLI exit !=0)."""
    monkeypatch.setattr(_sh.shutil, "which", lambda name: "/usr/bin/slm")
    monkeypatch.setattr(_sh, "_maybe_writeback_to_slm", lambda *a, **k: False)
    _seed(isolated_state, "f2")
    stdin_event({"session_id": "f2"})
    _sh.on_stop()
    assert "writeback=slm-failed" in capsys.readouterr().err


def test_maybe_writeback_to_memory_md_maps_complexity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Direct exercise of the complexity → EffortLevel map and the
    converged → quality fork to close coverage on both branches."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    for complexity in ("minimal", "low", "medium", "high", "max", "unknown"):
        env = {
            "user_prompt_redacted": f"q-{complexity}",
            "classification_complexity": complexity,
        }
        ok = _sh._maybe_writeback_to_memory_md(
            env,
            pre_count=0, post_count=0,
            duration_ms=0, converged=False,
        )
        assert ok is True
    body = (project_dir / "MEMORY.md").read_text()
    for complexity in ("minimal", "low", "medium", "high", "max", "unknown"):
        assert f"q-{complexity}" in body


def test_maybe_writeback_to_memory_md_converged_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    ok = _sh._maybe_writeback_to_memory_md(
        {"user_prompt_redacted": "x", "classification_complexity": "high"},
        pre_count=2, post_count=2,
        duration_ms=100, converged=True,
    )
    assert ok is True
    body = (project_dir / "MEMORY.md").read_text()
    assert "quality=0.85" in body  # converged path → 0.85


def test_slm_available_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_sh.shutil, "which", lambda name: "/usr/bin/slm")
    assert _sh._slm_available_for_writeback() is True
    monkeypatch.setattr(_sh.shutil, "which", lambda name: None)
    assert _sh._slm_available_for_writeback() is False


def test_on_stop_unknown_event_type_in_metrics(
    isolated_state: Path, stdin_event
) -> None:
    """An event row with an unrecognized event_type does not crash; falls
    through Pre/Post elif chain (else branch — neither pre nor post)."""
    _seed(isolated_state, "weird")
    s = _state.StateStore(isolated_state)
    # Inject a non-Pre/Post event type via record_event:
    s.record_event("weird", "CustomEvent", turn_id=1, tool_name="T", payload={})
    stdin_event({"session_id": "weird"})
    rc = _sh.on_stop()
    assert rc == 0
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        row = conn.execute(
            "SELECT finalize_report_json FROM outcomes WHERE session_id=?",
            ("weird",),
        ).fetchone()
    report = json.loads(row[0])
    # Custom event neither incremented pre nor post.
    assert report["tool_calls"] == 0
    assert report["tool_results"] == 0


# ---------------------------------------------------------------------------
# _infer_stop_reason — covers the previously-71% "unknown" attribution gap
# ---------------------------------------------------------------------------


def test_infer_stop_reason_trusts_host_supplied_value() -> None:
    """Host wins: a non-empty event.stop_reason is recorded verbatim."""
    assert _sh._infer_stop_reason(3, 3, "user_interrupt") == "user_interrupt"
    assert _sh._infer_stop_reason(0, 0, "rate_limited") == "rate_limited"


def test_infer_stop_reason_ignores_blank_host_value() -> None:
    """An empty / whitespace host value falls through to inference."""
    assert _sh._infer_stop_reason(2, 2, "") == "complete"
    assert _sh._infer_stop_reason(2, 2, "   ") == "complete"
    assert _sh._infer_stop_reason(2, 2, None) == "complete"


def test_infer_stop_reason_in_flight_when_pre_exceeds_post() -> None:
    """pre > post = tool calls started but not finished."""
    assert _sh._infer_stop_reason(5, 3, None) == "in_flight"
    assert _sh._infer_stop_reason(1, 0, None) == "in_flight"


def test_on_stop_writes_real_tokens_from_transcript(
    isolated_state: Path,
    stdin_event,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stop hook reads transcript JSONL and writes the delta to tokens_used."""
    from agent_amplifier.adapters.claude_code.transcript import transcript_path

    _seed(isolated_state, "sx")
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(tmp_path / "projects"))
    # Single assistant message — 100 tokens total.
    p = transcript_path("sx", Path("/cwd"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        '{"type": "assistant", "message": {"usage": {"input_tokens": 30, '
        '"output_tokens": 70}}}\n'
    )
    stdin_event({"session_id": "sx", "cwd": "/cwd", "stop_reason": "complete"})
    rc = _sh.on_stop()
    assert rc == 0
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        row = conn.execute(
            "SELECT tokens_used FROM outcomes WHERE session_id=? AND turn_id=?",
            ("sx", 1),
        ).fetchone()
    assert row[0] == 100


def test_on_stop_tokens_delta_subtracts_prior_outcomes(
    isolated_state: Path,
    stdin_event,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Second turn writes delta vs SUM(prior outcomes for same session)."""
    from agent_amplifier.adapters.claude_code.transcript import transcript_path

    s = _state.StateStore(isolated_state)
    s.upsert_session("sy", "/cwd")
    # Seed a prior outcome row (turn 1) with 200 tokens already accounted.
    s.record_envelope(
        "sy", 1,
        user_prompt_redacted="r",
        classification_complexity="low",
        classification_domain="d",
        thinking_trigger=None, persona=None, phase="P",
        envelope_text="env",
    )
    s.write_outcome(
        "sy", 1,
        iterations_completed=1, converged=True,
        tokens_used=200,
    )
    # New envelope for turn 2.
    s.record_envelope(
        "sy", 2,
        user_prompt_redacted="r",
        classification_complexity="low",
        classification_domain="d",
        thinking_trigger=None, persona=None, phase="P",
        envelope_text="env",
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(tmp_path / "projects"))
    # Transcript total = 500. Delta should be 500 - 200 = 300.
    p = transcript_path("sy", Path("/cwd"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        '{"type": "assistant", "message": {"usage": {"input_tokens": 500}}}\n'
    )
    stdin_event({"session_id": "sy", "cwd": "/cwd", "stop_reason": "complete"})
    assert _sh.on_stop() == 0
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        row = conn.execute(
            "SELECT tokens_used FROM outcomes WHERE session_id=? AND turn_id=?",
            ("sy", 2),
        ).fetchone()
    assert row[0] == 300


def test_on_stop_tokens_zero_when_transcript_missing(
    isolated_state: Path,
    stdin_event,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail-open: no transcript file → tokens_used = 0, hook still succeeds."""
    _seed(isolated_state, "sz")
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(tmp_path / "no-such-dir"))
    stdin_event({"session_id": "sz", "cwd": "/cwd", "stop_reason": "complete"})
    assert _sh.on_stop() == 0
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        row = conn.execute(
            "SELECT tokens_used FROM outcomes WHERE session_id=? AND turn_id=?",
            ("sz", 1),
        ).fetchone()
    assert row[0] == 0


def test_on_stop_tokens_clamps_at_zero_when_transcript_below_prior(
    isolated_state: Path,
    stdin_event,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Defensive: transcript total < prior outcome sum → delta clamps to 0."""
    from agent_amplifier.adapters.claude_code.transcript import transcript_path

    s = _state.StateStore(isolated_state)
    s.upsert_session("sw", "/cwd")
    s.record_envelope(
        "sw", 1,
        user_prompt_redacted="r",
        classification_complexity="low",
        classification_domain="d",
        thinking_trigger=None, persona=None, phase="P",
        envelope_text="env",
    )
    s.write_outcome("sw", 1, iterations_completed=1, converged=True, tokens_used=1000)
    s.record_envelope(
        "sw", 2,
        user_prompt_redacted="r",
        classification_complexity="low",
        classification_domain="d",
        thinking_trigger=None, persona=None, phase="P",
        envelope_text="env",
    )
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(tmp_path / "projects"))
    p = transcript_path("sw", Path("/cwd"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        '{"type": "assistant", "message": {"usage": {"input_tokens": 50}}}\n'
    )
    stdin_event({"session_id": "sw", "cwd": "/cwd", "stop_reason": "complete"})
    assert _sh.on_stop() == 0
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        row = conn.execute(
            "SELECT tokens_used FROM outcomes WHERE session_id=? AND turn_id=?",
            ("sw", 2),
        ).fetchone()
    assert row[0] == 0


def test_compute_turn_tokens_fail_open_when_transcript_raises(
    isolated_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If tokens_for_session itself raises, stop_hook returns 0, not propagates."""
    from agent_amplifier.adapters.claude_code import transcript as _t

    s = _state.StateStore(isolated_state)
    s.upsert_session("se", "/cwd")

    def _boom(*_a: object, **_k: object) -> int:
        raise RuntimeError("transcript module exploded")

    monkeypatch.setattr(_t, "tokens_for_session", _boom)
    assert _sh._compute_turn_tokens(s, "se", 1, "/cwd") == 0


def test_compute_turn_tokens_fail_open_when_prior_raises(
    isolated_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If prior_tokens_for_session raises, prior defaults to 0."""
    from agent_amplifier.adapters.claude_code import transcript as _t

    s = _state.StateStore(isolated_state)
    s.upsert_session("sf", "/cwd")

    def _ok(*_a: object, **_k: object) -> int:
        return 123

    def _boom(self: object, *_a: object, **_k: object) -> int:
        raise RuntimeError("prior query exploded")

    monkeypatch.setattr(_t, "tokens_for_session", _ok)
    monkeypatch.setattr(
        _state.StateStore, "prior_tokens_for_session", _boom
    )
    assert _sh._compute_turn_tokens(s, "sf", 1, "/cwd") == 123


def test_prior_tokens_for_session_helper(isolated_state: Path) -> None:
    """state.prior_tokens_for_session returns SUM of prior outcomes only."""
    s = _state.StateStore(isolated_state)
    s.upsert_session("st", "/cwd")
    for turn_id, tokens in [(1, 100), (2, 200), (3, 50)]:
        s.record_envelope(
            "st", turn_id,
            user_prompt_redacted="r",
            classification_complexity="low",
            classification_domain="d",
            thinking_trigger=None, persona=None, phase="P",
            envelope_text="env",
        )
        s.write_outcome(
            "st", turn_id,
            iterations_completed=1, converged=True, tokens_used=tokens,
        )
    # Sum of turns < 3 = 100 + 200 = 300
    assert s.prior_tokens_for_session("st", 3) == 300
    # Sum of turns < 1 (none) = 0
    assert s.prior_tokens_for_session("st", 1) == 0
    # Different session = 0
    assert s.prior_tokens_for_session("other", 99) == 0


def test_infer_stop_reason_empty_when_no_tool_use() -> None:
    """0 pre + 0 post = text-only response (refusal or short answer)."""
    assert _sh._infer_stop_reason(0, 0, None) == "empty"


def test_infer_stop_reason_complete_default_path() -> None:
    """pre == post >= 1 = happy-path completion."""
    assert _sh._infer_stop_reason(1, 1, None) == "complete"
    assert _sh._infer_stop_reason(20, 20, None) == "complete"


def test_infer_stop_reason_ignores_non_string_host_value() -> None:
    """A host value that is not a string falls through to inference."""
    assert _sh._infer_stop_reason(2, 2, 42) == "complete"  # type: ignore[arg-type]
    assert _sh._infer_stop_reason(0, 0, ["bogus"]) == "empty"  # type: ignore[arg-type]


def test_on_stop_writes_inferred_complete_when_event_omits_reason(
    isolated_state: Path, stdin_event
) -> None:
    """End-to-end: Claude Code's Stop event omits stop_reason; outcome row
    records the inferred ``"complete"`` instead of the legacy ``"unknown"``."""
    _seed(isolated_state, "infer-complete")
    s = _state.StateStore(isolated_state)
    s.record_event(
        "infer-complete", "PreToolUse", turn_id=1, tool_name="T", payload={}
    )
    s.record_event(
        "infer-complete", "PostToolUse", turn_id=1, tool_name="T", payload={}
    )
    stdin_event({"session_id": "infer-complete"})  # no stop_reason key
    assert _sh.on_stop() == 0
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        row = conn.execute(
            "SELECT finalize_report_json FROM outcomes WHERE session_id=?",
            ("infer-complete",),
        ).fetchone()
    report = json.loads(row[0])
    assert report["stop_reason"] == "complete"


def test_on_stop_writes_inferred_in_flight_when_pre_exceeds_post(
    isolated_state: Path, stdin_event
) -> None:
    """End-to-end: tools started but not completed = inferred ``"in_flight"``."""
    _seed(isolated_state, "infer-inflight")
    s = _state.StateStore(isolated_state)
    s.record_event(
        "infer-inflight", "PreToolUse", turn_id=1, tool_name="T", payload={}
    )
    s.record_event(
        "infer-inflight", "PreToolUse", turn_id=1, tool_name="T", payload={}
    )
    s.record_event(
        "infer-inflight", "PostToolUse", turn_id=1, tool_name="T", payload={}
    )
    stdin_event({"session_id": "infer-inflight"})
    assert _sh.on_stop() == 0
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        row = conn.execute(
            "SELECT finalize_report_json FROM outcomes WHERE session_id=?",
            ("infer-inflight",),
        ).fetchone()
    report = json.loads(row[0])
    assert report["stop_reason"] == "in_flight"


def test_on_stop_writes_inferred_empty_when_no_tool_events(
    isolated_state: Path, stdin_event
) -> None:
    """End-to-end: model produced text only, no tool calls = ``"empty"``."""
    _seed(isolated_state, "infer-empty")
    stdin_event({"session_id": "infer-empty"})
    assert _sh.on_stop() == 0
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        row = conn.execute(
            "SELECT finalize_report_json FROM outcomes WHERE session_id=?",
            ("infer-empty",),
        ).fetchone()
    report = json.loads(row[0])
    assert report["stop_reason"] == "empty"


def test_on_stop_passes_host_stop_reason_through(
    isolated_state: Path, stdin_event
) -> None:
    """End-to-end: a host-supplied event.stop_reason wins over inference."""
    _seed(isolated_state, "host-wins")
    stdin_event(
        {"session_id": "host-wins", "stop_reason": "user_interrupt"}
    )
    assert _sh.on_stop() == 0
    with contextlib.closing(sqlite3.connect(str(isolated_state))) as conn:
        row = conn.execute(
            "SELECT finalize_report_json FROM outcomes WHERE session_id=?",
            ("host-wins",),
        ).fetchone()
    report = json.loads(row[0])
    assert report["stop_reason"] == "user_interrupt"
