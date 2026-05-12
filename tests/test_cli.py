# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.cli`` (, )."""

from __future__ import annotations

import os
import pathlib
import unittest.mock

import pytest


def test_help_snapshot(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` MUST mention all 7 documented subcommands and 'agent-amp'."""
    from agent_amplifier.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "agent-amp" in out
    for cmd in (
        "install", "uninstall", "list", "status", "doctor", "config",
        "bench", "report", "dashboard", "demo",
    ):
        assert cmd in out, f"missing subcommand in --help: {cmd}"


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    from agent_amplifier import __version__
    from agent_amplifier.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_list_command_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    from agent_amplifier.cli import main

    rc = main(["list"])
    assert rc == 0


def test_status_command_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    from agent_amplifier.cli import main

    rc = main(["status"])
    assert rc == 0


def test_doctor_command_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    from agent_amplifier.cli import main

    rc = main(["doctor"])
    assert rc == 0
    out = capsys.readouterr().out
    # Doctor should print Python + OS diagnostics
    assert "Python" in out or "python" in out


def test_doctor_enumerates_all_six_adapters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MED-2 (QA-M01).

    Doctor MUST surface each bundled adapter's framework_name + display name
    so users immediately see which hosts the amplifier will bind to on this
    machine — not just whether the optional SLM binary is present.
    """
    from agent_amplifier.cli import main

    rc = main(["doctor"])
    assert rc == 0
    out = capsys.readouterr().out
    # The 6 framework slugs are the canonical machine-readable identifiers.
    for slug in (
        "claude_code",
        "cursor",
        "github_copilot",
        "langgraph",
        "crewai",
        "agentscope",
    ):
        assert slug in out, f"doctor missing adapter slug: {slug}"
    # Display names too — humans read this, not just slugs.
    for label in ("Claude Code", "Cursor", "GitHub Copilot",
                  "LangGraph", "CrewAI", "AgentScope"):
        assert label in out, f"doctor missing adapter label: {label}"
    # Each line must say DETECTED or missing — proves we ran detect().
    assert "DETECTED" in out or "missing" in out
    # The SLM hint should be demoted under the optional section, not the
    # headline (regression check vs the V2.0 doctor that led with it).
    assert "third-party memory providers" in out


def test_config_show_returns_zero() -> None:
    from agent_amplifier.cli import main

    rc = main(["config", "show"])
    assert rc == 0


def test_unknown_command_returns_nonzero() -> None:
    from agent_amplifier.cli import main

    # argparse exits with 2 on unknown subcommand
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code != 0


def test_install_unknown_target_returns_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    rc = main(["install", "nonexistent_framework_xyz"])
    assert rc == 2  # unknown target


def test_install_auto_with_no_adapters_is_safe() -> None:
    from agent_amplifier.cli import main

    rc = main(["install", "--auto"])
    # No bundled adapters yet; --auto should print and return 0.
    assert rc == 0


def test_no_command_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "USAGE" in out or "usage:" in out


# ---------------------------------------------------------------------------
# — install command honesty (no persistent ⇒ "ready")
# ---------------------------------------------------------------------------


def test_cli_install_message_for_marker_only_adapter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """File-based adapters (INSTALL_PERSISTENT=False) print 'ready: <name>',
    NOT 'installed: <name>'.  This tells the user nothing persistent
    happened on disk.

    day-0: ClaudeCodeAdapter flipped to INSTALL_PERSISTENT=True
    (it now persistently writes hook entries to ~/.claude/settings.json),
    so we exercise the marker-only path via CursorAdapter, which still
    inherits the default INSTALL_PERSISTENT=False.  We also force the
    claude_code adapter to NOT be detected so its install side-effect
    (which would touch settings.json) is skipped during this test.
    """
    from agent_amplifier import cli as _cli
    # Force claude_code OFF (persistent path; would touch settings.json)
    # and force cursor ON (marker-only path under test).
    monkeypatch.setattr(
        "agent_amplifier.adapters.claude_code.ClaudeCodeAdapter.detect",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "agent_amplifier.adapters.cursor.CursorAdapter.detect",
        classmethod(lambda cls: True),
    )
    rc = _cli.main(["install", "--auto"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ready:" in out
    assert "no persistent install" in out
    # Critically, we did NOT lie about cursor:
    assert "installed: cursor" not in out


# ---------------------------------------------------------------------------
# — config set returns non-zero when not implemented
# ---------------------------------------------------------------------------


def test_cli_config_set_returns_non_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``agent-amp config set foo=bar`` MUST return non-zero so wrapper
    scripts using ``set -e`` do not silently believe the config was changed.
    """
    from agent_amplifier import cli as _cli

    rc = _cli.main(["config", "set", "foo=bar"])
    err = capsys.readouterr().err
    assert rc != 0
    assert "not implemented" in err


def test_cli_config_set_missing_kv_returns_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bad usage stays at exit code 2 (argparse-style)."""
    from agent_amplifier import cli as _cli

    rc = _cli.main(["config", "set"])
    assert rc == 2


def test_report_command_no_db_returns_one(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``agent-amp report`` returns 1 when state.db is missing."""
    from agent_amplifier import cli as _cli
    from agent_amplifier.adapters.claude_code import state as _state

    monkeypatch.setattr(
        _state, "_DEFAULT_STATE_DIR", tmp_path / "no-amp-yet"
    )
    rc = _cli.main(["report"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err


def test_install_accepts_hyphenated_alias(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """``agent-amp install claude-code`` (hyphen, README-style) MUST resolve
    to the ``claude_code`` framework rather than returning unknown-target.
    Same for github-copilot."""
    from agent_amplifier import cli as _cli
    from agent_amplifier.adapters.claude_code import (
        installer as _ins,
    )
    from agent_amplifier.adapters.claude_code import (
        state as _state,
    )

    # Redirect both installer + state to tmp_path to avoid touching real
    # ~/.claude on the developer's machine.
    monkeypatch.setattr(
        _ins, "_DEFAULT_SETTINGS_PATH", tmp_path / "settings.json"
    )
    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path / "amp")

    rc = _cli.main(["install", "claude-code"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "claude_code" in out


def test_uninstall_accepts_hyphenated_alias(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Symmetric to install: ``agent-amp uninstall claude-code`` (hyphen)
    MUST resolve to the ``claude_code`` framework. We assert the alias is
    accepted (no ``unknown target`` error) — the adapter's per-instance
    install-flag bookkeeping is orthogonal to the alias resolution this
    test guards."""
    from agent_amplifier import cli as _cli
    from agent_amplifier.adapters.claude_code import (
        installer as _ins,
    )
    from agent_amplifier.adapters.claude_code import (
        state as _state,
    )

    monkeypatch.setattr(
        _ins, "_DEFAULT_SETTINGS_PATH", tmp_path / "settings.json"
    )
    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", tmp_path / "amp")

    _cli.main(["uninstall", "claude-code"])
    err = capsys.readouterr().err
    # The fix's contract: ``claude-code`` resolves to ``claude_code``;
    # we MUST NOT see the ``unknown target: claude-code`` branch.
    assert "unknown target" not in err


def test_demo_command_renders_full_preview(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``agent-amp demo <prompt>`` renders both halves + delta."""
    from agent_amplifier import cli as _cli

    rc = _cli.main(["demo", "Refactor the auth middleware to use JWT"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Baseline:" in out
    assert "Amplified:" in out
    assert "Delta:" in out


def test_demo_command_empty_prompt_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``agent-amp demo ""`` → 1 with stderr hint."""
    from agent_amplifier import cli as _cli

    rc = _cli.main(["demo", ""])
    assert rc == 1
    err = capsys.readouterr().err
    assert "must be non-empty" in err


def test_bench_command_with_prompt_default_shows_both_halves(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``bench --prompt P`` (no flags) defaults to both halves so the
    launch-GIF capture command stays simple."""
    from agent_amplifier import cli as _cli

    rc = _cli.main(["bench", "--prompt", "Refactor X"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Baseline:" in out
    assert "Amplified:" in out
    assert "Delta:" in out


def test_bench_command_with_prompt_baseline_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``bench --prompt P --baseline`` shows ONLY the baseline."""
    from agent_amplifier import cli as _cli

    rc = _cli.main(["bench", "--prompt", "Refactor X", "--baseline"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Baseline:" in out
    assert "Amplified:" not in out
    assert "Delta:" not in out


def test_bench_command_with_prompt_vs_amplified_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``bench --prompt P --vs-amplified`` shows ONLY the amplified envelope."""
    from agent_amplifier import cli as _cli

    rc = _cli.main(
        ["bench", "--prompt", "Refactor X", "--vs-amplified"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Baseline:" not in out
    assert "Amplified:" in out
    assert "Delta:" not in out


def test_report_command_with_data_returns_zero(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``agent-amp report --last 5`` renders dashboard for a populated DB."""
    from agent_amplifier import cli as _cli
    from agent_amplifier.adapters.claude_code import state as _state
    from agent_amplifier.adapters.claude_code.state import StateStore

    redirect = tmp_path / "amp"
    monkeypatch.setattr(_state, "_DEFAULT_STATE_DIR", redirect)
    db = redirect / _state._STATE_DB_FILENAME
    s = StateStore(db)
    s.upsert_session("sess-cli", "/proj")
    s.next_turn_id("sess-cli")
    s.record_envelope(
        "sess-cli", 1,
        user_prompt_redacted="x",
        classification_complexity="medium",
        classification_domain="api",
        thinking_trigger=None,
        persona=None,
        phase="EXPLORE",
        envelope_text="<env>",
    )
    rc = _cli.main(["report", "--last", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Last 5 turns" in out


# ---------------------------------------------------------------------------
# status --watch
# ---------------------------------------------------------------------------


def test_status_watch_flag_accepted(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent_amplifier.adapters.claude_code import state as _state
    from agent_amplifier.cli import main

    db = tmp_path / "state.db"
    store = _state.StateStore(db)
    store.upsert_session("s", "/p")
    with unittest.mock.patch.object(
        _state, "_DEFAULT_STATE_DIR", str(tmp_path)
    ):
        rc = main(["status", "--watch", "--once"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Tokens:" in out


def test_status_watch_renders_token_bar(capsys: pytest.CaptureFixture[str]) -> None:
    from agent_amplifier.cli import _render_token_bar

    line = _render_token_bar(used=50000, limit=250000)
    assert "50" in line or "50k" in line.lower() or "50,000" in line
    assert "250" in line or "250k" in line.lower()


def test_status_watch_renders_zero(capsys: pytest.CaptureFixture[str]) -> None:
    from agent_amplifier.cli import _render_token_bar

    line = _render_token_bar(used=0, limit=250000)
    assert "0" in line


def test_status_watch_renders_over_limit(capsys: pytest.CaptureFixture[str]) -> None:
    from agent_amplifier.cli import _render_token_bar

    line = _render_token_bar(used=300000, limit=250000)
    assert "300" in line or "OVER" in line.upper() or "!" in line


def test_status_watch_renders_no_limit(capsys: pytest.CaptureFixture[str]) -> None:
    from agent_amplifier.cli import _render_token_bar

    line = _render_token_bar(used=5000, limit=0)
    assert "no limit" in line.lower()


def test_status_watch_custom_budget_env(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent_amplifier.adapters.claude_code import state as _state
    from agent_amplifier.cli import main

    db = tmp_path / "state.db"
    store = _state.StateStore(db)
    store.upsert_session("s", "/p")
    with (
        unittest.mock.patch.object(_state, "_DEFAULT_STATE_DIR", str(tmp_path)),
        unittest.mock.patch.dict(os.environ, {"AGENT_AMP_WATCH_BUDGET": "500000"}),
    ):
        rc = main(["status", "--watch", "--once"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "500,000" in out


def test_status_watch_invalid_budget_env(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent_amplifier.adapters.claude_code import state as _state
    from agent_amplifier.cli import main

    db = tmp_path / "state.db"
    store = _state.StateStore(db)
    store.upsert_session("s", "/p")
    with (
        unittest.mock.patch.object(_state, "_DEFAULT_STATE_DIR", str(tmp_path)),
        unittest.mock.patch.dict(os.environ, {"AGENT_AMP_WATCH_BUDGET": "not_a_number"}),
    ):
        rc = main(["status", "--watch", "--once"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "250,000" in out


def test_status_watch_no_db(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent_amplifier.adapters.claude_code import state as _state
    from agent_amplifier.cli import main

    with unittest.mock.patch.object(
        _state, "_DEFAULT_STATE_DIR", str(tmp_path / "nonexistent")
    ):
        rc = main(["status", "--watch", "--once"])
    assert rc == 1
