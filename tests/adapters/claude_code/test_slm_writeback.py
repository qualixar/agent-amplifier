# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.adapters.claude_code.slm_writeback``.

Coverage targets: 100% line + 100% branch on slm_writeback.py.

Mocks the ``slm`` CLI via ``subprocess.run`` monkeypatching — no real SLM
binary needed.
"""
from __future__ import annotations

import subprocess
from typing import Any

import pytest

from agent_amplifier.adapters.claude_code import slm_writeback as _wb


class FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _envelope() -> dict[str, Any]:
    return {
        "user_prompt_redacted": "[REDACTED] refactor auth module",
        "classification_complexity": "high",
        "classification_domain": "architecture",
        "thinking_trigger": "ultrathink",
        "persona": "Senior security engineer, paranoid about OWASP",
        "phase": "EXPLORE",
    }


# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------


def test_slm_unavailable_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_wb.shutil, "which", lambda name: None)
    ok = _wb.write_outcome_to_slm(
        "s", 1,
        envelope=_envelope(),
        tool_calls=2, tool_results=2,
        duration_ms=300, converged=True,
    )
    assert ok is False


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_writeback_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    def fake_run(cmd, **kw):
        captured.append({"cmd": list(cmd), "kw": kw})
        return FakeProc(returncode=0)

    monkeypatch.setattr(_wb.shutil, "which", lambda name: "/usr/local/bin/slm")
    monkeypatch.setattr(_wb.subprocess, "run", fake_run)

    ok = _wb.write_outcome_to_slm(
        "sess-X", 7,
        envelope=_envelope(),
        tool_calls=4, tool_results=4,
        duration_ms=520, converged=True,
    )
    assert ok is True
    assert len(captured) == 1
    cmd = captured[0]["cmd"]
    assert cmd[:2] == ["slm", "remember"]
    summary = cmd[2]
    assert "[amp] turn-outcome" in summary
    assert "turn=7" in summary
    assert "complexity=high" in summary
    assert "domain=architecture" in summary
    assert "tools=4/4" in summary
    assert "converged=yes" in summary
    assert "duration_ms=520" in summary


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_writeback_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_wb.shutil, "which", lambda name: "/usr/local/bin/slm")

    def boom(*a: object, **kw: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["slm"], timeout=10)

    monkeypatch.setattr(_wb.subprocess, "run", boom)
    ok = _wb.write_outcome_to_slm(
        "s", 1, envelope=_envelope(),
        tool_calls=0, tool_results=0, duration_ms=0, converged=False,
    )
    assert ok is False


def test_writeback_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_wb.shutil, "which", lambda name: "/usr/local/bin/slm")

    def boom(*a: object, **kw: object) -> object:
        raise OSError("permission")

    monkeypatch.setattr(_wb.subprocess, "run", boom)
    ok = _wb.write_outcome_to_slm(
        "s", 1, envelope=_envelope(),
        tool_calls=0, tool_results=0, duration_ms=0, converged=False,
    )
    assert ok is False


def test_writeback_non_zero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_wb.shutil, "which", lambda name: "/usr/local/bin/slm")
    monkeypatch.setattr(
        _wb.subprocess, "run",
        lambda *a, **kw: FakeProc(stderr="slm: down", returncode=3),
    )
    ok = _wb.write_outcome_to_slm(
        "s", 1, envelope=_envelope(),
        tool_calls=0, tool_results=0, duration_ms=0, converged=False,
    )
    assert ok is False


def test_writeback_with_unconverged_and_empty_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_wb.shutil, "which", lambda name: "/usr/local/bin/slm")
    captured: list[Any] = []
    monkeypatch.setattr(
        _wb.subprocess, "run",
        lambda *a, **kw: captured.append(a[0]) or FakeProc(returncode=0),
    )
    # Envelope missing keys → should fall back to defaults, not crash.
    ok = _wb.write_outcome_to_slm(
        "s", 1, envelope={},
        tool_calls=1, tool_results=0, duration_ms=42, converged=False,
    )
    assert ok is True
    summary = captured[0][2]
    assert "complexity=unknown" in summary
    assert "domain=general" in summary
    assert "persona=default" in summary
    assert "phase=unknown" in summary
    assert "converged=no" in summary


def test_format_summary_truncates_long_persona_and_prompt() -> None:
    env = _envelope()
    env["persona"] = "X" * 500
    env["user_prompt_redacted"] = "Y" * 500
    s = _wb._format_summary(
        "sid", 1, env,
        tool_calls=0, tool_results=0,
        duration_ms=0, converged=True,
    )
    # persona caps at 80
    assert "persona=" in s and "X" * 80 in s and "X" * 81 not in s
    # prompt redacted to 160 chars before redact applies
    # (just assert the summary stays reasonable size)
    assert len(s) < 800
