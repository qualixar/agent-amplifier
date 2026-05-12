# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.adapters.slm.memory.SLMAdapter``.

Coverage targets: 100% line + 100% branch on slm/memory.py.

The ``slm`` CLI is mocked via subprocess.run monkeypatching — no real SLM
binary is required to run these tests.
"""
from __future__ import annotations

import subprocess
from typing import Any

import pytest

from agent_amplifier.adapters.slm import memory as _slm
from agent_amplifier.adapters.slm.memory import SLMAdapter
from agent_amplifier.types import EffortLevel, Outcome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProc:
    def __init__(
        self, stdout: str = "", stderr: str = "", returncode: int = 0
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture
def slm_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_slm.shutil, "which", lambda name: "/usr/local/bin/slm")


@pytest.fixture
def slm_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_slm.shutil, "which", lambda name: None)


@pytest.fixture
def mock_slm_recall(monkeypatch: pytest.MonkeyPatch):
    """Patch subprocess.run for the SLMAdapter module."""
    captured: dict[str, Any] = {"calls": []}

    def factory(stdout: str = "", stderr: str = "", returncode: int = 0):
        def fake_run(cmd, **kw):
            captured["calls"].append({"cmd": list(cmd), "kw": kw})
            return FakeProc(stdout=stdout, stderr=stderr, returncode=returncode)
        monkeypatch.setattr(_slm.subprocess, "run", fake_run)
        return captured

    return factory


# ---------------------------------------------------------------------------
# detect_slm
# ---------------------------------------------------------------------------


def test_detect_slm_when_present(slm_present: None) -> None:
    assert _slm.detect_slm() is True


def test_detect_slm_when_absent(slm_absent: None) -> None:
    assert _slm.detect_slm() is False


def test_adapter_detect_classmethod(slm_present: None) -> None:
    assert SLMAdapter.detect() is True


# ---------------------------------------------------------------------------
# install / uninstall (no-op markers)
# ---------------------------------------------------------------------------


def test_install_marks_installed() -> None:
    a = SLMAdapter(kernel=None)
    assert not a.is_installed()
    a.install()
    assert a.is_installed()


def test_uninstall_unmarks() -> None:
    a = SLMAdapter(kernel=None)
    a.install()
    a.uninstall()
    assert not a.is_installed()


def test_install_persistent_is_false() -> None:
    assert SLMAdapter.INSTALL_PERSISTENT is False


# ---------------------------------------------------------------------------
# default_memory_recall
# ---------------------------------------------------------------------------


_SAMPLE_CONTEXT = """\
# SLM Session Context — default

## Core Memory
[active_decisions] Some decision about Qualixar architecture.

## Recent Context (7 days)
- The Qualixar Cascade Pipeline has a quality gate.
- Qualixar arXiv papers covered 7 products.

## Recent Sessions
- Session handoff for Qualixar v1.0.0 launch.
"""


def test_recall_when_slm_absent_returns_empty(slm_absent: None) -> None:
    a = SLMAdapter(kernel=None)
    assert a.default_memory_recall("Qualixar") == []


def test_recall_happy_path(
    slm_present: None, mock_slm_recall
) -> None:
    captured = mock_slm_recall(stdout=_SAMPLE_CONTEXT)
    a = SLMAdapter(kernel=None)
    out = a.default_memory_recall("Qualixar", limit=2)
    assert len(out) == 2
    for p in out:
        assert "qualixar" in p.text.lower()
        assert p.source == "superlocalmemory:session-context"
    # Verify CLI call shape: ['slm', 'session-context', 'Qualixar']
    assert captured["calls"][0]["cmd"] == ["slm", "session-context", "Qualixar"]


def test_recall_empty_query_drops_positional(
    slm_present: None, mock_slm_recall
) -> None:
    captured = mock_slm_recall(stdout=_SAMPLE_CONTEXT)
    a = SLMAdapter(kernel=None)
    a.default_memory_recall("", limit=1)
    assert captured["calls"][0]["cmd"] == ["slm", "session-context"]


def test_recall_limit_truncates(
    slm_present: None, mock_slm_recall
) -> None:
    mock_slm_recall(stdout=_SAMPLE_CONTEXT)
    a = SLMAdapter(kernel=None)
    out = a.default_memory_recall("Qualixar", limit=1)
    assert len(out) == 1


def test_recall_chunk_per_chunk_byte_cap(
    slm_present: None, mock_slm_recall
) -> None:
    big_chunk = "## Big\n" + "A" * 10_000 + " Qualixar\n"
    mock_slm_recall(stdout=big_chunk)
    a = SLMAdapter(kernel=None)
    out = a.default_memory_recall("Qualixar")
    assert all(len(p.text) <= 4096 for p in out)


def test_recall_handles_timeout(
    slm_present: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **kw: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["slm"], timeout=5)
    monkeypatch.setattr(_slm.subprocess, "run", boom)
    a = SLMAdapter(kernel=None)
    assert a.default_memory_recall("x") == []


def test_recall_handles_oserror(
    slm_present: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **kw: object) -> object:
        raise OSError("permission denied")
    monkeypatch.setattr(_slm.subprocess, "run", boom)
    a = SLMAdapter(kernel=None)
    assert a.default_memory_recall("x") == []


def test_recall_non_zero_exit(
    slm_present: None, mock_slm_recall
) -> None:
    mock_slm_recall(stderr="slm: db locked", returncode=2)
    a = SLMAdapter(kernel=None)
    assert a.default_memory_recall("x") == []


def test_recall_empty_stdout(
    slm_present: None, mock_slm_recall
) -> None:
    mock_slm_recall(stdout="")
    a = SLMAdapter(kernel=None)
    assert a.default_memory_recall("x") == []


def test_rank_chunks_no_h2(slm_present: None) -> None:
    out = _slm._rank_chunks("just one block of text Qualixar", "qualixar")
    assert len(out) == 1
    assert "Qualixar" in out[0]


def test_rank_chunks_empty_text() -> None:
    assert _slm._rank_chunks("", "x") == []


def test_rank_chunks_no_query_keeps_all() -> None:
    text = "## A\nbody1\n\n## B\nbody2"
    chunks = _slm._rank_chunks(text, "")
    assert len(chunks) == 2


# ---------------------------------------------------------------------------
# default_memory_remember (Mode 3 via adapter; not the stop_hook path)
# ---------------------------------------------------------------------------


def _outcome(query: str = "build feature X") -> Outcome:
    return Outcome(
        query=query,
        effort=EffortLevel.MEDIUM,
        iterations=2,
        quality=0.85,
        converged=True,
        tokens_used=1234,
    )


def test_remember_when_slm_absent(slm_absent: None) -> None:
    a = SLMAdapter(kernel=None)
    a.default_memory_remember(_outcome())  # no-op, no raise


def test_remember_happy_path(
    slm_present: None, mock_slm_recall
) -> None:
    captured = mock_slm_recall(stdout="ok", returncode=0)
    a = SLMAdapter(kernel=None)
    a.default_memory_remember(_outcome())
    assert len(captured["calls"]) == 1
    assert captured["calls"][0]["cmd"][:2] == ["slm", "remember"]


def test_remember_empty_content_skips(
    slm_present: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the formatted summary is empty (defensive), don't shell out."""
    monkeypatch.setattr(
        SLMAdapter, "_format_remember_content", staticmethod(lambda o: "")
    )
    called = []
    monkeypatch.setattr(
        _slm.subprocess, "run",
        lambda *a, **kw: called.append(1) or FakeProc(),
    )
    a = SLMAdapter(kernel=None)
    a.default_memory_remember(_outcome())
    assert called == []


def test_remember_handles_timeout(
    slm_present: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **kw: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["slm"], timeout=10)
    monkeypatch.setattr(_slm.subprocess, "run", boom)
    a = SLMAdapter(kernel=None)
    a.default_memory_remember(_outcome())  # silent


def test_remember_handles_oserror(
    slm_present: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **kw: object) -> object:
        raise OSError("nope")
    monkeypatch.setattr(_slm.subprocess, "run", boom)
    a = SLMAdapter(kernel=None)
    a.default_memory_remember(_outcome())  # silent


def test_remember_non_zero_exit(
    slm_present: None, mock_slm_recall
) -> None:
    mock_slm_recall(stderr="slm: down", returncode=1)
    a = SLMAdapter(kernel=None)
    a.default_memory_remember(_outcome())  # silent


def test_format_summary_includes_key_fields() -> None:
    s = SLMAdapter._format_remember_content(_outcome("refactor auth"))
    assert "[amp] turn outcome" in s
    assert "effort=" in s
    assert "quality=0.85" in s
    assert "converged=yes" in s
    assert "iters=2" in s
    assert "tokens=1234" in s


def test_format_summary_truncates_long_query() -> None:
    long_q = "X" * 500
    s = SLMAdapter._format_remember_content(_outcome(long_q))
    # Query field is truncated to 120 chars before redact
    assert len(s) < 1000
