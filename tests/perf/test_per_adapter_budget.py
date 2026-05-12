# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Per-adapter ``default_memory_recall`` P99 budgets (H7 / ).

Closes audit finding ``H7`` ("Per-adapter perf budgets unimplemented") from

Each adapter is exercised through its public ``default_memory_recall`` entry
point under representative shapes:

* **Host adapters** (Claude Code, Cursor, GitHub Copilot) read real files
  from a ``tmp_path`` fixture. They include filesystem latency in the budget.
* **Framework adapters** (LangGraph, CrewAI, AgentScope) operate on mock
  user-supplied objects whose recall methods return immediately. The budget
  is therefore much tighter — anything beyond a few hundred microseconds
  signals that the adapter is doing real work it shouldn't (allocations,
  re-imports, stringification of large objects).

Budgets are deliberately generous for V1 — they're regression alarms, not
perf goals. will tighten them once the kernel-level memory plane
benchmark is wired in. Run explicitly with ``pytest -m perf``.

Methodology mirrors ``tests/perf/test_classify_p99.py``:

1. ``time.perf_counter()`` for sub-microsecond accuracy.
2. 50-iteration warm-up per adapter (excluded from timings) to settle JIT-
   like effects (regex caches, frozen-dataclass slots, ``Path.glob`` cache).
3. 100 measured iterations per adapter.
4. P99 = ``statistics.quantiles(timings_ms, n=100)[98]``.
5. Failures include the measured P99 and P50 for triage.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_amplifier.adapters.agentscope import AgentScopeAdapter
from agent_amplifier.adapters.claude_code import ClaudeCodeAdapter
from agent_amplifier.adapters.crewai import CrewAIAdapter
from agent_amplifier.adapters.cursor import CursorAdapter
from agent_amplifier.adapters.github_copilot import GitHubCopilotAdapter
from agent_amplifier.adapters.langgraph import LangGraphAdapter

# ---------------------------------------------------------------------------
# Budget table (per )
# ---------------------------------------------------------------------------

# File-host adapters do real I/O — 50 ms covers the worst-case filesystem
# round-trip on a slow CI runner. Framework adapters call mocks — 5 ms is
# tight enough to catch accidental real work without being flaky.
_FILE_HOST_P99_MS_BUDGET: float = 50.0
_FRAMEWORK_P99_MS_BUDGET: float = 5.0

_WARMUP_ITERS: int = 50
_MEASURED_ITERS: int = 100

# Representative query — non-empty so the keyword-rank branches execute,
# short so we measure adapter overhead not string-scan cost.
_QUERY: str = "test"


# ---------------------------------------------------------------------------
# Helper — common P99 measurement loop
# ---------------------------------------------------------------------------


def _measure_p99_ms(call: Any) -> tuple[float, float]:
    """Run ``call`` ``_WARMUP_ITERS + _MEASURED_ITERS`` times; return (P50, P99) ms."""
    for _ in range(_WARMUP_ITERS):
        call()
    timings_ms: list[float] = []
    for _ in range(_MEASURED_ITERS):
        t0 = time.perf_counter()
        call()
        timings_ms.append((time.perf_counter() - t0) * 1000)
    p50 = statistics.median(timings_ms)
    # ``statistics.quantiles(n=100)`` returns 99 cut-points; index 98 ≈ P99.
    p99 = statistics.quantiles(timings_ms, n=100)[98]
    return p50, p99


# ---------------------------------------------------------------------------
# Host adapters (real filesystem reads)
# ---------------------------------------------------------------------------


@pytest.mark.perf
def test_claude_code_recall_p99_under_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ClaudeCodeAdapter: P99 < 50 ms over 100 reads of a small CLAUDE.md."""
    # Pin CWD inside tmp_path; force a fake HOME so ``Path.home()/.claude``
    # cannot resolve to a real (large) user file outside the fixture.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(fake_home))

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "# Project\n\n## Test\nThis section is what queries match against.\n"
        "## Other\nUnrelated section to exercise the H2 split branch.\n",
        encoding="utf-8",
    )

    adapter = ClaudeCodeAdapter(kernel=None)
    p50, p99 = _measure_p99_ms(lambda: adapter.default_memory_recall(_QUERY, 3))
    assert p99 < _FILE_HOST_P99_MS_BUDGET, (
        f"ClaudeCodeAdapter recall P99 = {p99:.3f} ms exceeds "
        f"{_FILE_HOST_P99_MS_BUDGET} ms (P50 = {p50:.3f} ms)"
    )


@pytest.mark.perf
def test_cursor_recall_p99_under_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CursorAdapter: P99 < 50 ms over 100 reads of a small ``.mdc`` file."""
    monkeypatch.chdir(tmp_path)

    rules = tmp_path / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "python.mdc").write_text(
        "---\n"
        "description: Python rules\n"
        "alwaysApply: true\n"
        "---\n"
        "Use type hints on every public function. test test test.\n",
        encoding="utf-8",
    )

    adapter = CursorAdapter(kernel=None)
    p50, p99 = _measure_p99_ms(lambda: adapter.default_memory_recall(_QUERY, 3))
    assert p99 < _FILE_HOST_P99_MS_BUDGET, (
        f"CursorAdapter recall P99 = {p99:.3f} ms exceeds "
        f"{_FILE_HOST_P99_MS_BUDGET} ms (P50 = {p50:.3f} ms)"
    )


@pytest.mark.perf
def test_github_copilot_recall_p99_under_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHubCopilotAdapter: P99 < 50 ms over 100 reads of repo + scoped file."""
    monkeypatch.chdir(tmp_path)

    gh = tmp_path / ".github"
    gh.mkdir()
    (gh / "copilot-instructions.md").write_text(
        "# Repo rules\n\n## Test\nKeep tests deterministic.\n",
        encoding="utf-8",
    )
    scoped = gh / "instructions"
    scoped.mkdir()
    (scoped / "python.instructions.md").write_text(
        "## Test\nUse pytest fixtures for shared setup.\n",
        encoding="utf-8",
    )

    adapter = GitHubCopilotAdapter(kernel=None)
    p50, p99 = _measure_p99_ms(lambda: adapter.default_memory_recall(_QUERY, 3))
    assert p99 < _FILE_HOST_P99_MS_BUDGET, (
        f"GitHubCopilotAdapter recall P99 = {p99:.3f} ms exceeds "
        f"{_FILE_HOST_P99_MS_BUDGET} ms (P50 = {p50:.3f} ms)"
    )


# ---------------------------------------------------------------------------
# Framework adapters (mocked user-supplied objects)
# ---------------------------------------------------------------------------


def _make_langgraph_checkpointer() -> Any:
    """Build a mock ``BaseCheckpointSaver``-shaped object.

    Returns a single ``CheckpointTuple``-shaped namespace whose
    ``checkpoint["channel_values"]["messages"]`` is a 3-message list. Mock
    state is built once outside the timed loop so every call returns the
    same pre-built tuple — measuring adapter overhead, not mock construction.
    """
    tup = MagicMock()
    tup.checkpoint = {
        "channel_values": {
            "messages": [
                # ``.content`` attr style, the LangChain BaseMessage shape.
                _msg("hello test"),
                _msg("ack test"),
                _msg("noisy unrelated"),
            ]
        }
    }
    saver = MagicMock()
    saver.get_tuple = MagicMock(return_value=tup)
    return saver


def _msg(text: str) -> Any:
    """Return a duck-typed LangChain-style message object with ``.content``."""
    obj = MagicMock()
    obj.content = text
    return obj


@pytest.mark.perf
def test_langgraph_recall_p99_under_budget() -> None:
    """LangGraphAdapter: P99 < 5 ms over 100 calls (mocked checkpointer)."""
    adapter = LangGraphAdapter(checkpointer=_make_langgraph_checkpointer())
    p50, p99 = _measure_p99_ms(lambda: adapter.default_memory_recall(_QUERY, 3))
    assert p99 < _FRAMEWORK_P99_MS_BUDGET, (
        f"LangGraphAdapter recall P99 = {p99:.3f} ms exceeds "
        f"{_FRAMEWORK_P99_MS_BUDGET} ms (P50 = {p50:.3f} ms)"
    )


def _make_crewai_crew() -> Any:
    """Build a mock Crew with a ``.memory.search`` returning 3 dict items."""
    crew = MagicMock()
    crew.memory.search = MagicMock(
        return_value=[
            {"memory": "test entry one", "score": 0.9, "metadata": {"k": 1}},
            {"memory": "test entry two", "score": 0.7, "metadata": {}},
            {"memory": "noise", "score": 0.1, "metadata": {}},
        ]
    )
    return crew


@pytest.mark.perf
def test_crewai_recall_p99_under_budget() -> None:
    """CrewAIAdapter: P99 < 5 ms over 100 calls (mocked crew.memory.search)."""
    adapter = CrewAIAdapter(crew=_make_crewai_crew())
    p50, p99 = _measure_p99_ms(lambda: adapter.default_memory_recall(_QUERY, 3))
    assert p99 < _FRAMEWORK_P99_MS_BUDGET, (
        f"CrewAIAdapter recall P99 = {p99:.3f} ms exceeds "
        f"{_FRAMEWORK_P99_MS_BUDGET} ms (P50 = {p50:.3f} ms)"
    )


def _make_agentscope_memory() -> Any:
    """Build a mock AgentScope Memory whose ``get_memory()`` returns 3 Msgs."""

    def _build_msg(content: str, role: str = "user", name: str = "alice") -> Any:
        m = MagicMock()
        m.content = content
        m.role = role
        m.name = name
        return m

    memory = MagicMock()
    memory.get_memory = MagicMock(
        return_value=[
            _build_msg("hello test"),
            _build_msg("ack test"),
            _build_msg("noisy unrelated"),
        ]
    )
    return memory


@pytest.mark.perf
def test_agentscope_recall_p99_under_budget() -> None:
    """AgentScopeAdapter: P99 < 5 ms over 100 calls (mocked memory.get_memory)."""
    adapter = AgentScopeAdapter(memory=_make_agentscope_memory())
    p50, p99 = _measure_p99_ms(lambda: adapter.default_memory_recall(_QUERY, 3))
    assert p99 < _FRAMEWORK_P99_MS_BUDGET, (
        f"AgentScopeAdapter recall P99 = {p99:.3f} ms exceeds "
        f"{_FRAMEWORK_P99_MS_BUDGET} ms (P50 = {p50:.3f} ms)"
    )
