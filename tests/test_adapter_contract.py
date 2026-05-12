# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""MED-3 + QA-M05 — parametrized cross-adapter IP-9 contract test.

Every bundled adapter MUST satisfy the universal memory plane contract:

    1. ``default_memory_recall(query, limit)`` returns ``list[RecalledPattern]``
       (possibly empty), never raises.
    2. Every returned pattern carries a non-empty ``source`` string starting
       with ``HOST_NAME:``.
    3. ``default_memory_remember(Outcome(...))`` is a no-op or fire-and-forget;
       never raises regardless of input.

This file proves the V1 promise from Master Plan IP-9: the memory plane
works UNIVERSALLY across both file-based hosts (Claude Code / Cursor /
GitHub Copilot) AND framework adapters (LangGraph / CrewAI / AgentScope)
without each adapter test having to re-prove the shape.

Coverage cost: ~6 parametrize entries x 2 contract checks. Tests run in
<50 ms total because each adapter is exercised against a fixture fresh tmp
CWD or a duck-typed mock.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.adapters import (
    AgentScopeAdapter,
    ClaudeCodeAdapter,
    CrewAIAdapter,
    CursorAdapter,
    GitHubCopilotAdapter,
    LangGraphAdapter,
)
from agent_amplifier.types import EffortLevel, Outcome, RecalledPattern

# ---------------------------------------------------------------------------
# Duck-typed mocks for the framework adapters (no real framework imports).
# ---------------------------------------------------------------------------


class _FakeCheckpointTuple:
    """Minimal LangGraph checkpoint-tuple shape — duck-typed."""

    def __init__(self, messages: list[Any]) -> None:
        self.checkpoint = {"channel_values": {"messages": messages}}


class _FakeMessage:
    """LangGraph message duck shape: has ``content`` attr."""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeCheckpointer:
    def get_tuple(self, _config: dict[str, Any]) -> _FakeCheckpointTuple:
        return _FakeCheckpointTuple(
            [_FakeMessage("python refactor of the auth module")]
        )


class _FakeMemoryItem:
    """CrewAI memory shape: dict-ish with ``content`` key."""

    def __init__(self, content: str) -> None:
        self._d = {"content": content, "metadata": {}}

    def get(self, k: str, default: Any = None) -> Any:
        return self._d.get(k, default)


class _FakeCrewMemory:
    def search(self, query: str, limit: int = 3) -> list[Any]:
        # Return one match regardless of query so the adapter exercises the
        # "non-empty hit" branch on every contract check.
        return [{"content": f"crewai-stub: {query}", "metadata": {}}]

    def save(self, *_a: Any, **_kw: Any) -> None:
        return None


class _FakeCrew:
    memory = _FakeCrewMemory()


class _FakeAgentScopeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeAgentScopeMemory:
    def get_memory(self) -> list[Any]:
        return [_FakeAgentScopeMsg("agentscope-stub working memory")]

    def add(self, _msg: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Adapter factory — one constructor per adapter, kernel=None.
# ---------------------------------------------------------------------------


def _make_claude_code(_tmp_path: Path) -> AdapterBase:
    return ClaudeCodeAdapter(kernel=None)


def _make_cursor(_tmp_path: Path) -> AdapterBase:
    return CursorAdapter(kernel=None)


def _make_github_copilot(_tmp_path: Path) -> AdapterBase:
    return GitHubCopilotAdapter(kernel=None)


def _make_langgraph(_tmp_path: Path) -> AdapterBase:
    return LangGraphAdapter(_FakeCheckpointer(), thread_id="t1", kernel=None)


def _make_crewai(_tmp_path: Path) -> AdapterBase:
    return CrewAIAdapter(_FakeCrew(), kernel=None)


def _make_agentscope(_tmp_path: Path) -> AdapterBase:
    return AgentScopeAdapter(_FakeAgentScopeMemory(), kernel=None)


_ADAPTER_FACTORIES = (
    ("claude_code", _make_claude_code),
    ("cursor", _make_cursor),
    ("github_copilot", _make_github_copilot),
    ("langgraph", _make_langgraph),
    ("crewai", _make_crewai),
    ("agentscope", _make_agentscope),
)


# ---------------------------------------------------------------------------
# IP-9 contract tests (parametrized over the 6 V1 adapters)
# ---------------------------------------------------------------------------


@pytest.fixture
def cwd_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Switch CWD to an empty tmp dir so file-based adapters see no host data."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("name", "factory"),
    _ADAPTER_FACTORIES,
    ids=[n for n, _ in _ADAPTER_FACTORIES],
)
def test_recall_returns_list_of_recalled_pattern(
    name: str,
    factory: Any,
    cwd_isolated: Path,
) -> None:
    """IP-9 contract A — recall returns ``list[RecalledPattern]``, never raises."""
    adapter = factory(cwd_isolated)
    out = adapter.default_memory_recall("test", 3)
    assert isinstance(out, list)
    for pat in out:
        assert isinstance(pat, RecalledPattern), (
            f"{name} returned non-RecalledPattern: {type(pat).__name__}"
        )


@pytest.mark.parametrize(
    ("name", "factory"),
    _ADAPTER_FACTORIES,
    ids=[n for n, _ in _ADAPTER_FACTORIES],
)
def test_recall_patterns_have_nonempty_source_with_host_prefix(
    name: str,
    factory: Any,
    cwd_isolated: Path,
) -> None:
    """IP-9 contract B — every pattern's source starts with ``HOST_NAME:``.

    File-based adapters with no host files in CWD legitimately return ``[]``
    — that's contract-compliant. Framework adapters with mock backends DO
    return non-empty lists; we assert the source format on every returned
    item.
    """
    adapter = factory(cwd_isolated)
    out = adapter.default_memory_recall("test", 3)
    host_name = getattr(type(adapter), "HOST_NAME", "")
    assert host_name, f"{name} missing HOST_NAME class attribute"
    for pat in out:
        assert pat.source, (
            f"{name} returned RecalledPattern with empty source"
        )
        assert pat.source.startswith(f"{host_name}:"), (
            f"{name} source does not start with {host_name!r}: "
            f"got {pat.source!r}"
        )


@pytest.mark.parametrize(
    ("name", "factory"),
    _ADAPTER_FACTORIES,
    ids=[n for n, _ in _ADAPTER_FACTORIES],
)
def test_remember_does_not_raise(
    name: str,
    factory: Any,
    cwd_isolated: Path,
) -> None:
    """IP-9 contract C — remember is fire-and-forget, never raises.

    Adapters with no write target (no host files / no memory binding) MUST
    silently no-op. Adapters with a write target may write or skip; either
    is fine — we only care that the call returns without exception.
    """
    adapter = factory(cwd_isolated)
    outcome = Outcome(
        query="contract test",
        effort=EffortLevel.LOW,
        iterations=1,
        quality=0.5,
        converged=False,
        tokens_used=10,
    )
    # MUST NOT raise. Wrap in try/except to make the failure mode loud.
    try:
        adapter.default_memory_remember(outcome)
    except Exception as exc:  # pragma: no cover - defensive contract guard
        pytest.fail(
            f"{name} default_memory_remember raised "
            f"{type(exc).__name__}: {exc}"
        )


@pytest.mark.parametrize(
    ("name", "factory"),
    _ADAPTER_FACTORIES,
    ids=[n for n, _ in _ADAPTER_FACTORIES],
)
def test_framework_name_matches_regex(
    name: str,
    factory: Any,
    cwd_isolated: Path,
) -> None:
    """every adapter's framework_name matches the universal slug regex."""
    import re

    adapter = factory(cwd_isolated)
    assert re.match(
        r"^[a-z][a-z0-9_]{2,31}$", adapter.framework_name
    ), f"{name} framework_name violates  regex"
