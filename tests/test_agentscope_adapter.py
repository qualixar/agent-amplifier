# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for AgentScopeAdapter ().

Coverage targets: 100 % line + 100 % branch on
``src/agent_amplifier/adapters/agentscope.py``.

Critical: NO real ``agentscope`` install required. All Memory interaction
goes through mock objects implementing the documented working-memory
contract (``.get_memory()`` returning ``list[Msg]``; ``.add(msg)``).
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import pytest

from agent_amplifier.adapters.agentscope import (
    _PER_CHUNK_BYTES,
    AgentScopeAdapter,
)
from agent_amplifier.types import EffortLevel, Outcome, RecalledPattern

# ---------------------------------------------------------------------------
# Mock AgentScope Msg + Memory
# ---------------------------------------------------------------------------


@dataclass
class _MockMsg:
    """Mimics AgentScope Msg with .name / .content / .role."""

    name: str = "user"
    content: Any = ""
    role: str = "user"


@dataclass
class _MockMemory:
    """Mimics AgentScope working-memory shape."""

    messages: list[Any] = field(default_factory=list)
    get_raises: BaseException | None = None
    add_raises: BaseException | None = None
    return_non_iterable: bool = False
    return_none: bool = False
    added: list[Any] = field(default_factory=list)

    def get_memory(self) -> Any:
        if self.get_raises is not None:
            raise self.get_raises
        if self.return_none:
            return None
        if self.return_non_iterable:
            return 42
        return self.messages

    def add(self, msg: Any) -> None:
        if self.add_raises is not None:
            raise self.add_raises
        self.added.append(msg)


def _make_outcome(query: str = "amplifier note", quality: float = 0.5) -> Outcome:
    return Outcome(
        query=query,
        effort=EffortLevel.LOW,
        iterations=1,
        quality=quality,
    )


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


def test_detect_true_when_agentscope_spec_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object()
    )
    assert AgentScopeAdapter.detect() is True


def test_detect_false_when_agentscope_spec_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None
    )
    assert AgentScopeAdapter.detect() is False


def test_detect_false_when_find_spec_raises_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(name: str) -> Any:
        raise ImportError("blocked")

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert AgentScopeAdapter.detect() is False


def test_detect_false_when_find_spec_raises_modulenotfound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(name: str) -> Any:
        raise ModuleNotFoundError("missing")

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert AgentScopeAdapter.detect() is False


# ---------------------------------------------------------------------------
# default_memory_recall — happy paths
# ---------------------------------------------------------------------------


def test_recall_returns_recalledpattern() -> None:
    msgs = [
        _MockMsg(name="alice", content="hello python", role="user"),
        _MockMsg(name="bob", content="other text", role="assistant"),
    ]
    mem = _MockMemory(messages=msgs)
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert isinstance(res[0], RecalledPattern)
    assert "python" in res[0].text
    assert res[0].source == "agentscope:alice"
    assert "user" in res[0].tags


def test_recall_filters_by_query_substring_case_insensitive() -> None:
    msgs = [
        _MockMsg(content="Apple"),
        _MockMsg(content="banana"),
        _MockMsg(content="BANANA SPLIT"),
    ]
    mem = _MockMemory(messages=msgs)
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("BANANA")
    assert len(res) == 2


def test_recall_empty_query_returns_recent_messages() -> None:
    msgs = [_MockMsg(content=f"msg-{i}") for i in range(5)]
    mem = _MockMemory(messages=msgs)
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("", limit=2)
    assert len(res) == 2
    # most recent come back
    assert res[-1].text == "msg-4"


def test_recall_caps_chunk_size() -> None:
    huge = "z" * (_PER_CHUNK_BYTES * 4)
    mem = _MockMemory(messages=[_MockMsg(content=huge)])
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("")
    assert len(res[0].text) <= _PER_CHUNK_BYTES


def test_recall_skips_empty_messages() -> None:
    msgs = [
        _MockMsg(content=""),
        _MockMsg(content="real content"),
        _MockMsg(content=None),
    ]
    mem = _MockMemory(messages=msgs)
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("")
    assert len(res) == 1
    assert res[0].text == "real content"


def test_recall_handles_dict_messages() -> None:
    msgs: list[dict[str, Any]] = [
        {"content": "dict python", "name": "x", "role": "user"},
        {"content": "rust here", "name": "y", "role": "user"},
    ]
    mem = _MockMemory(messages=msgs)
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert "dict python" in res[0].text
    assert res[0].source == "agentscope:x"


def test_recall_handles_string_messages() -> None:
    """Plain strings → still recallable, source defaults to 'memory'."""
    mem = _MockMemory(messages=["plain python text", "rust"])
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert res[0].source == "agentscope:memory"
    # No name/role on plain string → empty tags
    assert res[0].tags == ()


def test_recall_handles_list_content_blocks() -> None:
    msg = _MockMsg(
        content=[
            {"text": "first part python"},
            "literal string",
            {"content": "third part"},
            {"unknown": "skip"},
        ]
    )
    mem = _MockMemory(messages=[msg])
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert "first part python" in res[0].text


def test_recall_skips_list_content_with_no_text_parts() -> None:
    """List of unrecognized blocks → no text → message skipped."""
    msg = _MockMsg(content=[{"unknown": "blob"}, 42, None])
    mem = _MockMemory(
        messages=[msg, _MockMsg(content="kept")]
    )
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("")
    assert len(res) == 1
    assert res[0].text == "kept"


def test_recall_skips_non_stringifiable_messages() -> None:
    class _Junk:
        pass

    mem = _MockMemory(messages=[_Junk(), _MockMsg(content="real")])
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("")
    assert len(res) == 1


def test_recall_dict_with_non_string_content_skipped() -> None:
    """Dict whose ``content`` is non-string → no extraction → skipped."""
    msgs: list[dict[str, Any]] = [
        {"content": 12345},
        {"content": "kept"},
    ]
    mem = _MockMemory(messages=msgs)
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("")
    assert len(res) == 1
    assert res[0].text == "kept"


def test_recall_message_without_name_uses_memory_default() -> None:
    """Object with content but no name → source falls back to 'memory'."""

    @dataclass
    class _Anon:
        content: str = "hello"

    mem = _MockMemory(messages=[_Anon()])
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("")
    assert res[0].source == "agentscope:memory"


def test_recall_empty_role_yields_empty_tags() -> None:
    @dataclass
    class _NoRole:
        name: str = "n"
        content: str = "hello"

    mem = _MockMemory(messages=[_NoRole()])
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("")
    assert res[0].tags == ()


# ---------------------------------------------------------------------------
# default_memory_recall — error / fallback paths
# ---------------------------------------------------------------------------


def test_recall_returns_empty_when_get_memory_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mem = _MockMemory(get_raises=RuntimeError("bad"))
    adapter = AgentScopeAdapter(memory=mem)
    with caplog.at_level(logging.WARNING):
        res = adapter.default_memory_recall("x")
    assert res == []
    assert any("get_memory failed" in r.message for r in caplog.records)


def test_recall_returns_empty_when_get_memory_returns_none() -> None:
    mem = _MockMemory(return_none=True)
    adapter = AgentScopeAdapter(memory=mem)
    assert adapter.default_memory_recall("x") == []


def test_recall_returns_empty_when_get_memory_non_iterable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mem = _MockMemory(return_non_iterable=True)
    adapter = AgentScopeAdapter(memory=mem)
    with caplog.at_level(logging.WARNING):
        res = adapter.default_memory_recall("x")
    assert res == []
    assert any("non-iterable" in r.message for r in caplog.records)


def test_recall_respects_limit() -> None:
    msgs = [_MockMsg(content=f"python {i}") for i in range(10)]
    mem = _MockMemory(messages=msgs)
    adapter = AgentScopeAdapter(memory=mem)
    res = adapter.default_memory_recall("python", limit=3)
    assert len(res) == 3


# ---------------------------------------------------------------------------
# default_memory_remember — fallback dict path
# ---------------------------------------------------------------------------


def test_remember_falls_back_to_dict_when_msg_unimportable(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No agentscope module installed → adapter writes a dict to memory."""
    # Ensure agentscope.message is NOT importable.
    monkeypatch.delitem(sys.modules, "agentscope.message", raising=False)
    monkeypatch.delitem(sys.modules, "agentscope", raising=False)
    mem = _MockMemory()
    adapter = AgentScopeAdapter(memory=mem)
    with caplog.at_level(logging.WARNING):
        adapter.default_memory_remember(_make_outcome("hello"))
    assert len(mem.added) == 1
    assert isinstance(mem.added[0], dict)
    assert mem.added[0] == {
        "name": "amplifier",
        "content": "hello",
        "role": "system",
    }
    assert any(
        "could not construct Msg" in r.message for r in caplog.records
    )


def test_remember_warns_when_memory_has_no_add_method(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _NoAdd:
        def get_memory(self) -> list[Any]:
            return []

    adapter = AgentScopeAdapter(memory=_NoAdd())
    with caplog.at_level(logging.WARNING):
        adapter.default_memory_remember(_make_outcome())
    assert any("no .add method" in r.message for r in caplog.records)


def test_remember_swallows_add_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mem = _MockMemory(add_raises=RuntimeError("locked"))
    adapter = AgentScopeAdapter(memory=mem)
    with caplog.at_level(logging.WARNING):
        adapter.default_memory_remember(_make_outcome())
    assert any("memory.add failed" in r.message for r in caplog.records)


def test_remember_truncates_long_query() -> None:
    """Outcome text exceeding cap → truncated before sending to memory."""
    mem = _MockMemory()
    adapter = AgentScopeAdapter(memory=mem)
    big = "x" * (_PER_CHUNK_BYTES + 500)
    adapter.default_memory_remember(_make_outcome(big, 0.5))
    # The dict path is used (no agentscope installed in test env)
    assert len(mem.added) == 1
    payload = mem.added[0]
    assert isinstance(payload, dict)
    assert len(payload["content"]) == _PER_CHUNK_BYTES


# ---------------------------------------------------------------------------
# default_memory_remember — Msg-construction success paths
# ---------------------------------------------------------------------------


class _FakeKwargMsg:
    """Constructed via kwargs only — first attempt succeeds."""

    def __init__(self, *, name: str, content: str, role: str) -> None:
        self.name = name
        self.content = content
        self.role = role


class _FakePositionalMsg:
    """Constructed via positional only — kwargs attempt raises TypeError."""

    def __init__(self, *args: Any) -> None:
        if len(args) != 3:
            raise TypeError("expected 3 positional args")
        self.name, self.content, self.role = args

    @classmethod
    def __init_subclass__(cls) -> None:  # pragma: no cover
        super().__init_subclass__()


def _install_fake_agentscope_message(
    monkeypatch: pytest.MonkeyPatch, msg_cls: type
) -> None:
    """Inject a fake ``agentscope.message`` module exposing ``Msg``."""
    pkg = ModuleType("agentscope")
    submod = ModuleType("agentscope.message")
    submod.Msg = msg_cls  # type: ignore[attr-defined]
    pkg.message = submod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agentscope", pkg)
    monkeypatch.setitem(sys.modules, "agentscope.message", submod)


def test_remember_uses_kwargs_msg_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agentscope_message(monkeypatch, _FakeKwargMsg)
    mem = _MockMemory()
    adapter = AgentScopeAdapter(memory=mem)
    adapter.default_memory_remember(_make_outcome("hello"))
    assert len(mem.added) == 1
    written = mem.added[0]
    assert isinstance(written, _FakeKwargMsg)
    assert written.name == "amplifier"
    assert written.content == "hello"
    assert written.role == "system"


def test_remember_falls_back_to_positional_msg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When kwargs constructor raises TypeError, positional is tried."""

    class _PositionalOnly:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if kwargs:
                raise TypeError("no kwargs supported")
            if len(args) != 3:
                raise TypeError("expected 3 positional")
            self.name, self.content, self.role = args

    _install_fake_agentscope_message(monkeypatch, _PositionalOnly)
    mem = _MockMemory()
    adapter = AgentScopeAdapter(memory=mem)
    adapter.default_memory_remember(_make_outcome("p-only"))
    assert len(mem.added) == 1
    written = mem.added[0]
    assert isinstance(written, _PositionalOnly)
    assert written.content == "p-only"


def test_remember_falls_back_to_dict_when_both_msg_signatures_fail(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both kwargs AND positional Msg constructors raise TypeError."""

    class _AlwaysBroken:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise TypeError("never works")

    _install_fake_agentscope_message(monkeypatch, _AlwaysBroken)
    mem = _MockMemory()
    adapter = AgentScopeAdapter(memory=mem)
    with caplog.at_level(logging.WARNING):
        adapter.default_memory_remember(_make_outcome("dict-fb"))
    assert len(mem.added) == 1
    assert isinstance(mem.added[0], dict)
    assert mem.added[0]["content"] == "dict-fb"


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------


def test_class_metadata() -> None:
    assert AgentScopeAdapter.framework_name == "agentscope"
    assert AgentScopeAdapter.HOST_NAME == "agentscope"
    assert AgentScopeAdapter.version == "1.0.0"


# ---------------------------------------------------------------------------
# — bounded materialization via islice
# ---------------------------------------------------------------------------


def test_recall_does_not_fully_materialize_huge_iterable() -> None:
    """the adapter must not consume an unbounded generator."""
    from unittest.mock import MagicMock

    from agent_amplifier.adapters.agentscope import AgentScopeAdapter

    yielded = {"n": 0}

    def big_gen():
        for i in range(100_000):
            yielded["n"] += 1
            m = MagicMock()
            m.content = f"item-{i}"
            yield m

    memory = MagicMock()
    memory.get_memory = MagicMock(return_value=big_gen())
    adapter = AgentScopeAdapter(memory=memory)
    out = adapter.default_memory_recall("item", limit=3)
    assert len(out) == 3
    # We should have consumed at most limit*8 + a tiny prefetch buffer,
    # never all 100K.
    assert yielded["n"] <= 30
