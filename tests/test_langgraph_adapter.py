# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for LangGraphAdapter ().

Coverage targets: 100 % line + 100 % branch on
``src/agent_amplifier/adapters/langgraph.py``.

Critical: NO real ``langgraph`` install required. All checkpointer
interaction goes through mock objects implementing the documented
``BaseCheckpointSaver.get_tuple`` contract.
"""
from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_amplifier.adapters.langgraph import _PER_CHUNK_BYTES, LangGraphAdapter
from agent_amplifier.types import EffortLevel, Outcome, RecalledPattern

# ---------------------------------------------------------------------------
# Mock LangGraph checkpointer + message types
# ---------------------------------------------------------------------------


@dataclass
class _MockMessage:
    """Mimics LangChain BaseMessage shape (``.content`` attribute)."""

    content: Any
    type: str = "human"


@dataclass
class _MockCheckpointTuple:
    checkpoint: dict[str, Any] = field(default_factory=dict)


class _MockCheckpointer:
    """Minimal BaseCheckpointSaver-shaped mock for tests."""

    def __init__(
        self,
        messages: list[Any] | None = None,
        raises: BaseException | None = None,
        return_none: bool = False,
        return_malformed: bool = False,
    ) -> None:
        self._messages = messages or []
        self._raises = raises
        self._return_none = return_none
        self._return_malformed = return_malformed
        self.last_config: Any = None

    def get_tuple(self, config: Any) -> Any:
        self.last_config = config
        if self._raises is not None:
            raise self._raises
        if self._return_none:
            return None
        if self._return_malformed:
            return _MockCheckpointTuple(checkpoint={"unrelated": "shape"})
        return _MockCheckpointTuple(
            checkpoint={"channel_values": {"messages": self._messages}}
        )


def _make_outcome(query: str = "test", quality: float = 0.5) -> Outcome:
    return Outcome(
        query=query,
        effort=EffortLevel.LOW,
        iterations=1,
        quality=quality,
    )


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


def test_detect_true_when_langgraph_spec_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_spec = object()  # truthy non-None
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: fake_spec
    )
    assert LangGraphAdapter.detect() is True


def test_detect_false_when_langgraph_spec_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None
    )
    assert LangGraphAdapter.detect() is False


def test_detect_false_when_find_spec_raises_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(name: str) -> Any:
        raise ImportError("blocked")

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert LangGraphAdapter.detect() is False


def test_detect_false_when_find_spec_raises_valueerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(name: str) -> Any:
        raise ValueError("bad name")

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert LangGraphAdapter.detect() is False


# ---------------------------------------------------------------------------
# default_memory_recall
# ---------------------------------------------------------------------------


def test_recall_returns_recalledpattern() -> None:
    msgs = [
        _MockMessage(content="hello python world"),
        _MockMessage(content="some other text"),
    ]
    ck = _MockCheckpointer(messages=msgs)
    adapter = LangGraphAdapter(checkpointer=ck, thread_id="t1")
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert isinstance(res[0], RecalledPattern)
    assert "python" in res[0].text
    assert res[0].source == "langgraph:thread-t1"
    assert "checkpoint" in res[0].tags
    assert "thread:t1" in res[0].tags


def test_recall_passes_correct_config_to_get_tuple() -> None:
    ck = _MockCheckpointer(messages=[_MockMessage(content="x")])
    adapter = LangGraphAdapter(checkpointer=ck, thread_id="custom-thread")
    adapter.default_memory_recall("x")
    assert ck.last_config == {"configurable": {"thread_id": "custom-thread"}}


def test_recall_returns_empty_when_get_tuple_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ck = _MockCheckpointer(raises=RuntimeError("backend down"))
    adapter = LangGraphAdapter(checkpointer=ck, thread_id="t1")
    with caplog.at_level(logging.WARNING):
        res = adapter.default_memory_recall("anything")
    assert res == []
    assert any("get_tuple failed" in r.message for r in caplog.records)


def test_recall_returns_empty_when_get_tuple_returns_none() -> None:
    ck = _MockCheckpointer(return_none=True)
    adapter = LangGraphAdapter(checkpointer=ck)
    assert adapter.default_memory_recall("anything") == []


def test_recall_returns_empty_when_checkpoint_shape_malformed() -> None:
    ck = _MockCheckpointer(return_malformed=True)
    adapter = LangGraphAdapter(checkpointer=ck)
    assert adapter.default_memory_recall("anything") == []


def test_recall_returns_empty_when_tup_checkpoint_attr_is_none() -> None:
    """``tup.checkpoint`` being None (rather than missing) → empty recall."""

    class _NoCheckpoint:
        checkpoint = None

    class _Ck:
        last_config: Any = None

        def get_tuple(self, config: Any) -> Any:
            self.last_config = config
            return _NoCheckpoint()

    ck = _Ck()
    adapter = LangGraphAdapter(checkpointer=ck)
    assert adapter.default_memory_recall("anything") == []


def test_recall_returns_empty_when_messages_list_empty() -> None:
    ck = _MockCheckpointer(messages=[])
    adapter = LangGraphAdapter(checkpointer=ck)
    assert adapter.default_memory_recall("anything") == []


def test_recall_filters_by_query_substring_case_insensitive() -> None:
    msgs = [
        _MockMessage(content="Apple Banana"),
        _MockMessage(content="cherry plum"),
        _MockMessage(content="banana split"),
    ]
    ck = _MockCheckpointer(messages=msgs)
    adapter = LangGraphAdapter(checkpointer=ck)
    res = adapter.default_memory_recall("BANANA")
    assert len(res) == 2
    assert all("banana" in r.text.lower() for r in res)


def test_recall_empty_query_returns_recent_messages_up_to_limit() -> None:
    msgs = [_MockMessage(content=f"msg-{i}") for i in range(10)]
    ck = _MockCheckpointer(messages=msgs)
    adapter = LangGraphAdapter(checkpointer=ck)
    res = adapter.default_memory_recall("", limit=3)
    assert len(res) == 3
    # Most recent (last) come back
    assert res[-1].text == "msg-9"
    assert res[0].text == "msg-7"


def test_recall_respects_limit() -> None:
    msgs = [_MockMessage(content=f"python {i}") for i in range(10)]
    ck = _MockCheckpointer(messages=msgs)
    adapter = LangGraphAdapter(checkpointer=ck)
    res = adapter.default_memory_recall("python", limit=2)
    assert len(res) == 2


def test_recall_caps_chunk_size() -> None:
    huge = "x" * (_PER_CHUNK_BYTES * 4)
    ck = _MockCheckpointer(messages=[_MockMessage(content=huge)])
    adapter = LangGraphAdapter(checkpointer=ck)
    res = adapter.default_memory_recall("")
    assert len(res) == 1
    assert len(res[0].text) <= _PER_CHUNK_BYTES


def test_recall_skips_empty_messages() -> None:
    msgs = [
        _MockMessage(content=""),
        _MockMessage(content="real content"),
        _MockMessage(content=None),
    ]
    ck = _MockCheckpointer(messages=msgs)
    adapter = LangGraphAdapter(checkpointer=ck)
    res = adapter.default_memory_recall("")
    assert len(res) == 1
    assert res[0].text == "real content"


def test_recall_handles_dict_messages() -> None:
    msgs: list[dict[str, str]] = [
        {"content": "hello python", "role": "user"},
        {"content": "no match here", "role": "assistant"},
    ]
    ck = _MockCheckpointer(messages=msgs)
    adapter = LangGraphAdapter(checkpointer=ck)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert res[0].text == "hello python"


def test_recall_handles_string_messages() -> None:
    msgs: list[Any] = ["plain string about python", "rust stuff"]
    ck = _MockCheckpointer(messages=msgs)
    adapter = LangGraphAdapter(checkpointer=ck)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert "python" in res[0].text


def test_recall_handles_list_content_blocks() -> None:
    """Multi-part content like ``[{"text": "a"}, {"text": "b"}]``."""
    msg = _MockMessage(
        content=[
            {"text": "first part python"},
            "literal string",
            {"content": "third part"},
            {"unknown": "skipped"},
        ]
    )
    ck = _MockCheckpointer(messages=[msg])
    adapter = LangGraphAdapter(checkpointer=ck)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    # Joined parts
    assert "first part python" in res[0].text


def test_recall_skips_non_stringifiable_messages() -> None:
    """Objects with no ``.content``, not dict, not string → skipped."""

    class _Junk:
        pass

    msgs = [_Junk(), _MockMessage(content="real")]
    ck = _MockCheckpointer(messages=msgs)
    adapter = LangGraphAdapter(checkpointer=ck)
    res = adapter.default_memory_recall("")
    assert len(res) == 1
    assert res[0].text == "real"


def test_recall_handles_list_content_with_no_text_parts() -> None:
    """Content list where no entries yield text → message skipped."""
    msg = _MockMessage(content=[{"unknown": "blob"}, 42, None])
    ck = _MockCheckpointer(messages=[msg, _MockMessage(content="kept")])
    adapter = LangGraphAdapter(checkpointer=ck)
    res = adapter.default_memory_recall("")
    assert len(res) == 1
    assert res[0].text == "kept"


def test_recall_handles_dict_content_non_string() -> None:
    """Dict with non-string ``content`` → falls through to other branches."""
    msgs: list[dict[str, Any]] = [
        {"content": 12345},  # int content → no string extraction
        {"content": "kept text"},
    ]
    ck = _MockCheckpointer(messages=msgs)
    adapter = LangGraphAdapter(checkpointer=ck)
    res = adapter.default_memory_recall("")
    assert len(res) == 1
    assert res[0].text == "kept text"


# ---------------------------------------------------------------------------
# default_memory_remember (V1 = no-op)
# ---------------------------------------------------------------------------


def test_remember_is_noop_returns_none() -> None:
    ck = _MockCheckpointer()
    adapter = LangGraphAdapter(checkpointer=ck)
    result = adapter.default_memory_remember(_make_outcome())
    assert result is None
    # Confirms we did NOT touch the checkpointer
    assert ck.last_config is None


def test_remember_does_not_raise_on_arbitrary_outcome() -> None:
    ck = _MockCheckpointer()
    adapter = LangGraphAdapter(checkpointer=ck)
    # Even with a "rich" outcome, remember stays a no-op
    outcome = Outcome(
        query="x" * 10000,
        effort=EffortLevel.MAX,
        iterations=10,
        quality=1.0,
        converged=True,
        tokens_used=99999,
    )
    adapter.default_memory_remember(outcome)


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------


def test_class_metadata() -> None:
    assert LangGraphAdapter.framework_name == "langgraph"
    assert LangGraphAdapter.HOST_NAME == "langgraph"
    assert LangGraphAdapter.version == "1.0.0"


def test_thread_id_default_is_default() -> None:
    ck = _MockCheckpointer(messages=[_MockMessage(content="hello")])
    adapter = LangGraphAdapter(checkpointer=ck)  # no thread_id
    adapter.default_memory_recall("hello")
    assert ck.last_config == {"configurable": {"thread_id": "default"}}


# ---------------------------------------------------------------------------
# — reverse-scan O(matches) instead of O(N)
# ---------------------------------------------------------------------------


def test_recall_reverse_scan_stops_early(monkeypatch: Any) -> None:
    """with limit=2, the adapter must stop scanning as soon as
    2 matches are found from the END of the message list.  Without
    reverse-scan, all N messages are inspected even if the answer is in
    the last 2.

    We assert by counting how many messages are inspected via a tracking
    iterator wrapper.
    """
    from unittest.mock import MagicMock

    from agent_amplifier.adapters.langgraph import LangGraphAdapter

    # 1000 messages, all matching "x"
    msgs = []
    for i in range(1000):
        m = MagicMock()
        m.content = f"x msg-{i}"
        msgs.append(m)

    inspected = {"count": 0}

    class TrackingList(list):
        def __reversed__(self):
            for m in super().__reversed__():
                inspected["count"] += 1
                yield m

    tup = MagicMock()
    tup.checkpoint = {"channel_values": {"messages": TrackingList(msgs)}}
    saver = MagicMock()
    saver.get_tuple = MagicMock(return_value=tup)

    adapter = LangGraphAdapter(checkpointer=saver, thread_id="t1")
    out = adapter.default_memory_recall("x", limit=2)
    assert len(out) == 2
    # We should have inspected at most ~limit messages (allowing some
    # iteration overhead), NEVER all 1000.
    assert inspected["count"] <= 5
