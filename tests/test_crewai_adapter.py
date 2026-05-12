# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for CrewAIAdapter ().

Coverage targets: 100 % line + 100 % branch on
``src/agent_amplifier/adapters/crewai.py``.

Critical: NO real ``crewai`` install required. All Crew/memory interaction
goes through mock objects implementing the documented unified ``Memory``
contract (``.search(query, limit)``, ``.save(value, metadata)``).
"""
from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_amplifier.adapters.crewai import _PER_CHUNK_BYTES, CrewAIAdapter
from agent_amplifier.types import EffortLevel, Outcome, RecalledPattern

# ---------------------------------------------------------------------------
# Mock CrewAI Memory + Crew
# ---------------------------------------------------------------------------


@dataclass
class _MockMemory:
    """Mimics CrewAI unified ``Memory.search()`` / ``.save()`` contract."""

    search_results: list[Any] = field(default_factory=list)
    search_raises: BaseException | None = None
    save_raises: BaseException | None = None
    return_non_iterable: bool = False
    last_search_args: dict[str, Any] = field(default_factory=dict)
    saved: list[dict[str, Any]] = field(default_factory=list)

    def search(self, *, query: str, limit: int = 5, **_: Any) -> Any:
        self.last_search_args = {"query": query, "limit": limit}
        if self.search_raises is not None:
            raise self.search_raises
        if self.return_non_iterable:
            return 12345  # int — not iterable
        return self.search_results

    def save(
        self, *, value: str, metadata: dict[str, Any] | None = None
    ) -> None:
        if self.save_raises is not None:
            raise self.save_raises
        self.saved.append({"value": value, "metadata": metadata or {}})


@dataclass
class _MockCrew:
    """Mimics CrewAI Crew with attached ``.memory``."""

    memory: Any = None


def _make_outcome(query: str = "test", quality: float = 0.5) -> Outcome:
    return Outcome(
        query=query,
        effort=EffortLevel.LOW,
        iterations=2,
        quality=quality,
        converged=True,
        tokens_used=100,
    )


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


def test_detect_true_when_crewai_spec_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object()
    )
    assert CrewAIAdapter.detect() is True


def test_detect_false_when_crewai_spec_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None
    )
    assert CrewAIAdapter.detect() is False


def test_detect_false_when_find_spec_raises_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(name: str) -> Any:
        raise ImportError("blocked")

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert CrewAIAdapter.detect() is False


def test_detect_false_when_find_spec_raises_valueerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(name: str) -> Any:
        raise ValueError("bad name")

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert CrewAIAdapter.detect() is False


# ---------------------------------------------------------------------------
# default_memory_recall — happy paths
# ---------------------------------------------------------------------------


def test_recall_returns_recalledpattern_from_dict_items() -> None:
    mem = _MockMemory(
        search_results=[
            {"memory": "python tip 1", "score": 0.9, "metadata": {"k": "v"}},
            {"memory": "python tip 2", "score": 0.7},
        ]
    )
    crew = _MockCrew(memory=mem)
    adapter = CrewAIAdapter(crew=crew)
    res = adapter.default_memory_recall("python", limit=3)
    assert len(res) == 2
    assert isinstance(res[0], RecalledPattern)
    assert res[0].text == "python tip 1"
    assert res[0].score == pytest.approx(0.9)
    assert res[0].source == "crewai:memory"
    assert "crew-memory" in res[0].tags
    assert res[0].metadata == {"k": "v"}


def test_recall_passes_query_and_limit_to_memory_search() -> None:
    mem = _MockMemory(search_results=[])
    crew = _MockCrew(memory=mem)
    adapter = CrewAIAdapter(crew=crew)
    adapter.default_memory_recall("hello", limit=7)
    assert mem.last_search_args == {"query": "hello", "limit": 7}


def test_recall_extracts_text_from_content_key() -> None:
    mem = _MockMemory(search_results=[{"content": "from content key"}])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].text == "from content key"


def test_recall_extracts_text_from_value_key() -> None:
    mem = _MockMemory(search_results=[{"value": "from value key"}])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].text == "from value key"


def test_recall_extracts_text_from_text_key() -> None:
    mem = _MockMemory(search_results=[{"text": "from text key"}])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].text == "from text key"


def test_recall_extracts_text_from_object_attrs() -> None:
    @dataclass
    class _Item:
        content: str = "obj content"
        score: float = 0.5

    mem = _MockMemory(search_results=[_Item()])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].text == "obj content"
    assert res[0].score == pytest.approx(0.5)


def test_recall_extracts_text_from_object_text_attr() -> None:
    @dataclass
    class _Item:
        text: str = "obj text"

    mem = _MockMemory(search_results=[_Item()])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].text == "obj text"


def test_recall_extracts_text_from_object_memory_attr() -> None:
    @dataclass
    class _Item:
        memory: str = "obj memory"

    mem = _MockMemory(search_results=[_Item()])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].text == "obj memory"


def test_recall_handles_raw_string_items() -> None:
    mem = _MockMemory(search_results=["plain string item"])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].text == "plain string item"


def test_recall_skips_items_with_no_extractable_text() -> None:
    @dataclass
    class _Junk:
        irrelevant: str = "no text-shaped attrs"

    mem = _MockMemory(
        search_results=[
            {"unknown": "key"},  # dict with no recognized text key
            _Junk(),  # object with no recognized text attr
            {"content": "real"},  # this one survives
        ]
    )
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert len(res) == 1
    assert res[0].text == "real"


def test_recall_respects_limit() -> None:
    items = [{"content": f"item-{i}", "score": 0.5} for i in range(10)]
    mem = _MockMemory(search_results=items)
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x", limit=3)
    assert len(res) == 3


def test_recall_caps_chunk_size() -> None:
    huge = "y" * (_PER_CHUNK_BYTES * 4)
    mem = _MockMemory(search_results=[{"content": huge}])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert len(res[0].text) <= _PER_CHUNK_BYTES


# ---------------------------------------------------------------------------
# default_memory_recall — error / fallback paths
# ---------------------------------------------------------------------------


def test_recall_returns_empty_when_crew_has_no_memory_attribute(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Bare:
        pass

    adapter = CrewAIAdapter(crew=_Bare())
    with caplog.at_level(logging.WARNING):
        res = adapter.default_memory_recall("anything")
    assert res == []
    assert any("no .memory attribute" in r.message for r in caplog.records)


def test_recall_returns_empty_when_memory_search_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mem = _MockMemory(search_raises=RuntimeError("backend down"))
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    with caplog.at_level(logging.WARNING):
        res = adapter.default_memory_recall("x")
    assert res == []
    assert any("memory.search failed" in r.message for r in caplog.records)


def test_recall_returns_empty_when_search_returns_non_iterable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mem = _MockMemory(return_non_iterable=True)
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    with caplog.at_level(logging.WARNING):
        res = adapter.default_memory_recall("x")
    assert res == []
    assert any(
        "non-iterable" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Score plucking: clamping + invalid types
# ---------------------------------------------------------------------------


def test_recall_score_default_zero_when_absent() -> None:
    mem = _MockMemory(search_results=[{"content": "no score here"}])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].score == 0.0


def test_recall_score_clamped_below_zero() -> None:
    mem = _MockMemory(search_results=[{"content": "x", "score": -0.5}])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].score == 0.0


def test_recall_score_clamped_above_one() -> None:
    mem = _MockMemory(search_results=[{"content": "x", "score": 1.5}])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].score == 1.0


def test_recall_score_invalid_type_falls_back_to_zero() -> None:
    mem = _MockMemory(search_results=[{"content": "x", "score": "garbage"}])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].score == 0.0


def test_recall_score_object_attr_path() -> None:
    @dataclass
    class _Item:
        content: str = "x"
        score: float = 0.42

    mem = _MockMemory(search_results=[_Item()])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].score == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Metadata plucking
# ---------------------------------------------------------------------------


def test_recall_metadata_default_empty_when_absent() -> None:
    mem = _MockMemory(search_results=[{"content": "x"}])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].metadata == {}


def test_recall_metadata_object_attr_path() -> None:
    @dataclass
    class _Item:
        content: str = "x"
        metadata: dict[str, Any] = field(
            default_factory=lambda: {"src": "obj"}
        )

    mem = _MockMemory(search_results=[_Item()])
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].metadata == {"src": "obj"}


def test_recall_metadata_non_dict_falls_back_to_empty() -> None:
    mem = _MockMemory(
        search_results=[{"content": "x", "metadata": "not a dict"}]
    )
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    res = adapter.default_memory_recall("x")
    assert res[0].metadata == {}


# ---------------------------------------------------------------------------
# default_memory_remember
# ---------------------------------------------------------------------------


def test_remember_calls_memory_save_with_quality_and_iterations() -> None:
    mem = _MockMemory()
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    adapter.default_memory_remember(_make_outcome("hello", 0.7))
    assert len(mem.saved) == 1
    saved = mem.saved[0]
    assert saved["value"] == "hello"
    assert saved["metadata"]["quality"] == pytest.approx(0.7)
    assert saved["metadata"]["iterations"] == 2
    assert saved["metadata"]["effort"] == "low"
    assert saved["metadata"]["converged"] is True
    assert saved["metadata"]["tokens_used"] == 100


def test_remember_swallows_save_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mem = _MockMemory(save_raises=RuntimeError("disk full"))
    adapter = CrewAIAdapter(crew=_MockCrew(memory=mem))
    with caplog.at_level(logging.WARNING):
        adapter.default_memory_remember(_make_outcome())
    assert any("memory.save failed" in r.message for r in caplog.records)


def test_remember_warns_when_no_memory_attr(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Bare:
        pass

    adapter = CrewAIAdapter(crew=_Bare())
    with caplog.at_level(logging.WARNING):
        adapter.default_memory_remember(_make_outcome())
    assert any("no .memory attribute" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------


def test_class_metadata() -> None:
    assert CrewAIAdapter.framework_name == "crewai"
    assert CrewAIAdapter.HOST_NAME == "crewai"
    assert CrewAIAdapter.version == "1.0.0"


# ---------------------------------------------------------------------------
# — bounded materialization via islice
# ---------------------------------------------------------------------------


def test_recall_does_not_fully_materialize_huge_search() -> None:
    """a buggy CrewAI memory.search returning a huge generator
    must not be fully materialized.  We cap at ``limit`` items.
    """
    from unittest.mock import MagicMock

    from agent_amplifier.adapters.crewai import CrewAIAdapter

    yielded = {"n": 0}

    def big_gen():
        for i in range(100_000):
            yielded["n"] += 1
            yield {"text": f"item-{i}", "score": 0.5}

    crew = MagicMock()
    crew.memory.search = MagicMock(return_value=big_gen())
    adapter = CrewAIAdapter(crew=crew)
    out = adapter.default_memory_recall("item", limit=3)
    assert len(out) == 3
    # Capped at limit (+ tiny prefetch slack)
    assert yielded["n"] <= 5
