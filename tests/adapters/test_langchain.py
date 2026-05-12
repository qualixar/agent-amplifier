# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for the LangChain adapter — BaseMemory + BaseChatMessageHistory."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from agent_amplifier.adapters.langchain import LangChainAdapter
from agent_amplifier.types import RecalledPattern

# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------


def _make_base_memory(
    variables: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock BaseMemory."""
    m = MagicMock()
    m.load_memory_variables = MagicMock(
        return_value=variables or {"history": ""}
    )
    m.save_context = MagicMock()
    # Remove messages attr so it's detected as base_memory
    del m.messages
    return m


def _make_chat_history(
    messages: list[Any] | None = None,
) -> MagicMock:
    """Create a mock BaseChatMessageHistory."""
    m = MagicMock()
    m.messages = messages or []
    m.add_message = MagicMock()
    # Remove load_memory_variables so it's detected as chat_history
    del m.load_memory_variables
    return m


def _make_message(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    return msg


# -----------------------------------------------------------------------
# mode detection
# -----------------------------------------------------------------------


class TestModeDetection:
    def test_detects_base_memory(self) -> None:
        mem = _make_base_memory()
        adapter = LangChainAdapter(memory=mem, kernel=None)
        assert adapter._mode == "base_memory"

    def test_detects_chat_history(self) -> None:
        mem = _make_chat_history()
        adapter = LangChainAdapter(memory=mem, kernel=None)
        assert adapter._mode == "chat_history"

    def test_detects_unknown(self) -> None:
        mem = MagicMock(spec=[])
        adapter = LangChainAdapter(memory=mem, kernel=None)
        assert adapter._mode == "unknown"


# -----------------------------------------------------------------------
# detect() classmethod
# -----------------------------------------------------------------------


class TestDetect:
    def test_detect_langchain_present(self) -> None:
        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.return_value = MagicMock()
            assert LangChainAdapter.detect() is True

    def test_detect_langchain_absent(self) -> None:
        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.return_value = None
            assert LangChainAdapter.detect() is False

    def test_detect_import_error(self) -> None:
        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.side_effect = ImportError("nope")
            assert LangChainAdapter.detect() is False

    def test_detect_value_error(self) -> None:
        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.side_effect = ValueError("bad")
            assert LangChainAdapter.detect() is False

    def test_detect_langchain_core_fallback(self) -> None:
        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.side_effect = [None, MagicMock()]
            assert LangChainAdapter.detect() is True


# -----------------------------------------------------------------------
# class attributes
# -----------------------------------------------------------------------


class TestClassAttrs:
    def test_framework_name(self) -> None:
        assert LangChainAdapter.framework_name == "langchain"

    def test_host_name(self) -> None:
        assert LangChainAdapter.HOST_NAME == "langchain"

    def test_version(self) -> None:
        assert LangChainAdapter.version == "1.0.0"

    def test_repr(self) -> None:
        mem = _make_base_memory()
        adapter = LangChainAdapter(memory=mem, kernel=None)
        r = repr(adapter)
        assert "LangChainAdapter" in r
        assert "langchain" in r


# -----------------------------------------------------------------------
# recall — BaseMemory path
# -----------------------------------------------------------------------


class TestRecallBaseMemory:
    def test_recall_empty(self) -> None:
        mem = _make_base_memory({"history": ""})
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("test")
        assert result == []

    def test_recall_single_chunk(self) -> None:
        mem = _make_base_memory({"history": "User asked about JWT tokens"})
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("jwt")
        assert len(result) == 1
        assert isinstance(result[0], RecalledPattern)
        assert "JWT" in result[0].text
        assert result[0].source == "langchain:memory"
        assert "langchain" in result[0].tags

    def test_recall_multi_chunk_filter(self) -> None:
        text = "First topic about dogs\n\nSecond about cats\n\nThird about dogs again"
        mem = _make_base_memory({"history": text})
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("dogs")
        assert len(result) == 2
        assert "dogs" in result[0].text.lower()
        assert "dogs" in result[1].text.lower()

    def test_recall_empty_query_returns_recent(self) -> None:
        text = "Chunk A\n\nChunk B\n\nChunk C\n\nChunk D"
        mem = _make_base_memory({"history": text})
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("", limit=2)
        assert len(result) == 2

    def test_recall_respects_limit(self) -> None:
        text = "A dogs\n\nB dogs\n\nC dogs\n\nD dogs"
        mem = _make_base_memory({"history": text})
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("dogs", limit=2)
        assert len(result) == 2

    def test_recall_custom_memory_key(self) -> None:
        mem = _make_base_memory({"chat_history": "Custom key content"})
        mem.load_memory_variables.return_value = {"chat_history": "Custom key content"}
        adapter = LangChainAdapter(
            memory=mem, memory_key="chat_history", kernel=None
        )
        result = adapter.default_memory_recall("custom")
        assert len(result) == 1

    def test_recall_non_string_value(self) -> None:
        mem = _make_base_memory()
        mem.load_memory_variables.return_value = {"history": 12345}
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("123")
        assert len(result) == 1

    def test_recall_exception_swallowed(self) -> None:
        mem = _make_base_memory()
        mem.load_memory_variables.side_effect = RuntimeError("boom")
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("test")
        assert result == []

    def test_recall_truncates_long_chunk(self) -> None:
        long_text = "x" * 10000
        mem = _make_base_memory({"history": long_text})
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("")
        assert len(result) == 1
        assert len(result[0].text) == 4096


# -----------------------------------------------------------------------
# recall — BaseChatMessageHistory path
# -----------------------------------------------------------------------


class TestRecallChatHistory:
    def test_recall_empty_messages(self) -> None:
        mem = _make_chat_history([])
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("test")
        assert result == []

    def test_recall_with_messages(self) -> None:
        msgs = [_make_message("Hello"), _make_message("JWT auth setup")]
        mem = _make_chat_history(msgs)
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("jwt")
        assert len(result) == 1
        assert "JWT" in result[0].text

    def test_recall_empty_query_returns_recent(self) -> None:
        msgs = [_make_message(f"msg {i}") for i in range(5)]
        mem = _make_chat_history(msgs)
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("", limit=2)
        assert len(result) == 2

    def test_recall_dict_message(self) -> None:
        mem = _make_chat_history([{"content": "dict-based message"}])
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("dict")
        assert len(result) == 1
        assert "dict-based" in result[0].text

    def test_recall_string_message(self) -> None:
        mem = _make_chat_history(["plain string message"])
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("plain")
        assert len(result) == 1

    def test_recall_list_content_message(self) -> None:
        msg = MagicMock()
        msg.content = [{"text": "block one"}, "block two"]
        mem = _make_chat_history([msg])
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("block")
        assert len(result) == 1
        assert "block one" in result[0].text

    def test_recall_list_content_with_content_key(self) -> None:
        msg = MagicMock()
        msg.content = [{"content": "inner content"}]
        mem = _make_chat_history([msg])
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("inner")
        assert len(result) == 1

    def test_recall_non_content_message_skipped(self) -> None:
        msg = MagicMock(spec=[])
        mem = _make_chat_history([msg])
        adapter = LangChainAdapter(memory=mem, kernel=None)
        result = adapter.default_memory_recall("test")
        assert result == []

    def test_recall_exception_swallowed(self) -> None:
        mem = _make_chat_history([_make_message("ok")])
        adapter = LangChainAdapter(memory=mem, kernel=None)
        # Make messages raise after construction
        type(mem).messages = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        result = adapter.default_memory_recall("test")
        assert result == []


# -----------------------------------------------------------------------
# recall — unknown mode
# -----------------------------------------------------------------------


class TestRecallUnknown:
    def test_recall_unknown_mode(self) -> None:
        mem = MagicMock(spec=[])
        adapter = LangChainAdapter(memory=mem, kernel=None)
        assert adapter._mode == "unknown"
        result = adapter.default_memory_recall("test")
        assert result == []


# -----------------------------------------------------------------------
# remember — BaseMemory path
# -----------------------------------------------------------------------


class TestRememberBaseMemory:
    def test_remember_calls_save_context(self) -> None:
        mem = _make_base_memory()
        adapter = LangChainAdapter(memory=mem, kernel=None)
        outcome = MagicMock()
        outcome.summary = "Turn completed successfully"
        adapter.default_memory_remember(outcome)
        mem.save_context.assert_called_once_with(
            {"input": "Turn completed successfully"}, {"output": ""}
        )

    def test_remember_custom_input_key(self) -> None:
        mem = _make_base_memory()
        adapter = LangChainAdapter(
            memory=mem, input_key="query", kernel=None
        )
        outcome = MagicMock()
        outcome.summary = "done"
        adapter.default_memory_remember(outcome)
        mem.save_context.assert_called_once_with(
            {"query": "done"}, {"output": ""}
        )

    def test_remember_exception_swallowed(self) -> None:
        mem = _make_base_memory()
        mem.save_context.side_effect = RuntimeError("boom")
        adapter = LangChainAdapter(memory=mem, kernel=None)
        outcome = MagicMock()
        outcome.summary = "test"
        adapter.default_memory_remember(outcome)

    def test_remember_no_summary_uses_str(self) -> None:
        mem = _make_base_memory()
        adapter = LangChainAdapter(memory=mem, kernel=None)

        class NoSummary:
            def __str__(self) -> str:
                return "fallback-str"

        adapter.default_memory_remember(NoSummary())
        mem.save_context.assert_called_once()
        call_args = mem.save_context.call_args[0]
        assert "fallback-str" in str(call_args)


# -----------------------------------------------------------------------
# remember — BaseChatMessageHistory path
# -----------------------------------------------------------------------


class TestRememberChatHistory:
    def test_remember_with_langchain_core(self) -> None:
        mem = _make_chat_history()
        adapter = LangChainAdapter(memory=mem, kernel=None)
        outcome = MagicMock()
        outcome.summary = "outcome text"

        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.return_value = MagicMock()
            with patch(
                "agent_amplifier.adapters.langchain.LangChainAdapter._remember_to_chat_history"
            ) as mock_remember:
                adapter.default_memory_remember(outcome)
                mock_remember.assert_called_once_with("outcome text")

    def test_remember_fallback_add_user_message(self) -> None:
        mem = _make_chat_history()
        del mem.add_message
        mem.add_user_message = MagicMock()
        adapter = LangChainAdapter(memory=mem, kernel=None)

        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.return_value = None
            adapter._remember_to_chat_history("test summary")
            mem.add_user_message.assert_called_once_with("test summary")

    def test_remember_fallback_add_message_string(self) -> None:
        mem = _make_chat_history()
        del mem.add_user_message
        adapter = LangChainAdapter(memory=mem, kernel=None)

        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.return_value = None
            adapter._remember_to_chat_history("test summary")
            mem.add_message.assert_called_once_with("test summary")

    def test_remember_no_method_logs_warning(self) -> None:
        mem = _make_chat_history()
        del mem.add_message
        del mem.add_user_message
        adapter = LangChainAdapter(memory=mem, kernel=None)

        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.return_value = None
            adapter._remember_to_chat_history("test")

    def test_remember_exception_in_langchain_core_import(self) -> None:
        mem = _make_chat_history()
        del mem.add_user_message
        adapter = LangChainAdapter(memory=mem, kernel=None)

        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.side_effect = ImportError("no langchain_core")
            adapter._remember_to_chat_history("test")
            mem.add_message.assert_called_once_with("test")

    def test_remember_exception_swallowed(self) -> None:
        mem = _make_chat_history()
        mem.add_message.side_effect = RuntimeError("boom")
        adapter = LangChainAdapter(memory=mem, kernel=None)
        outcome = MagicMock()
        outcome.summary = "test"

        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.return_value = None
            adapter.default_memory_remember(outcome)


# -----------------------------------------------------------------------
# remember — unknown mode
# -----------------------------------------------------------------------


class TestRememberUnknown:
    def test_remember_unknown_mode(self) -> None:
        mem = MagicMock(spec=[])
        adapter = LangChainAdapter(memory=mem, kernel=None)
        outcome = MagicMock()
        outcome.summary = "test"
        adapter.default_memory_remember(outcome)


# -----------------------------------------------------------------------
# stringify_message
# -----------------------------------------------------------------------


class TestStringifyMessage:
    def test_string_content(self) -> None:
        msg = _make_message("hello")
        assert LangChainAdapter._stringify_message(msg) == "hello"

    def test_list_content_strings(self) -> None:
        msg = MagicMock()
        msg.content = ["part1", "part2"]
        assert LangChainAdapter._stringify_message(msg) == "part1\npart2"

    def test_list_content_dicts_text(self) -> None:
        msg = MagicMock()
        msg.content = [{"text": "via text key"}]
        assert "via text key" in LangChainAdapter._stringify_message(msg)

    def test_list_content_dicts_content(self) -> None:
        msg = MagicMock()
        msg.content = [{"content": "via content key"}]
        assert "via content key" in LangChainAdapter._stringify_message(msg)

    def test_list_content_empty_dicts(self) -> None:
        msg = MagicMock()
        msg.content = [{"unrelated": "key"}]
        assert LangChainAdapter._stringify_message(msg) == ""

    def test_dict_message(self) -> None:
        assert LangChainAdapter._stringify_message({"content": "hi"}) == "hi"

    def test_plain_string(self) -> None:
        assert LangChainAdapter._stringify_message("raw string") == "raw string"

    def test_empty_content(self) -> None:
        msg = MagicMock()
        msg.content = None
        assert LangChainAdapter._stringify_message(msg) == ""

    def test_numeric_content(self) -> None:
        msg = MagicMock()
        msg.content = 42
        assert LangChainAdapter._stringify_message(msg) == ""

    def test_list_content_non_str_non_dict_skipped(self) -> None:
        msg = MagicMock()
        msg.content = [42, None, True]
        assert LangChainAdapter._stringify_message(msg) == ""

    def test_dict_message_non_string_content(self) -> None:
        assert LangChainAdapter._stringify_message({"content": 123}) == ""


# -----------------------------------------------------------------------
# dashboard adapter spec wiring
# -----------------------------------------------------------------------


class TestDashboardWiring:
    def test_langchain_in_adapter_specs(self) -> None:
        from agent_amplifier.dashboard.backend.adapters import ADAPTER_SPECS

        names = [s.name for s in ADAPTER_SPECS]
        assert "langchain" in names

    def test_langchain_factory_not_none(self) -> None:
        from agent_amplifier.dashboard.backend.adapters import ADAPTER_SPECS

        spec = next(s for s in ADAPTER_SPECS if s.name == "langchain")
        assert spec.factory is not None
        adapter = spec.factory()
        assert isinstance(adapter, LangChainAdapter)

    def test_langchain_in_all_exports(self) -> None:
        from agent_amplifier.adapters import LangChainAdapter as Exported

        assert Exported is LangChainAdapter
