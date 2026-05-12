# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for the Tune-tab persona section (load + render + add + remove)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from agent_amplifier.dashboard.ui.api import DashboardError
from agent_amplifier.dashboard.ui.components.tune_tab import (
    _load_personas,
    _render_persona_section,
)


def _make_session_state(**kwargs: Any) -> MagicMock:
    store: dict[str, Any] = dict(kwargs)
    ss = MagicMock()
    ss.__contains__ = lambda self, key: key in store
    ss.__getitem__ = lambda self, key: store[key]
    ss.__setitem__ = lambda self, key, val: store.__setitem__(key, val)
    ss.get = lambda key, default=None: store.get(key, default)
    ss.pop = lambda key, default=None: store.pop(key, default)
    for k, v in kwargs.items():
        setattr(ss, k, v)
    return ss


# ---------------------------------------------------------------------------
# _load_personas — defensive paths
# ---------------------------------------------------------------------------


class TestLoadPersonas:
    def test_happy_path_caches_personas(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = {
            "personas": [{"slug": "x", "label": "X", "custom": False}]
        }
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            result = _load_personas(client)
            assert result == [{"slug": "x", "label": "X", "custom": False}]

    def test_dashboard_error_returns_cached(self) -> None:
        client = MagicMock()
        client.get_personas.side_effect = DashboardError("backend down")
        cached = [{"slug": "cached", "label": "Cached", "custom": False}]
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state(
                amp_personas_cache=cached
            )
            result = _load_personas(client)
            assert result == cached
            mock_st.error.assert_called_once_with("backend down")

    def test_non_dict_response_returns_empty(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = "garbage"  # not a dict
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            assert _load_personas(client) == []

    def test_non_list_personas_returns_empty(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = {"personas": "not a list"}
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            assert _load_personas(client) == []

    def test_non_dict_entries_are_filtered_out(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = {
            "personas": [
                {"slug": "good", "label": "G", "custom": False},
                "garbage entry",
                42,
            ]
        }
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            result = _load_personas(client)
            assert result == [{"slug": "good", "label": "G", "custom": False}]


# ---------------------------------------------------------------------------
# _render_persona_section — full coverage of branches
# ---------------------------------------------------------------------------


def _builtin_entry() -> dict[str, Any]:
    return {
        "slug": "senior-engineer",
        "label": "Senior Engineer (normal mode)",
        "value_tagline": "Catches major correctness bugs cheaply.",
        "when_to_use": "Default for routine code review.",
        "level": 0,
        "role": "Senior software engineer",
        "strictness": 0.6,
        "focus": ["correctness", "logic"],
        "severity_threshold": "high",
        "custom": False,
    }


def _custom_entry() -> dict[str, Any]:
    return {
        "slug": "ml-eng",
        "label": "ML Engineer",
        "value_tagline": "PyTorch reviewer",
        "when_to_use": "Pick when ML code is involved.",
        "level": None,
        "role": "PyTorch reviewer",
        "strictness": None,
        "focus": ["pytorch"],
        "severity_threshold": None,
        "custom": True,
    }


def _ctx_mgr(*_a: Any, **_k: Any) -> MagicMock:
    """Return a fresh mock context manager."""
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    return m


class TestRenderPersonaSection:
    def _patched_st(
        self, mock_st: MagicMock, *, columns_count: int = 1
    ) -> None:
        """Wire common Streamlit primitives to context managers."""
        mock_st.container.return_value = _ctx_mgr()
        mock_st.expander.return_value = _ctx_mgr()
        mock_st.form.return_value = _ctx_mgr()

        # Each `st.columns([5, 1])` call returns a 2-tuple of context managers.
        col_pairs = [(_ctx_mgr(), _ctx_mgr()) for _ in range(columns_count)]
        mock_st.columns.side_effect = col_pairs

    def test_empty_personas_shows_info(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = {"personas": []}
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            _render_persona_section(client, {})
            mock_st.info.assert_called_once()

    def test_renders_builtin_with_value_tagline(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = {"personas": [_builtin_entry()]}
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            self._patched_st(mock_st)
            mock_st.selectbox.return_value = "Senior Engineer (normal mode)"
            mock_st.text_input.return_value = ""
            mock_st.text_area.return_value = ""
            mock_st.form_submit_button.return_value = False
            mock_st.button.return_value = False
            _render_persona_section(
                client, {"persona": "senior-engineer"}
            )
            # The value tagline phrase appears in one of the markdown calls.
            md_calls = " ".join(
                str(c.args[0]) for c in mock_st.markdown.call_args_list
            )
            assert "Catches major correctness" in md_calls

    def test_default_persona_when_config_missing_key(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = {"personas": [_builtin_entry()]}
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            self._patched_st(mock_st)
            mock_st.selectbox.return_value = "Senior Engineer (normal mode)"
            mock_st.text_input.return_value = ""
            mock_st.text_area.return_value = ""
            mock_st.form_submit_button.return_value = False
            mock_st.button.return_value = False
            _render_persona_section(client, {})
            assert mock_st.session_state.amp_persona_select == "senior-engineer"

    def test_config_with_unknown_persona_falls_back_to_first(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = {"personas": [_builtin_entry()]}
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            self._patched_st(mock_st)
            mock_st.selectbox.return_value = "Senior Engineer (normal mode)"
            mock_st.text_input.return_value = ""
            mock_st.text_area.return_value = ""
            mock_st.form_submit_button.return_value = False
            mock_st.button.return_value = False
            _render_persona_section(
                client, {"persona": "ghost-no-such-persona"}
            )
            assert mock_st.session_state.amp_persona_select == "senior-engineer"

    def test_renders_custom_persona_path(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = {
            "personas": [_builtin_entry(), _custom_entry()]
        }
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            self._patched_st(mock_st, columns_count=1)
            mock_st.selectbox.return_value = "ML Engineer"
            mock_st.text_input.return_value = ""
            mock_st.text_area.return_value = ""
            mock_st.form_submit_button.return_value = False
            mock_st.button.return_value = False
            _render_persona_section(client, {"persona": "ml-eng"})
            md_calls = " ".join(
                str(c.args[0]) for c in mock_st.markdown.call_args_list
            )
            assert "custom persona" in md_calls
            assert "ml-eng" in md_calls

    def test_custom_with_no_focus_renders_dash(self) -> None:
        client = MagicMock()
        custom = _custom_entry()
        custom["focus"] = []
        client.get_personas.return_value = {"personas": [custom]}
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            self._patched_st(mock_st, columns_count=1)
            mock_st.selectbox.return_value = custom["label"]
            mock_st.text_input.return_value = ""
            mock_st.text_area.return_value = ""
            mock_st.form_submit_button.return_value = False
            mock_st.button.return_value = False
            _render_persona_section(client, {})
            md_calls = " ".join(
                str(c.args[0]) for c in mock_st.markdown.call_args_list
            )
            assert "Focus axes:** —" in md_calls

    def test_add_form_submit_success_calls_create(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = {"personas": [_builtin_entry()]}
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state(
                amp_personas_cache=[_builtin_entry()]
            )
            self._patched_st(mock_st)
            mock_st.selectbox.return_value = "Senior Engineer (normal mode)"
            mock_st.text_input.side_effect = ["ml-eng", "ML Engineer", "pytorch,ml"]
            mock_st.text_area.return_value = "PyTorch reviewer"
            mock_st.form_submit_button.return_value = True
            mock_st.button.return_value = False
            _render_persona_section(client, {"persona": "senior-engineer"})
            client.create_persona.assert_called_once_with(
                name="ml-eng",
                label="ML Engineer",
                description="PyTorch reviewer",
                review_focus=["pytorch", "ml"],
            )
            mock_st.toast.assert_called()
            mock_st.rerun.assert_called()

    def test_add_form_submit_error_shows_error_banner(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = {"personas": [_builtin_entry()]}
        client.create_persona.side_effect = DashboardError("slug taken")
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            self._patched_st(mock_st)
            mock_st.selectbox.return_value = "Senior Engineer (normal mode)"
            mock_st.text_input.side_effect = ["x", "X", ""]
            mock_st.text_area.return_value = "d"
            mock_st.form_submit_button.return_value = True
            mock_st.button.return_value = False
            _render_persona_section(client, {})
            mock_st.error.assert_called_with("slug taken")

    def test_remove_button_calls_delete(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = {
            "personas": [_builtin_entry(), _custom_entry()]
        }
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state(
                amp_personas_cache=[_builtin_entry(), _custom_entry()]
            )
            self._patched_st(mock_st, columns_count=1)
            mock_st.selectbox.return_value = "Senior Engineer (normal mode)"
            mock_st.text_input.return_value = ""
            mock_st.text_area.return_value = ""
            mock_st.form_submit_button.return_value = False
            mock_st.button.return_value = True  # Remove clicked
            _render_persona_section(client, {"persona": "senior-engineer"})
            client.delete_persona.assert_called_once_with("ml-eng")
            mock_st.toast.assert_called()
            mock_st.rerun.assert_called()

    def test_remove_button_error_shows_banner(self) -> None:
        client = MagicMock()
        client.get_personas.return_value = {
            "personas": [_builtin_entry(), _custom_entry()]
        }
        client.delete_persona.side_effect = DashboardError("delete failed")
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            self._patched_st(mock_st, columns_count=1)
            mock_st.selectbox.return_value = "Senior Engineer (normal mode)"
            mock_st.text_input.return_value = ""
            mock_st.text_area.return_value = ""
            mock_st.form_submit_button.return_value = False
            mock_st.button.return_value = True
            _render_persona_section(client, {"persona": "senior-engineer"})
            mock_st.error.assert_called_with("delete failed")
