# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Unit tests for UI tab components (isolated from Streamlit runtime)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from agent_amplifier.dashboard.ui.api import DashboardError
from agent_amplifier.dashboard.ui.components.adapters_tab import (
    _status_badge,
    render_adapters_tab,
)
from agent_amplifier.dashboard.ui.components.health_tab import _run_doctor
from agent_amplifier.dashboard.ui.components.telemetry_tab import _fmt_pct, _fmt_tokens
from agent_amplifier.dashboard.ui.components.tune_tab import (
    _apply_changes,
    _revert_changes,
    _sync_from_backend,
)


def _make_session_state(**kwargs: Any) -> MagicMock:
    store: dict[str, Any] = dict(kwargs)
    ss = MagicMock()
    ss.__contains__ = lambda self, key: key in store
    ss.__getitem__ = lambda self, key: store[key]
    ss.__setitem__ = lambda self, key, val: store.__setitem__(key, val)
    ss.get = lambda key, default=None: store.get(key, default)
    for k, v in kwargs.items():
        setattr(ss, k, v)
    return ss


class TestTuneTabHelpers:
    def test_sync_from_backend_success(self) -> None:
        client = MagicMock()
        client.get_ips.return_value = {"ips": [{"id": "x", "name": "X"}]}
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            result = _sync_from_backend(client)
            assert result == [{"id": "x", "name": "X"}]

    def test_sync_from_backend_error(self) -> None:
        client = MagicMock()
        client.get_ips.side_effect = DashboardError("fail")
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            result = _sync_from_backend(client)
            assert result == []
            mock_st.error.assert_called_once_with("fail")

    def test_apply_changes_success(self) -> None:
        client = MagicMock()
        client.save_config.return_value = {"config": {"max_iterations": 6}}
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state(
                amp_config_cache={"max_iterations": 4},
                amp_max_iterations=6,
                amp_token_budget=1000,
                amp_persona_select="Senior Engineer",
            )
            _apply_changes(client)
            client.save_config.assert_called_once()
            mock_st.toast.assert_called_once()

    def test_apply_changes_error(self) -> None:
        client = MagicMock()
        client.save_config.side_effect = DashboardError("save fail")
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state(
                amp_config_cache={},
                amp_max_iterations=4,
                amp_token_budget=1000,
                amp_persona_select="Senior Engineer",
            )
            _apply_changes(client)
            mock_st.error.assert_called_once_with("save fail")

    def test_revert_changes_success(self) -> None:
        client = MagicMock()
        client.get_config.return_value = {"config": {"x": 1}}
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            _revert_changes(client)
            mock_st.toast.assert_called_once()

    def test_revert_changes_error(self) -> None:
        client = MagicMock()
        client.get_config.side_effect = DashboardError("get fail")
        with patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st:
            mock_st.session_state = _make_session_state()
            _revert_changes(client)
            mock_st.error.assert_called_once_with("get fail")

    def test_render_tune_tab_with_reorder(self) -> None:
        from agent_amplifier.dashboard.ui.components.tune_tab import render_tune_tab
        client = MagicMock()
        client.get_ips.return_value = {"ips": [
            {"id": "a", "name": "A", "enabled": True, "order": 1},
            {"id": "b", "name": "B", "enabled": False, "order": 2},
        ]}
        client.get_config.return_value = {"config": {"max_iterations": 4, "token_budget": 250000, "persona": "senior_engineer"}}

        with (
            patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st,
            patch("agent_amplifier.dashboard.ui.components.tune_tab.sort_items") as mock_sort,
        ):
            col_mock = MagicMock()
            mock_st.session_state = _make_session_state()
            mock_st.columns.side_effect = [
                [col_mock, col_mock],      # main 2-col layout
                [col_mock, col_mock, col_mock],  # 3-col toggles
                [col_mock, col_mock],      # apply/revert buttons
            ]
            mock_st.divider.return_value = None
            mock_sort.side_effect = [
                ["B", "A"],  # active list reorder
                ["B"],       # inactive list (no change)
            ]
            render_tune_tab(client)
            mock_st.session_state = _make_session_state(
                amp_ips_cache=[
                    {"id": "a", "name": "A", "enabled": True, "order": 1},
                    {"id": "b", "name": "B", "enabled": False, "order": 2},
                ],
                amp_config_cache={"max_iterations": 4, "token_budget": 250000, "persona": "senior_engineer"},
            )
            mock_st.columns.side_effect = [
                [col_mock, col_mock],
                [col_mock, col_mock, col_mock],
                [col_mock, col_mock],
            ]
            mock_sort.side_effect = [
                ["B", "A"],  # reordered
                ["B"],
            ]
            render_tune_tab(client)
            client.reorder_ips.assert_called()

    def test_render_tune_tab_toggle(self) -> None:
        from agent_amplifier.dashboard.ui.components.tune_tab import render_tune_tab
        client = MagicMock()
        client.get_ips.return_value = {"ips": [
            {"id": "a", "name": "A", "enabled": True, "order": 1},
        ]}
        client.get_config.return_value = {"config": {"max_iterations": 4}}

        with (
            patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st,
            patch("agent_amplifier.dashboard.ui.components.tune_tab.sort_items") as mock_sort,
        ):
            mock_sort.return_value = ["A"]
            col_mock = MagicMock()
            mock_st.session_state = _make_session_state(
                amp_ips_cache=[{"id": "a", "name": "A", "enabled": True, "order": 1}],
                amp_config_cache={"max_iterations": 4},
            )
            mock_st.columns.side_effect = [
                [col_mock, col_mock],      # main 2-col layout
                [col_mock, col_mock, col_mock],  # 3-col toggles
                [col_mock, col_mock],      # apply/revert buttons
            ]
            mock_st.divider.return_value = None
            mock_st.toggle.return_value = False  # toggled off
            render_tune_tab(client)
            client.toggle_ip.assert_called_once_with("a")


    def test_render_tune_tab_reorder_error(self) -> None:
        from agent_amplifier.dashboard.ui.components.tune_tab import render_tune_tab
        client = MagicMock()
        client.get_ips.return_value = {"ips": [
            {"id": "a", "name": "A", "enabled": True, "order": 1},
        ]}
        client.get_config.return_value = {"config": {"max_iterations": 4}}
        client.reorder_ips.side_effect = DashboardError("reorder fail")

        with (
            patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st,
            patch("agent_amplifier.dashboard.ui.components.tune_tab.sort_items") as mock_sort,
        ):
            col_mock = MagicMock()
            mock_st.session_state = _make_session_state(
                amp_ips_cache=[{"id": "a", "name": "A", "enabled": True, "order": 1}],
                amp_config_cache={"max_iterations": 4},
            )
            mock_st.columns.side_effect = [
                [col_mock, col_mock],
                [col_mock, col_mock, col_mock],
                [col_mock, col_mock],
            ]
            mock_st.divider.return_value = None
            mock_sort.return_value = ["X"]  # changed from original
            render_tune_tab(client)
            mock_st.error.assert_called()

    def test_render_tune_tab_inactive_reorder(self) -> None:
        from agent_amplifier.dashboard.ui.components.tune_tab import render_tune_tab
        client = MagicMock()
        client.get_ips.return_value = {"ips": [
            {"id": "a", "name": "A", "enabled": True, "order": 1},
            {"id": "b", "name": "B", "enabled": False, "order": 2},
            {"id": "c", "name": "C", "enabled": False, "order": 3},
        ]}
        client.get_config.return_value = {"config": {"max_iterations": 4}}

        with (
            patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st,
            patch("agent_amplifier.dashboard.ui.components.tune_tab.sort_items") as mock_sort,
        ):
            col_mock = MagicMock()
            mock_st.session_state = _make_session_state(
                amp_ips_cache=[
                    {"id": "a", "name": "A", "enabled": True, "order": 1},
                    {"id": "b", "name": "B", "enabled": False, "order": 2},
                    {"id": "c", "name": "C", "enabled": False, "order": 3},
                ],
                amp_config_cache={"max_iterations": 4},
            )
            mock_st.columns.side_effect = [
                [col_mock, col_mock],
                [col_mock, col_mock, col_mock],
                [col_mock, col_mock],
            ]
            mock_st.divider.return_value = None
            mock_sort.side_effect = [
                ["A"],           # active unchanged
                ["C", "B"],      # inactive reordered
            ]
            render_tune_tab(client)
            client.reorder_ips.assert_called_once()

    def test_render_tune_tab_inactive_reorder_error(self) -> None:
        from agent_amplifier.dashboard.ui.components.tune_tab import render_tune_tab
        client = MagicMock()
        client.get_ips.return_value = {"ips": [
            {"id": "a", "name": "A", "enabled": True, "order": 1},
            {"id": "b", "name": "B", "enabled": False, "order": 2},
            {"id": "c", "name": "C", "enabled": False, "order": 3},
        ]}
        client.get_config.return_value = {"config": {"max_iterations": 4}}
        client.reorder_ips.side_effect = DashboardError("reorder inactive fail")

        with (
            patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st,
            patch("agent_amplifier.dashboard.ui.components.tune_tab.sort_items") as mock_sort,
        ):
            col_mock = MagicMock()
            mock_st.session_state = _make_session_state(
                amp_ips_cache=[
                    {"id": "a", "name": "A", "enabled": True, "order": 1},
                    {"id": "b", "name": "B", "enabled": False, "order": 2},
                    {"id": "c", "name": "C", "enabled": False, "order": 3},
                ],
                amp_config_cache={"max_iterations": 4},
            )
            mock_st.columns.side_effect = [
                [col_mock, col_mock],
                [col_mock, col_mock, col_mock],
                [col_mock, col_mock],
            ]
            mock_st.divider.return_value = None
            mock_sort.side_effect = [
                ["A"],           # active unchanged
                ["C", "B"],      # inactive reordered
            ]
            render_tune_tab(client)
            mock_st.error.assert_called()

    def test_render_tune_tab_toggle_error(self) -> None:
        from agent_amplifier.dashboard.ui.components.tune_tab import render_tune_tab
        client = MagicMock()
        client.get_ips.return_value = {"ips": [
            {"id": "a", "name": "A", "enabled": True, "order": 1},
        ]}
        client.get_config.return_value = {"config": {"max_iterations": 4}}
        client.toggle_ip.side_effect = DashboardError("toggle fail")

        with (
            patch("agent_amplifier.dashboard.ui.components.tune_tab.st") as mock_st,
            patch("agent_amplifier.dashboard.ui.components.tune_tab.sort_items") as mock_sort,
        ):
            mock_sort.return_value = ["A"]
            col_mock = MagicMock()
            mock_st.session_state = _make_session_state(
                amp_ips_cache=[{"id": "a", "name": "A", "enabled": True, "order": 1}],
                amp_config_cache={"max_iterations": 4},
            )
            mock_st.columns.side_effect = [
                [col_mock, col_mock],
                [col_mock, col_mock, col_mock],
                [col_mock, col_mock],
            ]
            mock_st.divider.return_value = None
            mock_st.toggle.return_value = False
            render_tune_tab(client)
            mock_st.error.assert_called()


class TestAdaptersTabHelpers:
    def test_status_badge_active(self) -> None:
        assert _status_badge({"detected": True, "installed": True}) == "ACTIVE"

    def test_status_badge_ready(self) -> None:
        assert _status_badge({"detected": True, "installed": False}) == "READY"

    def test_status_badge_soon(self) -> None:
        assert _status_badge({"detected": False, "installed": False}) == "SOON"

    def test_render_adapters_tab_empty(self) -> None:
        client = MagicMock()
        client.get_adapters.return_value = {"adapters": []}
        with patch("agent_amplifier.dashboard.ui.components.adapters_tab.st") as mock_st:
            render_adapters_tab(client)
            mock_st.info.assert_called_once_with("No adapters registered")

    def test_render_adapters_tab_error(self) -> None:
        client = MagicMock()
        client.get_adapters.side_effect = DashboardError("fail")
        with patch("agent_amplifier.dashboard.ui.components.adapters_tab.st") as mock_st:
            render_adapters_tab(client)
            mock_st.error.assert_called_once_with("fail")

    def test_render_adapters_tab_install_error_404(self) -> None:
        from agent_amplifier.dashboard.ui.components.adapters_tab import render_adapters_tab
        client = MagicMock()
        client.get_adapters.return_value = {"adapters": [
            {"name": "langchain", "display_name": "LangChain", "detected": True, "installed": False}
        ]}
        exc = DashboardError("not found")
        exc.status_code = 404
        client.install_adapter.side_effect = exc

        with patch("agent_amplifier.dashboard.ui.components.adapters_tab.st") as mock_st:
            mock_st.button.return_value = True
            render_adapters_tab(client)
            mock_st.error.assert_called_once()
            assert "not yet available" in str(mock_st.error.call_args)

    def test_render_adapters_tab_install_error_non_404(self) -> None:
        from agent_amplifier.dashboard.ui.components.adapters_tab import render_adapters_tab
        client = MagicMock()
        client.get_adapters.return_value = {"adapters": [
            {"name": "langchain", "display_name": "LangChain", "detected": True, "installed": False}
        ]}
        exc = DashboardError("server error")
        exc.status_code = 500
        client.install_adapter.side_effect = exc

        with patch("agent_amplifier.dashboard.ui.components.adapters_tab.st") as mock_st:
            mock_st.button.return_value = True
            render_adapters_tab(client)
            mock_st.error.assert_called_once_with("server error")

    def test_render_adapters_tab_soon_disabled(self) -> None:
        from agent_amplifier.dashboard.ui.components.adapters_tab import (
            _action_label,
            render_adapters_tab,
        )
        assert _action_label("SOON") == "Notify Me"
        client = MagicMock()
        client.get_adapters.return_value = {"adapters": [
            {"name": "sk", "display_name": "Semantic Kernel", "detected": False, "installed": False}
        ]}
        with patch("agent_amplifier.dashboard.ui.components.adapters_tab.st") as mock_st:
            mock_st.button.return_value = False
            render_adapters_tab(client)
            mock_st.button.assert_called()


class TestHealthTabHelpers:
    def test_run_doctor_success(self) -> None:
        with patch("agent_amplifier.dashboard.ui.components.health_tab.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="all ok\n", stderr="", returncode=0)
            result = _run_doctor()
            assert "all ok" in result

    def test_run_doctor_file_not_found(self) -> None:
        with patch("agent_amplifier.dashboard.ui.components.health_tab.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            result = _run_doctor()
            assert "not found in PATH" in result

    def test_run_doctor_timeout(self) -> None:
        with patch("agent_amplifier.dashboard.ui.components.health_tab.subprocess.run") as mock_run:
            from subprocess import TimeoutExpired
            mock_run.side_effect = TimeoutExpired("cmd", 15)
            result = _run_doctor()
            assert "timed out" in result

    def test_run_doctor_generic_exception(self) -> None:
        with patch("agent_amplifier.dashboard.ui.components.health_tab.subprocess.run") as mock_run:
            mock_run.side_effect = RuntimeError("boom")
            result = _run_doctor()
            assert "Failed to run" in result

    def test_render_health_tab_doctor_subprocess(self) -> None:
        from agent_amplifier.dashboard.ui.components.health_tab import render_health_tab
        client = MagicMock()
        client.health.return_value = {"status": "ok"}
        client.telemetry_summary.return_value = {"coverage_rate": 0.9}
        with (
            patch("agent_amplifier.dashboard.ui.components.health_tab.st") as mock_st,
            patch("agent_amplifier.dashboard.ui.components.health_tab._run_doctor") as mock_doc,
        ):
            col_mock = MagicMock()
            mock_st.session_state = _make_session_state(amp_doctor_output="doctor output")
            mock_st.columns.return_value = [col_mock, col_mock]
            mock_doc.return_value = "doctor output"
            render_health_tab(client)
            mock_st.code.assert_called_once_with("doctor output", language="text")

    def test_render_health_tab_telemetry_error(self) -> None:
        from agent_amplifier.dashboard.ui.components.health_tab import render_health_tab
        client = MagicMock()
        client.health.return_value = {"status": "ok"}
        client.telemetry_summary.side_effect = DashboardError("telemetry down")
        with (
            patch("agent_amplifier.dashboard.ui.components.health_tab.st") as mock_st,
            patch("agent_amplifier.dashboard.ui.components.health_tab._run_doctor") as mock_doc,
        ):
            col_mock = MagicMock()
            mock_st.session_state = _make_session_state()
            mock_st.columns.return_value = [col_mock, col_mock]
            mock_doc.return_value = "ok"
            render_health_tab(client)

    def test_render_health_tab_health_error(self) -> None:
        from agent_amplifier.dashboard.ui.components.health_tab import render_health_tab
        client = MagicMock()
        client.health.side_effect = DashboardError("down")
        client.telemetry_summary.return_value = {"coverage_rate": 0.0}
        with patch("agent_amplifier.dashboard.ui.components.health_tab.st") as mock_st:
            col_mock = MagicMock()
            mock_st.session_state = _make_session_state()
            mock_st.columns.return_value = [col_mock, col_mock]
            render_health_tab(client)
            mock_st.error.assert_called_once_with("down")


class TestTelemetryTabHelpers:
    def test_fmt_pct(self) -> None:
        assert _fmt_pct(0.5) == "50.0%"
        assert _fmt_pct(1.0) == "100.0%"

    def test_fmt_tokens(self) -> None:
        assert _fmt_tokens(500) == "500"
        assert _fmt_tokens(1500) == "1.5k"
        assert _fmt_tokens(2_000_000) == "2.0M"

    def test_render_telemetry_summary_error(self) -> None:
        from agent_amplifier.dashboard.ui.components.telemetry_tab import render_telemetry_tab
        client = MagicMock()
        client.telemetry_summary.side_effect = DashboardError("fail")
        with patch("agent_amplifier.dashboard.ui.components.telemetry_tab.st") as mock_st:
            render_telemetry_tab(client)
            mock_st.error.assert_called_once_with("fail")

    def test_render_telemetry_convergence_error(self) -> None:
        from agent_amplifier.dashboard.ui.components.telemetry_tab import render_telemetry_tab
        client = MagicMock()
        client.telemetry_summary.return_value = {"counts": {}, "coverage_rate": 0, "convergence_rate": 0}
        client.convergence.side_effect = DashboardError("conv fail")
        with patch("agent_amplifier.dashboard.ui.components.telemetry_tab.st") as mock_st:
            mock_st.columns.side_effect = [
                [mock_st, mock_st, mock_st, mock_st],  # metrics
                [mock_st, mock_st],                    # charts
            ]
            render_telemetry_tab(client)
            mock_st.error.assert_called_once_with("conv fail")

    def test_render_telemetry_turns_error(self) -> None:
        from agent_amplifier.dashboard.ui.components.telemetry_tab import render_telemetry_tab
        client = MagicMock()
        client.telemetry_summary.return_value = {"counts": {}, "coverage_rate": 0, "convergence_rate": 0}
        client.convergence.return_value = {"points": []}
        client.turns.side_effect = DashboardError("turns fail")
        with patch("agent_amplifier.dashboard.ui.components.telemetry_tab.st") as mock_st:
            mock_st.columns.side_effect = [
                [mock_st, mock_st, mock_st, mock_st],  # metrics
                [mock_st, mock_st],                    # charts
            ]
            render_telemetry_tab(client)
            mock_st.error.assert_called_once_with("turns fail")

    def test_render_telemetry_no_data(self) -> None:
        from agent_amplifier.dashboard.ui.components.telemetry_tab import render_telemetry_tab
        client = MagicMock()
        client.telemetry_summary.return_value = {"counts": {}, "coverage_rate": 0, "convergence_rate": 0}
        client.convergence.return_value = {"points": []}
        client.turns.return_value = {"turns": []}
        with patch("agent_amplifier.dashboard.ui.components.telemetry_tab.st") as mock_st:
            mock_st.columns.side_effect = [
                [mock_st, mock_st, mock_st, mock_st],  # metrics
                [mock_st, mock_st],                    # charts
            ]
            render_telemetry_tab(client)
            infos = [c for c in mock_st.method_calls if c[0] == "info"]
            assert len(infos) >= 2  # no convergence data + no turn data + no recent sessions
