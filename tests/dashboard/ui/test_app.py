# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Integration tests for the Streamlit app using AppTest."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from agent_amplifier.dashboard.ui.api import DashboardError


def _make_mock_httpx(canned: dict[str, Any]) -> MagicMock:
    """Return a mock httpx.Client that routes requests by URL."""
    def _route_request(
        method: str, url: str, **kwargs: Any
    ) -> MagicMock:
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.text = "{}"

        if "health" in url:
            response.json.return_value = canned["health"]
        elif "/api/config" in url and method == "GET":
            response.json.return_value = canned["config"]
        elif "/api/config" in url and method == "POST":
            response.json.return_value = {"config": kwargs.get("json", {}).get("config", {})}
        elif "ips" in url and "toggle" not in url and "reorder" not in url:
            response.json.return_value = canned["ips"]
        elif "toggle" in url:
            ip_id = url.split("/")[-2]
            ips = list(canned["ips"]["ips"])
            for ip in ips:
                if ip["id"] == ip_id:
                    ip["enabled"] = not ip["enabled"]
            response.json.return_value = {"ip": next(ip for ip in ips if ip["id"] == ip_id)}
        elif "reorder" in url:
            response.json.return_value = canned["ips"]
        elif "telemetry/summary" in url:
            response.json.return_value = canned["telemetry"]
        elif "telemetry/turns" in url:
            response.json.return_value = canned["turns"]
        elif "telemetry/convergence" in url:
            response.json.return_value = canned["convergence"]
        elif "adapters" in url and "install" not in url:
            response.json.return_value = canned["adapters"]
        elif "install" in url:
            name = url.split("/")[-2]
            response.json.return_value = {"name": name, "status": f"installed:{name}"}
        else:
            response.json.return_value = {}
        return response

    mock_client = MagicMock()
    mock_client.request.side_effect = _route_request
    return mock_client


@pytest.fixture
def app_test(canned_responses: dict[str, Any]) -> AppTest:
    """Return an AppTest instance with backend calls mocked."""
    mock_client = _make_mock_httpx(canned_responses)
    with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
        MockHttpx.return_value = mock_client
        at = AppTest.from_file("src/agent_amplifier/dashboard/ui/app.py")
        at.run()
        return at


class TestNavigation:
    def test_default_tab_is_tune(self, app_test: AppTest) -> None:
        assert app_test.radio[0].value == "Tune"
        headers = [h.value for h in app_test.header]
        assert "Amplification Features" in headers

    def test_switch_to_telemetry(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        mock_client = _make_mock_httpx(canned_responses)
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            app_test.radio[0].set_value("Telemetry")
            app_test.run()
        headers = [h.value for h in app_test.header]
        assert "Telemetry" in headers

    def test_switch_to_adapters(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        mock_client = _make_mock_httpx(canned_responses)
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            app_test.radio[0].set_value("Adapters")
            app_test.run()
        headers = [h.value for h in app_test.header]
        assert "Host Adapters" in headers

    def test_switch_to_health(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        mock_client = _make_mock_httpx(canned_responses)
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            app_test.radio[0].set_value("Health")
            app_test.run()
        headers = [h.value for h in app_test.header]
        assert "System Health" in headers


class TestTuneTab:
    def test_shows_ip_lists(self, app_test: AppTest) -> None:
        assert app_test.radio[0].value == "Tune"
        texts = [s.value for s in app_test.subheader]
        assert any("Active" in t for t in texts)
        assert any("Inactive" in t for t in texts)

    def test_apply_changes_button(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        mock_client = _make_mock_httpx(canned_responses)
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            for btn in app_test.button:
                if btn.label == "Apply Changes":
                    btn.click()
                    break
            app_test.run()
        post_calls = [
            c for c in mock_client.request.call_args_list
            if c[0][0] == "POST" and "/api/config" in c[0][1]
        ]
        assert len(post_calls) >= 1

    def test_revert_button(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        mock_client = _make_mock_httpx(canned_responses)
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            for btn in app_test.button:
                if btn.label == "Revert to Defaults":
                    btn.click()
                    break
            app_test.run()
        get_calls = [
            c for c in mock_client.request.call_args_list
            if c[0][0] == "GET" and "/api/config" in c[0][1]
        ]
        assert len(get_calls) >= 1

    def test_backend_error_shows_banner(self) -> None:
        mock_client = MagicMock()
        mock_client.request.side_effect = DashboardError("backend down")
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            at = AppTest.from_file("src/agent_amplifier/dashboard/ui/app.py")
            at.run()
        errors = [e.value for e in at.error]
        assert any("backend down" in str(e) for e in errors)


class TestTelemetryTab:
    def test_metrics_render(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        mock_client = _make_mock_httpx(canned_responses)
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            app_test.radio[0].set_value("Telemetry")
            app_test.run()
        labels = [m.label for m in app_test.metric]
        assert any("Sessions" in str(lab) for lab in labels)
        assert any("Envelopes" in str(lab) for lab in labels)

    def test_charts_render(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        mock_client = _make_mock_httpx(canned_responses)
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            app_test.radio[0].set_value("Telemetry")
            app_test.run()
        assert len(app_test.subheader) >= 2


class TestAdaptersTab:
    def test_table_shows_adapters(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        mock_client = _make_mock_httpx(canned_responses)
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            app_test.radio[0].set_value("Adapters")
            app_test.run()
        texts = [t.value for t in app_test.markdown]
        assert any("Claude Code" in t for t in texts)

    def test_install_button(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        mock_client = _make_mock_httpx(canned_responses)
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            app_test.radio[0].set_value("Adapters")
            app_test.run()
            for btn in app_test.button:
                if btn.label == "Install":
                    btn.click()
                    break
            app_test.run()
        post_calls = [
            c for c in mock_client.request.call_args_list
            if c[0][0] == "POST" and "install" in c[0][1]
        ]
        assert len(post_calls) >= 1


class TestHealthTab:
    def test_doctor_output(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        mock_client = _make_mock_httpx(canned_responses)
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            app_test.radio[0].set_value("Health")
            app_test.run()
        texts = [t.value for t in app_test.subheader]
        assert any("Backend Status" in t for t in texts)

    def test_run_diagnostics_button(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        mock_client = _make_mock_httpx(canned_responses)
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            app_test.radio[0].set_value("Health")
            app_test.run()
            for btn in app_test.button:
                if btn.label == "Run Diagnostics":
                    btn.click()
                    break
            app_test.run()
        assert not app_test.exception


class TestTuneTabErrors:
    def test_get_config_error_shows_banner(self) -> None:
        mock_client = MagicMock()
        mock_client.request.side_effect = DashboardError("config down")
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            at = AppTest.from_file("src/agent_amplifier/dashboard/ui/app.py")
            at.run()
        errors = [e.value for e in at.error]
        assert any("config down" in str(e) for e in errors)

    def test_apply_changes_error(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        calls = []
        def _route(method: str, url: str, **kwargs: Any) -> MagicMock:
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status.return_value = None
            response.text = "{}"
            if "config" in url and method == "POST":
                calls.append((method, url))
                response.json.return_value = {"config": kwargs.get("json", {}).get("config", {})}
            elif "health" in url:
                response.json.return_value = canned_responses["health"]
            elif "config" in url and method == "GET":
                response.json.return_value = canned_responses["config"]
            elif "ips" in url:
                response.json.return_value = canned_responses["ips"]
            else:
                response.json.return_value = {}
            return response
        mc = MagicMock()
        mc.request.side_effect = _route
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mc
            for btn in app_test.button:
                if btn.label == "Apply Changes":
                    btn.click()
                    break
            app_test.run()
        assert len(calls) >= 1


class TestTelemetryTabErrors:
    def test_summary_error(self) -> None:
        mock_client = MagicMock()
        def _route(method: str, url: str, **kwargs: Any) -> MagicMock:
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status.return_value = None
            response.text = "{}"
            if "telemetry/summary" in url:
                raise DashboardError("telemetry down")
            response.json.return_value = {}
            return response
        mock_client.request.side_effect = _route
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            at = AppTest.from_file("src/agent_amplifier/dashboard/ui/app.py")
            at.run()
            at.radio[0].set_value("Telemetry")
            at.run()
        errors = [e.value for e in at.error]
        assert any("telemetry down" in str(e) for e in errors)


class TestAdaptersTabErrors:
    def test_empty_adapters(self) -> None:
        mock_client = MagicMock()
        def _route(method: str, url: str, **kwargs: Any) -> MagicMock:
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status.return_value = None
            response.text = "{}"
            if "adapters" in url:
                response.json.return_value = {"adapters": []}
            elif "health" in url:
                response.json.return_value = {"status": "ok", "amp_version": "1.1.1", "db_path": "/tmp/db"}
            else:
                response.json.return_value = {}
            return response
        mock_client.request.side_effect = _route
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            at = AppTest.from_file("src/agent_amplifier/dashboard/ui/app.py")
            at.run()
            at.radio[0].set_value("Adapters")
            at.run()
        infos = [i.value for i in at.info]
        assert any("No adapters" in str(i) for i in infos)

    def test_get_adapters_error(self) -> None:
        mock_client = MagicMock()
        def _route(method: str, url: str, **kwargs: Any) -> MagicMock:
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status.return_value = None
            response.text = "{}"
            if "adapters" in url:
                raise DashboardError("adapters down")
            elif "health" in url:
                response.json.return_value = {"status": "ok", "amp_version": "1.1.1", "db_path": "/tmp/db"}
            else:
                response.json.return_value = {}
            return response
        mock_client.request.side_effect = _route
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            at = AppTest.from_file("src/agent_amplifier/dashboard/ui/app.py")
            at.run()
            at.radio[0].set_value("Adapters")
            at.run()
        errors = [e.value for e in at.error]
        assert any("adapters down" in str(e) for e in errors)

    def test_install_404_error(self, app_test: AppTest, canned_responses: dict[str, Any]) -> None:
        def _route(method: str, url: str, **kwargs: Any) -> MagicMock:
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status.return_value = None
            response.text = "{}"
            if "install" in url:
                response.status_code = 404
                response.reason_phrase = "Not Found"
                response.json.return_value = {"detail": "not wired"}
                response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "404", request=MagicMock(), response=response
                )
            elif "adapters" in url:
                response.json.return_value = canned_responses["adapters"]
            elif "health" in url:
                response.json.return_value = canned_responses["health"]
            elif "config" in url and method == "GET":
                response.json.return_value = canned_responses["config"]
            elif "ips" in url:
                response.json.return_value = canned_responses["ips"]
            else:
                response.json.return_value = {}
            return response
        mc = MagicMock()
        mc.request.side_effect = _route
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mc
            app_test.radio[0].set_value("Adapters")
            app_test.run()
            for btn in app_test.button:
                if btn.label == "Install":
                    btn.click()
                    break
            app_test.run()
        errors = [e.value for e in app_test.error]
        assert any("not yet available" in str(e) for e in errors)


class TestHealthTabErrors:
    def test_health_error(self) -> None:
        mock_client = MagicMock()
        def _route(method: str, url: str, **kwargs: Any) -> MagicMock:
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status.return_value = None
            response.text = "{}"
            if "health" in url:
                raise DashboardError("health down")
            response.json.return_value = {}
            return response
        mock_client.request.side_effect = _route
        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
            MockHttpx.return_value = mock_client
            at = AppTest.from_file("src/agent_amplifier/dashboard/ui/app.py")
            at.run()
            at.radio[0].set_value("Health")
            at.run()
        errors = [e.value for e in at.error]
        assert any("health down" in str(e) for e in errors)
