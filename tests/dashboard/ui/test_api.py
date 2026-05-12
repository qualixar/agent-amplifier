# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Unit tests for the dashboard UI API client."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent_amplifier.dashboard.ui.api import (
    DashboardClient,
    DashboardError,
    get_backend_url,
)


class TestGetBackendUrl:
    def test_default_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AGENT_AMP_DASHBOARD_PORT", raising=False)
        assert get_backend_url() == "http://127.0.0.1:8765"

    def test_custom_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_AMP_DASHBOARD_PORT", "9999")
        assert get_backend_url() == "http://127.0.0.1:9999"

    def test_non_integer_port_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_AMP_DASHBOARD_PORT", "abc")
        with pytest.raises(DashboardError, match="must be an integer"):
            get_backend_url()

    def test_out_of_range_port_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_AMP_DASHBOARD_PORT", "70000")
        with pytest.raises(DashboardError, match="between 1 and 65535"):
            get_backend_url()


class TestDashboardClient:
    @pytest.fixture(autouse=True)
    def _patch_httpx(self) -> None:
        self.mock_response = MagicMock()
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {"ok": True}
        self.mock_response.raise_for_status.return_value = None

        self.mock_http_client = MagicMock()
        self.mock_http_client.request.return_value = self.mock_response

        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockClient:
            MockClient.return_value = self.mock_http_client
            yield

    def _make_client(self) -> DashboardClient:
        return DashboardClient(base_url="http://test")

    def _assert_request(self, method: str, url: str, **kwargs: Any) -> None:
        defaults: dict[str, Any] = {"timeout": 10.0}
        defaults.update(kwargs)
        self.mock_http_client.request.assert_called_with(method, url, **defaults)

    def test_health(self) -> None:
        client = self._make_client()
        result = client.health()
        assert result == {"ok": True}
        self._assert_request("GET", "http://test/api/health", json=None, params=None)

    def test_get_config(self) -> None:
        client = self._make_client()
        result = client.get_config()
        assert result == {"ok": True}
        self._assert_request("GET", "http://test/api/config", json=None, params=None)

    def test_save_config(self) -> None:
        self.mock_response.json.return_value = {"config": {"x": 1}}
        client = self._make_client()
        result = client.save_config({"x": 1})
        assert result == {"config": {"x": 1}}
        self._assert_request(
            "POST",
            "http://test/api/config",
            json={"config": {"x": 1}},
            params=None,
        )

    def test_get_ips(self) -> None:
        client = self._make_client()
        result = client.get_ips()
        assert result == {"ok": True}
        self._assert_request("GET", "http://test/api/ips", json=None, params=None)

    def test_toggle_ip(self) -> None:
        client = self._make_client()
        result = client.toggle_ip("kernel")
        assert result == {"ok": True}
        self._assert_request("POST", "http://test/api/ips/kernel/toggle", json=None, params=None)

    def test_reorder_ips(self) -> None:
        client = self._make_client()
        result = client.reorder_ips(["a", "b"])
        assert result == {"ok": True}
        self._assert_request(
            "POST",
            "http://test/api/ips/reorder",
            json={"ip_ids": ["a", "b"]},
            params=None,
        )

    def test_telemetry_summary(self) -> None:
        client = self._make_client()
        result = client.telemetry_summary()
        assert result == {"ok": True}
        self._assert_request("GET", "http://test/api/telemetry/summary", json=None, params=None)

    def test_turns(self) -> None:
        client = self._make_client()
        result = client.turns(limit=10)
        assert result == {"ok": True}
        self._assert_request(
            "GET",
            "http://test/api/telemetry/turns",
            params={"limit": 10},
            json=None,
        )

    def test_convergence(self) -> None:
        client = self._make_client()
        result = client.convergence(days=14)
        assert result == {"ok": True}
        self._assert_request(
            "GET",
            "http://test/api/telemetry/convergence",
            params={"days": 14},
            json=None,
        )

    def test_get_adapters(self) -> None:
        client = self._make_client()
        result = client.get_adapters()
        assert result == {"ok": True}
        self._assert_request("GET", "http://test/api/adapters", json=None, params=None)

    def test_install_adapter(self) -> None:
        client = self._make_client()
        result = client.install_adapter("cursor")
        assert result == {"ok": True}
        self._assert_request(
            "POST",
            "http://test/api/adapters/cursor/install",
            json=None,
            params=None,
        )

    def test_get_personas(self) -> None:
        client = self._make_client()
        result = client.get_personas()
        assert result == {"ok": True}
        self._assert_request(
            "GET", "http://test/api/personas", json=None, params=None
        )

    def test_create_persona(self) -> None:
        client = self._make_client()
        result = client.create_persona(
            name="ml-eng",
            label="ML Engineer",
            description="PyTorch reviewer",
            review_focus=["pytorch"],
        )
        assert result == {"ok": True}
        self._assert_request(
            "POST",
            "http://test/api/personas",
            json={
                "name": "ml-eng",
                "label": "ML Engineer",
                "description": "PyTorch reviewer",
                "review_focus": ["pytorch"],
            },
            params=None,
        )

    def test_delete_persona_204_returns_none(self) -> None:
        self.mock_response.status_code = 204
        client = self._make_client()
        assert client.delete_persona("ml-eng") is None

    def test_delete_persona_404_raises(self) -> None:
        self.mock_response.status_code = 404
        self.mock_response.json.return_value = {"detail": "persona not found: ghost"}
        client = self._make_client()
        with pytest.raises(DashboardError, match="persona not found"):
            client.delete_persona("ghost")

    def test_delete_persona_connection_error_raises(self) -> None:
        self.mock_http_client.request.side_effect = httpx.ConnectError("down")
        client = self._make_client()
        with pytest.raises(DashboardError, match="unreachable"):
            client.delete_persona("ml-eng")

    def test_delete_persona_403_uses_detail_message(self) -> None:
        self.mock_response.status_code = 403
        self.mock_response.json.return_value = {
            "detail": "cannot remove built-in persona: senior-engineer"
        }
        client = self._make_client()
        with pytest.raises(DashboardError, match="cannot remove built-in"):
            client.delete_persona("senior-engineer")

    def test_delete_persona_non_json_response_falls_back_to_text(self) -> None:
        self.mock_response.status_code = 500
        self.mock_response.json.side_effect = ValueError("not json")
        self.mock_response.text = "server boom"
        self.mock_response.reason_phrase = "Internal Server Error"
        client = self._make_client()
        with pytest.raises(DashboardError, match="server boom"):
            client.delete_persona("ml-eng")

    def test_context_manager(self) -> None:
        client = self._make_client()
        # Ensure client is created first so close() hits the mock
        client.health()
        with client as c:
            assert c is client
        self.mock_http_client.close.assert_called_once()

    def test_close_idempotent(self) -> None:
        client = self._make_client()
        client.health()  # force client creation
        client.close()
        client.close()
        self.mock_http_client.close.assert_called_once()


class TestRetryLogic:
    def test_connect_error_retries_then_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status.return_value = None

        mock_http_client = MagicMock()
        mock_http_client.request.side_effect = [
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
            mock_response,
        ]

        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockClient:
            MockClient.return_value = mock_http_client
            client = DashboardClient(base_url="http://test")
            with pytest.raises(DashboardError, match="Backend unreachable after 3 attempts"):
                client.health()
        assert mock_http_client.request.call_count == 3

    def test_connect_error_succeeds_on_retry(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"recovered": True}
        mock_response.raise_for_status.return_value = None

        mock_http_client = MagicMock()
        mock_http_client.request.side_effect = [
            httpx.ConnectError("refused"),
            mock_response,
        ]

        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockClient:
            MockClient.return_value = mock_http_client
            client = DashboardClient(base_url="http://test")
            result = client.health()
        assert result == {"recovered": True}
        assert mock_http_client.request.call_count == 2

    def test_timeout_retries(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status.return_value = None

        mock_http_client = MagicMock()
        mock_http_client.request.side_effect = [
            httpx.TimeoutException("slow"),
            mock_response,
        ]

        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockClient:
            MockClient.return_value = mock_http_client
            client = DashboardClient(base_url="http://test")
            result = client.health()
        assert result == {"ok": True}
        assert mock_http_client.request.call_count == 2


class TestHttpErrors:
    def test_4xx_raises_dashboard_error_with_status(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason_phrase = "Not Found"
        mock_response.json.return_value = {"detail": "unknown adapter"}
        mock_response.text = '{"detail": "unknown adapter"}'

        mock_http_client = MagicMock()
        mock_http_client.request.return_value = mock_response
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockClient:
            MockClient.return_value = mock_http_client
            client = DashboardClient(base_url="http://test")
            with pytest.raises(DashboardError, match="unknown adapter") as exc_info:
                client.install_adapter("nope")
            assert exc_info.value.status_code == 404

    def test_invalid_json_raises_dashboard_error(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")
        mock_response.raise_for_status.return_value = None

        mock_http_client = MagicMock()
        mock_http_client.request.return_value = mock_response

        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockClient:
            MockClient.return_value = mock_http_client
            client = DashboardClient(base_url="http://test")
            with pytest.raises(DashboardError, match="Invalid JSON"):
                client.health()

    def test_error_response_detail_from_text_when_json_fails(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.reason_phrase = "Server Error"
        mock_response.json.side_effect = ValueError("bad json")
        mock_response.text = "raw error body"

        mock_http_client = MagicMock()
        mock_http_client.request.return_value = mock_response
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockClient:
            MockClient.return_value = mock_http_client
            client = DashboardClient(base_url="http://test")
            with pytest.raises(DashboardError, match="raw error body") as exc_info:
                client.health()
            assert exc_info.value.status_code == 500


class TestPathValidation:
    def test_validate_rejects_traversal(self) -> None:
        from agent_amplifier.dashboard.ui.api import _validate_path_segment

        with pytest.raises(DashboardError, match="Invalid path segment"):
            _validate_path_segment("../../etc/passwd")

    def test_validate_accepts_normal_name(self) -> None:
        from agent_amplifier.dashboard.ui.api import _validate_path_segment

        assert _validate_path_segment("claude_code") == "claude_code"
        assert _validate_path_segment("langchain") == "langchain"
