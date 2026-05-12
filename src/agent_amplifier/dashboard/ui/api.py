# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""HTTP client for the Agent Amplifier dashboard backend."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import httpx

LOG = logging.getLogger(__name__)

_DEFAULT_PORT = 8765
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5


class DashboardError(Exception):
    """Raised when a backend request fails after retries or returns an error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def get_backend_url() -> str:
    """Return the base URL for the dashboard backend."""
    raw_port = os.environ.get("AGENT_AMP_DASHBOARD_PORT", str(_DEFAULT_PORT))
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise DashboardError(
            f"AGENT_AMP_DASHBOARD_PORT must be an integer: {raw_port!r}"
        ) from exc
    if not 1 <= port <= 65535:
        raise DashboardError(
            f"AGENT_AMP_DASHBOARD_PORT must be between 1 and 65535: {port}"
        )
    return f"http://127.0.0.1:{port}"


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    json: dict[str, object] | None = None,
    params: dict[str, str | int | float | bool | None] | None = None,
) -> dict[str, Any]:
    """Execute an HTTP request with exponential backoff on connection errors."""
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.request(
                method,
                url,
                json=json,
                params=params,
                timeout=10.0,
            )
            break
        except httpx.ConnectError as exc:
            last_error = exc
            wait = _BACKOFF_BASE * (2 ** attempt)
            LOG.debug(
                "Connection error on %s %s (attempt %d), retrying in %.1fs",
                method,
                url,
                attempt + 1,
                wait,
            )
            time.sleep(wait)
        except httpx.TimeoutException as exc:
            last_error = exc
            wait = _BACKOFF_BASE * (2 ** attempt)
            LOG.debug(
                "Timeout on %s %s (attempt %d), retrying in %.1fs",
                method,
                url,
                attempt + 1,
                wait,
            )
            time.sleep(wait)
    else:
        if last_error is not None:
            raise DashboardError(
                f"Backend unreachable after {_MAX_RETRIES} attempts: {last_error}"
            ) from last_error
        raise DashboardError("Backend unreachable after retries")  # pragma: no cover

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = exc.response.text or str(exc.response.status_code)
        raise DashboardError(
            f"Backend error: {detail or exc.response.reason_phrase}",
            status_code=exc.response.status_code,
        ) from exc

    try:
        result: dict[str, Any] = response.json()
        return result
    except Exception as exc:
        raise DashboardError(f"Invalid JSON from backend: {exc}") from exc


_SAFE_PATH_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")


def _validate_path_segment(value: str) -> str:
    if not _SAFE_PATH_RE.match(value):
        raise DashboardError(f"Invalid path segment: {value!r}")
    return value


class DashboardClient:
    """Thin wrapper around httpx that speaks to the FastAPI backend."""

    def __init__(self, base_url: str | None = None) -> None:
        self._base = (base_url or get_backend_url()).rstrip("/")
        self._client: httpx.Client | None = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> DashboardClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        """GET /api/health"""
        return _request_with_retry(
            self._ensure_client(), "GET", f"{self._base}/api/health"
        )

    def get_config(self) -> dict[str, Any]:
        """GET /api/config"""
        return _request_with_retry(
            self._ensure_client(), "GET", f"{self._base}/api/config"
        )

    def save_config(self, config: dict[str, object]) -> dict[str, Any]:
        """POST /api/config"""
        return _request_with_retry(
            self._ensure_client(),
            "POST",
            f"{self._base}/api/config",
            json={"config": config},
        )

    def get_ips(self) -> dict[str, Any]:
        """GET /api/ips"""
        return _request_with_retry(
            self._ensure_client(), "GET", f"{self._base}/api/ips"
        )

    def toggle_ip(self, ip_id: str) -> dict[str, Any]:
        """POST /api/ips/{ip_id}/toggle"""
        return _request_with_retry(
            self._ensure_client(),
            "POST",
            f"{self._base}/api/ips/{_validate_path_segment(ip_id)}/toggle",
        )

    def reorder_ips(self, ip_ids: list[str]) -> dict[str, Any]:
        """POST /api/ips/reorder"""
        return _request_with_retry(
            self._ensure_client(),
            "POST",
            f"{self._base}/api/ips/reorder",
            json={"ip_ids": ip_ids},
        )

    def telemetry_summary(self) -> dict[str, Any]:
        """GET /api/telemetry/summary"""
        return _request_with_retry(
            self._ensure_client(), "GET", f"{self._base}/api/telemetry/summary"
        )

    def turns(self, *, limit: int = 50) -> dict[str, Any]:
        """GET /api/telemetry/turns?limit=..."""
        return _request_with_retry(
            self._ensure_client(),
            "GET",
            f"{self._base}/api/telemetry/turns",
            params={"limit": limit},
        )

    def convergence(self, *, days: int = 7) -> dict[str, Any]:
        """GET /api/telemetry/convergence?days=..."""
        return _request_with_retry(
            self._ensure_client(),
            "GET",
            f"{self._base}/api/telemetry/convergence",
            params={"days": days},
        )

    def get_adapters(self) -> dict[str, Any]:
        """GET /api/adapters"""
        return _request_with_retry(
            self._ensure_client(), "GET", f"{self._base}/api/adapters"
        )

    def install_adapter(self, name: str) -> dict[str, Any]:
        """POST /api/adapters/{name}/install"""
        return _request_with_retry(
            self._ensure_client(),
            "POST",
            f"{self._base}/api/adapters/{_validate_path_segment(name)}/install",
        )

    def get_personas(self) -> dict[str, Any]:
        """GET /api/personas — list built-in + custom personas."""
        return _request_with_retry(
            self._ensure_client(), "GET", f"{self._base}/api/personas"
        )

    def create_persona(
        self,
        *,
        name: str,
        label: str,
        description: str,
        review_focus: list[str],
    ) -> dict[str, Any]:
        """POST /api/personas — add a custom persona."""
        return _request_with_retry(
            self._ensure_client(),
            "POST",
            f"{self._base}/api/personas",
            json={
                "name": name,
                "label": label,
                "description": description,
                "review_focus": review_focus,
            },
        )

    def delete_persona(self, name: str) -> None:
        """DELETE /api/personas/{name} — remove a custom persona.

        Returns ``None`` on success (HTTP 204). Raises ``DashboardError`` on
        any other status. Uses a one-shot request because 204 returns no body
        and the generic helper expects JSON.
        """
        url = f"{self._base}/api/personas/{_validate_path_segment(name)}"
        client = self._ensure_client()
        try:
            response = client.request("DELETE", url, timeout=10.0)
        except httpx.HTTPError as exc:
            raise DashboardError(f"Backend unreachable: {exc}") from exc
        if response.status_code == 204:
            return None
        # Map common error codes to DashboardError with the backend detail.
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except Exception:
            detail = response.text or response.reason_phrase
        raise DashboardError(
            f"Backend error: {detail or response.reason_phrase}",
            status_code=response.status_code,
        )


__all__ = [
    "DashboardClient",
    "DashboardError",
    "get_backend_url",
]
