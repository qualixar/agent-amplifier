# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Application service layer for the dashboard backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent_amplifier import __version__
from agent_amplifier.adapters.claude_code import state as _state
from agent_amplifier.config import USER_CONFIG_PATH
from agent_amplifier.dashboard.backend import adapters as adapter_registry
from agent_amplifier.dashboard.backend.config_store import (
    list_ips,
    load_config_dict,
    reorder_ips,
    save_config_dict,
    toggle_ip,
)
from agent_amplifier.dashboard.backend.models import (
    AdapterInstallResponse,
    AdaptersResponse,
    ConfigResponse,
    ConvergenceResponse,
    HealthResponse,
    IpInfo,
    IpsResponse,
    TelemetrySummaryResponse,
    TurnsResponse,
)
from agent_amplifier.dashboard.backend.telemetry import (
    convergence_series,
    recent_turns,
    telemetry_summary,
)


@dataclass(frozen=True, slots=True)
class DashboardSettings:
    config_path: Path = USER_CONFIG_PATH
    db_path: Path = Path(_state._DEFAULT_STATE_DIR) / _state._STATE_DB_FILENAME
    adapter_installer: Callable[[str], str] | None = None


class DashboardService:
    """Thin orchestrator over config, telemetry, and adapter helpers."""

    def __init__(self, settings: DashboardSettings) -> None:
        self._settings = settings

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            amp_version=__version__,
            db_path=str(self._settings.db_path),
        )

    def get_config(self) -> ConfigResponse:
        return ConfigResponse(config=load_config_dict(self._settings.config_path))

    def save_config(self, raw: dict[str, object]) -> ConfigResponse:
        return ConfigResponse(
            config=save_config_dict(self._settings.config_path, raw)
        )

    def ips(self) -> IpsResponse:
        return IpsResponse(ips=list_ips(self._settings.config_path))

    def toggle_ip(self, ip_id: str) -> IpInfo | None:
        return toggle_ip(self._settings.config_path, ip_id)

    def reorder_ips(self, ip_ids: list[str]) -> IpsResponse:
        return IpsResponse(ips=reorder_ips(self._settings.config_path, ip_ids))

    def telemetry_summary(self) -> TelemetrySummaryResponse:
        return telemetry_summary(self._settings.db_path)

    def turns(self, *, limit: int) -> TurnsResponse:
        return recent_turns(self._settings.db_path, limit=limit)

    def convergence(self, *, days: int) -> ConvergenceResponse:
        return convergence_series(self._settings.db_path, days=days)

    def adapters(self) -> AdaptersResponse:
        return AdaptersResponse(adapters=adapter_registry.list_adapters())

    def install_adapter(self, name: str) -> AdapterInstallResponse | None:
        if not adapter_registry.adapter_exists(name):
            return None
        if self._settings.adapter_installer is not None:
            return AdapterInstallResponse(
                name=name,
                status=self._settings.adapter_installer(name),
            )
        installed = adapter_registry.install_adapter(name)
        if installed is None:  # pragma: no cover - defensive; all v1.0 adapters have factories
            return None
        return AdapterInstallResponse(name=name, status=installed)


__all__ = ["DashboardService", "DashboardSettings"]
