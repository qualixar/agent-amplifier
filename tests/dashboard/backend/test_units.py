# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Focused branch coverage for dashboard backend helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.dashboard.backend import adapters as adapter_mod
from agent_amplifier.dashboard.backend import config_store, telemetry
from agent_amplifier.dashboard.backend.app import create_app
from agent_amplifier.dashboard.backend.services import DashboardSettings


class _FakeAdapter(AdapterBase):
    framework_name = "fake_dashboard_adapter"

    def install(self) -> None:
        self._mark_installed()

    def uninstall(self) -> None:
        self._mark_uninstalled()

    def on_before_step(self, context: dict[str, object]) -> dict[str, object]:
        return context

    def on_after_step(
        self,
        context: dict[str, object],
        result: dict[str, object] | str,
    ) -> dict[str, object]:
        return {"action": "continue"}


def test_get_config_endpoint_returns_defaults(
    client_factory: object,
) -> None:
    make_client = client_factory
    assert callable(make_client)
    client = make_client(None)
    res = client.get("/api/config")
    assert res.status_code == 200
    assert res.json()["config"]["max_iterations"] == 4


def test_config_read_errors_are_http_400(
    config_path: Path,
    state_db: Path,
) -> None:
    config_path.parent.mkdir(parents=True)
    config_path.write_text("max_iterations = 0\n", encoding="utf-8")
    client = TestClient(create_app(DashboardSettings(config_path=config_path, db_path=state_db)))
    assert client.get("/api/config").status_code == 400
    assert client.get("/api/ips").status_code == 400
    assert client.post("/api/ips/kernel/toggle").status_code == 400


def test_adapter_install_exception_is_http_400(
    client_factory: object,
) -> None:
    make_client = client_factory
    assert callable(make_client)

    def installer(name: str) -> str:
        raise RuntimeError(f"boom:{name}")

    client = make_client(installer)
    res = client.post("/api/adapters/cursor/install")
    assert res.status_code == 400
    assert "boom:cursor" in res.json()["detail"]


def test_toggle_twice_reenables_ip(
    client_factory: object,
) -> None:
    make_client = client_factory
    assert callable(make_client)
    client = make_client(None)
    assert client.post("/api/ips/kernel/toggle").json()["ip"]["enabled"] is False
    assert client.post("/api/ips/kernel/toggle").json()["ip"]["enabled"] is True


def test_config_store_rejects_unsafe_backup(
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_store.save_config_dict(config_path, {"max_iterations": 4})
    monkeypatch.setattr(config_store, "safe_read_text", lambda path, root: None)
    with pytest.raises(Exception, match="unsafe config path"):
        config_store.save_config_dict(config_path, {"max_iterations": 5})


def test_config_store_rejects_unsafe_write(
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_store, "safe_open_write", lambda path, root: None)
    with pytest.raises(Exception, match="unsafe config write"):
        config_store.save_config_dict(config_path, {"max_iterations": 4})


def test_safe_writer_exit_without_handle_is_noop(config_path: Path) -> None:
    writer = config_store.safe_writer(config_path, config_path.parent)
    writer.__exit__(None, None, None)


def test_normalized_order_drops_duplicates_and_unknowns(config_path: Path) -> None:
    config_store.save_config_dict(
        config_path,
        {
            "max_iterations": 4,
            "ip_order": ["kernel", "unknown", "kernel", "tool_selector"],
        },
    )
    ips = config_store.list_ips(config_path)
    assert [ip.id for ip in ips[:2]] == ["kernel", "tool_selector"]
    assert len(ips) == 11


def test_telemetry_missing_db_for_turns_and_convergence(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    assert telemetry.recent_turns(missing, limit=999).limit == 500
    assert telemetry.recent_turns(missing, limit=0).limit == 1
    assert telemetry.convergence_series(missing, days=999).days == 365
    assert telemetry.convergence_series(missing, days=0).days == 1


def test_telemetry_helpers_handle_zero_and_bad_values() -> None:
    assert telemetry._rate(1, 0) == 0.0
    assert telemetry._optional_int(None) is None
    assert telemetry._optional_int(True) == 1
    assert telemetry._optional_bool(None) is None
    with pytest.raises(TypeError):
        telemetry._optional_int(object())


def test_adapter_registry_install_and_error_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raises() -> bool:
        raise RuntimeError("detect fail")

    def broken_factory() -> AdapterBase:
        raise RuntimeError("factory fail")

    specs = (
        adapter_mod.AdapterSpec("ok", "OK", lambda: _FakeAdapter(kernel=None), lambda: True),
        adapter_mod.AdapterSpec("detect_fail", "Detect Fail", None, raises),
        adapter_mod.AdapterSpec("factory_fail", "Factory Fail", broken_factory, lambda: True),
        adapter_mod.AdapterSpec("none", "None", None, lambda: False),
    )
    monkeypatch.setattr(adapter_mod, "ADAPTER_SPECS", specs)

    listed = adapter_mod.list_adapters()
    assert [item.name for item in listed] == ["ok", "detect_fail", "factory_fail", "none"]
    assert listed[1].detected is False
    assert listed[2].installed is False
    assert adapter_mod.install_adapter("ok") == "installed:ok"
    assert adapter_mod.install_adapter("none") is None
    assert adapter_mod.install_adapter("missing") is None
    assert adapter_mod.adapter_exists("missing") is False


def test_service_uses_default_adapter_installer(
    config_path: Path,
    state_db: Path,
) -> None:
    client = TestClient(create_app(DashboardSettings(config_path=config_path, db_path=state_db)))
    assert client.post("/api/adapters/cursor/install").json() == {
        "name": "cursor",
        "status": "installed:cursor",
    }
    assert client.post("/api/adapters/langchain/install").json() == {
        "name": "langchain",
        "status": "installed:langchain",
    }
