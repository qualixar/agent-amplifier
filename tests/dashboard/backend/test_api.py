# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""FastAPI contract tests for the dashboard backend."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient


def test_health_reports_version_and_db_path(
    client_factory: Callable[[Callable[[str], str] | None], TestClient],
    state_db: Path,
) -> None:
    client = client_factory(None)
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["amp_version"] == "1.1.1"
    assert body["db_path"] == str(state_db)


def test_config_round_trips_with_backup_and_validation(
    client_factory: Callable[[Callable[[str], str] | None], TestClient],
    config_path: Path,
) -> None:
    client = client_factory(None)

    res = client.post(
        "/api/config",
        json={"config": {"max_iterations": 6, "budget_mode": "balanced"}},
    )
    assert res.status_code == 200
    assert res.json()["config"]["max_iterations"] == 6
    assert config_path.read_text(encoding="utf-8").count("max_iterations = 6") == 1

    res = client.post(
        "/api/config",
        json={"config": {"max_iterations": 7, "budget_mode": "balanced"}},
    )
    assert res.status_code == 200
    assert (config_path.parent / "config.toml.bak").is_file()
    assert "max_iterations = 6" in (config_path.parent / "config.toml.bak").read_text(
        encoding="utf-8"
    )

    bad = client.post("/api/config", json={"config": {"max_iterations": 0}})
    assert bad.status_code == 400
    assert "max_iterations" in bad.json()["detail"]


def test_ips_list_toggle_and_reorder_persist(
    client_factory: Callable[[Callable[[str], str] | None], TestClient],
) -> None:
    client = client_factory(None)
    initial = client.get("/api/ips")
    assert initial.status_code == 200
    ips = initial.json()["ips"]
    assert len(ips) == 11
    assert ips[0] == {
        "id": "kernel",
        "name": "Runtime Kernel",
        "file": "src/agent_amplifier/kernel.py",
        "enabled": True,
        "order": 1,
    }

    toggled = client.post("/api/ips/kernel/toggle")
    assert toggled.status_code == 200
    assert toggled.json()["ip"]["enabled"] is False
    assert client.get("/api/ips").json()["ips"][0]["enabled"] is False

    ids = [ip["id"] for ip in ips]
    reordered_ids = list(reversed(ids))
    reordered = client.post("/api/ips/reorder", json={"ip_ids": reordered_ids})
    assert reordered.status_code == 200
    after = client.get("/api/ips").json()["ips"]
    assert [ip["id"] for ip in after] == reordered_ids
    assert after[0]["order"] == 1


def test_invalid_ip_operations_reject_cleanly(
    client_factory: Callable[[Callable[[str], str] | None], TestClient],
) -> None:
    client = client_factory(None)
    assert client.post("/api/ips/nope/toggle").status_code == 404
    bad = client.post("/api/ips/reorder", json={"ip_ids": ["kernel"]})
    assert bad.status_code == 400
    assert "all IP ids" in bad.json()["detail"]


def test_telemetry_summary_turns_and_convergence(
    client_factory: Callable[[Callable[[str], str] | None], TestClient],
) -> None:
    client = client_factory(None)
    summary = client.get("/api/telemetry/summary")
    assert summary.status_code == 200
    assert summary.json()["counts"]["envelopes"] == 2
    assert summary.json()["coverage_rate"] == 1.0
    assert summary.json()["convergence_rate"] == 0.5

    turns = client.get("/api/telemetry/turns?limit=2")
    assert turns.status_code == 200
    body = turns.json()
    assert body["limit"] == 2
    assert len(body["turns"]) == 2
    assert body["turns"][0]["session_id"] == "session-a"
    assert body["turns"][0]["stop_reason"] == "max_iterations"
    assert "user_prompt_redacted" not in body["turns"][0]
    # IP-10 v2: tokens_used flows from outcomes through the API.
    # Turn 2 (most recent) wrote 0 tokens (v1.0 default); Turn 1 wrote 1200.
    assert body["turns"][0]["tokens_used"] == 0
    assert body["turns"][1]["tokens_used"] == 1200

    series = client.get("/api/telemetry/convergence?days=7")
    assert series.status_code == 200
    points = series.json()["points"]
    assert len(points) == 1
    assert points[0]["total"] == 2
    assert points[0]["converged"] == 1
    assert points[0]["rate"] == 0.5


def test_telemetry_missing_db_returns_empty_summary(
    client_factory: Callable[[Callable[[str], str] | None], TestClient],
    state_db: Path,
) -> None:
    state_db.unlink()
    client = client_factory(None)
    res = client.get("/api/telemetry/summary")
    assert res.status_code == 200
    assert res.json()["db_exists"] is False
    assert res.json()["counts"] == {
        "sessions": 0,
        "envelopes": 0,
        "events": 0,
        "outcomes": 0,
    }


def test_adapters_list_seven_and_install_calls_programmatic_installer(
    client_factory: Callable[[Callable[[str], str] | None], TestClient],
) -> None:
    calls: list[str] = []

    def installer(name: str) -> str:
        calls.append(name)
        return f"installed:{name}"

    client = client_factory(installer)
    listed = client.get("/api/adapters")
    assert listed.status_code == 200
    adapters = listed.json()["adapters"]
    # Host-adapter registry exposes seven hosts in v1.0.
    assert len(adapters) == 7
    expected_names = {
        "claude_code",
        "cursor",
        "github_copilot",
        "langgraph",
        "crewai",
        "agentscope",
        "langchain",
    }
    assert {adapter["name"] for adapter in adapters} == expected_names
    assert all("detected" in adapter and "installed" in adapter for adapter in adapters)

    installed = client.post("/api/adapters/cursor/install")
    assert installed.status_code == 200
    assert installed.json() == {"name": "cursor", "status": "installed:cursor"}
    assert calls == ["cursor"]

    missing = client.post("/api/adapters/nope/install")
    assert missing.status_code == 404
