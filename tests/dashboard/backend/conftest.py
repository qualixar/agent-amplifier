# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Fixtures for dashboard backend tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_amplifier.adapters.claude_code.state import StateStore
from agent_amplifier.dashboard.backend.app import create_app
from agent_amplifier.dashboard.backend.services import DashboardSettings


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from agent_amplifier import config as cfg_mod

    fake_home = tmp_path / "home"
    monkeypatch.setattr(cfg_mod, "_allowed_roots", lambda: [fake_home.resolve()])
    return fake_home / ".config" / "agent-amplifier" / "config.toml"


@pytest.fixture
def state_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "home" / ".claude" / "agent-amp" / "state.db"
    store = StateStore(db_path)
    store.upsert_session("session-a", "/repo", model="sonnet", model_provider="anthropic")
    turn_one = store.next_turn_id("session-a")
    store.record_envelope(
        "session-a",
        turn_one,
        user_prompt_redacted="build backend",
        classification_complexity="high",
        classification_domain="backend",
        thinking_trigger="PERSONA",
        persona="Senior Python backend engineer",
        phase="EXECUTE",
        envelope_text="redacted envelope",
    )
    store.write_outcome(
        "session-a",
        turn_one,
        iterations_completed=3,
        converged=True,
        drift_at_end=0.02,
        tokens_used=1200,
        duration_ms=2500,
        quality_estimate=0.96,
        finalize_report={"stop_reason": "done"},
    )
    turn_two = store.next_turn_id("session-a")
    store.record_envelope(
        "session-a",
        turn_two,
        user_prompt_redacted="ship dashboard",
        classification_complexity="medium",
        classification_domain="frontend",
        thinking_trigger=None,
        persona=None,
        phase="VERIFY",
        envelope_text="second envelope",
    )
    store.write_outcome(
        "session-a",
        turn_two,
        iterations_completed=1,
        converged=False,
        duration_ms=900,
        finalize_report={"stop_reason": "max_iterations"},
    )
    return db_path


@pytest.fixture
def client_factory(
    config_path: Path,
    state_db: Path,
) -> Callable[[Callable[[str], str] | None], TestClient]:
    def _make(installer: Callable[[str], str] | None = None) -> TestClient:
        settings = DashboardSettings(
            config_path=config_path,
            db_path=state_db,
            adapter_installer=installer,
        )
        return TestClient(create_app(settings))

    return _make
