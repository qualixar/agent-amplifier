# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""CLI coverage for ``agent-amp dashboard``."""

from __future__ import annotations

import types

import pytest


def test_dashboard_help_mentions_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    from agent_amplifier.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    assert "dashboard" in capsys.readouterr().out


def test_dashboard_launch_uses_env_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_amplifier import cli

    captured: dict[str, object] = {}

    def run(app_path: str, *, factory: bool, host: str, port: int) -> None:
        captured.update(
            {
                "app_path": app_path,
                "factory": factory,
                "host": host,
                "port": port,
            }
        )

    fake_uvicorn = types.SimpleNamespace(run=run)
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", fake_uvicorn)
    monkeypatch.setenv("AGENT_AMP_DASHBOARD_PORT", "9012")
    assert cli.main(["dashboard"]) == 0
    assert captured == {
        "app_path": "agent_amplifier.dashboard.backend.app:create_app",
        "factory": True,
        "host": "127.0.0.1",
        "port": 9012,
    }


def test_dashboard_launch_rejects_bad_port(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    monkeypatch.setenv("AGENT_AMP_DASHBOARD_PORT", "not-a-port")
    assert main(["dashboard"]) == 2
    assert "AGENT_AMP_DASHBOARD_PORT" in capsys.readouterr().err


def test_dashboard_launch_rejects_out_of_range_port(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Covers cli.py:449-454 — port range validation branch."""
    from agent_amplifier.cli import main

    monkeypatch.setenv("AGENT_AMP_DASHBOARD_PORT", "70000")
    assert main(["dashboard"]) == 2
    err = capsys.readouterr().err
    assert "between 1 and 65535" in err


def test_dashboard_launch_handles_missing_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Covers cli.py:457-462 — uvicorn ImportError branch."""
    import builtins
    import sys

    from agent_amplifier.cli import main

    # Pop any cached uvicorn so the import statement re-runs and raises.
    monkeypatch.delitem(sys.modules, "uvicorn", raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "uvicorn":
            raise ImportError("no uvicorn")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert main(["dashboard"]) == 1
    assert "uvicorn" in capsys.readouterr().err
