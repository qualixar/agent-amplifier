# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for the Stage-13-day-0 install/uninstall delegation in
``ClaudeCodeAdapter`` (memory.py).

The existing ``tests/test_claude_code_adapter.py`` covers the memory-plane
behavior at 100%. This file adds the new install/uninstall delegation
paths so the adapter's lifecycle methods stay at 100% line coverage.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_amplifier.adapters.claude_code import installer as _ins
from agent_amplifier.adapters.claude_code import uninstaller as _un
from agent_amplifier.adapters.claude_code.memory import ClaudeCodeAdapter


@pytest.fixture(autouse=True)
def _redirect_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    redirect = tmp_path / "settings.json"
    monkeypatch.setattr(_ins, "_DEFAULT_SETTINGS_PATH", redirect)
    return redirect


def test_install_delegates_to_installer(
    _redirect_default: Path,
) -> None:
    """ClaudeCodeAdapter.install() must invoke installer.install() — meaning
    settings.json gets written and the adapter is marked installed."""
    a = ClaudeCodeAdapter(kernel=None)
    assert not a.is_installed()
    a.install()
    assert a.is_installed()
    assert _redirect_default.exists()
    assert _redirect_default.stat().st_size > 0


def test_uninstall_delegates_to_uninstaller(
    _redirect_default: Path,
) -> None:
    a = ClaudeCodeAdapter(kernel=None)
    a.install()
    a.uninstall()
    assert not a.is_installed()
    # Marker is gone.
    res = _un.uninstall(_redirect_default)
    assert res["removed_events"] == []  # already removed by adapter.uninstall


def test_install_persistent_class_attr_is_true() -> None:
    assert ClaudeCodeAdapter.INSTALL_PERSISTENT is True
