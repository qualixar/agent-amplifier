# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Defensive conftest for the Claude Code adapter test package.

Belt-and-suspenders safety: even if a per-test fixture forgets to redirect
``installer._DEFAULT_SETTINGS_PATH`` and ``state._DEFAULT_STATE_DIR``, this
session-scoped autouse fixture ensures NO test in this directory can ever
write to the real ``~/.claude/settings.json`` or ``~/.claude/agent-amp/``.

The redirect target lives under ``/tmp/agent-amp-test-fail-safe-<pid>``,
which is a path no production code ever consults — if a write lands there
during testing, the test is buggy but the user's machine is unharmed.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _claude_code_test_failsafe_redirects() -> None:
    """Repoint adapter module-level defaults at a per-process /tmp path so
    bad per-test fixtures cannot leak writes to the real ``~/.claude``."""
    safe_root = Path(tempfile.gettempdir()) / f"agent-amp-test-failsafe-{os.getpid()}"
    safe_root.mkdir(parents=True, exist_ok=True)

    from agent_amplifier.adapters.claude_code import (
        installer as _ins,
    )
    from agent_amplifier.adapters.claude_code import (
        state as _state,
    )
    # Repoint defaults — process-wide for this test session.
    _ins._DEFAULT_SETTINGS_PATH = safe_root / "settings.json"
    _state._DEFAULT_STATE_DIR = safe_root / "agent-amp"
