# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``python -m agent_amplifier`` ().

Covers ``src/agent_amplifier/__main__.py`` end-to-end via subprocess + import.
The 3-line module is trivial but ships in the wheel, so it gets an explicit
test (Anti-Rationalization: "If a code path exists in shipped code, it
gets a test.").
"""

from __future__ import annotations

import importlib
import subprocess
import sys


def test_dunder_main_imports_cleanly() -> None:
    """In-process import covers the module-level statements (lines 3-7).

    The `if __name__ == "__main__":` guard lines 9-10 are excluded from
    coverage (see pyproject.toml [tool.coverage.report] exclude_lines).
    """
    # Force a fresh import to make sure coverage records the run.
    if "agent_amplifier.__main__" in sys.modules:
        del sys.modules["agent_amplifier.__main__"]
    mod = importlib.import_module("agent_amplifier.__main__")
    assert hasattr(mod, "main")


def test_module_invocation_version_flag() -> None:
    """``python -m agent_amplifier --version`` prints the version + exits 0."""
    from agent_amplifier import __version__

    proc = subprocess.run(
        [sys.executable, "-m", "agent_amplifier", "--version"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    # argparse writes --version to stdout on success.
    assert __version__ in (proc.stdout or "")


def test_module_invocation_no_args_returns_zero() -> None:
    """Bare ``python -m agent_amplifier`` prints help and exits 0."""
    proc = subprocess.run(
        [sys.executable, "-m", "agent_amplifier"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "agent-amp" in out
