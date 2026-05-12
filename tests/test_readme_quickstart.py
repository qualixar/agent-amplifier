# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Smoke test: README Quickstart MUST stay runnable.

(BLOCKER B1) — README V2 Quickstart shipped a non-runnable
snippet (``before_step({"query": ...})`` instead of the real ``before_step(query: str)``
signature, and ``envelope.system_addendum`` instead of ``envelope.envelope``).

This test imports ``examples/quickstart_demo.py`` and executes its ``main()``
function. The example file is the canonical Quickstart — README must mirror
it. If this test fails, the README is broken.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_quickstart_module():  # type: ignore[no-untyped-def]
    """Load examples/quickstart_demo.py as an in-memory module."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "examples" / "quickstart_demo.py"
    assert path.exists(), f"Quickstart demo missing at {path}"
    spec = importlib.util.spec_from_file_location(
        "agent_amplifier_quickstart_demo", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_readme_quickstart_runs_cleanly() -> None:
    """``examples/quickstart_demo.py`` (= the README Quickstart) executes
    without raising."""
    module = _load_quickstart_module()
    # main() must exist and complete without error.
    assert hasattr(module, "main"), "quickstart_demo.py must expose main()"
    module.main()


def test_readme_quickstart_uses_real_api() -> None:
    """Lock the runnable API surface so future README edits cannot regress.

    These are the EXACT names/types the README and the demo use:
        * AgentAmplifier(...) constructible with no args
        * before_step(query: str) -> StepEnvelope
        * StepEnvelope.envelope: str   (NOT system_addendum)
        * after_step(envelope, result: str) -> dict with key 'action'
    """
    from agent_amplifier import AgentAmplifier, StepEnvelope

    amp = AgentAmplifier()
    try:
        env = amp.before_step("refactor the auth module")
        assert isinstance(env, StepEnvelope)
        # Field MUST be `envelope`, not `system_addendum`.
        assert hasattr(env, "envelope")
        assert not hasattr(env, "system_addendum")
        assert isinstance(env.envelope, str)

        decision = amp.after_step(env, "done")
        assert isinstance(decision, dict)
        assert "action" in decision
    finally:
        amp.close()
