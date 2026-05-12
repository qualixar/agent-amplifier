# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Observer branch coverage (.10 — closes coverage gaps).

Targets:
  * Lines 56-59 — PrintObserver double-fallback when even keypair-string
    join fails (outer try/except).
  * Lines 96-98 — RichConsoleObserver swallows dispatch failure.
  * Line 130 — ON_BUDGET_LOW dispatch path in RichConsoleObserver.
  * Lines 150-151 — make_default_observer fallback when rich missing.
"""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from agent_amplifier.types import AmplifierEvent

# ---------------------------------------------------------------------------
# PrintObserver — outer try/except path (lines 56-59)
# ---------------------------------------------------------------------------


def test_print_observer_outer_except_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Force the outer try to fail by passing a non-dict payload.

    ``payload.items()`` raises AttributeError on non-dict → falls into the
    outer except block (line 56-59).
    """
    from agent_amplifier.observers import PrintObserver

    obs = PrintObserver()
    # str payload doesn't have .items() → outer try raises AttributeError.
    obs(AmplifierEvent.AFTER_STEP, "not a dict")  # type: ignore[arg-type]
    out = capsys.readouterr().out
    # Inner fallback prints just `[amp] <event>` with no keypairs.
    assert "[amp] after_step" in out


def test_print_observer_inner_print_failure_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the fallback ``print`` itself raises, contextlib.suppress eats it.

    Confirms line 58 (``with contextlib.suppress(Exception):``) actually
    suppresses — i.e. the observer never raises out.
    """
    from agent_amplifier import observers

    calls = {"count": 0}
    real_print = builtins.print

    def kaboom(*args: Any, **kwargs: Any) -> None:
        calls["count"] += 1
        raise OSError("disk go bye-bye")

    monkeypatch.setattr(observers, "print", kaboom, raising=False)
    # Also force the outer try to fail so we hit the inner suppress.
    obs = observers.PrintObserver()
    # MUST NOT raise.
    obs(AmplifierEvent.AFTER_STEP, "not a dict")  # type: ignore[arg-type]
    # At least one call to print attempted (the fallback).
    assert calls["count"] >= 1
    _ = real_print  # silence unused


# ---------------------------------------------------------------------------
# RichConsoleObserver dispatch failure (lines 96-98)
# ---------------------------------------------------------------------------


def test_rich_observer_swallows_dispatch_failure() -> None:
    """If _dispatch raises (e.g. payload member access blows up), __call__
    must not propagate (lines 96-98)."""
    pytest.importorskip("rich")
    from agent_amplifier.observers import RichConsoleObserver

    obs = RichConsoleObserver()

    class _Boom:
        def __getitem__(self, k: str) -> Any:
            raise RuntimeError("explosion")

        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("explosion")

    # Force dispatch to fail by handing it a payload whose .get() raises.
    obs(AmplifierEvent.BEFORE_STEP, _Boom())  # type: ignore[arg-type]
    # If we got here without an exception, the swallow worked.


def test_rich_observer_budget_low_branch_dispatched() -> None:
    """ON_BUDGET_LOW must hit the styled-print line (line 130)."""
    pytest.importorskip("rich")
    from agent_amplifier.observers import RichConsoleObserver

    obs = RichConsoleObserver()
    obs(AmplifierEvent.ON_BUDGET_LOW, {"used": 50, "limit": 100})


# ---------------------------------------------------------------------------
# make_default_observer fallback (lines 150-151)
# ---------------------------------------------------------------------------


def test_make_default_observer_falls_back_to_print_when_rich_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If RichConsoleObserver() raises ImportError, make_default_observer
    returns a PrintObserver (lines 150-151)."""
    from agent_amplifier import observers

    def boom_init(self: Any) -> None:
        raise ImportError("simulated rich missing")

    monkeypatch.setattr(
        observers.RichConsoleObserver, "__init__", boom_init
    )

    obs = observers.make_default_observer()
    assert isinstance(obs, observers.PrintObserver)
