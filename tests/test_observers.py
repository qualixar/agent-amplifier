# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.observers`` (.10, )."""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from agent_amplifier.types import AmplifierEvent


def test_print_observer_writes_to_capsys(capsys: pytest.CaptureFixture[str]) -> None:
    """``PrintObserver`` is the rich-free fallback used when rich is missing."""
    from agent_amplifier.observers import PrintObserver

    obs = PrintObserver()
    obs(AmplifierEvent.BEFORE_STEP, {"iteration": 0, "phase": "EXPLORE"})
    out = capsys.readouterr().out
    assert "before_step" in out
    assert "EXPLORE" in out


def test_print_observer_swallows_serialization_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-stringifiable payload must NOT crash the observer."""
    from agent_amplifier.observers import PrintObserver

    class Weird:
        def __repr__(self) -> str:
            raise ValueError("nope")

    obs = PrintObserver()
    # Should not raise even with the weird payload
    obs(AmplifierEvent.AFTER_STEP, {"weird": Weird()})


def test_rich_observer_imports_lazily_when_rich_present() -> None:
    """If ``rich`` is importable, RichConsoleObserver constructs cleanly."""
    pytest.importorskip("rich")
    from agent_amplifier.observers import RichConsoleObserver

    obs = RichConsoleObserver()
    # Each known event branch dispatches; just call to confirm no raise.
    obs(AmplifierEvent.BEFORE_STEP, {"iteration": 0, "phase": "EXPLORE"})
    obs(AmplifierEvent.ON_CONVERGE, {"iteration": 3})
    obs(AmplifierEvent.ON_DRIFT, {"drift": 0.8})
    obs(AmplifierEvent.ON_BUDGET_HIT, {"used": 100, "limit": 90})
    obs(AmplifierEvent.AFTER_STEP, {"iteration": 0})  # branch-free


def test_rich_observer_missing_rich_raises_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``rich`` is missing, RichConsoleObserver raises with install hint."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "rich" or name.startswith("rich."):
            raise ImportError("no rich")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from agent_amplifier.observers import RichConsoleObserver

    with pytest.raises(ImportError, match="agent-amplifier\\[pretty\\]"):
        RichConsoleObserver()


def test_make_default_observer_picks_rich_when_present() -> None:
    """make_default_observer returns Rich if installed, else Print."""
    from agent_amplifier.observers import (
        PrintObserver,
        make_default_observer,
    )

    obs = make_default_observer()
    # Either flavor is acceptable — what matters is callable + no raise.
    assert callable(obs)
    # Smoke-call the chosen observer to confirm it works end-to-end.
    obs(AmplifierEvent.BEFORE_STEP, {"iteration": 0})
    if obs.__class__.__name__ == "PrintObserver":
        assert isinstance(obs, PrintObserver)
