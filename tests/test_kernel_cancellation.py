# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""MED-15 — ``asyncio.CancelledError`` propagation tests.

DistSys CRIT carry-over: under Python 3.8+, ``CancelledError``
inherits from ``BaseException`` (not ``Exception``). The kernel's outer
``except Exception`` blocks do NOT swallow it — meaning a cancelled
``before_step`` / ``after_step`` MUST propagate ``CancelledError`` to the
caller instead of being silently converted to a degenerate envelope.

We use ``asyncio.run`` directly rather than pytest-asyncio so the test
file has zero plugin dependencies. Anyio's pytest plugin is already
installed but we don't need its facilities here — only direct event-loop
scheduling.
"""
from __future__ import annotations

import asyncio
from typing import Any

from agent_amplifier.kernel import AsyncAgentAmplifier
from agent_amplifier.types import AmplifierConfig, BudgetMode


def test_before_step_propagates_cancelled_error() -> None:
    """``CancelledError`` raised inside a memory_recall callback MUST
    propagate past the kernel's ``except Exception`` boundary.

    The kernel's outer try/except is ``except Exception`` (not
    ``BaseException``); ``CancelledError`` inherits from ``BaseException``
    in Python 3.8+ so it is NOT caught and MUST surface to the caller.
    Without this propagation, a host that cancels the surrounding task
    would observe a "degenerate envelope" instead of the cancellation.
    """

    def recall(_q: str, _l: int) -> Any:
        # Simulate "outer task got cancelled while we were inside the
        # callback". Note: the kernel's `_resolve_recall` wraps the user
        # callback in `try/except Exception` — which does NOT catch
        # CancelledError. So this raises through.
        raise asyncio.CancelledError()

    cfg = AmplifierConfig(budget_mode=BudgetMode.UNLIMITED)

    async def _drive() -> None:
        amp = AsyncAgentAmplifier(config=cfg, memory_recall=recall)
        await amp.before_step("query that cancels")

    raised: BaseException | None = None
    try:
        asyncio.run(_drive())
    except BaseException as exc:
        raised = exc

    assert raised is not None, "expected CancelledError to propagate"
    assert isinstance(raised, asyncio.CancelledError), (
        f"expected CancelledError, got {type(raised).__name__}"
    )


def test_before_step_swallows_regular_exception() -> None:
    """Inverse of the cancellation test — sanity-check that regular
    ``Exception`` from a callback IS swallowed.

    Without this dual proof, a future regression that widens the catch to
    ``BaseException`` would silently eat shutdown signals.
    """

    def recall(_q: str, _l: int) -> Any:
        raise RuntimeError("boom")  # Exception, NOT BaseException

    cfg = AmplifierConfig(budget_mode=BudgetMode.UNLIMITED)

    async def _drive() -> Any:
        amp = AsyncAgentAmplifier(config=cfg, memory_recall=recall)
        return await amp.before_step("query that errors")

    # Must NOT raise — the kernel logs a WARNING and returns an empty
    # recall list; the rest of before_step proceeds normally.
    env = asyncio.run(_drive())
    assert env.envelope, "expected envelope, not degenerate fallback"


def test_remember_swallows_regular_exception() -> None:
    """``_resolve_remember`` mirrors ``_resolve_recall`` — Exception
    swallowed, BaseException NOT swallowed.
    """

    def remember(_o: Any) -> None:
        raise RuntimeError("write failed")  # Exception

    cfg = AmplifierConfig(budget_mode=BudgetMode.UNLIMITED)

    async def _drive() -> Any:
        amp = AsyncAgentAmplifier(
            config=cfg,
            memory_remember=remember,
        )
        # Drive a complete session to invoke finalize.
        env = await amp.before_step("setup")
        await amp.after_step(env, "result")
        return await amp._core.finalize()

    # Must NOT raise — finalize() catches the exception and logs.
    report = asyncio.run(_drive())
    assert report["iterations_completed"] >= 1


def test_remember_propagates_cancelled_error() -> None:
    """MED-15 — CancelledError from a ``memory_remember`` callback
    propagates the same way as in ``memory_recall``.
    """

    def remember(_o: Any) -> None:
        raise asyncio.CancelledError()

    cfg = AmplifierConfig(budget_mode=BudgetMode.UNLIMITED)

    async def _drive() -> None:
        amp = AsyncAgentAmplifier(
            config=cfg,
            memory_remember=remember,
        )
        env = await amp.before_step("setup")
        await amp.after_step(env, "result")
        await amp._core.finalize()

    raised: BaseException | None = None
    try:
        asyncio.run(_drive())
    except BaseException as exc:
        raised = exc
    assert raised is not None
    assert isinstance(raised, asyncio.CancelledError)


def test_cancellederror_inherits_from_baseexception() -> None:
    """Doc-test: ``asyncio.CancelledError`` inherits from ``BaseException``.

    Python 3.8+. This invariant is what makes ``except Exception`` correct
    for the kernel — we want to propagate cancellation, never swallow it.
    """
    assert issubclass(asyncio.CancelledError, BaseException)
    # And critically NOT from Exception in 3.8+:
    assert not issubclass(asyncio.CancelledError, Exception)
