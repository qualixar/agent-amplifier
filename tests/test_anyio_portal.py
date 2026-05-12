# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier._anyio_portal`` (.14, ).

RED→GREEN: every test below should FAIL until ``_anyio_portal.PortalHolder``
lands. Each test names exactly which behavior it locks.
"""

from __future__ import annotations

import threading
import time

import pytest

# anyio.from_thread.start_blocking_portal stable
# in anyio 4.x (verified 4.13.0 stable, no deprecation in 4.4-4.13). Used as
# context manager; module-level alias removed in 4.0 — must import from
# ``anyio.from_thread``.


def test_portal_lazy_start_no_thread_until_first_call() -> None:
    """PortalHolder MUST NOT start a portal at construction time."""
    from agent_amplifier._anyio_portal import PortalHolder

    holder = PortalHolder()
    # No portal => no extra threads beyond the test's main thread.
    assert holder._portal is None
    holder.close()


def test_portal_run_sync_executes_async_callable() -> None:
    """PortalHolder.run_sync must run an async callable from sync code."""
    from agent_amplifier._anyio_portal import PortalHolder

    async def double(x: int) -> int:
        return x * 2

    holder = PortalHolder()
    try:
        assert holder.run_sync(double, 21) == 42
    finally:
        holder.close()


def test_portal_run_sync_propagates_exception() -> None:
    """If the async callable raises, run_sync must surface the exception."""
    from agent_amplifier._anyio_portal import PortalHolder

    async def boom() -> None:
        raise ValueError("boom")

    holder = PortalHolder()
    try:
        with pytest.raises(ValueError, match="boom"):
            holder.run_sync(boom)
    finally:
        holder.close()


def test_portal_close_is_idempotent() -> None:
    """close() called twice does not raise."""
    from agent_amplifier._anyio_portal import PortalHolder

    holder = PortalHolder()

    async def noop() -> None:
        return None

    holder.run_sync(noop)
    holder.close()
    holder.close()  # second call must be a no-op


def test_portal_thread_safe_lazy_start() -> None:
    """Concurrent first calls from many threads must end up sharing one portal."""
    from agent_amplifier._anyio_portal import PortalHolder

    holder = PortalHolder()
    results: list[int] = []
    lock = threading.Lock()

    async def echo(i: int) -> int:
        return i

    def worker(i: int) -> None:
        r = holder.run_sync(echo, i)
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    holder.close()

    assert sorted(results) == list(range(20))


def test_portal_run_sync_after_close_raises_runtime_error() -> None:
    """H13: after close(), run_sync must raise RuntimeError.

    Previous behavior was to lazily re-spawn the portal — that masked
    use-after-close bugs and surprised users who held references to a
    "closed" amp expecting resources to stay released. locks
    the holder closed; create a new ``AgentAmplifier`` to start another
    session.
    """
    from agent_amplifier._anyio_portal import PortalHolder

    holder = PortalHolder()

    async def one() -> int:
        return 1

    assert holder.run_sync(one) == 1
    holder.close()
    with pytest.raises(RuntimeError, match="portal closed"):
        holder.run_sync(one)


def test_portal_run_sync_after_close_without_use_raises() -> None:
    """H13: closing a never-used holder still locks it.

    The fast path (``self._closed`` set during ``close()`` even if the
    portal was never started) is what protects against the "construct,
    close, then try to use" pattern.
    """
    from agent_amplifier._anyio_portal import PortalHolder

    holder = PortalHolder()
    holder.close()  # never started the portal

    async def one() -> int:
        return 1

    with pytest.raises(RuntimeError, match="portal closed"):
        holder.run_sync(one)


def test_portal_close_remains_idempotent_after_h13() -> None:
    """H13: ``close()`` is still idempotent after the closed-flag fix."""
    from agent_amplifier._anyio_portal import PortalHolder

    holder = PortalHolder()
    holder.close()
    holder.close()  # must not raise


def test_portal_runs_in_separate_thread() -> None:
    """The async callable runs on a thread different from the calling thread."""
    from agent_amplifier._anyio_portal import PortalHolder

    holder = PortalHolder()
    main_ident = threading.get_ident()

    async def get_thread_ident() -> int:
        # asyncio default loop thread when launched by start_blocking_portal
        return threading.get_ident()

    try:
        portal_ident = holder.run_sync(get_thread_ident)
        assert portal_ident != main_ident
    finally:
        holder.close()


def test_portal_call_after_construction_completes_quickly() -> None:
    """Sanity check: portal startup is bounded (< 2s) so tests don't hang."""
    from agent_amplifier._anyio_portal import PortalHolder

    holder = PortalHolder()

    async def quick() -> str:
        return "ok"

    t0 = time.monotonic()
    try:
        assert holder.run_sync(quick) == "ok"
    finally:
        holder.close()
    assert time.monotonic() - t0 < 2.0
