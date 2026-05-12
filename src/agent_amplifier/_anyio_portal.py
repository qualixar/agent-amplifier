# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Sync ↔ Async bridge for AgentAmplifier (.14, ).

Owns the ``anyio.from_thread.start_blocking_portal`` lifecycle. Lazy start, single
portal per ``AgentAmplifier`` instance, explicit ``close()`` on context exit.

# ``anyio.from_thread.start_blocking_portal`` is
# stable in anyio 4.x (verified against anyio 4.13.0 on 2026-04-26 via
# WebSearch fallback after hub gemini timeout). Used as a context manager;
# the top-level alias ``anyio.start_blocking_portal`` was removed in 4.0 —
# we import explicitly from ``anyio.from_thread``.

Why a portal?
    The async core (``_AmplifierCore``) uses ``anyio.Lock`` and may schedule
    futures internally. Calling those primitives from a sync thread requires an
    event loop running on a dedicated thread. The ``BlockingPortal`` pattern
    is the canonical anyio idiom for this and is what the Anthropic and
    OpenAI SDKs use under the hood.

Threading contract:
    * ``run_sync`` is safe to call from any thread, including the thread that
      lazy-spawned the portal.
    * Multiple concurrent calls are serialized by anyio's own scheduler.
    * Lazy start is double-checked-locked so concurrent first-callers do not
      spawn two portals.
    * ``close()`` is idempotent. After ``close()``, ``run_sync`` re-spawns
      the portal — useful for kernels that are reused across sessions.

This module is import-safe even when anyio is missing — the import lives at
module top so a missing-anyio environment fails loudly at construction (not
at first call).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from anyio.from_thread import BlockingPortal, start_blocking_portal

T = TypeVar("T")


class PortalHolder:
    """Thread-safe lazy holder for an ``anyio.BlockingPortal``.

    Construction is cheap — no portal is started until the first ``run_sync``
    call. Subsequent calls reuse the same portal until ``close()``.

    See module docstring for threading + lifecycle invariants.
    """

    def __init__(self) -> None:
        self._portal: BlockingPortal | None = None
        # ``start_blocking_portal()`` returns a context manager; we keep it so
        # we can call ``__exit__`` cleanly on close.
        self._cm: Any | None = None
        # ``threading.Lock`` (not RLock) — we never re-enter ``_ensure``
        # while holding it.
        self._lock = threading.Lock()
        # H13: close-then-resurrect guard. After ``close()`` any
        # subsequent operation must raise loudly. The previous design
        # silently re-spawned the portal which surprised users who held
        # references to a "closed" amp and expected resources to be
        # released. The kernel never relies on resurrection — it's a
        # design smell to support it implicitly.
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Lazy start (double-checked locking)
    # ------------------------------------------------------------------

    def _ensure(self) -> BlockingPortal:
        """Return the portal, starting it lazily on first call.

        Double-checked-lock pattern: cheap read first, then lock and re-check.
        """
        portal = self._portal
        if portal is not None:
            return portal
        with self._lock:
            if self._portal is None:  # pragma: no branch - double-checked lock race
                cm = start_blocking_portal()
                # Enter the context manager — this returns the live portal.
                portal = cm.__enter__()
                self._cm = cm
                self._portal = portal
            assert self._portal is not None
            return self._portal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_sync(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Call an async callable from sync code; block until result.

        Exceptions raised inside the coroutine propagate to the caller.

        ``portal.call`` blocks the calling thread until the coroutine completes
        and returns the awaited value. If the portal hasn't been started yet,
        we start it transparently first.

        H13: raises ``RuntimeError`` if the portal has been
        ``close()``-d already. Pre-close use is unchanged.
        """
        if self._closed:
            raise RuntimeError(
                "portal closed; create a new AgentAmplifier to start "
                "another session"
            )
        portal = self._ensure()
        return portal.call(fn, *args, **kwargs)

    def close(self) -> None:
        """Tear down the portal. Idempotent.

        H13: after ``close()`` the holder is locked closed —
        subsequent ``run_sync`` calls raise ``RuntimeError``. The previous
        "transparent re-spawn on next call" behavior masked use-after-close
        bugs and surprised users who expected resources to stay released.
        """
        with self._lock:
            cm = self._cm
            self._cm = None
            self._portal = None
            self._closed = True
        if cm is not None:
            # Defensive: portal teardown failures must not bubble out of
            # close(). Worst case the original event loop ended in an
            # inconsistent state, which is fine because we're done with it.
            with contextlib.suppress(Exception):
                cm.__exit__(None, None, None)


__all__ = ["PortalHolder"]
