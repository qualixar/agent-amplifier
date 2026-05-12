# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Optional observability sinks (.10,  ship-along).

Two ship-along observer classes:
    * :class:`RichConsoleObserver` — pretty-printed terminal output. Requires
      the optional extra ``agent-amplifier[pretty]`` (depends on ``rich``).
    * :class:`PrintObserver` — zero-dependency fallback that writes one
      line per event to stdout.

Both are drop-in ``observability_callback`` values. Plug into
``AmplifierConfig`` like::

    from agent_amplifier import AmplifierConfig, AgentAmplifier
    from agent_amplifier.observers import RichConsoleObserver

    cfg = AmplifierConfig(observability_callback=RichConsoleObserver())
    amp = AgentAmplifier(cfg)

The kernel guarantees observers are invoked OUTSIDE its lock, so observer
work runs on the same OS thread as the originating ``before_step`` /
``after_step``. Implementations MUST complete in <5 ms or push to a
``queue.Queue`` to avoid blocking the kernel hot path.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from agent_amplifier.types import AmplifierEvent

LOG = logging.getLogger("agent_amplifier.observers")


# ---------------------------------------------------------------------------
# PrintObserver — zero-dependency fallback
# ---------------------------------------------------------------------------


class PrintObserver:
    """Zero-dependency stdout observer. Always available.

    One line per event. Errors during stringification are swallowed
    (the observer never crashes the kernel).
    """

    def __call__(
        self, event: AmplifierEvent, payload: dict[str, Any]
    ) -> None:
        try:
            # Compact, grep-able single-line format.
            keypairs = " ".join(
                f"{k}={_safe_str(v)}" for k, v in payload.items()
            )
            print(f"[amp] {event.value}\t{keypairs}")
        except Exception:
            # Belt-and-suspenders — observer crashes are never user-visible.
            with contextlib.suppress(Exception):
                print(f"[amp] {event.value}")


# ---------------------------------------------------------------------------
# RichConsoleObserver — pretty terminal output (optional extra)
# ---------------------------------------------------------------------------


class RichConsoleObserver:
    """Pretty-print amplifier events to the terminal. Requires ``rich``.

    Usage::

        pip install agent-amplifier[pretty]

        from agent_amplifier.observers import RichConsoleObserver
        cfg = AmplifierConfig(observability_callback=RichConsoleObserver())

    The constructor lazily imports ``rich`` so the module remains importable
    in environments without the optional extra.
    """

    def __init__(self) -> None:
        try:
            # STAGE-5C-COV-01: ``rich`` is in the optional ``[pretty]`` extra
            # but is also a dev dep so the rich path gets coverage. The
            # ``unused-ignore`` suppression keeps strict mypy happy in BOTH
            # environments (rich installed + rich missing).
            from rich.console import Console  # type: ignore[import-not-found, unused-ignore]
        except ImportError as e:
            raise ImportError(
                "RichConsoleObserver requires rich. Install with "
                "pip install agent-amplifier[pretty]"
            ) from e
        self._console: Any = Console()

    def __call__(
        self, event: AmplifierEvent, payload: dict[str, Any]
    ) -> None:
        try:
            self._dispatch(event, payload)
        except Exception:
            # Observer must not crash the kernel.
            LOG.warning("RichConsoleObserver: render failed; continuing.")

    def _dispatch(
        self, event: AmplifierEvent, payload: dict[str, Any]
    ) -> None:
        if event is AmplifierEvent.BEFORE_STEP:
            self._console.print(
                f"[amp] iter {payload.get('iteration')} "
                f"{payload.get('phase')}\t"
                f"effort={payload.get('effort')}\t"
                f"trigger={payload.get('thinking_trigger')}\t"
                f"tools={','.join(payload.get('recommended_groups', []) or [])}"
            )
        elif event is AmplifierEvent.ON_CONVERGE:
            self._console.print(
                f"[amp] iter {payload.get('iteration')} "
                f"converged at iteration {payload.get('iteration')}",
                style="green",
            )
        elif event is AmplifierEvent.ON_DRIFT:
            drift = payload.get("drift", 0.0)
            self._console.print(
                f"[amp] DRIFT  drift={float(drift):.2f}  re-anchoring",
                style="yellow",
            )
        elif event is AmplifierEvent.ON_BUDGET_HIT:
            self._console.print(
                f"[amp] BUDGET HIT used={payload.get('used')} "
                f"of {payload.get('limit')}",
                style="red",
            )
        elif event is AmplifierEvent.ON_BUDGET_LOW:
            self._console.print(
                f"[amp] BUDGET LOW {payload}",
                style="dim",
            )
        # Other events: silent — the user can still grep INFO logs.


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_default_observer() -> Any:
    """Return :class:`RichConsoleObserver` when ``rich`` is importable, else
    :class:`PrintObserver`.

    Used by ``AGENT_AMP_VERBOSE=1`` env-var auto-wiring (.10).
    """
    try:
        return RichConsoleObserver()
    except ImportError:
        return PrintObserver()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_str(v: Any) -> str:
    try:
        return str(v)
    except Exception:
        return f"<unprintable {type(v).__name__}>"


__all__ = [
    "PrintObserver",
    "RichConsoleObserver",
    "make_default_observer",
]
