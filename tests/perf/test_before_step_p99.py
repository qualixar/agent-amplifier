# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Wall-clock performance gate for ``AgentAmplifier.before_step``.

H10 — closes Skeptic-H finding "kernel before_step P99 perf
gate test file does not exist ( §5.5)".

Marked ``@pytest.mark.perf`` and excluded from the default pytest run via
``pyproject.toml``. Gate separately::

    .venv/bin/python -m pytest -m perf

Budget: P99 < 5 ms per ``before_step`` call on a stub adapter (no real
memory I/O). The kernel is pure CPU + lock + dispatch; anything slower is
a regression worth investigating.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import pytest

from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.kernel import AgentAmplifier
from agent_amplifier.types import Outcome, RecalledPattern


class _StubAdapter(AdapterBase):
    """Zero-cost adapter for the perf gate. No I/O, returns empty recall."""

    framework_name = "perf_stub"
    version = "0.0.1"

    def install(self) -> None:  # pragma: no cover - perf gate
        self._mark_installed()

    def uninstall(self) -> None:  # pragma: no cover - perf gate
        self._mark_uninstalled()

    def on_before_step(self, context: dict[str, Any]) -> dict[str, Any]:
        return context

    def on_after_step(
        self, context: dict[str, Any], result: dict[str, Any] | str
    ) -> dict[str, Any]:
        return {"action": "continue"}

    def default_memory_recall(
        self, query: str, limit: int = 3
    ) -> list[RecalledPattern]:
        return []

    def default_memory_remember(self, outcome: Outcome) -> None:
        return None


PERF_QUERIES: list[str] = [
    "hello world",
    "fix typo in docstring",
    "refactor function to use async",
    "audit security of authentication",
    "x" * 256,
]


@pytest.mark.perf
def test_before_step_p99_under_5ms() -> None:
    """``before_step`` P99 < 5 ms across representative queries.

    100 warm-up iterations per query (excluded), then 200 timed iterations.
    Budget chosen to leave headroom for noisy CI runners while still
    catching real regressions (typical local: P99 ~1-2 ms).
    """
    amp = AgentAmplifier(adapter=_StubAdapter(kernel=None))
    timings_ms: list[float] = []
    try:
        # Warm-up
        for q in PERF_QUERIES:
            for _ in range(20):
                amp.before_step(q, {"available_tools": []})
        # Measure
        for q in PERF_QUERIES:
            for _ in range(200):
                t0 = time.perf_counter()
                amp.before_step(q, {"available_tools": []})
                timings_ms.append((time.perf_counter() - t0) * 1000)
    finally:
        amp.close()

    p50 = statistics.median(timings_ms)
    p99 = statistics.quantiles(timings_ms, n=100)[98]
    assert p99 < 5.0, (
        f"before_step P99 = {p99:.3f} ms > 5.0 ms budget "
        f"(P50 = {p50:.3f} ms)"
    )


@pytest.mark.perf
def test_before_step_throughput_1000_calls_under_3s() -> None:
    """1000 ``before_step`` calls must complete in < 3 s wall-clock."""
    amp = AgentAmplifier(adapter=_StubAdapter(kernel=None))
    queries = (PERF_QUERIES * 200)[:1000]
    try:
        t0 = time.perf_counter()
        for q in queries:
            amp.before_step(q, {"available_tools": []})
        elapsed = time.perf_counter() - t0
    finally:
        amp.close()
    assert elapsed < 3.0, (
        f"1000 before_step calls took {elapsed:.2f} s > 3.0 s budget"
    )
