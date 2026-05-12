# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Perf gate: ``classify`` P99 < 2 ms (.8 / ).

Marked ``@pytest.mark.perf`` and excluded from the default pytest invocation
(see ``pyproject.toml``: ``addopts = "-ra --strict-markers"`` + ``-m "not
perf"`` in CI default). Run explicitly with ``pytest -m perf``.
"""

from __future__ import annotations

import statistics
import time

import pytest

from agent_amplifier.effort_router import classify

PERF_QUERIES: list[str] = [
    "fix typo",                                  # MINIMAL
    "fix this bug in my function",               # LOW
    "refactor this function to use async",       # MEDIUM
    "audit security of authentication module",   # HIGH (single MAX kw)
    "audit security cve owasp injection",        # MAX (>=2 distinct)
    "x" * 1000,                                  # length stress
    "the " * 250,                                # 1000-char filler
]


@pytest.mark.perf
def test_classify_p99_under_2ms() -> None:
    """ perf gate: P99 < 2 ms across representative queries.

    Burns 200 iterations per query then computes P99 across all timings.
    Failures here block merge (gated separately from the unit suite).
    """
    timings_ms: list[float] = []
    # Warm-up — exclude from timings.
    for q in PERF_QUERIES:
        for _ in range(20):
            classify(q)
    for q in PERF_QUERIES:
        for _ in range(200):
            t0 = time.perf_counter()
            classify(q)
            timings_ms.append((time.perf_counter() - t0) * 1000)

    p50 = statistics.median(timings_ms)
    # statistics.quantiles(n=100) returns 99 cut-points; index 98 ≈ P99.
    p99 = statistics.quantiles(timings_ms, n=100)[98]
    # Always include the numbers so a CI failure has the data inline.
    assert p99 < 2.0, (
        f"classify P99 = {p99:.3f} ms > 2.0 ms target (P50 = {p50:.3f} ms)"
    )


@pytest.mark.perf
def test_classify_throughput_1000_calls_under_200ms() -> None:
    """1000 random classifications must complete in < 200 ms (LLD §1.8)."""
    queries = (PERF_QUERIES * 200)[:1000]
    t0 = time.perf_counter()
    for q in queries:
        classify(q)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 200, (
        f"1000 classifications took {elapsed_ms:.1f} ms > 200 ms"
    )
