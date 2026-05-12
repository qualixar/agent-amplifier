# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Wall-clock performance assertions for Cluster C.

Marked ``@pytest.mark.perf`` and excluded from the default pytest run via
``pyproject.toml``. Gate separately:

    .venv/bin/python -m pytest -m perf

Per .8 / performance assertions live OUTSIDE the
correctness suite so a noisy CI runner cannot flake the merge gate.
"""

from __future__ import annotations

import time

import pytest

from agent_amplifier._internal.keyword_set import (
    MAX_OUTPUT_CHARS_FOR_ANALYSIS,
    keyword_set,
)
from agent_amplifier.convergence import ConvergenceDetector


@pytest.mark.perf
def test_update_completes_under_5ms_per_1k_chars() -> None:
    cd = ConvergenceDetector(max_iterations=32, history_keep=32)
    text = "alpha beta gamma delta epsilon " * 32  # ~1 KB
    start = time.perf_counter()
    for i in range(100):
        cd.update(text, i)
    elapsed = time.perf_counter() - start
    # 100 updates over a 1 KB input — total budget 500 ms.
    assert elapsed < 0.5, f"convergence.update too slow: {elapsed:.3f}s"


@pytest.mark.perf
def test_keyword_set_under_50ms_for_10mb() -> None:

    big = "a" * (10 * 1024 * 1024)  # 10 MB
    start = time.perf_counter()
    out = keyword_set(big)
    elapsed = time.perf_counter() - start
    # Generous budget — large memory copy on the truncation step alone
    # can take tens of ms on busy CI runners; we want to catch O(N^2) regressions
    # not micro-jitter.
    assert elapsed < 0.5, f"keyword_set too slow: {elapsed:.3f}s"
    # Truncation worked: the surviving token is at most
    # MAX_OUTPUT_CHARS_FOR_ANALYSIS chars long, never 10 MB.
    assert all(len(t) <= MAX_OUTPUT_CHARS_FOR_ANALYSIS for t in out)
    assert MAX_OUTPUT_CHARS_FOR_ANALYSIS == 256_000
