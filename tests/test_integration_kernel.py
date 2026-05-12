# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Integration test for the Cluster-E kernel pipeline (.4, ).

Per the spec rewrite of §5.4: SLM disabled, monotonic clock frozen, canned
outputs, parsed envelope assertions (NOT substring) where feasible.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from agent_amplifier.kernel import AgentAmplifier
from agent_amplifier.types import EffortLevel


def test_kernel_4_iteration_convergence_with_canned_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    clock_iter: Iterator[float] = iter(
        [1.0, 1.05, 2.0, 2.05, 3.0, 3.05, 4.0, 4.05, 5.0]
    )

    def _fake_monotonic() -> float:
        try:
            return next(clock_iter)
        except StopIteration:
            return 6.0

    monkeypatch.setattr(time, "monotonic", _fake_monotonic)
    monkeypatch.setenv("AGENT_AMP_FALLBACK_PHASE", "EXPLORE")

    # V2.1: no memory provider wired — universal memory plane returns []
    # by default, so the integration test does no I/O.
    kernel = AgentAmplifier()
    try:
        canned = [
            "Initial exploration of payment service security.",
            "Evaluation: identified 3 risk vectors.",
            "Execute: apply the 3 fixes; security posture: GOOD.",
            "Execute: apply the 3 fixes; security posture: GOOD.",
        ]
        query = "audit security of payment service"
        decision = None
        last_env_iteration = -1
        for i in range(4):
            env = kernel.before_step(
                query,
                {"available_tools": ["Read", "WebSearch", "Bash"]},
            )
            # Iteration 0 must classify HIGH or MAX given the security
            # keywords. (Effort router is content-driven, not iteration.)
            if i == 0:
                assert env.classification.complexity in (
                    EffortLevel.HIGH,
                    EffortLevel.MAX,
                )
                # Modifier envelope present at HIGH/MAX.
                assert "<system-reminder" in env.envelope
            assert env.iteration >= last_env_iteration
            last_env_iteration = env.iteration
            decision = kernel.after_step(env, canned[i])
            if decision.get("action") == "stop":
                break

        assert decision is not None
        assert decision["action"] == "stop"
        # Identical canned outputs at i=2,i=3 must converge.
        assert decision["reason"] == "converged"

        report = kernel.finalize()
        assert report["iterations_completed"] >= 1
        assert report["final_state"] == "converged"
    finally:
        kernel.close()


def test_kernel_step_id_distinct_under_concurrent_async() -> None:
    """parallel before_step calls each get a unique, monotone step_id."""
    import asyncio

    from agent_amplifier.kernel import AsyncAgentAmplifier

    async def _go() -> list[int]:
        amp = AsyncAgentAmplifier()
        try:
            envs = await asyncio.gather(
                *(amp.before_step(f"q{i}") for i in range(10))
            )
            return [e.step_id for e in envs]
        finally:
            await amp.aclose()

    ids = asyncio.run(_go())
    assert sorted(ids) == list(range(1, 11))
    assert len(set(ids)) == 10
