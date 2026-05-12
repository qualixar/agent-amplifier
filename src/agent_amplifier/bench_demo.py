# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Single-prompt amplification preview for the launch GIF (H-6).

Backs two CLI surfaces that share one rendering function:

  * ``agent-amp bench --prompt PROMPT [--baseline] [--vs-amplified]``
    — matches the H-6 spec verbatim; integrates with the existing
    ``bench`` subcommand for users who already know it.
  * ``agent-amp demo <PROMPT>`` — friendly alias that always renders the
    full before/after; positional prompt for fast launch-GIF capture.

Both delegate to :func:`run_demo`. No LLM and no network — the function
calls the local kernel's ``before_step`` to materialize a real envelope,
then renders it side-by-side with the unamplified prompt so the reader
can see exactly what amplification adds.
"""
from __future__ import annotations

import sys

from agent_amplifier import AgentAmplifier
from agent_amplifier.kernel import StepEnvelope


def run_demo(
    prompt: str,
    *,
    show_baseline: bool = True,
    show_amplified: bool = True,
) -> int:
    """Render a baseline-vs-amplified preview for ``prompt``.

    Returns 0 on success, 1 on empty / whitespace-only prompt.

    When both ``show_baseline`` and ``show_amplified`` are True (the
    default), prints a one-line delta summary at the end so the GIF can
    cut on a quantitative beat.
    """
    if not prompt or not prompt.strip():
        print("demo: prompt must be non-empty", file=sys.stderr)
        return 1

    env: StepEnvelope | None = None
    if show_amplified:
        amp = AgentAmplifier(adapter=None)
        try:
            env = amp.before_step(prompt)
        finally:
            amp.close()

    if show_baseline:
        _print_baseline(prompt)
    if env is not None:
        _print_amplified(prompt, env)
    if show_baseline and env is not None:
        _print_delta(prompt, env)
    return 0


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def _print_baseline(prompt: str) -> None:
    print("Baseline:")
    print(f"  {prompt}")
    print(f"  ({len(prompt)} chars, no amplification)")
    print()


def _print_amplified(prompt: str, env: StepEnvelope) -> None:
    print("Amplified:")
    print(
        f"  [classification: complexity={env.classification.complexity.value}, "
        f"domain={env.classification.domain}]"
    )
    print(f"  [phase: {env.phase}]")
    if env.thinking_trigger:
        print(f"  [thinking trigger: {env.thinking_trigger}]")
    if env.persona:
        persona = env.persona if len(env.persona) <= 80 else env.persona[:80] + "..."
        print(f"  [persona: {persona}]")
    if env.recommended_groups:
        print(f"  [tool groups: {', '.join(env.recommended_groups)}]")
    print()
    print("  --- envelope text (what the model sees) ---")
    for line in env.envelope.splitlines():
        print(f"  {line}")
    print()
    print(
        f"  ({len(env.envelope)} chars of amplification framing + "
        f"{len(prompt)} chars of prompt)"
    )
    print()


def _print_delta(prompt: str, env: StepEnvelope) -> None:
    amp_chars = len(env.envelope) + len(prompt)
    ratio = round(amp_chars / max(1, len(prompt)), 1)
    print(
        f"Delta: {ratio}x prompt size; "
        f"adds classification, phase, persona, modifiers, goal anchor"
    )


__all__ = ["run_demo"]
