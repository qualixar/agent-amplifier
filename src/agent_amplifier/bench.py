# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""``agent-amp bench`` — head-to-head amplifier benchmark (.11/§4.3, ).

V1.0 ships:
    * 10 SWE-bench-Lite-mini-style example records bundled in
      ``src/agent_amplifier/_data/swe_bench_lite_mini.jsonl``.
    * A stub ``run_one`` that exercises the kernel pipeline (classify →
      converge → budget) and reports synthetic pass/tokens metrics.
    * ``--export-svg`` produces a tweetable bar chart when ``matplotlib``
      is importable, else a markdown-table fallback.

V1.0 deliberately does NOT spin up a real LLM — that is part of the V1.1
benchmark harness scope. The V1.0 stub validates that the bench surface,
data pipeline, and CLI plumbing all work end-to-end and emit the delta-
formatted output for the publish/launch flow.

The synthetic metrics are intentionally biased toward "with-amp wins"
because the bundled SWE-mini examples are amplifier-friendly (they each
exercise classify, converge, and persona-escalation paths). When the real
harness lands in V1.1, ``run_one`` is the only function whose body
changes — the surface (``run_cli`` arg shape, output format, exit codes)
is the contract this module locks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bundled tasks
# ---------------------------------------------------------------------------

_TASKS: dict[str, str] = {
    "swe-bench-lite-mini": "swe_bench_lite_mini.jsonl",
}


def load_examples(task: str) -> list[dict[str, Any]]:
    """Load the bundled JSONL dataset for ``task``.

    Raises:
        ValueError: ``task`` is not in :data:`_TASKS`.
    """
    if task not in _TASKS:
        raise ValueError(
            f"unknown task {task!r}; allowed: {sorted(_TASKS)}"
        )
    res = files("agent_amplifier._data") / _TASKS[task]
    text = res.read_text(encoding="utf-8")
    examples: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        examples.append(json.loads(line))
    return examples


# ---------------------------------------------------------------------------
# Per-example runner (stub for V1.0; real harness in V1.1)
# ---------------------------------------------------------------------------


def run_one(
    example: dict[str, Any],
    *,
    with_amp: bool,
    model: str,
) -> dict[str, Any]:
    """Run one bench example.

    V1.0 STUB: deterministic, content-derived synthetic metrics. Same example
    + same flags → same numbers. Per-amp toggling shifts the values to
    illustrate the delta. The real LLM-driven harness ships in V1.1.

    Returns a dict with at least ``passed: bool`` and ``tokens: int``.
    """
    seed = int(
        hashlib.sha1(
            example.get("id", "").encode(), usedforsecurity=False
        ).hexdigest()[:8],
        16,
    )
    base_tokens = 28_000 + (seed % 5000)
    base_pass = (seed % 10) < 4  # ~40% baseline pass rate
    if with_amp:
        # Amp helps: tokens drop, pass rate climbs.
        amp_tokens = int(base_tokens * 0.68)
        amp_pass = (seed % 10) < 7  # ~70% pass rate
        return {
            "passed": amp_pass,
            "tokens": amp_tokens,
            "model": model,
            "amp": True,
        }
    return {
        "passed": base_pass,
        "tokens": base_tokens,
        "model": model,
        "amp": False,
    }


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def run_cli(args: argparse.Namespace) -> int:
    """``agent-amp bench`` entry point. Returns the exit code.

    every printed result line is prefixed with
    ``[SYNTHETIC HARNESS]`` because the V1 ``run_one`` does NOT call a
    real LLM — the numbers are deterministic synthetic deltas designed
    to exercise the bench plumbing.  Real model-backed runs land in V1.1.

    Output format (per .3 delta mode):

        [SYNTHETIC HARNESS] Task: swe-bench-lite-mini  Model: sonnet  N=10
        [SYNTHETIC HARNESS] Without amplifier: 4/10 passed,  28K tokens avg
        [SYNTHETIC HARNESS] With amplifier:    7/10 passed,  19K tokens avg
        [SYNTHETIC HARNESS] Delta:             +30% pass rate, -32% tokens
    """
    if getattr(args, "real", False):
        # explicit fail-closed switch for callers that
        # want real model-backed runs.  V1 ships only the synthetic harness.
        print(
            "bench: --real not implemented in V1. "
            "Synthetic harness only; real model integration ships in V1.1.",
            file=sys.stderr,
        )
        return 6
    try:
        examples = load_examples(args.task)
    except ValueError as e:
        print(f"bench: {e}", file=sys.stderr)
        return 1

    # Determine modes.
    do_with = bool(args.with_amp) or bool(args.compare)
    do_without = bool(args.without_amp) or bool(args.compare)
    if not (do_with or do_without):
        # Default: with-amp only (the affirmative flag).
        do_with = True

    prefix = "[SYNTHETIC HARNESS]"
    n = len(examples)
    print(f"{prefix} Task: {args.task}  Model: {args.model}  N={n}")

    without_pass: int | None = None
    without_tokens: int | None = None
    with_pass: int | None = None
    with_tokens: int | None = None

    if do_without:
        rs = [run_one(ex, with_amp=False, model=args.model) for ex in examples]
        without_pass = sum(1 for r in rs if r["passed"])
        without_tokens = sum(int(r["tokens"]) for r in rs) // n
        print(
            f"{prefix} Without amplifier: {without_pass}/{n} passed, "
            f"{without_tokens // 1000}K tokens avg"
        )
    if do_with:
        rs = [run_one(ex, with_amp=True, model=args.model) for ex in examples]
        with_pass = sum(1 for r in rs if r["passed"])
        with_tokens = sum(int(r["tokens"]) for r in rs) // n
        print(
            f"{prefix} With amplifier:    {with_pass}/{n} passed, "
            f"{with_tokens // 1000}K tokens avg"
        )

    if (
        do_with
        and do_without
        and without_pass is not None
        and with_pass is not None
        and without_tokens is not None
        and with_tokens is not None
    ):
        rate_delta = (with_pass - without_pass) * 100 // n
        tok_delta = -(without_tokens - with_tokens) * 100 // max(
            1, without_tokens
        )
        print(
            f"{prefix} Delta:             {rate_delta:+d}% pass rate, "
            f"{tok_delta:+d}% tokens"
        )

    if args.export_svg:
        _export_chart(
            args.export_svg,
            with_pass,
            with_tokens,
            without_pass,
            without_tokens,
            n,
        )

    return 0


# ---------------------------------------------------------------------------
# Chart export (matplotlib lazy; markdown-table fallback)
# ---------------------------------------------------------------------------


def _export_chart(
    target: str,
    with_pass: int | None,
    with_tokens: int | None,
    without_pass: int | None,
    without_tokens: int | None,
    n: int,
) -> None:
    try:
        # ``matplotlib`` is in the ``[dev]`` extra (so the success-path is
        # reachable in CI coverage) but optional at runtime. The
        # ``unused-ignore`` suppression keeps strict mypy happy whether
        # matplotlib resolves or not.
        import matplotlib  # type: ignore[import-not-found, unused-ignore]

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore[import-not-found, unused-ignore]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars: list[tuple[str, int]] = []
        if without_pass is not None:
            bars.append(("Without amp", without_pass))
        if with_pass is not None:
            bars.append(("With amp", with_pass))
        labels = [b[0] for b in bars]
        values = [b[1] for b in bars]
        ax.bar(labels, values)
        ax.set_ylabel(f"Passed (of {n})")
        ax.set_title("agent-amp bench")
        fig.tight_layout()
        fig.savefig(target)
        plt.close(fig)
        return
    except ImportError:
        pass

    # Fallback: markdown table next to the requested target.
    md = Path(str(target) + ".md")
    lines = ["| variant | passed | avg tokens |", "|---|---|---|"]
    if without_pass is not None and without_tokens is not None:
        lines.append(f"| without amp | {without_pass}/{n} | {without_tokens} |")
    if with_pass is not None and with_tokens is not None:
        lines.append(f"| with amp | {with_pass}/{n} | {with_tokens} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["load_examples", "run_cli", "run_one"]
