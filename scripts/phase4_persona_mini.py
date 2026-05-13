#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Phase 4 supplementary — persona-variance mini-bench (Sonnet 4.6 + AA).

Tests whether forcing the AA envelope to use different persona levels
(LEVEL_0 = generalist senior eng → LEVEL_3 = AI safety distinguished
engineer) changes Sonnet 4.6 output quality on the same task.

Scope: 3 tasks × 4 personas × 1 arm (Sonnet+AA only) = 12 candidate +
12 judge = 24 calls. Sonnet-raw and Opus-raw baselines are NOT re-run
here — those numbers come from the main bench (phase4_results.json).

Reuses ``foundry_call``, ``parse_judge_score``, ``build_judge_prompt``,
``JUDGE_SYSTEM`` from ``phase4_agentic_bench`` for consistency. Persona
override happens by replacing the ``PERSONA: ...`` line in AA's default
envelope with the chosen level's role string.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from agent_amplifier import AgentAmplifier  # noqa: E402
from agent_amplifier.personas import LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3  # noqa: E402
from phase4_agentic_bench import (  # noqa: E402
    JUDGE_SYSTEM,
    _foundry_endpoints,
    build_judge_prompt,
    foundry_call,
    parse_judge_score,
    PRICING_USD_PER_M,
)

LEVELS = (LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3)
DEFAULT_TASK_IDS = ("ag-001", "ag-002", "ag-005")


_PERSONA_LINE_RE = re.compile(r"^PERSONA:.*$", re.MULTILINE)


def amplified_prompt_for_level(prompt: str, level_role: str) -> str:
    """Build the Sonnet+AA prompt with the persona line overridden."""
    aa = AgentAmplifier()
    step = aa.before_step(prompt)
    envelope = step.envelope
    if _PERSONA_LINE_RE.search(envelope):
        envelope = _PERSONA_LINE_RE.sub(f"PERSONA: {level_role}", envelope, count=1)
    else:
        envelope = envelope.rstrip() + f"\nPERSONA: {level_role}\n"
    return f"{envelope}\n\n{prompt}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--task-ids", nargs="*", default=list(DEFAULT_TASK_IDS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-spend", type=float, default=5.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--candidate-max-tokens", type=int, default=2048)
    ap.add_argument("--judge-max-tokens", type=int, default=256)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    all_tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    by_id = {t["task_id"]: t for t in all_tasks}
    tasks = [by_id[t] for t in args.task_ids if t in by_id]
    missing = [t for t in args.task_ids if t not in by_id]
    if missing:
        print(f"ERROR: task ids not in corpus: {missing}", file=sys.stderr)
        return 2

    subs = _foundry_endpoints()
    if not subs:
        print("ERROR: no Foundry subs configured", file=sys.stderr)
        return 2

    # Cost projection
    n_candidate = len(tasks) * len(LEVELS)
    n_judge = n_candidate
    # heuristic: amplified prompt ~1000 input tokens, output ~700, judge ~2000 in, 80 out
    proj_in_c = 1000 * n_candidate
    proj_out_c = 700 * n_candidate
    cost_c = (proj_in_c / 1_000_000) * PRICING_USD_PER_M["claude-sonnet-4-6"]["in"] + \
             (proj_out_c / 1_000_000) * PRICING_USD_PER_M["claude-sonnet-4-6"]["out"]
    proj_in_j = 2000 * n_judge
    proj_out_j = 80 * n_judge
    cost_j = (proj_in_j / 1_000_000) * PRICING_USD_PER_M["claude-opus-4-7"]["in"] + \
             (proj_out_j / 1_000_000) * PRICING_USD_PER_M["claude-opus-4-7"]["out"]
    total = cost_c + cost_j

    print(f"Phase 4 persona mini — {len(tasks)} task(s) × {len(LEVELS)} personas × Sonnet+AA")
    print(f"projected calls: {n_candidate + n_judge} ({n_candidate} candidate + {n_judge} judge)")
    print(f"projected spend: ${total:.4f} USD (max-spend ${args.max_spend:.2f})")
    print()

    if args.dry_run:
        print("--dry-run: not dispatching.")
        return 0
    if total > args.max_spend:
        print(f"ABORT: projected ${total:.4f} > max-spend ${args.max_spend:.2f}", file=sys.stderr)
        return 3
    if not args.yes:
        if input(f"Spend ~${total:.2f} USD? [y/N] ").strip().lower() != "y":
            print("aborted.")
            return 0
    else:
        print(f"--yes: dispatching ${total:.2f} USD spend.")

    started = time.monotonic()
    rows: list[dict[str, Any]] = []

    # Phase A — candidate calls
    candidate_jobs: list[tuple[dict[str, Any], Any, dict[str, str]]] = []
    for i, task in enumerate(tasks):
        for j, level in enumerate(LEVELS):
            sub = subs[(i * len(LEVELS) + j) % len(subs)]
            candidate_jobs.append((task, level, sub))

    print(f"\n[A] dispatching {len(candidate_jobs)} Sonnet+AA candidate calls ...")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {}
        for task, level, sub in candidate_jobs:
            prompt = amplified_prompt_for_level(task["prompt"], level.role)
            futs[pool.submit(
                foundry_call,
                model="claude-sonnet-4-6",
                prompt=prompt,
                max_tokens=args.candidate_max_tokens,
                sub=sub,
            )] = (task, level)
        for fut in as_completed(futs):
            task, level = futs[fut]
            r = fut.result()
            err = f" ERR={r.error}" if r.error else ""
            print(f"  [A] {task['task_id']} L{level.level} sub={r.sub_name} in={r.input_tokens} out={r.output_tokens} dt={r.elapsed_s:.1f}s{err}")
            rows.append({
                "task_id": task["task_id"],
                "category": task["category"],
                "persona_level": level.level,
                "persona_strictness": level.strictness,
                "persona_role": level.role,
                "candidate": {
                    "text": r.text,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "sub": r.sub_name,
                    "elapsed_s": r.elapsed_s,
                    "error": r.error,
                },
            })

    # Phase B — judge calls
    print(f"\n[B] dispatching {len(rows)} judge calls ...")
    tasks_by_id = {t["task_id"]: t for t in tasks}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs2 = {}
        for idx, row in enumerate(rows):
            sub = subs[idx % len(subs)]
            task = tasks_by_id[row["task_id"]]
            judge_prompt = build_judge_prompt(
                task_prompt=task["prompt"],
                gold_summary=task["gold_summary"],
                candidate=row["candidate"]["text"] or "(empty)",
            )
            futs2[pool.submit(
                foundry_call,
                model="claude-opus-4-7",
                prompt=judge_prompt,
                max_tokens=args.judge_max_tokens,
                sub=sub,
                system=JUDGE_SYSTEM,
            )] = (idx, row)
        for fut in as_completed(futs2):
            idx, row = futs2[fut]
            r = fut.result()
            score, just = parse_judge_score(r.text)
            err = f" ERR={r.error}" if r.error else ""
            print(f"  [B] {row['task_id']} L{row['persona_level']} score={score}/5 in={r.input_tokens} out={r.output_tokens}{err}")
            row["judge"] = {
                "raw": r.text,
                "score": score,
                "justification": just,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "sub": r.sub_name,
                "elapsed_s": r.elapsed_s,
                "error": r.error,
            }

    total_s = time.monotonic() - started

    out_doc = {
        "schema": "agent-amplifier.phase4-persona-mini.v1",
        "started_at": int(time.time()),
        "total_elapsed_s": round(total_s, 2),
        "task_ids": args.task_ids,
        "personas": [
            {"level": L.level, "strictness": L.strictness, "role": L.role}
            for L in LEVELS
        ],
        "arm": "sonnet_amp",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2))
    print(f"\nresults → {args.out}")

    # Quick per-persona summary
    print("\nPer-persona mean (n=3 per persona):")
    for L in LEVELS:
        scores = [r["judge"]["score"] for r in rows if r["persona_level"] == L.level and "judge" in r]
        if scores:
            print(f"  L{L.level} (strictness {L.strictness}): mean={sum(scores)/len(scores):.3f}  scores={scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
