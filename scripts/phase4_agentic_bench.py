#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Phase 4 — hybrid agentic A/B benchmark for Agent Amplifier v1.1 → v1.2 verdict.

Three arms × 12 tasks × Opus-as-judge with blind output ordering and a 5-point
rubric. Verdict via AgentAssay's ``evaluate_quality_distributions`` (Mann-Whitney
U on continuous judge scores).

Foundry routing
---------------
Azure AI Foundry endpoints (URL pattern verified 2026-05-13):

    {endpoint}/anthropic/v1/messages?api-version=2024-10-22-preview

with header ``Authorization: Bearer <api-key>`` and ``anthropic-version: 2023-06-01``.

Three subscriptions configured (load-balanced round-robin in live mode):

  * gtic      — gtic-resource.services.ai.azure.com           (Sweden Central)
  * tap       — tap-main-project-resource.cognitiveservices…  (East US 2)
  * hm        — tap-aoai-dev-resource.cognitiveservices…      (East US 2)

Cost-gated
----------
``--dry-run`` stages all 72 calls, counts tokens (input deterministic, output
estimated at ``--est-output-tokens``), prints a per-arm + total cost projection,
and exits without dispatching anything.

Live mode requires the projected cost to be ≤ ``--max-spend``. The script also
prompts for an interactive ``y`` before any paid call fires. Defense in depth.

Usage
-----
    # dry-run (no spend)
    python scripts/phase4_agentic_bench.py \\
        --tasks .backup/revive-plan-2026-05-13/agentic_eval_v1.jsonl \\
        --out  .backup/revive-plan-2026-05-13/phase4_results.json \\
        --dry-run

    # live run
    python scripts/phase4_agentic_bench.py \\
        --tasks .backup/revive-plan-2026-05-13/agentic_eval_v1.jsonl \\
        --out  .backup/revive-plan-2026-05-13/phase4_results.json \\
        --max-spend 25.0 \\
        --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error

# Ensure src/ is importable so we can use agent_amplifier without installing.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from agent_amplifier import AgentAmplifier  # noqa: E402
from agent_amplifier.bench_verdict import (  # noqa: E402
    evaluate_quality_distributions,
    verdict_summary_line,
)


# ---------------------------------------------------------------------------
# Pricing (per 1M tokens, USD) — Anthropic public list pricing for the Foundry
# routes used here. Numbers are conservative upper bounds; actual Azure Foundry
# invoicing may be a few % below depending on subscription tier. The dry-run
# uses these to project spend.
# ---------------------------------------------------------------------------

PRICING_USD_PER_M = {
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0},
    "claude-opus-4-7": {"in": 15.0, "out": 75.0},
}


# ---------------------------------------------------------------------------
# Foundry routing
# ---------------------------------------------------------------------------


def _foundry_endpoints() -> list[dict[str, str]]:
    """Read configured Foundry subscriptions from environment.

    Returns up to three ``{name, endpoint, key}`` triples in routing order.
    Skipped if any env var is missing.
    """
    candidates = [
        ("gtic", "ANTHROPIC_FOUNDRY_API_KEY", "https://gtic-resource.services.ai.azure.com"),
        ("tap", "AZURE_TAP_API_KEY", os.environ.get("AZURE_TAP_ENDPOINT", "")),
        ("hm", "AZURE_HM_API_KEY", os.environ.get("AZURE_HM_ENDPOINT", "")),
    ]
    out: list[dict[str, str]] = []
    for name, key_env, endpoint in candidates:
        key = os.environ.get(key_env, "").strip()
        endpoint = (endpoint or "").rstrip("/")
        if key and endpoint:
            out.append({"name": name, "endpoint": endpoint, "key": key})
    return out


def _foundry_url(endpoint: str) -> str:
    return f"{endpoint}/anthropic/v1/messages?api-version=2024-10-22-preview"


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    sub_name: str
    elapsed_s: float
    error: str | None = None


def foundry_call(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    sub: dict[str, str],
    system: str | None = None,
    timeout_s: int = 600,
    enable_thinking: bool = False,
    effort: str = "medium",
) -> CallResult:
    """Single Anthropic-protocol call against an Azure Foundry endpoint.

    Note: ``temperature`` is intentionally omitted — Opus 4.7 returns
    HTTP 400 ``"temperature is deprecated for this model"`` if the field
    is present. Default sampling is used across arms.

    When ``enable_thinking`` is True, the payload includes
    ``thinking.type=adaptive`` plus ``output_config.effort`` per the
    verified v1.1.1 spec. The sonnet_amp arm sets this to True so the
    Phase 4 re-run exercises the full single-turn envelope + adaptive
    thinking together.
    """
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system is not None:
        payload["system"] = system
    if enable_thinking:
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": effort}

    req = urllib.request.Request(
        _foundry_url(sub["endpoint"]),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {sub['key']}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
        elapsed = time.monotonic() - started
        data = json.loads(body)
        content = data.get("content") or []
        text = "".join(part.get("text", "") for part in content if part.get("type") == "text")
        usage = data.get("usage") or {}
        return CallResult(
            text=text,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            model=model,
            sub_name=sub["name"],
            elapsed_s=elapsed,
        )
    except urllib.error.HTTPError as e:  # pragma: no cover - network branch
        body = e.read().decode("utf-8", errors="replace")
        return CallResult(
            text="",
            input_tokens=0,
            output_tokens=0,
            model=model,
            sub_name=sub["name"],
            elapsed_s=time.monotonic() - started,
            error=f"HTTP {e.code}: {body[:300]}",
        )
    except Exception as e:  # pragma: no cover - network branch
        return CallResult(
            text="",
            input_tokens=0,
            output_tokens=0,
            model=model,
            sub_name=sub["name"],
            elapsed_s=time.monotonic() - started,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Envelope builder (Sonnet + AA arm)
# ---------------------------------------------------------------------------


def aa_envelope(prompt: str) -> str:
    """Build the AA envelope prefix using the kernel + Claude Code adapter.

    The Claude Code adapter sets ``is_single_iteration=True``, which routes
    the kernel through the v1.1.1 single-turn envelope (XML phase staging,
    stage-wise persona escalation, subagent dispatch for MAX-complexity
    tasks). This is the envelope the Sonnet+AA arm of Phase 4 is measuring.
    """
    from agent_amplifier.adapters import ClaudeCodeAdapter

    aa = AgentAmplifier()
    _ = ClaudeCodeAdapter(aa)
    # Re-wrap so the adapter is bound to the kernel that .before_step uses.
    aa_with_adapter = AgentAmplifier(adapter=ClaudeCodeAdapter(AgentAmplifier()))
    step = aa_with_adapter.before_step(prompt)
    return step.envelope


def aa_amplified_prompt(prompt: str) -> str:
    """Sonnet+AA arm input = envelope + blank line + user prompt."""
    env = aa_envelope(prompt)
    return f"{env}\n\n{prompt}"


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


JUDGE_SYSTEM = (
    "You are a rigorous, calibrated evaluator scoring an AI assistant's response "
    "to an agentic engineering task. Score on this rubric (total 0-5):\n"
    "  Correctness   0-2: Is the technical content accurate and bug-free?\n"
    "  Completeness  0-2: Does it address every part of the asked task?\n"
    "  Goal-adherence 0-1: Does the response stay on the user's stated goal "
    "without drifting?\n"
    "\n"
    "Output EXACTLY one line: 'score=<int> | <one-sentence justification>'. "
    "No preamble, no other text."
)


def build_judge_prompt(*, task_prompt: str, gold_summary: str, candidate: str) -> str:
    return (
        f"TASK:\n{task_prompt}\n\n"
        f"REFERENCE (for calibration only; not shown to candidate):\n{gold_summary}\n\n"
        f"CANDIDATE RESPONSE:\n{candidate}\n"
    )


def parse_judge_score(text: str) -> tuple[int, str]:
    """Parse 'score=<int> | <reason>' — fail-open to score=0 if malformed."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    if not line.startswith("score="):
        return 0, f"unparseable judge response: {text[:120]!r}"
    rest = line[len("score=") :].strip()
    pipe = rest.find("|")
    score_str = rest[:pipe].strip() if pipe >= 0 else rest.strip()
    justification = rest[pipe + 1 :].strip() if pipe >= 0 else ""
    try:
        score = int(score_str)
    except ValueError:
        return 0, f"unparseable score field: {score_str!r}"
    score = max(0, min(5, score))
    return score, justification


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------


ARMS = (
    ("sonnet_raw", "claude-sonnet-4-6", False),
    ("sonnet_amp", "claude-sonnet-4-6", True),
    ("opus_raw", "claude-opus-4-7", False),
)


@dataclass
class TaskRecord:
    task_id: str
    category: str
    source: str
    arm_results: dict[str, dict[str, Any]] = field(default_factory=dict)


def run_task_arm(*, task: dict[str, Any], arm_name: str, model: str, amped: bool, sub: dict[str, str], max_tokens: int) -> dict[str, Any]:
    prompt = task["prompt"]
    if amped:
        prompt = aa_amplified_prompt(prompt)
    # Adaptive thinking is enabled only on the amped arm, matching v1.1.1's
    # design: the envelope and the API-layer thinking config land together.
    res = foundry_call(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        sub=sub,
        enable_thinking=amped,
        effort="high" if amped else "medium",
    )
    return {
        "arm": arm_name,
        "model": model,
        "amped": amped,
        "sub": sub["name"],
        "text": res.text,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "elapsed_s": res.elapsed_s,
        "error": res.error,
    }


def judge_arm(*, task: dict[str, Any], arm_output: dict[str, Any], sub: dict[str, str], max_tokens: int) -> dict[str, Any]:
    judge_prompt = build_judge_prompt(
        task_prompt=task["prompt"],
        gold_summary=task["gold_summary"],
        candidate=arm_output["text"] or "(empty)",
    )
    res = foundry_call(
        model="claude-opus-4-7",
        prompt=judge_prompt,
        max_tokens=max_tokens,
        sub=sub,
        system=JUDGE_SYSTEM,
    )
    score, justification = parse_judge_score(res.text)
    return {
        "raw": res.text,
        "score": score,
        "justification": justification,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "elapsed_s": res.elapsed_s,
        "error": res.error,
    }


# ---------------------------------------------------------------------------
# Cost projection
# ---------------------------------------------------------------------------


def estimate_input_tokens(text: str) -> int:
    # 1 token ≈ 4 chars heuristic. Good enough for projection (Anthropic
    # returns actual usage in the response).
    return max(1, len(text) // 4)


def project_costs(tasks: list[dict[str, Any]], est_output_tokens: int) -> dict[str, Any]:
    """Project spend for a 3-arm × 12-task × judge run.

    Returns a dict with per-arm and total USD estimates plus per-step token
    breakdown. Conservative — uses public list pricing.
    """
    rows: list[dict[str, Any]] = []
    grand_total = 0.0

    # Candidate calls
    for task in tasks:
        for arm_name, model, amped in ARMS:
            user_prompt = task["prompt"]
            if amped:
                user_prompt = aa_amplified_prompt(task["prompt"])
            in_toks = estimate_input_tokens(user_prompt)
            out_toks = est_output_tokens
            in_usd = (in_toks / 1_000_000) * PRICING_USD_PER_M[model]["in"]
            out_usd = (out_toks / 1_000_000) * PRICING_USD_PER_M[model]["out"]
            cost = in_usd + out_usd
            grand_total += cost
            rows.append({
                "task_id": task["task_id"],
                "kind": "candidate",
                "arm": arm_name,
                "model": model,
                "in_toks": in_toks,
                "out_toks": out_toks,
                "usd": cost,
            })

    # Judge calls — one per (task, arm), 36 total
    for task in tasks:
        for arm_name, _, _ in ARMS:
            judge_prompt = build_judge_prompt(
                task_prompt=task["prompt"],
                gold_summary=task["gold_summary"],
                candidate="X" * (est_output_tokens * 4),  # placeholder for size
            )
            in_toks = estimate_input_tokens(judge_prompt) + estimate_input_tokens(JUDGE_SYSTEM)
            out_toks = 80  # judge output is one line ~ 80 tokens
            in_usd = (in_toks / 1_000_000) * PRICING_USD_PER_M["claude-opus-4-7"]["in"]
            out_usd = (out_toks / 1_000_000) * PRICING_USD_PER_M["claude-opus-4-7"]["out"]
            cost = in_usd + out_usd
            grand_total += cost
            rows.append({
                "task_id": task["task_id"],
                "kind": "judge",
                "arm_judged": arm_name,
                "model": "claude-opus-4-7",
                "in_toks": in_toks,
                "out_toks": out_toks,
                "usd": cost,
            })

    # Aggregate by kind+model
    summary: dict[str, float] = {}
    for r in rows:
        key = f"{r['kind']}_{r['model']}"
        summary[key] = summary.get(key, 0.0) + r["usd"]

    return {
        "rows": rows,
        "summary": summary,
        "grand_total_usd": round(grand_total, 4),
        "total_calls": len(rows),
    }


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------


def run_live(
    *,
    tasks: list[dict[str, Any]],
    subs: list[dict[str, str]],
    out_path: Path,
    concurrency: int,
    candidate_max_tokens: int,
    judge_max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """Execute the full benchmark and persist results."""
    rng = random.Random(seed)
    started = time.monotonic()
    results: dict[str, TaskRecord] = {
        t["task_id"]: TaskRecord(task_id=t["task_id"], category=t["category"], source=t["source"])
        for t in tasks
    }

    # ---- Phase A: candidate calls (3 per task = 36 calls) ----
    candidate_jobs: list[tuple[dict[str, Any], str, str, bool, dict[str, str]]] = []
    for i, task in enumerate(tasks):
        for j, (arm_name, model, amped) in enumerate(ARMS):
            sub = subs[(i * len(ARMS) + j) % len(subs)]
            candidate_jobs.append((task, arm_name, model, amped, sub))

    print(f"[A] dispatching {len(candidate_jobs)} candidate calls (concurrency={concurrency}) ...")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {
            pool.submit(run_task_arm, task=t, arm_name=a, model=m, amped=p, sub=s, max_tokens=candidate_max_tokens): (t["task_id"], a)
            for (t, a, m, p, s) in candidate_jobs
        }
        done = 0
        for fut in as_completed(futs):
            tid, arm = futs[fut]
            out = fut.result()
            results[tid].arm_results[arm] = {"candidate": out}
            done += 1
            err = f" ERR={out['error']}" if out.get("error") else ""
            print(f"  [A {done:>2}/{len(candidate_jobs)}] {tid:<8} {arm:<11} in={out['input_tokens']:>4} out={out['output_tokens']:>4} dt={out['elapsed_s']:>5.1f}s{err}")

    # ---- Phase B: judge calls (36) — blind: shuffle arm order per task ----
    judge_jobs: list[tuple[dict[str, Any], str, dict[str, Any], dict[str, str]]] = []
    for i, task in enumerate(tasks):
        arm_order = [a for a, _, _ in ARMS]
        rng.shuffle(arm_order)
        for j, arm in enumerate(arm_order):
            sub = subs[(i * len(ARMS) + j) % len(subs)]
            judge_jobs.append((task, arm, results[task["task_id"]].arm_results[arm]["candidate"], sub))

    print(f"\n[B] dispatching {len(judge_jobs)} judge calls (concurrency={concurrency}) ...")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs2 = {
            pool.submit(judge_arm, task=t, arm_output=o, sub=s, max_tokens=judge_max_tokens): (t["task_id"], a)
            for (t, a, o, s) in judge_jobs
        }
        done = 0
        for fut in as_completed(futs2):
            tid, arm = futs2[fut]
            j = fut.result()
            results[tid].arm_results[arm]["judge"] = j
            done += 1
            err = f" ERR={j['error']}" if j.get("error") else ""
            print(f"  [B {done:>2}/{len(judge_jobs)}] {tid:<8} {arm:<11} score={j['score']}/5 in={j['input_tokens']:>4} out={j['output_tokens']:>4}{err}")

    total_elapsed = time.monotonic() - started

    # ---- Serialize ----
    out_doc: dict[str, Any] = {
        "schema": "agent-amplifier.phase4.v1",
        "started_at": int(time.time()),
        "total_elapsed_s": round(total_elapsed, 2),
        "arms": [a for a, _, _ in ARMS],
        "model_map": {a: m for a, m, _ in ARMS},
        "subscriptions": [s["name"] for s in subs],
        "tasks": [
            {
                "task_id": r.task_id,
                "category": r.category,
                "source": r.source,
                "arms": r.arm_results,
            }
            for r in results.values()
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_doc, indent=2))
    print(f"\nresults saved → {out_path}")
    return out_doc


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def compute_verdict(results: dict[str, Any]) -> str:
    """Run the AgentAssay verdict on Sonnet-raw vs Sonnet+AA and Opus-raw vs Sonnet+AA."""
    arms_scores: dict[str, list[float]] = {a: [] for a, _, _ in ARMS}
    for task in results["tasks"]:
        for arm in arms_scores:
            j = (task["arms"].get(arm) or {}).get("judge") or {}
            score = j.get("score")
            if isinstance(score, (int, float)):
                arms_scores[arm].append(float(score))

    out_lines: list[str] = []
    out_lines.append("# Phase 4 Verdict — Agent Amplifier v1.1 → v1.2")
    out_lines.append("")
    out_lines.append("## Per-arm score summary")
    out_lines.append("")
    out_lines.append("| Arm | n | mean | min | max |")
    out_lines.append("|---|---|---|---|---|")
    for arm in arms_scores:
        s = arms_scores[arm]
        n = len(s)
        mean = sum(s) / n if n else 0.0
        out_lines.append(f"| {arm} | {n} | {mean:.3f} | {min(s) if s else 0} | {max(s) if s else 0} |")
    out_lines.append("")

    # Comparison 1: Sonnet+AA vs Sonnet raw (the main claim)
    if arms_scores["sonnet_raw"] and arms_scores["sonnet_amp"]:
        v1 = evaluate_quality_distributions(
            baseline_scores=arms_scores["sonnet_raw"],
            current_scores=arms_scores["sonnet_amp"],
        )
        out_lines.append("## Comparison 1 — Sonnet 4.6 + Agent Amplifier vs raw Sonnet 4.6")
        out_lines.append("")
        out_lines.append(f"`{verdict_summary_line(v1)}`")
        out_lines.append("")

    # Comparison 2: Sonnet+AA vs Opus raw (the ceiling)
    if arms_scores["opus_raw"] and arms_scores["sonnet_amp"]:
        v2 = evaluate_quality_distributions(
            baseline_scores=arms_scores["opus_raw"],
            current_scores=arms_scores["sonnet_amp"],
        )
        out_lines.append("## Comparison 2 — Sonnet 4.6 + Agent Amplifier vs raw Opus 4.7 (ceiling)")
        out_lines.append("")
        out_lines.append(f"`{verdict_summary_line(v2)}`")
        out_lines.append("")

    out_lines.append("## Per-task scores")
    out_lines.append("")
    out_lines.append("| task_id | category | sonnet_raw | sonnet_amp | opus_raw |")
    out_lines.append("|---|---|---|---|---|")
    for task in results["tasks"]:
        row = [task["task_id"], task["category"]]
        for arm in ("sonnet_raw", "sonnet_amp", "opus_raw"):
            j = (task["arms"].get(arm) or {}).get("judge") or {}
            row.append(str(j.get("score", "—")))
        out_lines.append("| " + " | ".join(row) + " |")
    out_lines.append("")

    out_lines.append("## Methodology")
    out_lines.append("")
    out_lines.append("- **Corpus:** 12 tasks (6 from WildBench-Hard / Ai2 `allenai/WildBench`, paraphrased; 6 custom Qualixar agentic seeds). Saved at `agent-amplifier/.backup/revive-plan-2026-05-13/agentic_eval_v1.jsonl`.")
    out_lines.append("- **Arms:** 3 — raw Sonnet 4.6, Sonnet 4.6 + Agent Amplifier envelope, raw Opus 4.7.")
    out_lines.append("- **Judge:** Opus 4.7 as LLM-as-judge. 5-point rubric (correctness 0-2 / completeness 0-2 / goal-adherence 0-1). Blind judging (judge sees only task + reference + candidate; arm label hidden; arm order shuffled per task).")
    out_lines.append("- **Verdict:** AgentAssay `evaluate_quality_distributions` (Mann-Whitney U on continuous judge scores).")
    out_lines.append("- **n:** 12 per arm — INCONCLUSIVE is an honest outcome; this is a tight sample size.")
    out_lines.append("")
    out_lines.append("Benchmarked using AgentAssay's stochastic-test framework — see https://github.com/qualixar/agentassay")
    return "\n".join(out_lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, required=True, help="JSONL corpus path")
    ap.add_argument("--out", type=Path, required=True, help="results JSON output path")
    ap.add_argument("--dry-run", action="store_true", help="stage all calls, print cost, do not dispatch")
    ap.add_argument("--max-spend", type=float, default=25.0, help="abort live run if projection > this (USD)")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel API calls in live mode")
    ap.add_argument("--est-output-tokens", type=int, default=600, help="planning budget per candidate response")
    ap.add_argument("--candidate-max-tokens", type=int, default=2048, help="cap on candidate response tokens")
    ap.add_argument("--judge-max-tokens", type=int, default=256, help="cap on judge response tokens")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for blind shuffle")
    ap.add_argument("--smoke", action="store_true", help="run only the first task end to end (~$0.50)")
    ap.add_argument("--yes", action="store_true", help="skip the interactive spend confirmation (--max-spend still gates)")
    ap.add_argument("--verdict-only", type=Path, help="skip the run, compute verdict from existing results JSON")
    args = ap.parse_args()

    if args.verdict_only:
        results = json.loads(args.verdict_only.read_text())
        verdict_md = compute_verdict(results)
        verdict_path = args.verdict_only.with_name(args.verdict_only.stem.replace("results", "verdict") + ".md")
        verdict_path.write_text(verdict_md)
        print(f"verdict written → {verdict_path}")
        print()
        print(verdict_md)
        return 0

    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if args.smoke:
        tasks = tasks[:1]

    subs = _foundry_endpoints()
    if not subs:
        print("ERROR: no Foundry subscriptions configured in env (need ANTHROPIC_FOUNDRY_API_KEY or AZURE_TAP_API_KEY+AZURE_TAP_ENDPOINT or AZURE_HM_API_KEY+AZURE_HM_ENDPOINT)", file=sys.stderr)
        return 2

    print(f"Phase 4 harness — {len(tasks)} task(s), {len(subs)} Foundry sub(s) ({', '.join(s['name'] for s in subs)})")
    proj = project_costs(tasks, est_output_tokens=args.est_output_tokens)
    print()
    print(f"projected calls: {proj['total_calls']}")
    print(f"projected spend: ${proj['grand_total_usd']:.4f} USD (max-spend gate: ${args.max_spend:.2f})")
    print()
    print("per-step breakdown:")
    for k, v in sorted(proj["summary"].items()):
        print(f"  {k:<35} ${v:.4f}")
    print()

    if args.dry_run:
        print("--dry-run: not dispatching. Re-run without --dry-run to execute.")
        return 0

    if proj["grand_total_usd"] > args.max_spend:
        print(f"ABORT: projected ${proj['grand_total_usd']:.4f} exceeds max-spend ${args.max_spend:.2f}", file=sys.stderr)
        return 3

    if args.yes:
        print(f"--yes: bypassing interactive confirmation. Spending ~${proj['grand_total_usd']:.2f} USD on Foundry now.")
    else:
        confirm = input(f"Spend ~${proj['grand_total_usd']:.2f} USD on Foundry now? [y/N] ").strip().lower()
        if confirm != "y":
            print("aborted.")
            return 0

    results = run_live(
        tasks=tasks,
        subs=subs,
        out_path=args.out,
        concurrency=args.concurrency,
        candidate_max_tokens=args.candidate_max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        seed=args.seed,
    )

    verdict_md = compute_verdict(results)
    verdict_path = args.out.with_name(args.out.stem.replace("results", "verdict") + ".md")
    verdict_path.write_text(verdict_md)
    print(f"verdict written → {verdict_path}")
    print()
    print(verdict_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
