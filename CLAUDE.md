# Agent Amplifier — Project Guidelines

## What this is

Agent Amplifier is a runtime amplification layer for AI coding agents. It intercepts at the hook layer (UserPromptSubmit, PreToolUse, PostToolUse, Stop, PreCompact) and applies dynamic effort routing, goal anchoring, convergence detection, and token budgeting — all in deterministic Python, no extra LLM calls.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,dashboard,tokenizer]"

# Run tests (100% coverage required)
.venv/bin/pytest --cov=src/agent_amplifier --cov-branch --cov-fail-under=100 -p no:randomly --ignore=tests/perf -q

# Type checking
.venv/bin/mypy --strict src/agent_amplifier/

# Linting
.venv/bin/ruff check src/agent_amplifier/ tests/
```

## Code style

- Immutable data: frozen dataclasses with `__slots__`
- Many small files over few large files (200-400 lines typical, 800 max)
- Error handling: fail-open in hooks, explicit at system boundaries
- No mutation of kernel state outside the lock
- Lazy imports for optional dependencies (LangGraph, CrewAI, AgentScope, LangChain)

## Architecture

```
kernel.py        — orchestrator (classify → anchor → select → phase → persona → converge → budget)
effort_router.py — 5-tier complexity classifier
goal_anchor.py   — anti-drift re-injection
convergence.py   — LTI stability detection
phase_prompts.py — EXPLORE / EVALUATE / EXECUTE / VERIFY / REFINE
personas.py      — escalating audit per iteration
token_budget.py  — cost-bounded amplification
tool_selector.py — MoE-inspired tool shortlisting
model_router.py  — complexity → model tier suggestion
bench_verdict.py — AgentAssay verdict wrapper (Phase 4 A/B benchmarks)
_internal/embedding.py — Ollama HTTP client for Tier 2 quality score
adapters/        — 7 host adapters + SLM memory adapter
dashboard/       — FastAPI backend + Streamlit UI
```

## Metric semantics (v1.1)

AA has two execution paths and they measure different things — do not confuse them when reading dashboards.

**Claude Code adapter (hook-time, no extra LLM calls):**

  * One amplification cycle per user turn — `iterations_completed=1` is correct by design, not a bug.
  * `outcomes.completed` mirrors v1.0's `converged` (`in_flight==0` at Stop). Liveness only.
  * `outcomes.quality_score` is the layered metric: Tier 1 Jaccard always; Tier 2 nomic embedding (via Ollama) only when Tier 1 is in the [0.30, 0.70] ambiguous band; Tier 3 trajectory delta penalises looping + missing-recon. All bounded to [0, 1].
  * `outcomes.convergence_state` is the per-session trajectory class derived from the rolling `quality_score` history. None when current quality_score is None.
  * `is_synthetic=1` sessions are excluded from dashboards by default (env var `AGENT_AMP_SYNTHETIC=1` or non-existent cwd triggers the flag).

**Kernel path (CrewAI / LangGraph / AgentScope / LangChain — opt-in per adapter):**

  * Multi-iteration loop using `convergence.py::ConvergenceDetector` (max_iterations=4 default, Jaccard threshold 0.95, Gompertz damping).
  * `iterations_completed > 1` reflects real critic-loop iterations.

**Cross-version benchmark verdicts** (Phase 4) use `bench_verdict.py` which wraps AgentAssay's `VerdictFunction`. The viral claim in any release post must cite the AgentAssay verdict, not a raw average — Wilson CI + Fisher / Mann-Whitney with a 3-valued PASS / FAIL / INCONCLUSIVE result is the bar.

## Adapters

7 host adapters at v1.0: Claude Code, Cursor, GitHub Copilot, LangGraph, CrewAI, AgentScope, LangChain. Each implements `AdapterBase` with `detect()`, `install()`, `default_memory_recall()`, `default_memory_remember()`.

## CLI

```
agent-amp install <host>     # attach hooks
agent-amp uninstall <host>   # remove hooks
agent-amp status [--watch]   # live token bar
agent-amp doctor             # environment diagnostics
agent-amp report             # telemetry dashboard
agent-amp demo "<prompt>"    # preview amplified envelope
agent-amp dashboard          # launch Streamlit UI
```

## License

AGPL-3.0-or-later. See LICENSE file.
