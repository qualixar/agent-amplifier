# Changelog

All notable changes to **Agent Amplifier** are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-05-13

Engineering hardening release. Layered quality metric, AgentAssay verdict
wrapper, doctor v1.1 telemetry. Fully backward-compatible with v1.0
state.db files (additive schema migrations only).

### Added

- **npm install parity** — `npm install -g agent-amplifier` now bootstraps the Python package via `pipx` and exposes the real `agent-amp` CLI. `pip install`, `pipx install`, `npm install -g`, and `git clone && pip install -e .` all converge on the same on-disk binary and same feature set. The npm postinstall detects Python 3.11+, installs `pipx` if missing, and runs `pipx install agent-amplifier==<version>`. Opt-out via `AGENT_AMP_SKIP_POSTINSTALL=1` for CI / Docker. Prior v1.0 npm package was a name-reservation stub; v1.1 ships a working bootstrap.
- **Layered `quality_score`** — bounded `[0, 1]`, composed deterministically
  from three tiers and persisted per outcome:
  - Tier 1: Jaccard similarity between envelope goal and the final
    assistant message extracted from the session transcript JSONL
    (always runs, ~5 ms).
  - Tier 2: `nomic-embed-text` via local Ollama, cosine blend
    `0.3·lex + 0.7·cos`, fires only when Tier 1 is in the `[0.30, 0.70]`
    ambiguous band (~40 ms; opt-out via `AGENT_AMP_EMBED_ENABLED=0`).
  - Tier 3: trajectory delta penalty for tool-call loops and missing-
    recon (Edit/Write without prior Read of the same path); capped at
    `-0.20`.
- **`convergence_state` classifier** — per-session trajectory class
  (`improving` / `stagnant` / `oscillating` / `converged`) derived from
  rolling `quality_score` history.
- **`bench_verdict.py`** — AgentAssay verdict wrapper for A/B benchmarks.
  Three public functions: `evaluate_completion_regression`,
  `evaluate_quality_distributions`, `verdict_summary_line`. Adds
  `agentassay>=0.1` to the `[bench]` extra.
- **Synthetic-session quarantine** — `is_synthetic` flag on sessions,
  auto-set when `AGENT_AMP_SYNTHETIC=1` is exported or when the
  recorded cwd does not exist on disk. Dashboards (`agent-amp report`,
  telemetry tab, `agent-amp doctor`) exclude synthetic sessions by
  default; pass `--include-synthetic` / `--synthetic-only` to override.
- **`suggested_model` persistence** — the `UserPromptSubmit` hook now
  stores the `ModelRouter` tier suggestion per envelope.
- **`agent-amp doctor` v1.1** — adds a telemetry health block
  (state.db size, real-vs-synthetic session split, quality coverage %,
  last activity, SLM daemon probe). New `--json` flag with a stable
  schema for programmatic consumers.
- **Transcript reader** — `final_assistant_message()` and
  `list_events_for_turn()` helpers used by the Stop hook to score
  against on-disk transcript JSONL with zero extra LLM calls.
- **`_internal/embedding.py`** — zero-dep Ollama HTTP client used by the
  Tier 2 embedding blend. Falls back cleanly to Tier 1 + 3 when Ollama
  is unreachable.

### Changed

- `outcomes.completed` is the canonical replacement for v1.0's
  `outcomes.converged`. The Stop hook continues to write **both**
  columns this release so v1.0 dashboards and external consumers keep
  working. The deprecation contract: v1.2 emits a `DeprecationWarning`
  on `converged` reads; v1.3 removes the column entirely.
- Stop hook now also computes `quality_score` and `convergence_state`
  from telemetry already on disk — no extra LLM calls, no extra
  latency in the hot path.

### Backward compatibility

- Every schema change is an additive `ALTER TABLE` behind an idempotent
  `PRAGMA table_info` guard.
- v1.0 `state.db` files migrate on first open with zero breakage.
- v1.0 `converged` column remains both readable and writable.

### Tests & quality

- 1,899 tests passing, 1 skipped (platform-specific).
- 100% line + 100% branch coverage on every file touched in this
  release: `state.py`, `stop_hook.py`, `transcript.py`, `embedding.py`,
  `bench_verdict.py`, `cli.py`.
- `mypy --strict` clean on all touched `src/` modules.
- `ruff check` clean across `src/` and `tests/`.

### Coming next

A 12-task A/B benchmark (raw Sonnet 4.6 vs Sonnet 4.6 + Agent Amplifier
envelope vs raw Opus 4.7), scored by Opus 4.7 as judge with a 5-point
rubric and an AgentAssay statistical verdict (Wilson 95% CI +
Mann-Whitney U), ships with v1.2. Agent Amplifier is part of Qualixar's
AI Reliability Engineering category, and every public claim must clear
that statistical bar.

## [1.0.0] — 2026-05-15

Initial public release.

### Features

- **Runtime Harness** — runs inside the agent's live loop, not offline
- **Dynamic Effort Router** — classifies prompt complexity into 5 tiers, auto-selects thinking budget
- **Goal Anchor Protocol** — re-injects original request every N tool calls to prevent drift
- **LTI Convergence Detection** — stops the loop when output stabilizes
- **Semantic Modifier Injection** — 97 validated keywords selected per task type
- **Cross-Framework Adapters** — 7 host adapters (Claude Code, Cursor, GitHub Copilot, LangGraph, CrewAI, AgentScope, LangChain)
- **Phase-Aware Prompting** — EXPLORE / EVALUATE / EXECUTE / VERIFY / REFINE
- **Escalating Audit Personas** — stricter review per iteration depth
- **Cross-Host Memory Plane** — every user gets memory recall at turn start + outcome write at turn end
- **Cost-Bounded Amplification** — hard token ceiling with graceful finalize
- **Intelligent Tool Selector** — shortlists relevant tools per turn
- **Model Router** — maps complexity to suggested model tier (haiku / sonnet / opus)
- **CLI** — `agent-amp install`, `report`, `demo`, `doctor`, `status --watch`, `dashboard`
- **Streamlit Dashboard** — 4-tab UI (Tune, Telemetry, Adapters, Health)
- **FastAPI Backend** — REST API for the dashboard
- **SuperLocalMemory Composition** — 3 modes (adjacent, composed, closed-loop)

### Quality

- 1,570 tests, 100% line + branch coverage
- mypy --strict clean (61 source files)
- ruff clean
- Fail-open design: if the amplifier crashes, your agent runs normally

### License

AGPL-3.0-or-later

---

Built by [Qualixar](https://qualixar.com) — AI Reliability Engineering.
[@varunPbhardwaj](https://x.com/varunPbhardwaj)
