# Changelog

All notable changes to Agent Amplifier are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [1.0.0] -- 2026-05-15

### Added

- **Runtime Harness Amplification** -- online amplification inside the agent's
  live execution loop with fail-open design and sub-100ms hook latency.
- **Dynamic Effort Router** -- 5-tier automatic effort classification
  (minimal / low / medium / high / max) with thinking-budget trigger
  selection.
- **Goal Anchor Protocol** -- re-injects original user request every N tool
  calls to prevent drift in long agent runs.
- **LTI Convergence Detection** -- stops the iteration loop when output
  stabilizes, using Linear-Time-Invariant stability bounds. Default hard cap:
  4 iterations.
- **Semantic Modifier Injection** -- 97 validated keyword library, dynamically
  selected per task complexity, phase, and persona.
- **Cross-Framework Universal Adapter** -- 7 host adapters: Claude Code,
  Cursor, GitHub Copilot, LangGraph, CrewAI, AgentScope, LangChain.
- **Phase-Aware Prompting** -- EXPLORE / EXPLOIT / FINALIZE system-prompt
  headers per iteration depth.
- **Escalating Audit Personas** -- reviewer strictness increases per iteration.
- **Cross-Host Memory Plane** -- pluggable memory recall and write for every
  host. SLM integration for 4-channel retrieval when installed.
- **Cost-Bounded Amplification** -- token budget with prefix-delta tracking.
  Budget modes: strict, soft, off. Optional `[tokenizer]` extra for real BPE
  counting.
- **Intelligent Tool Selector** -- per-turn tool shortlisting based on prompt
  relevance.
- **`agent-amp` CLI** -- `install`, `uninstall`, `list`, `status`, `doctor`,
  `config`, `bench`, `demo`, `report`, `dashboard`.
- **Claude Code hook installer** -- official 5-hook integration via
  `agent-amp install claude-code`.
- **PreCompact observe-only hook** -- paper-data on compaction overlap with
  amp turns (active deferral planned for v1.0.1).
- AGPL-3.0-or-later license.
- 100% line + branch test coverage, mypy --strict, ruff clean, bandit
  baseline, pip-audit clean.
- Zero telemetry. All data stored locally at `~/.claude/agent-amp/state.db`.

---

*Built by [Qualixar](https://qualixar.com) -- AI Reliability Engineering.*
*[@varunPbhardwaj](https://x.com/varunPbhardwaj)*
