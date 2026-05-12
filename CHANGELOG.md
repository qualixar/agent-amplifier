# Changelog

All notable changes to **Agent Amplifier** are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
