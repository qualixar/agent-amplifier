# Agent Amplifier

**Runtime amplification for AI coding agents.**
Same model. Higher effort. Verified.

---

Agent Amplifier is a small, local Python layer that sits between you and your
AI coding agent. It makes the agent reason harder, drift less, and stop when
it is actually done -- using deterministic Python, no LLM calls, no network
dependency.

It does not make the model smarter. It changes **what the model sees** and
**when the loop ends**. Think of it as a coach standing next to a
brilliant-but-sloppy intern: same intern, much better output.

## The four problems it solves

1. **Wrong effort level.** You ask something simple -- the model ultra-thinks
   for 90 seconds. You ask something complex -- it skims and ships a half-fix.
2. **Goal drift.** After ~50 tool calls the model forgets the original ask and
   starts riffing on the latest sub-task.
3. **No convergence stop.** The iteration loop keeps "improving" until the
   token budget runs out or you press Ctrl-C.
4. **Memory amnesia.** Each new conversation starts from zero despite every
   host already having a memory file.

## Quick start

```bash
pip install agent-amplifier
agent-amp install claude-code
# restart Claude Code -- next prompt is amplified
```

Verify it is working:

```bash
agent-amp doctor            # environment diagnostics
agent-amp demo "Refactor auth to use JWT"   # preview the amplification envelope
agent-amp report            # real telemetry after a few sessions
```

## Key facts

| Property | Value |
|---|---|
| Install time | ~60 seconds |
| Hook latency | P50 = 72 ms, P99 = 77 ms (M-series Mac) |
| Supported hosts | Claude Code, Cursor, GitHub Copilot, LangGraph, CrewAI, AgentScope, LangChain |
| License | AGPL-3.0-or-later |
| Telemetry | None. `state.db` lives only on your machine. |
| Dependencies | Python 3.10+, anyio. Optional `[tokenizer]` extra for real BPE counting. |

## What ships at v1.0.0

- All 11 features (see [Features](features.md))
- 7 host adapters (see [Adapters](adapters.md))
- `agent-amp` CLI: `install`, `uninstall`, `list`, `status`, `doctor`,
  `config show`, `bench`, `demo`, `report`, `dashboard`
- Optional SLM composition for richer memory (see [SLM Composition](composition.md))
- 100% line + branch test coverage, mypy --strict, ruff clean, bandit baseline, pip-audit clean

## Links

- [GitHub](https://github.com/qualixar/agent-amplifier)
- [PyPI](https://pypi.org/project/agent-amplifier/)
- [Qualixar](https://qualixar.com) -- AI Reliability Engineering
- [@varunPbhardwaj](https://x.com/varunPbhardwaj)
