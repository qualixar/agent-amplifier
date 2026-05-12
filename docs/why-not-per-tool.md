# FAQ: Why Agent Amplifier?

Common questions about why Agent Amplifier exists and how it compares to
alternative approaches.

---

## Why not just use better system prompts?

You can. Agent Amplifier **is** better system prompts -- applied dynamically,
per-turn, conditioned on task complexity and iteration depth.

The difference: a static system prompt uses the same framing whether you are
asking "format this string" or "refactor authentication to use JWT." Amp
classifies the prompt, picks the right effort tier, selects phase-appropriate
modifiers, escalates the audit persona per iteration, and anchors against your
original goal. You get the benefit of carefully-tuned system prompts without
manually writing them.

If you already maintain a sophisticated per-task prompt library and swap
system prompts manually, amp automates what you do by hand.

## Why not fine-tune the model?

Fine-tuning changes the model's weights. Agent Amplifier changes what the
model sees. These are complementary, not competing.

Limitations of fine-tuning for this problem:

- **Access.** You cannot fine-tune Claude Opus or most frontier models.
- **Latency.** Fine-tuning cycles take hours to days. Amp activates in 60
  seconds.
- **Generality.** A fine-tuned model is better at its training distribution.
  Amp applies to every prompt, regardless of domain.
- **Composability.** Fine-tuning is per-model. Amp works across seven hosts
  with the same kernel.

If you have a fine-tuned model, amp still improves it. The amplification is
model-agnostic -- it shapes the input, not the weights.

## Why not just use a bigger model?

Bigger models are better at raw reasoning. They are not better at effort
calibration, goal persistence, or knowing when to stop.

Agent Amplifier addresses failures that scale with context length, not model
size: drift after 50K tokens, wasted iterations on already-correct output,
wrong effort on simple prompts. A bigger model exhibits the same failure
modes; it just takes longer to hit them.

The launch benchmark: **Sonnet + amp consistently outperforms Sonnet alone on
real engineering tasks.** Same model. Higher effort. Verified.

## Why not DSPy?

DSPy is a framework for optimizing LLM programs through automatic prompt
tuning. Agent Amplifier occupies a different niche.

| Dimension | DSPy | Agent Amplifier |
|---|---|---|
| Optimization target | Prompt templates via meta-optimization | Per-turn execution quality via deterministic classification |
| Runtime model | Compile-time optimization + runtime inference | Pure runtime, no compilation step |
| LLM calls for optimization | Yes (meta-prompting) | None (deterministic Python) |
| Scope | LLM program pipelines | Agent execution loops |
| Integration | DSPy programs | 7 agent host frameworks |
| Memory | Not a concern | Cross-host memory plane (Feature 9) |

DSPy optimizes *what prompt to use*. Amp decides *how hard to think*, *what
goal to anchor*, *when to stop*, and *what context to recall* -- per turn, at
runtime, without LLM calls.

If you use DSPy to build your agent pipeline, amp can still amplify the
agent's execution loop on top of DSPy's optimized prompts.

## Why not per-tool system prompts in Claude Code?

Claude Code lets you write tool-specific instructions via
`.claude/settings.json`. This is useful for scoping tool behavior but does not
address:

- **Effort routing.** Per-tool prompts do not adjust the model's thinking
  budget based on query complexity.
- **Goal drift.** Per-tool prompts do not re-inject your original request
  after 50 tool calls.
- **Convergence.** Per-tool prompts do not detect when output has stabilized
  and stop the iteration loop.
- **Cross-host.** Per-tool prompts are Claude Code only. Amp works across
  seven hosts.

Per-tool prompts and amp are complementary. Your tool instructions stay; amp
adds execution-quality amplification on top.

## Why not Cursor Rules / .cursorrules?

Cursor Rules are static instruction files. They apply the same framing to
every prompt. The same limitations as static system prompts apply: no effort
routing, no convergence detection, no phase adaptation.

If you use Cursor, `agent-amp install cursor` writes amplification rules that
layer on top of your existing `.cursor/rules/` files.

## Does amp work with models other than Claude?

The kernel is model-agnostic. It produces a `StepEnvelope` containing effort
tier, phase, persona, modifiers, and goal anchor. The adapter translates this
into the host's injection format.

At v1.0, the adapters target:

- Claude (via Claude Code adapter)
- Any model behind Cursor
- Any model behind GitHub Copilot
- Any model behind LangGraph / CrewAI / AgentScope / LangChain

The semantic modifiers (L99, CRIT, FINISH, etc.) are tested primarily on
Claude and GPT-family models. Effectiveness on other model families may vary.

## What data does amp collect?

None that leaves your machine. Telemetry is stored locally at
`~/.claude/agent-amp/state.db` (SQLite). The `agent-amp report` command reads
this database. The `agent-amp dashboard` command serves a local web UI.

No analytics. No phone-home. No cloud dependency. This is an AI Reliability
Engineering tool built by [Qualixar](https://qualixar.com) -- local-first is
a brand-level decision, not a technical convenience.

## What is the performance overhead?

Hook latency on M-series MacBook Pro:

- `UserPromptSubmit`: P50 = 72 ms, P99 = 77 ms
- `Stop`: P50 = 61 ms

Sub-perceptual. The model's inference time (seconds to minutes) dwarfs the
amplifier's computation (tens of milliseconds).

## How does this compare to Microsoft Amplifier?

Microsoft's Amplifier is a different product solving a different problem.
Microsoft Amplifier is an enterprise security posture management platform for
SaaS application permissions -- it manages OAuth consent, API permissions, and
access control across Microsoft 365 tenants.

Agent Amplifier is a runtime execution-quality layer for AI coding agents.
The name collision is coincidental; the domains do not overlap.

| | Microsoft Amplifier | Agent Amplifier |
|---|---|---|
| Domain | Enterprise SaaS security | AI agent execution quality |
| What it manages | OAuth permissions and app consent | Effort, drift, convergence, budget |
| Runtime | Cloud-hosted Microsoft 365 service | Local Python, runs on your machine |
| Audience | IT administrators | AI engineers and developers |

## How does this compare to Sourcegraph Amp?

Sourcegraph Amp (formerly Cody) is a cloud-hosted AI coding agent built on
Sourcegraph's code intelligence graph. It is a **product** -- a full AI coding
agent that you use instead of (or alongside) other agents.

Agent Amplifier is a **layer** -- it makes your existing agent better without
replacing it. You keep Claude Code, Cursor, or Copilot as your agent; amp
shapes how that agent reasons.

| | Sourcegraph Amp | Agent Amplifier |
|---|---|---|
| Type | AI coding agent (product) | Amplification layer (library) |
| Replaces your agent? | Yes (it IS the agent) | No (wraps your existing agent) |
| Code intelligence | Sourcegraph's code graph | Host-native memory + optional SLM |
| Model | Sourcegraph-managed | Whatever your host uses |
| Hosts supported | Amp only | 7 hosts at v1.0 |
| Install | SaaS signup | `pip install agent-amplifier` |

The CLI command is `agent-amp` (not `amp`) to avoid collision with
Sourcegraph's `amp` binary.

## Can I use amp and SLM separately?

Yes. Each product installs and works independently.

- `pip install agent-amplifier` -- amplification without SLM memory
- `pip install superlocalmemory` -- memory without amplification
- Install both -- they compose automatically (see
  [SLM Composition](composition.md))

---

Built by [Qualixar](https://qualixar.com) -- AI Reliability Engineering for
AI agents.
