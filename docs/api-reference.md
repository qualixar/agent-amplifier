# API Reference

Public API surface for Agent Amplifier v1.0. Import paths are stable across
minor versions. This is the reference for integrating amp into your own
AI Reliability Engineering workflows.

---

## AgentAmplifier

The synchronous kernel. Use this when your host framework is synchronous.

```python
from agent_amplifier import AgentAmplifier
```

### Constructor

```python
amp = AgentAmplifier(
    config=None,                # AmplifierConfig or None (loads from TOML)
    adapter=None,               # AdapterBase subclass or None
    memory_recall=None,         # Callable[[str, int], list[RecalledPattern]]
    memory_remember=None,       # Callable[[Outcome], None]
    observability_callback=None # Callable[[AmplifierEvent, dict], None]
)
```

### Methods

#### `before_step(query, tool_names=None, context=None) -> StepEnvelope`

Run the full amplification pipeline for a user prompt. Returns a frozen
`StepEnvelope` containing all decisions.

```python
envelope = amp.before_step(
    query="Refactor auth to use JWT",
    tool_names=["Read", "Write", "Bash", "Edit"],
)
print(envelope.effort_tier)       # "high"
print(envelope.thinking_trigger)  # "megathink"
print(envelope.phase)             # "EXPLORE"
print(envelope.modifiers)         # ("L99", "CRIT", "OODA")
```

#### `after_step(result) -> dict`

Feed the model's output back to the kernel. Returns a decision dict with
`"action"` key: `"continue"`, `"stop"`, or `"re_anchor"`.

```python
decision = amp.after_step(result=model_output)
if decision["action"] == "stop":
    amp.finalize()
```

#### `finalize() -> Outcome`

End the current turn. Writes outcome to memory (if configured), logs
telemetry to `state.db`, returns an `Outcome` dataclass.

---

## AsyncAgentAmplifier

The asynchronous kernel. Use this when your host framework is async
(LangGraph, async LangChain).

```python
from agent_amplifier import AsyncAgentAmplifier
```

Same constructor signature as `AgentAmplifier`. All methods are async:

```python
envelope = await amp.before_step(query="...", tool_names=[...])
decision = await amp.after_step(result=output)
outcome = await amp.finalize()
```

---

## StepEnvelope

Frozen dataclass representing all amplification decisions for a single turn.

```python
from agent_amplifier.kernel import StepEnvelope
```

| Field | Type | Description |
|---|---|---|
| `step_id` | `str` | Unique identifier for this step |
| `effort_tier` | `str` | `minimal` / `low` / `medium` / `high` / `max` |
| `thinking_trigger` | `str` | Keyword for the model's thinking budget |
| `phase` | `str` | `EXPLORE` / `EXPLOIT` / `FINALIZE` |
| `persona` | `str` | Escalating reviewer description |
| `modifiers` | `tuple[str, ...]` | Selected semantic modifiers |
| `goal_anchor` | `str` | Original user request |
| `recalled_context` | `str` | Memory chunks from recall |
| `tool_shortlist` | `tuple[str, ...]` | Recommended tool subset |
| `iteration` | `int` | Current iteration index |
| `budget_remaining` | `int` | Estimated tokens remaining |

---

## AmplifierConfig

Configuration dataclass. Loaded from
`~/.config/agent-amplifier/config.toml` by default.

```python
from agent_amplifier.types import AmplifierConfig
```

Key fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_iterations` | `int` | `4` | Hard cap on convergence loop |
| `goal_anchor_interval` | `int` | `5` | Re-inject goal every N tool calls |
| `budget_mode` | `str` | `"strict"` | `strict` / `soft` / `off` |
| `budget_ceiling` | `int` | `100000` | Max tokens per turn |
| `tool_selector_threshold` | `float` | `0.3` | Tool selection aggressiveness |
| `recall_limit` | `int` | `3` | Max memory chunks per recall |

---

## agent-amp CLI

The CLI entry point. Always invoked as `agent-amp` (never `amp` -- collision
with Sourcegraph Amp).

### Commands

```
agent-amp install <target>        Attach to a host framework
agent-amp install --auto          Auto-detect and install all available hosts
agent-amp uninstall <target>      Detach from a host framework
agent-amp list                    List adapters and detection status
agent-amp status                  Show installation status
agent-amp doctor                  Environment diagnostics
agent-amp config show             Print current configuration
agent-amp config set <key> <val>  Update a config value
agent-amp config path             Print config file path
agent-amp bench [options]         Run benchmarks
agent-amp demo "<prompt>"         Preview amplification envelope for a prompt
agent-amp report                  Show telemetry dashboard from state.db
agent-amp dashboard               Launch local web dashboard
```

### Targets

Valid install/uninstall targets:

- `claude-code` (or `claude_code`)
- `cursor`
- `github_copilot`
- `langgraph`
- `crewai`
- `agentscope`
- `langchain`

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Generic error |
| 2 | Unknown command or target |
| 3 | Already installed |
| 4 | Permission denied |
| 5 | Host framework not detected |

### Environment variables

| Variable | Effect |
|---|---|
| `AGENT_AMP_MAX_ITERATIONS` | Override `max_iterations` for this session |
| `AGENT_AMP_FALLBACK_PHASE` | Override fallback phase on contract error |

---

## ModelRouter

Stateless router that maps effort classification to a suggested model tier.
Adapters can read `ModelSuggestion.tier` to auto-select the model for the
next LLM call. The `agent-amp report` dashboard shows what model each prompt
would route to.

```python
from agent_amplifier.model_router import ModelRouter, create_router
```

```python
from agent_amplifier.model_router import ModelRouter, create_router
from agent_amplifier.types import EffortLevel

router = create_router(enabled=True)
suggestion = router.suggest(EffortLevel.HIGH, domain="security")
print(suggestion.tier)       # "opus"
print(suggestion.display)    # "Claude Opus (deep reasoning)"
print(suggestion.reason)     # "Complexity=HIGH → opus [domain=security]"
```

Default mapping: MINIMAL/LOW -> haiku, MEDIUM -> sonnet, HIGH/MAX -> opus.
Override via `AGENT_AMP_MODEL_MAP` env var (JSON):

```bash
export AGENT_AMP_MODEL_MAP='{"MEDIUM": "opus"}'
```

`ModelSuggestion` is a frozen dataclass with fields: `tier` (str), `display`
(str), `reason` (str), `overridden` (bool).

---

## Exceptions

```python
from agent_amplifier.adapter_base import (
    AdapterError,
    AdapterNotInstalledError,
    AdapterAlreadyInstalledError,
)
from agent_amplifier.kernel import KernelContractError
```

| Exception | When raised |
|---|---|
| `AdapterError` | Base for all adapter failures |
| `AdapterNotInstalledError` | `uninstall()` on a non-installed adapter |
| `AdapterAlreadyInstalledError` | `install()` called twice without uninstall |
| `KernelContractError` | Kernel invariant violated (includes 3-field diagnostic) |

---

Built by [Qualixar](https://qualixar.com) -- AI Reliability Engineering for
AI agents.
