# Getting Started

This page walks you from zero to your first amplified turn in under 60 seconds.

---

## Prerequisites

- Python 3.10 or later
- One of the supported agent hosts installed (Claude Code, Cursor, GitHub
  Copilot, LangGraph, CrewAI, AgentScope, or LangChain)

## Install

```bash
pip install agent-amplifier
```

For real BPE token counting (recommended for cost tracking):

```bash
pip install agent-amplifier[tokenizer]
```

The `[tokenizer]` extra adds `tiktoken` with `o200k_base` encoding for modern
frontier models and `cl100k_base` for legacy GPT-3.5/4. Without it, token
counts use the `len(text) // 4` estimate.

## Attach to your agent host

=== "Claude Code"

    ```bash
    agent-amp install claude-code
    ```

    This writes five hooks into `~/.claude/settings.json`:
    `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`, and `PreCompact`.

    Restart Claude Code after install.

=== "Cursor"

    ```bash
    agent-amp install cursor
    ```

    Writes amplification rules to `.cursor/rules/`.

=== "GitHub Copilot"

    ```bash
    agent-amp install github_copilot
    ```

    Writes amplification instructions to `.github/copilot-instructions.md`.

=== "LangGraph"

    ```bash
    agent-amp install langgraph
    ```

    Registers the adapter so `before_step()` / `after_step()` calls
    integrate with your LangGraph checkpointer.

=== "CrewAI"

    ```bash
    agent-amp install crewai
    ```

=== "AgentScope"

    ```bash
    agent-amp install agentscope
    ```

=== "LangChain"

    ```bash
    agent-amp install langchain
    ```

    Wires into LangChain's `BaseMemory` / `BaseChatMessageHistory` APIs.

## Auto-detect all available hosts

```bash
agent-amp install --auto
```

The CLI probes each adapter's `detect()` method and installs every host it
finds on your system.

## Verify

```bash
# Check environment health
agent-amp doctor

# Preview what amplification does to a single prompt
agent-amp demo "Refactor the auth module to use JWT"

# After using your agent for a few sessions, check telemetry
agent-amp report
```

### What `agent-amp doctor` checks

- Python version and platform
- Whether `anyio` is available
- Whether each adapter's host is detected
- Whether SuperLocalMemory (SLM) is installed (enables richer composition)
- Whether `tiktoken` is available for real token counting

### What `agent-amp demo` shows

The `demo` command runs a single prompt through the full kernel pipeline
without executing it against a live model. It prints:

1. The classified effort tier (minimal / low / medium / high / max)
2. The selected phase (EXPLORE / EXPLOIT / FINALIZE)
3. The persona assignment
4. The semantic modifiers injected
5. The goal anchor text
6. The complete `StepEnvelope` that would be injected into the prompt

This is the fastest way to see what amp does before committing to a session.

## Uninstall

```bash
agent-amp uninstall claude-code
```

Surgical removal: only Agent Amplifier's hooks are removed. Your other hooks,
settings, and memory files are never touched.

## Configuration

Amp ships with sane defaults. Configuration lives at
`~/.config/agent-amplifier/config.toml`.

```bash
# Show current config
agent-amp config show

# Show config file path
agent-amp config path
```

Key knobs:

| Setting | Default | What it controls |
|---|---|---|
| `max_iterations` | `4` | Hard cap on convergence iterations per turn |
| `goal_anchor_interval` | `5` | Re-inject original request every N tool calls |
| `budget_mode` | `"strict"` | Token budget enforcement: `strict`, `soft`, or `off` |
| `tool_selector_threshold` | `0.3` | How aggressively to trim irrelevant tools |

You can also set `AGENT_AMP_MAX_ITERATIONS` as an environment variable for
per-session overrides.

## Next steps

- [Architecture](architecture.md) -- understand the kernel pipeline
- [Features](features.md) -- the 11 features in detail
- [Adapters](adapters.md) -- per-host integration guide
- [SLM Composition](composition.md) -- how SuperLocalMemory multiplies amp
