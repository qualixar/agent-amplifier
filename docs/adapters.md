# Adapters

Agent Amplifier ships seven host adapters at v1.0. Each adapter is a thin
shim (~200 lines) that translates between the host framework's event model
and the amp kernel's `before_step()` / `after_step()` contract. The adapter
layer is how AI Reliability Engineering reaches every major agent host through
one kernel.

---

## Adapter contract

Every adapter extends `AdapterBase` (defined in `adapter_base.py`) and
implements:

| Method | Purpose |
|---|---|
| `install()` | Attach amplifier hooks to the host framework |
| `uninstall()` | Detach amplifier hooks (surgical: only ours) |
| `on_before_step(context)` | Translate host event into kernel call, return modified context |
| `on_after_step(context, result)` | Feed host result back to kernel, return decision dict |
| `detect()` (class method) | Return `True` if the host is installed on this system |
| `default_memory_recall(query)` | Read host-native memory (Feature 9) |
| `default_memory_remember(outcome)` | Write outcome to host-native memory (Feature 9) |

Async hosts (LangGraph, async LangChain) override `aon_before_step()` and
`aon_after_step()` directly. The base class provides a sync-bridge default via
`anyio.to_thread.run_sync`.

---

## 1. Claude Code

**Primary launch target.** The deepest integration.

```bash
agent-amp install claude-code
```

Installs five hooks into `~/.claude/settings.json`:

- `UserPromptSubmit` -- full pipeline, returns `StepEnvelope` as `additionalContext`
- `PreToolUse` -- goal re-injection at N-call intervals, tool-selector gating
- `PostToolUse` -- convergence state update, budget tracking
- `Stop` -- outcome write to memory, telemetry to `state.db`
- `PreCompact` -- observe-only in v1.0 (active deferral in v1.0.1)

**Memory:** reads `CLAUDE.md`, `MEMORY.md`, and `~/.claude/CLAUDE.md`. If
SuperLocalMemory is installed, upgrades to SLM's 4-channel recall
automatically.

```python
from agent_amplifier.adapters.claude_code import ClaudeCodeAdapter

adapter = ClaudeCodeAdapter(kernel=None)
assert adapter.detect()  # True if ~/.claude/ exists
```

---

## 2. Cursor

```bash
agent-amp install cursor
```

Writes amplification rules to `.cursor/rules/agent-amp.mdc`. The rules are
read by Cursor at every prompt.

**Memory:** reads `.cursor/rules/*.mdc` and legacy `.cursorrules`.

```python
from agent_amplifier.adapters.cursor import CursorAdapter

adapter = CursorAdapter(kernel=amp)
adapter.install()
```

---

## 3. GitHub Copilot

```bash
agent-amp install github_copilot
```

Writes amplification instructions to `.github/copilot-instructions.md` and
scoped instruction files under `.github/instructions/`.

**Memory:** reads `.github/copilot-instructions.md` and scoped
`.github/instructions/*.instructions.md`.

```python
from agent_amplifier.adapters.github_copilot import GitHubCopilotAdapter

adapter = GitHubCopilotAdapter(kernel=amp)
adapter.install()
```

---

## 4. LangGraph

```bash
agent-amp install langgraph
```

Registers as a node in your LangGraph graph. The adapter hooks into the
checkpointer for memory read/write.

**Memory:** reads from whatever `BaseCheckpointSaver` you have configured.

```python
from agent_amplifier.adapters.langgraph import LangGraphAdapter

adapter = LangGraphAdapter(kernel=amp)
adapter.install()

# In your graph definition:
context = adapter.on_before_step({"query": user_input})
# ... run your LLM node ...
decision = adapter.on_after_step(context, llm_output)
```

For async graphs, use the async siblings:

```python
context = await adapter.aon_before_step({"query": user_input})
decision = await adapter.aon_after_step(context, llm_output)
```

---

## 5. CrewAI

```bash
agent-amp install crewai
```

Integrates with CrewAI's `Crew.memory` system.

**Memory:** reads from whatever `Crew.memory` you have configured.

```python
from agent_amplifier.adapters.crewai import CrewAIAdapter

adapter = CrewAIAdapter(kernel=amp)
adapter.install()
```

---

## 6. AgentScope

```bash
agent-amp install agentscope
```

Integrates with AgentScope's `Memory` instance.

**Memory:** reads from the framework's `Memory` instance you wired.

```python
from agent_amplifier.adapters.agentscope import AgentScopeAdapter

adapter = AgentScopeAdapter(kernel=amp)
adapter.install()
```

---

## 7. LangChain

```bash
agent-amp install langchain
```

Hooks into LangChain's `BaseMemory` and `BaseChatMessageHistory` APIs.

**Memory:** calls `memory.load_memory_variables({"input": query})` for recall
and writes outcomes through the configured memory backend.

```python
from agent_amplifier.adapters.langchain import LangChainAdapter

adapter = LangChainAdapter(kernel=amp)
adapter.install()
```

---

## Writing a custom adapter

Implement `AdapterBase` with your host's event model:

```python
from agent_amplifier.adapter_base import AdapterBase
from typing import Any, ClassVar

class MyHostAdapter(AdapterBase):
    framework_name: ClassVar[str] = "my_host"
    version: ClassVar[str] = "0.1.0"

    def install(self) -> None:
        # Attach your hooks
        self._mark_installed()

    def uninstall(self) -> None:
        # Detach your hooks
        self._mark_uninstalled()

    def on_before_step(self, context: dict[str, Any]) -> dict[str, Any]:
        # Call kernel.before_step(), return modified context
        envelope = self.kernel.before_step(
            query=context["query"],
            tool_names=context.get("tools", []),
        )
        context["amplification"] = envelope
        return context

    def on_after_step(
        self, context: dict[str, Any], result: dict[str, Any] | str
    ) -> dict[str, Any]:
        # Call kernel.after_step(), return decision
        return self.kernel.after_step(result=result)
```

**Constraints:**

- `framework_name` must match `^[a-z][a-z0-9_]{2,31}$`
- `install()` must be idempotent or raise `AdapterAlreadyInstalledError`
- `uninstall()` must remove only your hooks, never others'
- Do not route raw tool output into prompt slots without applying
  `_neutralize_xml` (see the docstring on `AdapterBase` for the full safety
  contract)

The full adapter specification is at [Adapter Spec](adapter-spec.md).

---

Built by [Qualixar](https://qualixar.com) -- AI Reliability Engineering for
AI agents.
