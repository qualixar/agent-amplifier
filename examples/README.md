# Agent Amplifier — examples

This directory contains optional, copy-paste reference implementations that
plug into the **universal memory plane** (LLD-04 V2.1 §3.5). These files are
**not** part of the `agent_amplifier` package — they are working examples
that demonstrate the contract.

For the full adapter contract (when to build a class-based adapter vs a
callback pair, testing rules, anti-patterns), read
[`../docs/adapter-spec.md`](../docs/adapter-spec.md).

---

## Memory plane contract

Any provider that exposes two callables works as a memory backend:

```python
def recall(query: str, limit: int = 3) -> list[RecalledPattern]: ...
def remember(outcome: Outcome) -> None: ...
```

- `RecalledPattern` is a frozen dataclass exported from `agent_amplifier`
  (fields: `text`, `score`, `tags`, `source`, `metadata`).
- `Outcome` is a frozen dataclass exported from `agent_amplifier` (fields:
  `query`, `effort`, `iterations`, `quality`, `converged`, `tokens_used`).
- The kernel applies universal injection-defense (cap to 8 KB + neutralize
  + smuggling-signal log) to every recalled `text` automatically; providers
  return raw text.
- Both callbacks must NEVER raise. The kernel logs and returns safe defaults
  if they do.
- For mypy strict, annotate `RecalledPattern.metadata` arguments as
  `Mapping[str, Any]`, not `dict[str, Any]` — the dataclass field is
  covariant.

---

## Example 1 — SuperLocalMemory provider (`slm_provider.py`)

Drop-in callbacks for [SuperLocalMemory](https://qualixar.com/superlocalmemory).
Wires SLM as the recall + remember source for Agent Amplifier without
introducing SLM as a core dependency.

```python
from agent_amplifier import AgentAmplifier, ClaudeCodeAdapter
from examples.slm_provider import SLMProvider

slm = SLMProvider()
amp = AgentAmplifier(
    adapter=ClaudeCodeAdapter(kernel=None),
    memory_recall=slm.recall,
    memory_remember=slm.remember,
)
```

If SLM is not installed, the provider sets itself disabled and the callbacks
return safely (`recall` -> `[]`, `remember` -> no-op). The amplifier still
runs — just without cross-session learning.

The provider preserves SLM-specific defenses (HMAC signing, tag-allowlist,
sentinel-prefixed argv). Universal injection-defense (cap + neutralize +
smuggling-detect) is performed by the kernel via
`agent_amplifier._internal.recall_safety` regardless of which provider
produced the text.

---

## Example 2 — Mem0 callback (~15 lines)

[Mem0](https://github.com/mem0ai/mem0) is a hosted long-term memory service
for agents. Wire it without writing an adapter class:

```python
from mem0 import Memory
from agent_amplifier import AgentAmplifier, ClaudeCodeAdapter, Outcome, RecalledPattern

mem = Memory()  # uses MEM0_API_KEY from env
USER_ID = "alice"

def mem0_recall(query: str, limit: int = 3) -> list[RecalledPattern]:
    try:
        rows = mem.search(query=query, user_id=USER_ID, limit=limit)
    except Exception:
        return []
    return [
        RecalledPattern(
            text=str(r.get("memory", ""))[:4096],
            score=float(r.get("score", 0.0)),
            source=f"mem0:{r.get('id', '')}",
        )
        for r in rows.get("results", [])
    ]

def mem0_remember(outcome: Outcome) -> None:
    try:
        mem.add(
            messages=[{"role": "user", "content": outcome.query[:200]}],
            user_id=USER_ID,
            metadata={"effort": outcome.effort.value, "quality": outcome.quality},
        )
    except Exception:
        pass  # never raise

amp = AgentAmplifier(
    adapter=ClaudeCodeAdapter(kernel=None),
    memory_recall=mem0_recall,
    memory_remember=mem0_remember,
)
```

The same pattern applies to Letta, Zep, Pinecone, Chroma, or any vector
store — replace `mem.search` / `mem.add` with the equivalent calls.

---

## Example 3 — Flat-file callback (~15 lines, zero deps)

For local prototyping or air-gapped environments, a JSONL file is enough:

```python
import json
from pathlib import Path
from agent_amplifier import AgentAmplifier, ClaudeCodeAdapter, Outcome, RecalledPattern

LOG = Path(".agent-amplifier-memory.jsonl")

def file_recall(query: str, limit: int = 3) -> list[RecalledPattern]:
    if not LOG.is_file():
        return []
    q = query.lower()
    hits: list[RecalledPattern] = []
    try:
        for line in LOG.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if q in row.get("query", "").lower():
                hits.append(RecalledPattern(text=row["query"], source=f"jsonl:{LOG.name}"))
    except Exception:
        return []
    return hits[-limit:]

def file_remember(outcome: Outcome) -> None:
    try:
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(outcome.to_dict()) + "\n")
    except Exception:
        pass

amp = AgentAmplifier(
    adapter=ClaudeCodeAdapter(kernel=None),
    memory_recall=file_recall,
    memory_remember=file_remember,
)
```

This is the simplest possible memory backend. Use it as a sanity check that
your wiring works before integrating a real store.

---

## When to write a callback vs an adapter class

- **Callback pair** — when you are wiring an existing memory store for your
  own use. Stays in your own codebase. No detection, no lifecycle.
- **Adapter class** — when you are shipping support for a host or framework
  others will reuse. Implements `detect()` for `agent-amp install --auto`
  and binds at zero config. See [`../docs/adapter-spec.md`](../docs/adapter-spec.md)
  for the full contract and the `LlamaIndexAdapter` worked example.

---

## Bringing your own memory

V1 ships first-class adapters for Claude Code, Cursor, GitHub Copilot,
LangGraph, CrewAI, and AgentScope. The remaining systems (LlamaIndex,
AutoGen, Pydantic AI, Mem0/Letta/Zep, Aider, Cline, Continue.dev, Windsurf,
Antigravity, Anthropic Agent SDK, OpenAI Agents Session, Semantic Kernel,
DSPy) are V1.1 community-PR targets. Until then, the callback pattern above
covers them.

For each system's memory API, see the docstrings on the corresponding
adapter under `src/agent_amplifier/adapters/`. Each adapter cites the
upstream API in its module docstring.
