# Agent Amplifier — Adapter Specification

> Public contract for community adapter authors.
> Version: V1.0 (locked 2026-04-27).
> Audience: anyone shipping `agent-amplifier-<host>` on PyPI or a PR into this repo.

---

## 1. What an adapter is

An adapter binds Agent Amplifier to one specific agent host or framework. It does three jobs:

1. **Detect** whether the host is installed on this machine (so `agent-amp install --auto` can pick the right one).
2. **Read** the host's native memory and surface it as `RecalledPattern` rows.
3. **Write** an amplification outcome back to that same memory store.

Adapters are intentionally thin. They are GLUE. The kernel owns effort routing, goal anchoring, convergence, prompt-injection defense, and observability. The adapter just speaks the host's local dialect.

If the host has no native memory (e.g., the Anthropic Agent SDK), there is no adapter — users wire it through the `memory_recall` / `memory_remember` callbacks instead. Document the wiring; do not invent a memory store.

---

## 2. The contract

`AdapterBase` lives in `src/agent_amplifier/adapter_base.py`. Every adapter subclasses it and implements four methods plus two class attributes.

### 2.1 Class attributes

```python
framework_name: ClassVar[str]   # must match ^[a-z][a-z0-9_]{2,31}$
HOST_NAME:      ClassVar[str]   # public slug used in RecalledPattern.source
version:        ClassVar[str]   # semver, your adapter's version
```

`framework_name` is validated by regex in `AdapterBase.__init__` — bad slugs raise `TypeError` before any I/O. Underscores, no dashes. `HOST_NAME` is the user-visible slug (dashes allowed, e.g., `"claude-code"`); it appears in `RecalledPattern.source` like `"claude-code:CLAUDE.md"`.

### 2.2 Required methods

```python
@classmethod
def detect(cls) -> bool:
    """True iff this adapter's host is present on the current machine."""

def install(self) -> None:
    """Attach amplifier hooks (or just mark installed for file-based hosts)."""

def uninstall(self) -> None:
    """Detach amplifier hooks. Surgical: only OUR hooks, never others'."""

def on_before_step(self, context: dict[str, Any]) -> dict[str, Any]:
    """Translate framework event -> kernel call -> modified context."""

def on_after_step(
    self,
    context: dict[str, Any],
    result: dict[str, Any] | str,
) -> dict[str, Any]:
    """Feed framework result back; return decision dict with 'action' key."""

def default_memory_recall(
    self, query: str, limit: int = 3
) -> list[RecalledPattern]:
    """Read host-native memory. Default: empty list. MUST NOT raise."""

def default_memory_remember(self, outcome: Outcome) -> None:
    """Write outcome to host-native memory. Default: no-op. MUST NOT raise."""
```

The async siblings `aon_before_step` / `aon_after_step` exist on `AdapterBase` and bridge to the sync versions via `anyio.to_thread.run_sync` by default — override only when the host is genuinely async (LangGraph, async LangChain callbacks).

For file-based hosts (Claude Code, Cursor, GitHub Copilot) the lifecycle methods are markers — there are no callbacks to attach. See `src/agent_amplifier/adapters/claude_code.py:95-111` for the canonical no-op pattern.

### 2.3 The two iron rules

1. **`default_memory_recall` and `default_memory_remember` MUST NEVER RAISE.** On any error, log at WARNING and return `[]` / `None`. The kernel applies a defensive try/except around the callback as a last-resort guard (the kernel), but you should never rely on it — the kernel may downgrade the entire `before_step` if your method misbehaves. See [`ClaudeCodeAdapter.default_memory_remember`](../src/agent_amplifier/adapters/claude_code.py) for the canonical try/except pattern.
2. **Adapters MUST NOT call `recall_safety.apply_recall_safety` themselves.** The kernel applies it to every chunk's `text` after the adapter returns. Return raw text; let the kernel handle capping + neutralization + smuggling-signal logging. (`src/agent_amplifier/kernel.py:374-419`.)

---

## 3. `RecalledPattern` reference

Defined as `RecalledPattern` in [`src/agent_amplifier/types.py`](../src/agent_amplifier/types.py). Frozen dataclass, slots, immutable. Field validation runs in `__post_init__`.

| Field | Type | Default | When to populate |
|---|---|---|---|
| `text` | `str` | required | Always. Raw recalled content. Kernel caps to 8KB. |
| `score` | `float` | `0.0` | If your store ranks results (vector search, BM25). `0.0` means "I don't know". |
| `tags` | `tuple[str, ...]` | `()` | Source-specific labels. Examples: `("project-rule",)`, `("checkpoint", "thread:42")`. |
| `source` | `str` | `""` | Provenance. Format `"<HOST_NAME>:<sub-source>"`. Always populate. |
| `metadata` | `Mapping[str, Any]` | `{}` | Adapter-specific extension. Type as `Mapping`, not `dict`, to satisfy mypy strict. |

**Provenance examples from V1 adapters:**

- `claude-code:/path/to/CLAUDE.md` (Claude Code adapter)
- `cursor:.cursor/rules/python.mdc` (Cursor adapter)
- `langgraph:thread-default` (LangGraph adapter)
- `crewai:short-term` (CrewAI adapter)
- `agentscope:TemporaryMemory` (AgentScope adapter)

**Don't:** populate `metadata` with secrets, tokens, file contents, or anything you wouldn't paste into a public bug report. The kernel does not redact `metadata`.

---

## 4. `Outcome` reference

Defined in `src/agent_amplifier/types.py:234-280`. Frozen dataclass, passed to your `default_memory_remember`.

| Field | Type | Notes |
|---|---|---|
| `query` | `str` | The original goal anchor. Cap to ~200 chars before persisting; never log full. |
| `effort` | `EffortLevel` | Enum: `MINIMAL`, `LOW`, `MEDIUM`, `HIGH`, `MAX`. |
| `iterations` | `int` | `>= 0`. |
| `quality` | `float` | `0.0..1.0`. |
| `converged` | `bool` | Did the loop converge? |
| `tokens_used` | `int` | `>= 0`. |

`Outcome.to_dict()` returns a JSON-safe representation. Use it for storage; do not pickle.

---

## 5. Worked example: `LlamaIndexAdapter`

This is a V1.1 community-PR target. Here is what someone shipping it would write.

### 5.1 Module skeleton (`src/agent_amplifier/adapters/llamaindex.py`)

```python
"""LlamaIndexAdapter — vector-store-backed memory binding for LlamaIndex.

Deferred to V1.1. Memory shape: vector store with similarity
search. Read: index.as_retriever().retrieve(query). Write: insert a Document.

Lazy import: ``llama_index`` is NEVER imported at module top — only inside
``detect()`` via ``importlib.util.find_spec`` so users without LlamaIndex keep
``import agent_amplifier`` cheap.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.types import RecalledPattern

if TYPE_CHECKING:  # pragma: no cover
    from agent_amplifier.types import Outcome

LOG = logging.getLogger("agent_amplifier.adapters.llamaindex")

_PER_CHUNK_BYTES: int = 4096


class LlamaIndexAdapter(AdapterBase):
    """Adapter for LlamaIndex (VectorStoreIndex-backed memory)."""

    framework_name: ClassVar[str] = "llama_index"
    HOST_NAME: ClassVar[str] = "llamaindex"
    version: ClassVar[str] = "1.0.0"

    def __init__(self, index: Any, *, kernel: Any = None) -> None:
        """Bind a user-supplied VectorStoreIndex."""
        super().__init__(kernel=kernel)
        self._index = index  # duck-typed; do NOT isinstance-check

    @classmethod
    def detect(cls) -> bool:
        try:
            import importlib.util
            return importlib.util.find_spec("llama_index") is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False

    def install(self) -> None:
        self._mark_installed()

    def uninstall(self) -> None:
        self._mark_uninstalled()

    def on_before_step(self, context: dict[str, Any]) -> dict[str, Any]:
        return context

    def on_after_step(
        self, context: dict[str, Any], result: dict[str, Any] | str,
    ) -> dict[str, Any]:
        return {"action": "continue"}

    def default_memory_recall(
        self, query: str, limit: int = 3
    ) -> list[RecalledPattern]:
        """Vector-search the index. Returns up to ``limit`` ranked patterns."""
        if not query.strip():
            return []
        try:
            retriever = self._index.as_retriever(similarity_top_k=limit)
            nodes = retriever.retrieve(query)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("llamaindex recall: retrieve failed: %r", exc)
            return []
        return [
            RecalledPattern(
                text=str(getattr(n, "text", ""))[:_PER_CHUNK_BYTES],
                score=float(getattr(n, "score", 0.0) or 0.0),
                source=f"{self.HOST_NAME}:{getattr(n, 'node_id', '')}",
            )
            for n in nodes
            if getattr(n, "text", None)
        ]

    def default_memory_remember(self, outcome: Outcome) -> None:
        """Insert a Document so future recalls can match this outcome."""
        try:
            # Lazy import: only when actually writing.
            from llama_index.core import Document  # type: ignore[import-not-found]
            doc = Document(
                text=f"{outcome.query} -> quality={outcome.quality:.2f}",
                metadata={"effort": outcome.effort.value, "iters": outcome.iterations},
            )
            self._index.insert(doc)
        except Exception as exc:
            LOG.warning("llamaindex remember: insert failed: %r", exc)
```

### 5.2 What this gives the user

```python
from llama_index.core import VectorStoreIndex
from agent_amplifier import AgentAmplifier
from agent_amplifier_llamaindex import LlamaIndexAdapter  # community package

index = VectorStoreIndex.from_documents(my_docs)
amp = AgentAmplifier(adapter=LlamaIndexAdapter(index, kernel=None))
# amp now reads/writes against the user's existing LlamaIndex store.
```

Forty lines of glue. The user did not migrate to a new memory system. That is the point.

---

## 6. Lazy imports rule

**Framework imports MUST be inside methods. NEVER at module top.**

Why: `from agent_amplifier import AgentAmplifier` runs the package `__init__`, which imports every adapter module. If an adapter top-imports `langgraph` and the user does not have LangGraph installed, the import fails and `agent_amplifier` itself becomes unimportable.

**Reference implementation:** `src/agent_amplifier/adapters/langgraph.py:79-89` shows the canonical `detect()` using `importlib.util.find_spec`. Note the broad except — `find_spec` can raise `ImportError`, `ValueError`, or `ModuleNotFoundError` depending on Python version and the exact failure mode.

For framework symbols used inside methods (rare — most adapters duck-type the user-supplied object), import inside the method body:

```python
def default_memory_remember(self, outcome: Outcome) -> None:
    try:
        from llama_index.core import Document  # lazy
        ...
```

For type hints only, use `if TYPE_CHECKING` to avoid runtime cost:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover
    from agent_amplifier.types import Outcome
```

---

## 7. Testing your adapter

The kernel ships with 100% line + branch coverage. Adapters held to the same bar.

### 7.1 Mock objects, not framework imports

Tests must run without the framework installed. Inject duck-typed mocks:

```python
class _FakeCheckpointer:
    def get_tuple(self, config):
        return _FakeTuple(checkpoint={"channel_values": {"messages": [...]}})

def test_recall_substring_match():
    adapter = LangGraphAdapter(checkpointer=_FakeCheckpointer(), thread_id="t1", kernel=None)
    out = adapter.default_memory_recall("auth", limit=3)
    assert len(out) == 1
    assert out[0].source == "langgraph:thread-t1"
```

See [`tests/test_langgraph_adapter.py`](../tests/test_langgraph_adapter.py) for the full pattern.

### 7.2 Required test surface per adapter

- `test_detect_returns_true_when_host_present` (mock `find_spec` or fixture)
- `test_detect_returns_false_when_host_absent`
- `test_recall_empty_query_returns_recent` (or empty, depending on adapter)
- `test_recall_substring_or_score_filter_works`
- `test_recall_swallows_exception` — your method MUST NOT raise
- `test_remember_writes_to_target` (or no-op, with assertion)
- `test_remember_swallows_exception`
- `test_install_uninstall_lifecycle`

### 7.3 Integration-smoke test

One test that constructs a real (or near-real) host object and verifies one round trip. For file-based hosts: write a fixture `CLAUDE.md`, recall, assert. For framework hosts: instantiate the framework's in-memory variant (`MemorySaver`, `crewai.Crew(memory=True)`, `agentscope.memory.TemporaryMemory`) and round-trip one entry.

---

## 8. Anti-patterns — what NOT to do

| Don't | Why |
|---|---|
| Reimplement memory storage in your adapter | The user already has memory. Read theirs; do not invent another. |
| Reach into kernel internals (`amp._memory_recall`, `amp._convergence_state`) | Private attributes break without notice. Use the public callback hooks. |
| Log secrets, tokens, file paths with PII, or full `outcome.query` | Cap query at ~100 chars in log lines. Never log credentials. |
| Raise from `default_memory_recall` / `default_memory_remember` | Kernel catches as a last-resort guard but degrades the step. Log + return `[]` / `None` instead. |
| Top-import `langgraph` / `crewai` / `agentscope` at module level | Breaks `import agent_amplifier` for users without that framework. |
| Call `recall_safety.apply_recall_safety` yourself | The kernel applies it after your method returns. Doing it twice is harmless but signals you misread the contract. |
| Auto-create user files (`CLAUDE.md`, `.cursor/rules/`) on first write | Anti-surprise. Write only if the directory/file already exists; otherwise no-op-and-log. |
| Use `isinstance` against framework types | Forward-incompatible. Duck-type. See `LangGraphAdapter.__init__` for the pattern. |
| Pickle `Outcome` for storage | Use `Outcome.to_dict()`. |

---

## 9. Naming + packaging

### 9.1 If you ship as a separate PyPI package

Name: `agent-amplifier-<host>` (kebab-case on PyPI; underscore in the import path).

```
agent-amplifier-llamaindex/
├── pyproject.toml          # depends on agent-amplifier>=1.0
├── src/
│   └── agent_amplifier_llamaindex/
│       └── __init__.py     # exports LlamaIndexAdapter
└── tests/
```

Users install via `pip install agent-amplifier[llamaindex]` if you also register your package as an extra in the core repo (PR welcome).

### 9.2 If you PR into this repo

Add the module under `src/agent_amplifier/adapters/<host>.py`, register the export in `src/agent_amplifier/adapters/__init__.py` and `src/agent_amplifier/__init__.py`, add tests under `tests/test_<host>_adapter.py`. Match the style of the existing six adapters; the reviewer is mechanical.

Coverage gate: 100% line + branch on the new module. Run:

```bash
.venv/bin/python -m pytest tests/test_<host>_adapter.py \
    --cov=src/agent_amplifier/adapters/<host> --cov-report=term-missing
```

---

## 10. Reference adapters in this repo

| File | Memory shape | Key learnings |
|---|---|---|
| `src/agent_amplifier/adapters/claude_code.py` | File-based markdown | Anti-surprise write rule; H2-section ranking; multi-source priority list |
| `src/agent_amplifier/adapters/cursor.py` | File-based MDC + frontmatter | Manual YAML-ish frontmatter parsing without PyYAML dep; legacy `.cursorrules` fallback |
| `src/agent_amplifier/adapters/github_copilot.py` | File-based markdown + path-scoped | Path-scoping via `applyTo` pattern; multi-file aggregation |
| `src/agent_amplifier/adapters/langgraph.py` | Checkpointer state | Duck-typed checkpointer; no-op remember (graph runtime owns lifecycle); message-shape tolerance |
| `src/agent_amplifier/adapters/crewai.py` | Unified Memory class | Lazy import inside method body; chat-buffer shape |
| `src/agent_amplifier/adapters/agentscope.py` | Memory class | Async-bridging via existing `_anyio_portal.py`; Asian-market coverage |

Reading these in order takes 15 minutes and shows every pattern you need.

### 10.1 Cursor `globs:` accepts both string and list forms

The Cursor adapter parses `globs:` in MDC frontmatter as either a single
string or a YAML-style flow list and normalizes both to `list[str]`:

```yaml
---
description: Python rules
globs: src/**/*.py                # single-string form
---
```

```yaml
---
description: Python rules
globs: ["src/**/*.py", "tests/**/*.py"]   # list form
---
```

In both cases the parsed list is surfaced via
`RecalledPattern.metadata["globs"]`, so a downstream scoped-recall layer
(or the kernel in a future version) can filter by current-file context.
The kernel does not currently filter by globs — the metadata simply
preserves the signal.

### 10.2 GitHub Copilot `applyTo:` is preserved in metadata

Copilot's `*.instructions.md` files may carry an `applyTo:` glob in
their YAML frontmatter:

```yaml
---
applyTo: "**/*.ts"
---

## TypeScript rules
...
```

The adapter parses this and surfaces it via
`RecalledPattern.metadata["apply_to"]`. Like the Cursor `globs:` case,
the kernel does not yet filter recalls by current-file context — the
metadata preserves the signal so a future scoped-recall layer can use
it without breaking the contract.

---

## 11. Versioning + deprecation

The contract above is locked at V1.0. Adapter authors can rely on `AdapterBase`, `RecalledPattern`, and `Outcome` signatures not changing within the V1.x line. New optional fields on `RecalledPattern` may be added (with sensible defaults) — additive only.

Breaking changes will land in V2.0 with a one-version deprecation window and a migration guide.

---

## 12. Getting help

- Open a GitHub issue with `[adapter:<host>]` in the title
- Read the source: `src/agent_amplifier/adapters/` ships six worked examples (Claude Code, Cursor, GitHub Copilot, LangGraph, CrewAI, AgentScope) and is the canonical reference for the contract.

---

Built by [Qualixar](https://qualixar.com). AI Reliability Engineering for AI agents.
