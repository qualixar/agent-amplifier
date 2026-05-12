# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Built-in host + framework adapters (+ 7.6.C).

(file-based hosts) — bind to host-native memory files:
    * ``ClaudeCodeAdapter``     — ``CLAUDE.md`` / ``MEMORY.md`` / ``~/.claude/CLAUDE.md``
    * ``CursorAdapter``         — ``.cursor/rules/*.mdc`` (+ legacy ``.cursorrules``)
    * ``GitHubCopilotAdapter``  — ``.github/copilot-instructions.md`` (+ scoped ``*.instructions.md``)

(framework adapters) — wrap a user-supplied runtime object:
    * ``LangGraphAdapter``      — ``BaseCheckpointSaver.get_tuple(...)``
    * ``CrewAIAdapter``         — ``crew.memory.search(query, limit)``
    * ``AgentScopeAdapter``     — ``memory.get_memory()`` (working memory)

Each adapter:
    * sets ``framework_name`` matching ``AdapterBase`` regex (underscores only),
    * exposes ``HOST_NAME`` (slug) used as the ``RecalledPattern.source`` prefix,
    * implements ``detect()`` for ``agent-amp install --auto``,
    * implements ``install`` / ``uninstall`` as no-op lifecycle markers (file-based
      hosts have nothing to attach hooks to; framework adapters compose via
      user-supplied objects rather than mutating callback registries),
    * implements ``on_before_step`` / ``on_after_step`` as identity / continue,
    * overrides ``default_memory_recall`` / ``default_memory_remember`` to read /
      append host-native memory.

The kernel re-applies ``recall_safety.apply_recall_safety`` (cap + neutralize +
smuggling-signal detect) to every chunk's ``text``; adapters DO NOT call it
themselves. Adapters MUST NOT raise from ``default_memory_*``: on any error they
log at WARNING and return ``[]`` / ``None``.

Critical: framework adapters use **lazy imports** — the framework module
(``langgraph``, ``crewai``, ``agentscope``) is NEVER imported at module top
so ``import agent_amplifier`` stays cheap and side-effect-free for users
who don't have the framework installed. Imports happen inside ``detect()``
(via ``importlib.util.find_spec``) and lazily inside method bodies.

Source: .1 (the 6 V1 adapters), §2.5 (memory hooks),
"""
from __future__ import annotations

from agent_amplifier.adapters.agentscope import AgentScopeAdapter
from agent_amplifier.adapters.claude_code import ClaudeCodeAdapter
from agent_amplifier.adapters.crewai import CrewAIAdapter
from agent_amplifier.adapters.cursor import CursorAdapter
from agent_amplifier.adapters.github_copilot import GitHubCopilotAdapter
from agent_amplifier.adapters.langchain import LangChainAdapter
from agent_amplifier.adapters.langgraph import LangGraphAdapter
from agent_amplifier.adapters.slm import SLMAdapter

__all__ = [
    "AgentScopeAdapter",
    "ClaudeCodeAdapter",
    "CrewAIAdapter",
    "CursorAdapter",
    "GitHubCopilotAdapter",
    "LangChainAdapter",
    "LangGraphAdapter",
    "SLMAdapter",
]
