# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Adapter registry facade for the dashboard backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.adapters import (
    AgentScopeAdapter,
    ClaudeCodeAdapter,
    CrewAIAdapter,
    CursorAdapter,
    GitHubCopilotAdapter,
    LangChainAdapter,
    LangGraphAdapter,
)
from agent_amplifier.dashboard.backend.models import AdapterInfo


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    name: str
    display_name: str
    factory: Callable[[], AdapterBase] | None
    detector: Callable[[], bool]


class _RuntimeObject:
    pass


ADAPTER_SPECS: tuple[AdapterSpec, ...] = (
    AdapterSpec(
        "claude_code",
        "Claude Code",
        lambda: ClaudeCodeAdapter(kernel=None),
        ClaudeCodeAdapter.detect,
    ),
    AdapterSpec("cursor", "Cursor", lambda: CursorAdapter(kernel=None), CursorAdapter.detect),
    AdapterSpec(
        "github_copilot",
        "GitHub Copilot",
        lambda: GitHubCopilotAdapter(kernel=None),
        GitHubCopilotAdapter.detect,
    ),
    AdapterSpec(
        "langgraph",
        "LangGraph",
        lambda: LangGraphAdapter(checkpointer=_RuntimeObject(), kernel=None),
        LangGraphAdapter.detect,
    ),
    AdapterSpec(
        "crewai",
        "CrewAI",
        lambda: CrewAIAdapter(crew=_RuntimeObject(), kernel=None),
        CrewAIAdapter.detect,
    ),
    AdapterSpec(
        "agentscope",
        "AgentScope",
        lambda: AgentScopeAdapter(memory=_RuntimeObject(), kernel=None),
        AgentScopeAdapter.detect,
    ),
    AdapterSpec(
        "langchain",
        "LangChain",
        lambda: LangChainAdapter(memory=_RuntimeObject(), kernel=None),
        LangChainAdapter.detect,
    ),
)


def list_adapters() -> list[AdapterInfo]:
    infos: list[AdapterInfo] = []
    for spec in ADAPTER_SPECS:
        infos.append(
            AdapterInfo(
                name=spec.name,
                display_name=spec.display_name,
                detected=_detect(spec),
                installed=_is_installed(spec),
            )
        )
    return infos


def install_adapter(name: str) -> str | None:
    spec = _find(name)
    if spec is None or spec.factory is None:
        return None
    adapter = spec.factory()
    adapter.install()
    return f"installed:{name}"


def adapter_exists(name: str) -> bool:
    return _find(name) is not None


def _find(name: str) -> AdapterSpec | None:
    for spec in ADAPTER_SPECS:
        if spec.name == name:
            return spec
    return None


def _detect(spec: AdapterSpec) -> bool:
    try:
        return spec.detector()
    except Exception:
        return False


def _is_installed(spec: AdapterSpec) -> bool:
    if spec.factory is None:
        return False
    try:
        return spec.factory().is_installed()
    except Exception:
        return False


__all__ = ["adapter_exists", "install_adapter", "list_adapters"]
