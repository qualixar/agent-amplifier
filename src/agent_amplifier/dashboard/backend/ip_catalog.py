# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Catalog of the 11 runtime amplification components exposed to the UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IpCatalogEntry:
    id: str
    name: str
    file: str


IP_CATALOG: tuple[IpCatalogEntry, ...] = (
    IpCatalogEntry("kernel", "Runtime Kernel", "src/agent_amplifier/kernel.py"),
    IpCatalogEntry("effort_router", "Dynamic Effort Router", "src/agent_amplifier/effort_router.py"),
    IpCatalogEntry("goal_anchor", "Goal Anchor Protocol", "src/agent_amplifier/goal_anchor.py"),
    IpCatalogEntry("convergence", "Convergence Detector", "src/agent_amplifier/convergence.py"),
    IpCatalogEntry("semantic_modifiers", "Semantic Modifier Injection", "src/agent_amplifier/semantic_modifiers.py"),
    IpCatalogEntry("adapters", "Cross-Framework Adapter Layer", "src/agent_amplifier/adapter_base.py"),
    IpCatalogEntry("phase_prompts", "Phase Prompt Engine", "src/agent_amplifier/phase_prompts.py"),
    IpCatalogEntry("personas", "Persona Escalation", "src/agent_amplifier/personas.py"),
    IpCatalogEntry("memory_plane", "Cross-Host Memory Plane", "src/agent_amplifier/types.py"),
    IpCatalogEntry("token_budget", "Token Budget Controller", "src/agent_amplifier/token_budget.py"),
    IpCatalogEntry("tool_selector", "Tool Selector", "src/agent_amplifier/tool_selector.py"),
)

IP_IDS: frozenset[str] = frozenset(entry.id for entry in IP_CATALOG)


__all__ = ["IP_CATALOG", "IP_IDS", "IpCatalogEntry"]
