# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Fixtures for dashboard UI tests."""

from __future__ import annotations

from typing import Any

import pytest

_CANNED_HEALTH: dict[str, Any] = {
    "status": "ok",
    "amp_version": "1.1.1",
    "db_path": "/tmp/state.db",
}

_CANNED_CONFIG: dict[str, Any] = {
    "config": {
        "max_iterations": 4,
        "token_budget": 250000,
        "persona": "senior_engineer",
    }
}

_CANNED_IPS: dict[str, Any] = {
    "ips": [
        {"id": "kernel", "name": "Runtime Kernel", "file": "kernel.py", "enabled": True, "order": 1},
        {"id": "effort_router", "name": "Dynamic Effort Router", "file": "effort_router.py", "enabled": True, "order": 2},
        {"id": "goal_anchor", "name": "Goal Anchor Protocol", "file": "goal_anchor.py", "enabled": True, "order": 3},
        {"id": "convergence", "name": "Convergence Detector", "file": "convergence.py", "enabled": True, "order": 4},
        {"id": "semantic_modifiers", "name": "Semantic Modifier Injection", "file": "semantic_modifiers.py", "enabled": True, "order": 5},
        {"id": "adapters", "name": "Cross-Framework Adapter Layer", "file": "adapter_base.py", "enabled": True, "order": 6},
        {"id": "phase_prompts", "name": "Phase Prompt Engine", "file": "phase_prompts.py", "enabled": True, "order": 7},
        {"id": "personas", "name": "Persona Escalation", "file": "personas.py", "enabled": True, "order": 8},
        {"id": "memory_plane", "name": "Cross-Host Memory Plane", "file": "types.py", "enabled": True, "order": 9},
        {"id": "token_budget", "name": "Token Budget Controller", "file": "token_budget.py", "enabled": True, "order": 10},
        {"id": "tool_selector", "name": "Tool Selector", "file": "tool_selector.py", "enabled": True, "order": 11},
    ]
}

_CANNED_TELEMETRY: dict[str, Any] = {
    "db_exists": True,
    "counts": {"sessions": 5, "envelopes": 12, "events": 20, "outcomes": 10},
    "coverage_rate": 0.95,
    "convergence_rate": 0.85,
}

_CANNED_TURNS: dict[str, Any] = {
    "limit": 50,
    "turns": [
        {
            "session_id": "session-a",
            "turn_id": 1,
            "complexity": "high",
            "domain": "backend",
            "trigger": "PERSONA",
            "phase": "EXECUTE",
            "created_at": 1715500000.0,
            "duration_ms": 2500,
            "converged": True,
            "stop_reason": "done",
            "tokens_used": 4200,
            "quality_estimate": 0.94,
        },
        {
            "session_id": "session-b",
            "turn_id": 1,
            "complexity": "medium",
            "domain": "frontend",
            "trigger": None,
            "phase": "VERIFY",
            "created_at": 1715490000.0,
            "duration_ms": 900,
            "converged": False,
            "stop_reason": "max_iterations",
            "tokens_used": 3100,
            "quality_estimate": 0.88,
        },
        {
            # IP-4: turn with no outcome row → converged=None → "—" in table.
            "session_id": "session-c",
            "turn_id": 1,
            "complexity": "low",
            "domain": "general",
            "trigger": None,
            "phase": "EXPLORE",
            "created_at": 1715480000.0,
            "duration_ms": None,
            "converged": None,
            "stop_reason": None,
            "tokens_used": None,
            "quality_estimate": None,
        },
    ],
}

_CANNED_CONVERGENCE: dict[str, Any] = {
    "days": 7,
    "points": [
        {"date": "2026-05-06", "total": 10, "converged": 8, "rate": 0.8},
        {"date": "2026-05-07", "total": 12, "converged": 10, "rate": 0.83},
        {"date": "2026-05-08", "total": 8, "converged": 7, "rate": 0.875},
        {"date": "2026-05-09", "total": 15, "converged": 14, "rate": 0.93},
        {"date": "2026-05-10", "total": 11, "converged": 10, "rate": 0.91},
        {"date": "2026-05-11", "total": 9, "converged": 8, "rate": 0.89},
        {"date": "2026-05-12", "total": 10, "converged": 9, "rate": 0.9},
    ],
}

_CANNED_ADAPTERS: dict[str, Any] = {
    "adapters": [
        {"name": "claude_code", "display_name": "Claude Code", "detected": True, "installed": True},
        {"name": "cursor", "display_name": "Cursor", "detected": True, "installed": False},
        {"name": "github_copilot", "display_name": "GitHub Copilot", "detected": True, "installed": False},
        {"name": "langgraph", "display_name": "LangGraph", "detected": True, "installed": False},
        {"name": "crewai", "display_name": "CrewAI", "detected": True, "installed": False},
        {"name": "agentscope", "display_name": "AgentScope", "detected": True, "installed": False},
        {"name": "langchain", "display_name": "LangChain", "detected": True, "installed": False},
    ]
}


@pytest.fixture
def canned_responses() -> dict[str, Any]:
    return {
        "health": _CANNED_HEALTH,
        "config": _CANNED_CONFIG,
        "ips": _CANNED_IPS,
        "telemetry": _CANNED_TELEMETRY,
        "turns": _CANNED_TURNS,
        "convergence": _CANNED_CONVERGENCE,
        "adapters": _CANNED_ADAPTERS,
    }
