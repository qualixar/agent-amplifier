# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Agent Amplifier - Runtime amplification for AI agents.

By Qualixar (https://qualixar.com).

V1.0 surface (.1 / Cluster E):
    * AgentAmplifier        — sync facade
    * AsyncAgentAmplifier   — async-native facade
    * StepEnvelope          — typed return shape
    * AdapterBase           — ABC for framework adapters
    * KernelContractError   — adapter contract violation
    * KernelReentrancyError — re-entry guard violation
    * AmplifierConfig + enums, Cluster A surface (re-exported for convenience)
"""

from __future__ import annotations

from agent_amplifier.adapter_base import (
    AdapterAlreadyInstalledError,
    AdapterBase,
    AdapterError,
    AdapterNotInstalledError,
)
from agent_amplifier.adapters import (
    AgentScopeAdapter,
    ClaudeCodeAdapter,
    CrewAIAdapter,
    CursorAdapter,
    GitHubCopilotAdapter,
    LangChainAdapter,
    LangGraphAdapter,
)
from agent_amplifier.kernel import (
    AgentAmplifier,
    AsyncAgentAmplifier,
    KernelContractError,
    KernelReentrancyError,
    StepEnvelope,
    amplify,
)
from agent_amplifier.types import (
    AmplifierConfig,
    AmplifierEvent,
    BudgetMode,
    ConvergenceState,
    EffortLevel,
    ObservabilityCallback,
    Outcome,
    PhaseIndex,
    QualityScore,
    RecalledPattern,
    TaskClassification,
)

__version__ = "1.1.0"
__author__ = "Qualixar"
__license__ = "AGPL-3.0-or-later"
__url__ = "https://qualixar.com"

__all__ = [
    "AdapterAlreadyInstalledError",
    "AdapterBase",
    "AdapterError",
    "AdapterNotInstalledError",
    "AgentAmplifier",
    "AgentScopeAdapter",
    "AmplifierConfig",
    "AmplifierEvent",
    "AsyncAgentAmplifier",
    "BudgetMode",
    "ClaudeCodeAdapter",
    "ConvergenceState",
    "CrewAIAdapter",
    "CursorAdapter",
    "EffortLevel",
    "GitHubCopilotAdapter",
    "KernelContractError",
    "KernelReentrancyError",
    "LangChainAdapter",
    "LangGraphAdapter",
    "ObservabilityCallback",
    "Outcome",
    "PhaseIndex",
    "QualityScore",
    "RecalledPattern",
    "StepEnvelope",
    "TaskClassification",
    "__author__",
    "__license__",
    "__url__",
    "__version__",
    "amplify",
]
