# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Model router — maps classification to suggested model tier.

Option A: adapters read ``suggested_model`` from the StepEnvelope
and auto-select the model for the next LLM call.

Option C: ``agent-amp report`` and the Streamlit dashboard show what
model the prompt WOULD route to without actually switching.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, ClassVar

from agent_amplifier.types import EffortLevel

LOG = logging.getLogger("agent_amplifier.model_router")

_DEFAULT_MAP: dict[str, str] = {
    "MINIMAL": "haiku",
    "LOW": "haiku",
    "MEDIUM": "sonnet",
    "HIGH": "opus",
    "MAX": "opus",
}

_TIER_DISPLAY: dict[str, str] = {
    "haiku": "Claude Haiku (fast, cost-efficient)",
    "sonnet": "Claude Sonnet (balanced)",
    "opus": "Claude Opus (deep reasoning)",
}


@dataclass(frozen=True, slots=True)
class ModelSuggestion:
    """Immutable model routing suggestion."""

    tier: str
    display: str
    reason: str
    overridden: bool = False


class ModelRouter:
    """Stateless router: effort level -> model tier."""

    DEFAULT_MAP: ClassVar[dict[str, str]] = dict(_DEFAULT_MAP)

    def __init__(
        self,
        *,
        model_map: dict[str, str] | None = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._map = dict(model_map or _DEFAULT_MAP)
        self._env_override = self._load_env_override()
        if self._env_override:
            self._map.update(self._env_override)

    @staticmethod
    def _load_env_override() -> dict[str, str]:
        raw = os.environ.get("AGENT_AMP_MODEL_MAP", "")
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {k.upper(): str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            LOG.warning("AGENT_AMP_MODEL_MAP parse error: %r", exc)
        return {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def model_map(self) -> dict[str, str]:
        return dict(self._map)

    def suggest(
        self,
        effort: EffortLevel | str,
        *,
        domain: str = "general",
    ) -> ModelSuggestion:
        if not self._enabled:
            return ModelSuggestion(
                tier="none",
                display="Model routing disabled",
                reason="Model routing is disabled in configuration.",
            )

        key = effort.name if isinstance(effort, EffortLevel) else str(effort).upper()
        tier = self._map.get(key)
        if tier is None:
            LOG.debug("Unknown effort level %r, defaulting to sonnet", key)
            tier = "sonnet"
        display = _TIER_DISPLAY.get(tier, tier)
        overridden = self._env_override.get(key) is not None
        reason = (
            f"Complexity={key} → {tier}"
            + (" (env override)" if overridden else "")
            + (f" [domain={domain}]" if domain != "general" else "")
        )
        return ModelSuggestion(
            tier=tier,
            display=display,
            reason=reason,
            overridden=overridden,
        )

    def suggest_for_report(
        self,
        effort: EffortLevel | str,
        domain: str = "general",
    ) -> dict[str, Any]:
        suggestion = self.suggest(effort, domain=domain)
        return {
            "tier": suggestion.tier,
            "display": suggestion.display,
            "reason": suggestion.reason,
            "overridden": suggestion.overridden,
        }


def create_router(
    *,
    model_map: dict[str, str] | None = None,
    enabled: bool = True,
) -> ModelRouter:
    """Factory for creating a ModelRouter."""
    return ModelRouter(model_map=model_map, enabled=enabled)


__all__ = [
    "ModelRouter",
    "ModelSuggestion",
    "create_router",
]
