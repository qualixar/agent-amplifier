# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Anthropic Messages API payload builder (verified against Azure Foundry).

The Foundry endpoint pattern (verified 2026-05-13 against ``gtic-resource``,
``tap-main-project-resource``, ``tap-aoai-dev-resource``):

    {endpoint}/anthropic/v1/messages?api-version=2024-10-22-preview

with the ``Authorization: Bearer <api-key>`` header (NOT ``api-key:``) and
``anthropic-version: 2023-06-01``.

This module produces ONLY the request body. Endpoint, auth headers, and the
HTTP call are the caller's responsibility (the Phase 4 benchmark harness owns
those details).

Verified rules baked into ``build_request``:

* ``thinking`` always uses ``{"type": "adaptive"}`` — the canonical 2026 form.
  The legacy ``{"type": "enabled", "budget_tokens": N}`` shape works on
  Sonnet 4.6 but is rejected on Opus 4.7 with HTTP 400.
* ``output_config.effort`` (NOT nested inside ``thinking``) carries the
  effort level. ``"low" | "medium" | "high" | "max"``. Verified on Sonnet
  4.6 for all four levels and on Opus 4.7 for ``high`` and ``max``.
* ``temperature`` is intentionally omitted. Opus 4.7 returns HTTP 400 if it
  appears in the payload (deprecated). Sonnet 4.6 still accepts it but we
  omit for consistency.
* When ``enable_thinking=False`` the payload contains neither ``thinking``
  nor ``output_config`` — useful for trivial/lookup queries that do not
  warrant the extra latency or token cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

EffortLevel = Literal["low", "medium", "high", "max"]
"""Allowed values for ``output_config.effort`` — verified spec."""

ModelName = Literal["claude-sonnet-4-6", "claude-opus-4-7"]
"""Foundry-routed Anthropic models in scope for Agent Amplifier v1.1+."""

ComplexityTier = Literal["MINIMAL", "LOW", "MEDIUM", "HIGH", "MAX"]
"""Names from :class:`agent_amplifier.effort_router.EffortLevel`."""


_TIER_TO_EFFORT: Mapping[str, EffortLevel] = {
    "MINIMAL": "low",
    "LOW": "low",
    "MEDIUM": "medium",
    "HIGH": "high",
    "MAX": "high",
}
"""Default mapping from ``effort_router`` complexity tier to thinking effort.

``MAX`` is capped at ``"high"`` here because some Foundry deployments treat
``effort="max"`` as Opus-only. ``build_request`` promotes it to ``"max"``
automatically when ``model="claude-opus-4-7"``.
"""


def tier_to_effort(tier: str, *, model: ModelName) -> EffortLevel:
    """Resolve the effort level for a given complexity tier + model.

    The default mapping caps at ``"high"`` for Sonnet 4.6. When the tier
    is ``"MAX"`` AND the model is Opus 4.7, the effort is promoted to
    ``"max"`` so the deepest available reasoning depth is used.

    Unknown tiers fall through to ``"medium"`` (defensive default).
    """
    base = _TIER_TO_EFFORT.get(tier, "medium")
    if tier == "MAX" and model == "claude-opus-4-7":
        return "max"
    return base


def build_request(
    *,
    model: ModelName,
    user_prompt: str,
    system_prompt: str | None = None,
    tier: str = "MEDIUM",
    max_tokens: int = 4096,
    enable_thinking: bool = True,
) -> dict[str, Any]:
    """Build a Foundry-ready Anthropic Messages API payload.

    Parameters
    ----------
    model
        ``"claude-sonnet-4-6"`` or ``"claude-opus-4-7"``.
    user_prompt
        The prompt that goes into ``messages[0].content`` as a user-role
        message. Callers compose the AA envelope + the user's original
        prompt themselves and pass the concatenation here.
    system_prompt
        Optional system-level instruction. Used by the benchmark judge
        path. Omit for normal hot-path calls.
    tier
        ``effort_router`` complexity name: one of
        ``"MINIMAL"``, ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"MAX"``.
        Unknown values map to ``"medium"`` effort.
    max_tokens
        Upper bound on the model's reply length.
    enable_thinking
        When ``True`` (default), payload includes
        ``thinking.type=adaptive`` and the tier-derived
        ``output_config.effort``. When ``False``, the model produces a
        normal non-thinking reply (suitable for trivial/lookup tasks).

    Returns
    -------
    dict[str, Any]
        JSON-serialisable Anthropic Messages API request body.
    """
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if system_prompt is not None:
        payload["system"] = system_prompt
    if enable_thinking:
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": tier_to_effort(tier, model=model)}
    return payload


__all__ = [
    "ComplexityTier",
    "EffortLevel",
    "ModelName",
    "build_request",
    "tier_to_effort",
]
