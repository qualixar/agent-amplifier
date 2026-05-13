# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier._internal.foundry_payload``.

Covers every branch of the payload builder against the verified Foundry
Anthropic Messages API contract (live-tested 2026-05-13). The contract:

* ``thinking`` uses ``{"type": "adaptive"}`` — legacy ``budget_tokens`` is
  rejected by Opus 4.7.
* ``output_config.effort`` lives outside ``thinking`` (NOT nested inside).
* ``temperature`` MUST be absent — Opus 4.7 returns HTTP 400 if present.
* ``enable_thinking=False`` strips both ``thinking`` and ``output_config``.
"""

from __future__ import annotations

import pytest

from agent_amplifier._internal.foundry_payload import (
    build_request,
    tier_to_effort,
)

# ---------------------------------------------------------------------------
# tier_to_effort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "expected_sonnet"),
    [
        ("MINIMAL", "low"),
        ("LOW", "low"),
        ("MEDIUM", "medium"),
        ("HIGH", "high"),
        ("MAX", "high"),  # Sonnet caps at high
    ],
)
def test_tier_to_effort_sonnet_mapping(tier: str, expected_sonnet: str) -> None:
    assert tier_to_effort(tier, model="claude-sonnet-4-6") == expected_sonnet


def test_tier_to_effort_max_upgrades_on_opus() -> None:
    assert tier_to_effort("MAX", model="claude-opus-4-7") == "max"


@pytest.mark.parametrize("tier", ["MINIMAL", "LOW", "MEDIUM", "HIGH"])
def test_tier_to_effort_opus_non_max_matches_sonnet(tier: str) -> None:
    sonnet = tier_to_effort(tier, model="claude-sonnet-4-6")
    opus = tier_to_effort(tier, model="claude-opus-4-7")
    assert sonnet == opus


def test_tier_to_effort_unknown_tier_defaults_medium() -> None:
    assert tier_to_effort("UNKNOWN_TIER", model="claude-sonnet-4-6") == "medium"


# ---------------------------------------------------------------------------
# build_request — shape invariants
# ---------------------------------------------------------------------------


def test_build_request_default_shape_for_sonnet_medium() -> None:
    p = build_request(model="claude-sonnet-4-6", user_prompt="hi")
    assert p["model"] == "claude-sonnet-4-6"
    assert p["max_tokens"] == 4096
    assert p["messages"] == [{"role": "user", "content": "hi"}]
    assert p["thinking"] == {"type": "adaptive"}
    assert p["output_config"] == {"effort": "medium"}
    assert "system" not in p
    assert "temperature" not in p


def test_build_request_includes_system_when_provided() -> None:
    p = build_request(
        model="claude-sonnet-4-6",
        user_prompt="hi",
        system_prompt="You are a strict judge.",
    )
    assert p["system"] == "You are a strict judge."


def test_build_request_no_system_when_none() -> None:
    p = build_request(model="claude-sonnet-4-6", user_prompt="hi", system_prompt=None)
    assert "system" not in p


def test_build_request_never_emits_temperature() -> None:
    # Opus 4.7 returns HTTP 400 if `temperature` is present at all.
    for model in ("claude-sonnet-4-6", "claude-opus-4-7"):
        p = build_request(model=model, user_prompt="hi")  # type: ignore[arg-type]
        assert "temperature" not in p


def test_build_request_never_emits_budget_tokens() -> None:
    # Legacy `thinking.budget_tokens` is rejected by Opus 4.7.
    for model in ("claude-sonnet-4-6", "claude-opus-4-7"):
        p = build_request(model=model, user_prompt="hi")  # type: ignore[arg-type]
        thinking = p.get("thinking", {})
        assert "budget_tokens" not in thinking


def test_build_request_respects_max_tokens_override() -> None:
    p = build_request(model="claude-sonnet-4-6", user_prompt="hi", max_tokens=8192)
    assert p["max_tokens"] == 8192


# ---------------------------------------------------------------------------
# build_request — thinking / effort routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "expected_effort"),
    [
        ("MINIMAL", "low"),
        ("LOW", "low"),
        ("MEDIUM", "medium"),
        ("HIGH", "high"),
        ("MAX", "high"),  # Sonnet caps at high
    ],
)
def test_build_request_effort_levels_on_sonnet(tier: str, expected_effort: str) -> None:
    p = build_request(model="claude-sonnet-4-6", user_prompt="x", tier=tier)
    assert p["output_config"]["effort"] == expected_effort


def test_build_request_max_promotes_to_max_on_opus() -> None:
    p = build_request(model="claude-opus-4-7", user_prompt="x", tier="MAX")
    assert p["output_config"]["effort"] == "max"


def test_build_request_high_stays_high_on_opus() -> None:
    p = build_request(model="claude-opus-4-7", user_prompt="x", tier="HIGH")
    assert p["output_config"]["effort"] == "high"


def test_build_request_disable_thinking_strips_both_fields() -> None:
    p = build_request(model="claude-sonnet-4-6", user_prompt="x", enable_thinking=False)
    assert "thinking" not in p
    assert "output_config" not in p


def test_build_request_thinking_type_is_always_adaptive_when_enabled() -> None:
    for tier in ("MINIMAL", "LOW", "MEDIUM", "HIGH", "MAX"):
        p = build_request(model="claude-opus-4-7", user_prompt="x", tier=tier)
        assert p["thinking"] == {"type": "adaptive"}


def test_build_request_unknown_tier_falls_through_to_medium_effort() -> None:
    p = build_request(model="claude-sonnet-4-6", user_prompt="x", tier="UNKNOWN")
    assert p["output_config"]["effort"] == "medium"


# ---------------------------------------------------------------------------
# build_request — payload is JSON-serialisable end-to-end
# ---------------------------------------------------------------------------


def test_build_request_is_json_serialisable() -> None:
    import json

    p = build_request(
        model="claude-opus-4-7",
        user_prompt="hello world",
        system_prompt="be terse",
        tier="MAX",
        max_tokens=2048,
    )
    encoded = json.dumps(p)
    decoded = json.loads(encoded)
    assert decoded == p
