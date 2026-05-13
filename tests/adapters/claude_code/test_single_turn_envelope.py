# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for the v1.1.1 Claude Code single-turn envelope builder.

The envelope must:

* Contain all five XML phase-staging tags (``<plan>``, ``<execute>``,
  ``<reflection>``, ``<refine>``, ``<final_answer>``) for inline path.
* Never contain ``PHASE: EXPLORE`` or ``AWAITING-EVALUATION`` — those are
  the deferred-handoff markers that caused the v1.1.0 single-turn regression.
* Use ``string.Template`` substitution (XML braces must NOT break formatting).
* Inject persona role strings stage-wise (LEVEL_0 / LEVEL_2 / LEVEL_3).
* Produce identical envelope ids for identical queries (idempotency).
* For the subagent path, include the Task tool dispatch instructions AND
  embed the inline envelope verbatim AND surface an explicit fallback clause.
"""

from __future__ import annotations

import pytest

from agent_amplifier.adapters.claude_code.single_turn_envelope import (
    build_inline_envelope,
    build_subagent_envelope,
    modifier_line,
    stable_envelope_id,
)

# Personas fixture matches what :func:`compose_single_turn_personas` returns.
PERSONAS = {
    "plan": "Senior software engineer (8 years), well-rested.",
    "execute": "Principal engineer + open-source maintainer.",
    "reflection": "Distinguished engineer + AI safety reviewer.",
}


# ---------------------------------------------------------------------------
# stable_envelope_id
# ---------------------------------------------------------------------------


def test_stable_envelope_id_idempotent() -> None:
    assert stable_envelope_id("hello world") == stable_envelope_id("hello world")


def test_stable_envelope_id_differs_per_query() -> None:
    assert stable_envelope_id("refactor auth") != stable_envelope_id("refactor cache")


def test_stable_envelope_id_is_16_hex_chars() -> None:
    eid = stable_envelope_id("anything")
    assert len(eid) == 16
    int(eid, 16)  # parses as hex


# ---------------------------------------------------------------------------
# modifier_line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "expected_substring"),
    [
        ("MINIMAL", ""),
        ("LOW", ""),
        ("MEDIUM", "think hard"),
        ("HIGH", "ultrathink"),
        ("MAX", "ultrathink"),
    ],
)
def test_modifier_line_per_tier(tier: str, expected_substring: str) -> None:
    line = modifier_line(tier)
    if expected_substring:
        assert expected_substring in line
    else:
        assert line == ""


def test_modifier_line_max_mentions_subagent() -> None:
    assert "subagent" in modifier_line("MAX")


def test_modifier_line_unknown_tier_returns_empty() -> None:
    assert modifier_line("UNKNOWN_TIER") == ""


# ---------------------------------------------------------------------------
# build_inline_envelope — structure invariants
# ---------------------------------------------------------------------------


def test_inline_envelope_contains_all_five_stage_tags() -> None:
    env = build_inline_envelope(query="x", tier="MEDIUM", personas=PERSONAS)
    for tag in ("<plan>", "<execute>", "<reflection>", "<refine>", "<final_answer>"):
        assert tag in env, f"missing tag: {tag}"
        # Closing tag too
        assert tag.replace("<", "</") in env, f"missing closing tag: {tag}"


def test_inline_envelope_no_awaiting_evaluation() -> None:
    env = build_inline_envelope(query="x", tier="HIGH", personas=PERSONAS)
    assert "AWAITING-EVALUATION" not in env
    assert "AWAITING EVALUATION" not in env


def test_inline_envelope_no_phase_explore_keyword() -> None:
    env = build_inline_envelope(query="x", tier="HIGH", personas=PERSONAS)
    assert "PHASE: EXPLORE" not in env
    assert "PHASE EXPLORE" not in env
    # Note: the word "EXPLORE" alone is fine — only the legacy header marker is banned.


def test_inline_envelope_has_system_reminder_wrapper() -> None:
    env = build_inline_envelope(query="x", tier="MEDIUM", personas=PERSONAS)
    assert env.startswith('<system-reminder id="amp:')
    assert env.endswith("</system-reminder>")


def test_inline_envelope_uses_stable_envelope_id() -> None:
    env1 = build_inline_envelope(query="refactor auth", tier="MEDIUM", personas=PERSONAS)
    env2 = build_inline_envelope(query="refactor auth", tier="MEDIUM", personas=PERSONAS)
    assert env1 == env2  # full idempotency for same inputs


def test_inline_envelope_persona_strings_injected_correctly() -> None:
    env = build_inline_envelope(query="x", tier="MEDIUM", personas=PERSONAS)
    assert PERSONAS["plan"] in env
    assert PERSONAS["execute"] in env
    assert PERSONAS["reflection"] in env


def test_inline_envelope_modifier_line_present_per_tier() -> None:
    for tier in ("MEDIUM", "HIGH", "MAX"):
        env = build_inline_envelope(query="x", tier=tier, personas=PERSONAS)
        line = modifier_line(tier)
        assert line and line in env


def test_inline_envelope_low_tier_omits_modifier() -> None:
    env = build_inline_envelope(query="x", tier="LOW", personas=PERSONAS)
    assert "think hard" not in env
    assert "ultrathink" not in env


def test_inline_envelope_has_no_format_string_leftovers() -> None:
    # ``string.Template`` substitution should never leave ``$word`` placeholders.
    env = build_inline_envelope(query="x", tier="MEDIUM", personas=PERSONAS)
    assert "$eid" not in env
    assert "$modifier" not in env
    assert "$persona_plan" not in env
    assert "$persona_execute" not in env
    assert "$persona_reflection" not in env


def test_inline_envelope_no_temperature_or_budget_token_references() -> None:
    # Envelope text should not leak deprecated thinking-config keywords.
    env = build_inline_envelope(query="x", tier="HIGH", personas=PERSONAS)
    assert "temperature" not in env.lower()
    assert "budget_tokens" not in env


def test_inline_envelope_hard_rules_section_present() -> None:
    env = build_inline_envelope(query="x", tier="HIGH", personas=PERSONAS)
    assert "Hard rules:" in env
    assert "<final_answer>" in env  # already asserted above but explicit here


# ---------------------------------------------------------------------------
# build_subagent_envelope — structure invariants
# ---------------------------------------------------------------------------


def test_subagent_envelope_contains_task_tool_dispatch() -> None:
    env = build_subagent_envelope(query="x", tier="MAX", personas=PERSONAS)
    assert "Task tool" in env
    assert "general-purpose" in env
    assert 'subagent_type:' in env or 'subagent_type"' in env


def test_subagent_envelope_embeds_inline_envelope_verbatim() -> None:
    env = build_subagent_envelope(query="x", tier="MAX", personas=PERSONAS)
    inline = build_inline_envelope(query="x", tier="MAX", personas=PERSONAS)
    assert inline in env


def test_subagent_envelope_contains_fallback_clause() -> None:
    env = build_subagent_envelope(query="x", tier="MAX", personas=PERSONAS)
    # The envelope must instruct fallback when Task tool is unavailable.
    assert "Fallback:" in env or "fallback" in env
    assert "Task tool is unavailable" in env or "Task tool unavailable" in env


def test_subagent_envelope_instructs_final_answer_extraction() -> None:
    env = build_subagent_envelope(query="x", tier="MAX", personas=PERSONAS)
    assert "<final_answer>" in env
    assert "extract" in env.lower()


def test_subagent_envelope_no_awaiting_evaluation_either() -> None:
    env = build_subagent_envelope(query="x", tier="MAX", personas=PERSONAS)
    assert "AWAITING-EVALUATION" not in env


def test_subagent_envelope_idempotent_per_query() -> None:
    env1 = build_subagent_envelope(query="refactor", tier="MAX", personas=PERSONAS)
    env2 = build_subagent_envelope(query="refactor", tier="MAX", personas=PERSONAS)
    assert env1 == env2


def test_subagent_envelope_has_no_format_string_leftovers() -> None:
    env = build_subagent_envelope(query="x", tier="MAX", personas=PERSONAS)
    assert "$inline_envelope" not in env
    assert "$eid" not in env
    assert "$modifier" not in env


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["MINIMAL", "LOW", "MEDIUM", "HIGH", "MAX"])
def test_inline_envelope_builds_for_all_tiers(tier: str) -> None:
    # Should not raise / should produce a non-empty system-reminder block.
    env = build_inline_envelope(query="x", tier=tier, personas=PERSONAS)
    assert env.startswith('<system-reminder')
    assert env.endswith("</system-reminder>")


def test_subagent_envelope_intended_for_max_tier_but_works_for_any() -> None:
    # The builder itself does not enforce tier=MAX; the dispatcher in
    # ``kernel.py`` is responsible for picking the right builder per tier.
    env = build_subagent_envelope(query="x", tier="HIGH", personas=PERSONAS)
    assert "Task tool" in env
