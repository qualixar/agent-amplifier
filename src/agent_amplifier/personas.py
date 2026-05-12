# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Escalating audit personas (IP-8).

Per . Mechanizes Multi-Agent Verification (arXiv 2502.20379)
at runtime: each iteration audits more strictly than the last. Uses a
control-theory clamp pattern (``clamp(loop_t, max_t)``) to bound how far
the persona ladder can escalate within a single session.


    * ``PersonaConfig`` is ``frozen=True, slots=True``.
    * LEVEL_1 role string degenericized — "Stripe" replaced with
      "top-tier payments company".
    * NEW ``BANNED_COMPANY_NAMES`` table for the static lint test.
    * ``format_persona_prompt`` neutralizes hostile content via
      ``_neutralize_xml`` (defense-in-depth).

Anti-drift rule: any persona role/rationale change MUST pass the static
lint test in ``tests/test_personas.py`` (no branded company names).
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_amplifier.semantic_modifiers import _neutralize_xml

# ---------------------------------------------------------------------------
# 1. Severity constants
# ---------------------------------------------------------------------------

SEVERITY_HIGH: str = "high"
SEVERITY_MEDIUM: str = "medium"
SEVERITY_LOW: str = "low"
SEVERITY_ANY: str = "any"


# ---------------------------------------------------------------------------
# 2. PersonaConfig — frozen + slotted dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PersonaConfig:
    """Immutable persona definition.

    Fields:
        level: 0..3 escalation rung.
        role: short expert descriptor (no branded companies — see lint).
        strictness: 0..1 — higher = catches more.
        focus: tuple of audit axes this persona prioritizes.
        severity_threshold: lowest severity this persona will report.
        rationale: short justification — used in docs and the rendered
            persona prompt.
    """

    level: int
    role: str
    strictness: float
    focus: tuple[str, ...]
    severity_threshold: str
    rationale: str


# ---------------------------------------------------------------------------
# 3. Persona ladder — 4 levels of escalating strictness
# ---------------------------------------------------------------------------

LEVEL_0 = PersonaConfig(
    level=0,
    role="Senior software engineer, 8 years, well-rested, reviewing in normal mode",
    strictness=0.6,
    focus=("correctness", "logic"),
    severity_threshold=SEVERITY_HIGH,
    rationale=(
        "First iteration: catch big bugs cheaply. MAV section 3.1 — broad "
        "first verifier."
    ),
)


LEVEL_1 = PersonaConfig(
    level=1,
    role=(
        "Senior security engineer with 12 years at a top-tier payments "
        "company, paranoid about OWASP Top 10 + race conditions"
    ),
    strictness=0.8,
    focus=("security", "performance", "edge_cases", "input_validation"),
    severity_threshold=SEVERITY_MEDIUM,
    rationale=(
        "Second iteration: tighten the gates. Maps to MAV section 3.2 "
        "specialized verifier."
    ),
)

LEVEL_2 = PersonaConfig(
    level=2,
    role=(
        "Principal engineer + open-source maintainer who has shipped 4 "
        "frameworks; reviews as if a competing OSS project will fork this "
        "code tomorrow"
    ),
    strictness=0.95,
    focus=(
        "market_comparison",
        "IP_risk",
        "competitor_parity",
        "API_design",
        "DX",
    ),
    severity_threshold=SEVERITY_LOW,
    rationale=(
        "Third iteration: harsh re-audit. Mechanized stage of the audit "
        "cascade. MAV verifier ensemble."
    ),
)

LEVEL_3 = PersonaConfig(
    level=3,
    role=(
        "Distinguished engineer + AI safety researcher who has shipped to "
        ">100M users; reviews like the VP-of-Engineering pre-launch gate"
    ),
    strictness=1.0,
    focus=(
        "everything",
        "regression_risk",
        "ops_burden",
        "documentation_completeness",
        "rollback_plan",
    ),
    severity_threshold=SEVERITY_ANY,
    rationale=(
        "Terminal iteration: catch what 100x re-audit would catch. "
        "Maps to WORSTCASE modifier."
    ),
)

PERSONA_LADDER: tuple[PersonaConfig, ...] = (LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3)
MAX_LEVEL: int = len(PERSONA_LADDER) - 1


# ---------------------------------------------------------------------------
# 4. Banned company names —  lint table
# ---------------------------------------------------------------------------

#: Branded company names that MUST NOT appear in persona role/rationale.
#: Used by ``tests/test_personas.py`` static lint. Public so callers can
#: extend (e.g., for a domain-specific deployment that bans more brands).
BANNED_COMPANY_NAMES: tuple[str, ...] = (
    "Stripe",
    "Anthropic",
    "OpenAI",
    "Google",
    "Meta",
    "Apple",
    "Microsoft",
    "AWS",
    "Netflix",
    "Cloudflare",
    "Temporal",
    "Databricks",
    "Snowflake",
    "Datadog",
    "GitHub",
    "GitLab",
    "Atlassian",
)


# ---------------------------------------------------------------------------
# 5. Selection + rendering API
# ---------------------------------------------------------------------------


def get_persona(
    iteration: int, *, override_level: int | None = None
) -> PersonaConfig:
    """Linear escalation: ``iteration N -> level min(N, MAX_LEVEL)``.

    Per D-3 (locked decision). ``override_level`` lets the kernel/user
    pin a specific level (paranoid mode = always LEVEL_3).

    Negative iterations clamp to LEVEL_0 (defensive — kernel must not
    crash on a misconfigured loop counter).
    """
    if override_level is not None:
        clamped = max(0, min(override_level, MAX_LEVEL))
        return PERSONA_LADDER[clamped]
    if iteration < 0:
        return PERSONA_LADDER[0]
    return PERSONA_LADDER[min(iteration, MAX_LEVEL)]


def format_persona_prompt(persona: PersonaConfig) -> str:
    """Render a persona block for injection.

    Output shape (per .3):

    .. code-block:: text

        PERSONA: <role>
        Strictness: <0..1>
        Focus: a, b, c
        Anti-conformity rule: Do NOT mark a finding "no issue" just because
        the previous iteration did. Each iteration audits independently.

    The role string is neutralized via :func:`_neutralize_xml` so a hostile
    or programmatically-constructed PersonaConfig cannot inject tags into
    the rendered output (defense in depth — even if the kernel forgets).
    """
    safe_role = _neutralize_xml(persona.role)
    focus_line = ", ".join(persona.focus)
    return (
        f"PERSONA: {safe_role}\n"
        f"Strictness: {persona.strictness}\n"
        f"Focus: {focus_line}\n"
        "Severity threshold: "
        f"{persona.severity_threshold}\n"
        "Anti-conformity rule: Do NOT mark a finding 'no issue' just "
        "because the previous iteration did. Each iteration audits "
        "independently from prior verifiers."
    )


# ---------------------------------------------------------------------------
# 6. Two-axis composition API (IP-8 v2)
#
# StrictnessProfile captures the iteration-driven escalation axis.
# compose_persona() merges a user-chosen flavor description with a
# StrictnessProfile to produce the full rendered PERSONA block.
# This is the correct long-term design — the previous approach coupled
# flavor + strictness into a single PersonaConfig, making user-selected
# personas impossible to wire into the kernel.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrictnessProfile:
    """Iteration-driven strictness axis (orthogonal to flavor/domain).

    Derived from the existing LEVEL_0..3 data but without the role text,
    which now belongs to ``PersonaFlavor`` (a.k.a. ``CustomPersona``).
    """

    level: int
    strictness: float
    focus: tuple[str, ...]
    severity_threshold: str
    iteration_directive: str


# One profile per LEVEL — matches PERSONA_LADDER length.
STRICTNESS_PROFILES: tuple[StrictnessProfile, ...] = (
    StrictnessProfile(
        level=0,
        strictness=LEVEL_0.strictness,
        focus=LEVEL_0.focus,
        severity_threshold=LEVEL_0.severity_threshold,
        iteration_directive=(
            "First pass — catch major bugs cheaply. Wide net, broad scope."
        ),
    ),
    StrictnessProfile(
        level=1,
        strictness=LEVEL_1.strictness,
        focus=LEVEL_1.focus,
        severity_threshold=LEVEL_1.severity_threshold,
        iteration_directive=(
            "Second pass — tighten the gates. Escalate focus on security, "
            "performance, and edge cases."
        ),
    ),
    StrictnessProfile(
        level=2,
        strictness=LEVEL_2.strictness,
        focus=LEVEL_2.focus,
        severity_threshold=LEVEL_2.severity_threshold,
        iteration_directive=(
            "Third pass — adversarial re-audit. Report 3 specific flaws "
            "minimum. No politeness budget."
        ),
    ),
    StrictnessProfile(
        level=3,
        strictness=LEVEL_3.strictness,
        focus=LEVEL_3.focus,
        severity_threshold=LEVEL_3.severity_threshold,
        iteration_directive=(
            "Terminal pass — pre-launch gate. Enumerate regression risk, "
            "ops burden, rollback plan, and any gap the prior iterations missed."
        ),
    ),
)

_MAX_STRICTNESS_LEVEL: int = len(STRICTNESS_PROFILES) - 1


def get_strictness_profile(iteration: int) -> StrictnessProfile:
    """Return the strictness profile for ``iteration``.

    Clamps to [0, _MAX_STRICTNESS_LEVEL] — same contract as ``get_persona``.
    """
    if iteration < 0:
        return STRICTNESS_PROFILES[0]
    return STRICTNESS_PROFILES[min(iteration, _MAX_STRICTNESS_LEVEL)]


def compose_persona(
    description: str,
    review_focus: tuple[str, ...],
    profile: StrictnessProfile,
) -> str:
    """Compose a PERSONA block from a flavor description + strictness profile.

    Defense-in-depth: ``description`` is neutralized via ``_neutralize_xml``
    even though ``save_custom_persona`` already sanitizes it. A misconfigured
    caller that constructs a description in memory cannot inject tags.

    Focus axes are the union of ``review_focus`` (flavor) and ``profile.focus``
    (strictness level), deduplicated in insertion order.
    """
    safe_description = _neutralize_xml(description)
    # Merge flavor focus + profile focus, dedup preserving insertion order.
    seen: set[str] = set()
    merged_focus: list[str] = []
    for axis in (*review_focus, *profile.focus):
        if axis not in seen:
            seen.add(axis)
            merged_focus.append(axis)
    focus_line = ", ".join(merged_focus) if merged_focus else "(none)"
    return (
        f"PERSONA: {safe_description}\n"
        f"Strictness: {profile.strictness}\n"
        f"Focus: {focus_line}\n"
        f"Severity threshold: {profile.severity_threshold}\n"
        f"Iteration directive: {profile.iteration_directive}\n"
        "Anti-conformity rule: Do NOT mark a finding 'no issue' just "
        "because the previous iteration did. Each iteration audits "
        "independently from prior verifiers."
    )


# ---------------------------------------------------------------------------
# 7. Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "BANNED_COMPANY_NAMES",
    "LEVEL_0",
    "LEVEL_1",
    "LEVEL_2",
    "LEVEL_3",
    "MAX_LEVEL",
    "PERSONA_LADDER",
    "SEVERITY_ANY",
    "SEVERITY_HIGH",
    "SEVERITY_LOW",
    "SEVERITY_MEDIUM",
    "STRICTNESS_PROFILES",
    "PersonaConfig",
    "StrictnessProfile",
    "compose_persona",
    "format_persona_prompt",
    "get_persona",
    "get_strictness_profile",
]
