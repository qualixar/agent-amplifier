# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""User-education catalog for personas.

Surfaces a short **value tagline** and **when to use** hint for every built-in
persona, plus a helper that combines this with the underlying ``PersonaConfig``
to produce a dict the CLI / Dashboard / Backend can render uniformly.

Why this exists:
The raw ``personas.py`` ladder describes WHO is auditing (role text), HOW
strictly (severity threshold), and WHAT axes (focus). It does NOT explain to
the end user *why they'd want it* or *when to pick it*. This module fills
that gap without touching the kernel's ``PersonaConfig`` schema (deferred to
v1.1 per ``.backup/decisions/DECISION-2026-05-13-persona-architecture-v1.1.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_amplifier.custom_personas import CustomPersona
from agent_amplifier.custom_personas import (
    load_custom_personas as _load_custom_personas,
)
from agent_amplifier.personas import MAX_LEVEL, PERSONA_LADDER

# Re-export so callers can clamp without importing personas.py directly.
MAX_LEVEL_FOR_DOCS: int = MAX_LEVEL


@dataclass(frozen=True, slots=True)
class PersonaDoc:
    """User-facing documentation for a built-in persona level.

    Fields:
        slug: URL/CLI-safe identifier (matches naming rules used for custom).
        label: human-readable display label.
        value_tagline: one-sentence "what this catches" pitch.
        when_to_use: one-sentence "pick this when..." guidance.
    """

    slug: str
    label: str
    value_tagline: str
    when_to_use: str


# ---------------------------------------------------------------------------
# Catalog — one entry per LEVEL_0..LEVEL_3
# ---------------------------------------------------------------------------

BUILTIN_PERSONA_DOCS: tuple[PersonaDoc, ...] = (
    PersonaDoc(
        slug="senior-engineer",
        label="Senior Engineer (normal mode)",
        value_tagline=(
            "Catches major correctness bugs and logic errors on the first "
            "pass. Cheap and broad."
        ),
        when_to_use=(
            "Default for routine code review — refactors, new features, "
            "and anything that does not touch auth, payments, or migrations."
        ),
    ),
    PersonaDoc(
        slug="security-paranoid-engineer",
        label="Security-Paranoid Engineer",
        value_tagline=(
            "Hunts OWASP Top 10, race conditions, and input-validation gaps "
            "as if every input is hostile."
        ),
        when_to_use=(
            "Pick this when reviewing auth flows, payment paths, user-input "
            "handlers, or anything that crosses a trust boundary."
        ),
    ),
    PersonaDoc(
        slug="principal-oss-maintainer",
        label="Principal Engineer + OSS Maintainer",
        value_tagline=(
            "Reviews API design, backward compatibility, IP risk, and "
            "competitor parity as if a fork is being prepared tomorrow."
        ),
        when_to_use=(
            "Use before declaring a public API frozen, before a v1.0 cut, "
            "or when sweat-testing developer-experience choices."
        ),
    ),
    PersonaDoc(
        slug="distinguished-ai-safety-reviewer",
        label="Distinguished Engineer + AI-Safety Reviewer",
        value_tagline=(
            "Pre-launch gate — surfaces regression risk, ops burden, "
            "rollback plans, and documentation gaps the rest of the team "
            "missed."
        ),
        when_to_use=(
            "Run this right before a public push, a migration, or any "
            "deploy that would be expensive to roll back."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def get_builtin_doc(level: int) -> PersonaDoc:
    """Return the doc for ``level``. Clamps to the valid range."""
    if level < 0:
        level = 0
    if level > MAX_LEVEL:
        level = MAX_LEVEL
    return BUILTIN_PERSONA_DOCS[level]


def describe_builtin_persona(level: int) -> dict[str, Any]:
    """Return a merged dict (ladder + value tagline) suitable for API/UI."""
    if level < 0:
        clamped = 0
    elif level > MAX_LEVEL:
        clamped = MAX_LEVEL
    else:
        clamped = level
    config = PERSONA_LADDER[clamped]
    doc = BUILTIN_PERSONA_DOCS[clamped]
    return {
        "slug": doc.slug,
        "label": doc.label,
        "value_tagline": doc.value_tagline,
        "when_to_use": doc.when_to_use,
        "level": config.level,
        "role": config.role,
        "strictness": config.strictness,
        "focus": list(config.focus),
        "severity_threshold": config.severity_threshold,
        "custom": False,
    }


def describe_custom_persona(persona: CustomPersona) -> dict[str, Any]:
    """Return a merged dict for a user-defined persona."""
    return {
        "slug": persona.name,
        "label": persona.label,
        "value_tagline": persona.description,
        "when_to_use": (
            "User-defined persona — pick this when your task matches the "
            "description above."
        ),
        "level": None,
        "role": persona.description,
        "strictness": None,
        "focus": list(persona.review_focus),
        "severity_threshold": None,
        "custom": True,
    }


def list_all_personas() -> list[dict[str, Any]]:
    """Return all personas (built-in first, then custom), ready to render."""
    builtins = [describe_builtin_persona(i) for i in range(len(BUILTIN_PERSONA_DOCS))]
    customs = [describe_custom_persona(p) for p in _load_custom_personas()]
    return builtins + customs


# ---------------------------------------------------------------------------
# Built-in flavor catalog (maps slugs → CustomPersona-shaped objects)
#
# Each built-in flavor derives its description from the matching PERSONA_LADDER
# role text so the kernel produces identical output to v1.0 when the user has
# not configured a custom persona.
# ---------------------------------------------------------------------------

#: Four built-in flavors as ``CustomPersona`` instances — same shape as
#: user-defined personas so ``resolve_flavor`` can return either without
#: a type-switch.
BUILTIN_FLAVORS: tuple[CustomPersona, ...] = (
    CustomPersona(
        name="senior-engineer",
        label="Senior Engineer (normal mode)",
        description=PERSONA_LADDER[0].role,
        review_focus=PERSONA_LADDER[0].focus,
    ),
    CustomPersona(
        name="security-paranoid-engineer",
        label="Security-Paranoid Engineer",
        description=PERSONA_LADDER[1].role,
        review_focus=PERSONA_LADDER[1].focus,
    ),
    CustomPersona(
        name="principal-oss-maintainer",
        label="Principal Engineer + OSS Maintainer",
        description=PERSONA_LADDER[2].role,
        review_focus=PERSONA_LADDER[2].focus,
    ),
    CustomPersona(
        name="distinguished-ai-safety-reviewer",
        label="Distinguished Engineer + AI-Safety Reviewer",
        description=PERSONA_LADDER[3].role,
        review_focus=PERSONA_LADDER[3].focus,
    ),
)

_BUILTIN_FLAVOR_MAP: dict[str, CustomPersona] = {f.name: f for f in BUILTIN_FLAVORS}
_DEFAULT_FLAVOR: CustomPersona = BUILTIN_FLAVORS[0]


def resolve_flavor(slug: str) -> CustomPersona:
    """Return the ``CustomPersona`` for ``slug``.

    Resolution order:
    1. Built-in catalog (``BUILTIN_FLAVORS``).
    2. User-defined custom personas from ``~/.config/agent-amplifier/personas.toml``.
    3. Default fallback (``senior-engineer``) if not found.

    Never raises — unknown or empty slug silently falls back to the default.
    This keeps the kernel fail-open on misconfigured TOML values.
    """
    if not slug:
        return _DEFAULT_FLAVOR
    builtin = _BUILTIN_FLAVOR_MAP.get(slug)
    if builtin is not None:
        return builtin
    for custom in _load_custom_personas():
        if custom.name == slug:
            return custom
    return _DEFAULT_FLAVOR


__all__ = [
    "BUILTIN_FLAVORS",
    "BUILTIN_PERSONA_DOCS",
    "MAX_LEVEL_FOR_DOCS",
    "PersonaDoc",
    "describe_builtin_persona",
    "describe_custom_persona",
    "get_builtin_doc",
    "list_all_personas",
    "resolve_flavor",
]
