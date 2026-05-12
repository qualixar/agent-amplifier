# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Agent Amplifier — semantic modifier registry & injection (IP-5).

Per . Dynamically injects A/B-tested semantic modifiers into
the agent's prompt envelope based on ``(EffortLevel, PhaseIndex)``. Dynamic
counterpart to Karpathy-skills' static CLAUDE.md.


    * ``ModifierSpec`` is ``frozen=True, slots=True``.
    * Registry tuples are immutable by construction; ``MODIFIER_REGISTRY`` is
      an alias for ``EXTENDED_MODIFIERS``.
    * ``THINKING_GROUP`` is a ``frozenset`` (mutex enforcement).
    * NEW ``_neutralize_xml`` helper — *syntactic* mitigation
      against ``<system-reminder>`` tag smuggling. Defense-in-depth requires
      complementary kernel-side enforcement ( ); see
      spot-fix STAGE-5C-003 for the residual-risk note.
    * ``inject_modifiers`` accepts ``session_nonce`` kwarg.
    * ``inject_modifiers`` uses ``"\\n".join`` instead of multi-segment concat
 — single-pass C-level allocator.
    * CRIT Flaw 3 mitigation applied: hostile ``persona_role`` is neutralized
      via ``_neutralize_xml`` *inside* ``inject_modifiers`` before
      ``.format(role=...)`` — defense-in-depth even if the kernel forgets.

Anti-drift rule (per .4): module-level state MUST be
immutable. Any new constant goes through ``MappingProxyType`` (for dicts) or
remains ``tuple``/``frozenset``. Plain ``dict``/``list``/``set`` at module
scope is BANNED.

Adapter contract (per .5): adapters MUST NOT route raw tool
output / SLM-recalled patterns into prompt slots without ``_neutralize_xml``.
The kernel applies ``_neutralize_xml`` at the OUTER boundary; this module
applies it as the INNER defense layer.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
from collections.abc import Iterable
from dataclasses import dataclass

from agent_amplifier.types import EffortLevel, PhaseIndex

LOG = logging.getLogger("agent_amplifier.semantic_modifiers")


# ---------------------------------------------------------------------------
# 1. _neutralize_xml — syntactic tag-smuggling defense
# ---------------------------------------------------------------------------

# Pre-compiled regex matching opening AND closing variants of hostile tags.
# Hostile tags = those Claude has training-time priors for, plus the
# ``amp_*`` namespace that an attacker might forge.
_HOSTILE_TAG_RE: re.Pattern[str] = re.compile(
    r"</?\s*(system-reminder|system|user|assistant|tool_result|tool_use|amp_[a-z_]+)\b[^>]*>",
    re.IGNORECASE,
)


def _neutralize_xml(s: object) -> str:
    """Replace hostile XML/HTML-style tags with Unicode look-alikes.

    Substitutes ``<`` -> ``‹`` (U+2039) and ``>`` -> ``›`` (U+203A) inside
    matched hostile-tag spans. Idempotent. Pure function. Safe on already-
    neutralized strings and on non-string inputs (returns ``""``).

    This is a **syntactic** mitigation. Frontier LLMs trained on text where
    ``‹`` / ``›`` appear as visual angle-bracket alternatives may still
    semantically equate ``‹system›`` with ``<system>`` (see spot-fix
    STAGE-5C-003-lookalike-residual-risk.md). Complementary defenses are:

        1. Per-session HMAC-bound nonce envelope on emitted blocks.
        2. Kernel-side smuggling detector ( ).
        3. Adapter contract forbidding raw tool output in prompt slots.

    The parameter is typed ``object`` (not ``str``) deliberately — the
    function runs at the trust boundary between adapter input and the
    prompt envelope, and must NEVER crash on a foreign type.

    Args:
        s: untrusted text — may contain ``</system-reminder>`` etc.

    Returns:
        ``s`` with hostile tags rewritten as ``‹/system-reminder›``-style
        markup. Non-strings yield ``""``.
    """
    if not isinstance(s, str):
        return ""
    return _HOSTILE_TAG_RE.sub(
        lambda m: m.group(0).replace("<", "‹").replace(">", "›"),
        s,
    )


# ---------------------------------------------------------------------------
# 2. ModifierSpec — frozen + slotted dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModifierSpec:
    """Frozen, slot-backed spec — immutable + memory-efficient.

    Fields:
        name: short identifier — used for mutex (THINKING_GROUP) and logs.
        text: verbatim text injected into the prompt envelope. May contain
              the literal placeholder ``{role}`` for the PERSONA modifier;
              every other modifier's text is plain.
        min_effort: lowest tier at which this modifier emits.
        applicable_phases: frozenset of phases where the modifier emits.
        category: one of {"thinking","reasoning","format","audit","persona"}.
        source: provenance string for the placebo lint (B-08 / §1.2).
        placebo_check: True for every modifier except hardcoded thinking
            triggers (those have non-semantic effects parsed by Claude Code).
    """

    name: str
    text: str
    min_effort: EffortLevel
    applicable_phases: frozenset[PhaseIndex]
    category: str
    source: str
    placebo_check: bool = True


# ---------------------------------------------------------------------------
# 3. Modifier definitions — verbatim from .3
# ---------------------------------------------------------------------------

# Hardcoded thinking triggers (parsed by Claude Code CLI — non-semantic).
THINK_4K = ModifierSpec(
    name="think",
    text="think",
    min_effort=EffortLevel.LOW,
    applicable_phases=frozenset({PhaseIndex.EXECUTE, PhaseIndex.REFINE}),
    category="thinking",
    source="magic-words entry 2 [hardcoded 4000 tok]",
    placebo_check=False,
)
THINK_HARD_10K = ModifierSpec(
    name="think hard",
    text="think hard",
    min_effort=EffortLevel.MEDIUM,
    applicable_phases=frozenset(
        {PhaseIndex.EVALUATE, PhaseIndex.EXECUTE, PhaseIndex.VERIFY}
    ),
    category="thinking",
    source="magic-words entry 3 [hardcoded 1e4 tok]",
    placebo_check=False,
)
ULTRATHINK_31K = ModifierSpec(
    name="ultrathink",
    text="ultrathink",
    min_effort=EffortLevel.HIGH,
    applicable_phases=frozenset(
        {
            PhaseIndex.EXPLORE,
            PhaseIndex.EVALUATE,
            PhaseIndex.EXECUTE,
            PhaseIndex.VERIFY,
            PhaseIndex.REFINE,
        }
    ),
    category="thinking",
    source="magic-words entry 1 [hardcoded 31999 tok]",
    placebo_check=False,
)

# Validated semantic modifiers (CLSkillsHub A/B tested — >=53% efficacy).
L99 = ModifierSpec(
    name="L99",
    text="L99",
    min_effort=EffortLevel.HIGH,
    applicable_phases=frozenset(
        {PhaseIndex.EVALUATE, PhaseIndex.VERIFY, PhaseIndex.REFINE}
    ),
    category="reasoning",
    source="magic-words entry 15 [hedging -73%]",
)
OODA = ModifierSpec(
    name="OODA",
    text="OODA",
    min_effort=EffortLevel.HIGH,
    applicable_phases=frozenset({PhaseIndex.EXPLORE, PhaseIndex.EVALUATE}),
    category="reasoning",
    source="magic-words entry 16",
)
PERSONA = ModifierSpec(
    name="PERSONA",
    text="PERSONA: {role}",
    min_effort=EffortLevel.MEDIUM,
    applicable_phases=frozenset(
        {
            PhaseIndex.EXPLORE,
            PhaseIndex.EVALUATE,
            PhaseIndex.EXECUTE,
            PhaseIndex.VERIFY,
            PhaseIndex.REFINE,
        }
    ),
    category="persona",
    source="magic-words entry 18",
)
CRIT = ModifierSpec(
    name="CRIT",
    text="CRIT: identify 3 specific flaws a senior reviewer would catch.",
    min_effort=EffortLevel.HIGH,
    applicable_phases=frozenset({PhaseIndex.VERIFY, PhaseIndex.REFINE}),
    category="audit",
    source="magic-words entry 26",
)
FINISH = ModifierSpec(
    name="FINISH",
    text="FINISH the task. Do not explain what you would do — do it.",
    min_effort=EffortLevel.LOW,
    applicable_phases=frozenset({PhaseIndex.EXECUTE, PhaseIndex.REFINE}),
    category="format",
    source="magic-words entry 28",
)
AUDIT = ModifierSpec(
    name="AUDIT",
    text="AUDIT",
    min_effort=EffortLevel.MAX,
    applicable_phases=frozenset({PhaseIndex.VERIFY}),
    category="audit",
    source="magic-words entry 51",
)
WORSTCASE = ModifierSpec(
    name="WORSTCASE",
    text=(
        "WORSTCASE: enumerate catastrophic failures, data-at-risk, "
        "rollback plan."
    ),
    min_effort=EffortLevel.MAX,
    applicable_phases=frozenset({PhaseIndex.VERIFY, PhaseIndex.REFINE}),
    category="audit",
    source="magic-words entry 33",
)
SKEPTIC = ModifierSpec(
    name="/skeptic",
    text="/skeptic",
    min_effort=EffortLevel.HIGH,
    applicable_phases=frozenset({PhaseIndex.EXPLORE}),
    category="reasoning",
    source="magic-words entry 19",
)
GHOST = ModifierSpec(
    name="/ghost",
    text="/ghost",
    min_effort=EffortLevel.LOW,
    applicable_phases=frozenset({PhaseIndex.EXECUTE}),
    category="format",
    source="magic-words entry 17 [content-creation adapters only]",
)


# ---------------------------------------------------------------------------
# 4. Registries — immutable tuples + a mutex frozenset
# ---------------------------------------------------------------------------

CORE_MODIFIERS: tuple[ModifierSpec, ...] = (
    L99,
    CRIT,
    FINISH,
    PERSONA,
    OODA,
    AUDIT,
    WORSTCASE,
    THINK_4K,
    THINK_HARD_10K,
    ULTRATHINK_31K,
)
EXTENDED_MODIFIERS: tuple[ModifierSpec, ...] = (*CORE_MODIFIERS, SKEPTIC, GHOST)

#: Mutually exclusive group — selector picks AT MOST ONE thinking trigger.
THINKING_GROUP: frozenset[str] = frozenset({"think", "think hard", "ultrathink"})

#: Canonical name  / tests refer to.
MODIFIER_REGISTRY: tuple[ModifierSpec, ...] = EXTENDED_MODIFIERS

# ---------------------------------------------------------------------------
# 5. Effort ranking helper (private — selection algorithm)
# ---------------------------------------------------------------------------

_EFFORT_RANK: dict[EffortLevel, int] = {
    EffortLevel.MINIMAL: 0,
    EffortLevel.LOW: 1,
    EffortLevel.MEDIUM: 2,
    EffortLevel.HIGH: 3,
    EffortLevel.MAX: 4,
}

# Stable category ordering for emitted modifier sequence.
_CATEGORY_ORDER: tuple[str, ...] = (
    "thinking",  # always first (Claude Code parser sees it early)
    "reasoning",
    "audit",
    "persona",
    "format",
)


def _category_rank(cat: str) -> int:
    try:
        return _CATEGORY_ORDER.index(cat)
    except ValueError:
        return len(_CATEGORY_ORDER)  # unknown -> last


# ---------------------------------------------------------------------------
# 6. select_modifiers — pure selection function
# ---------------------------------------------------------------------------


def select_modifiers(
    effort: EffortLevel,
    phase: PhaseIndex,
    *,
    persona_role: str | None = None,
) -> tuple[ModifierSpec, ...]:
    """Choose modifiers applicable to ``(effort, phase)``.

    Algorithm (per .4):
        1. MINIMAL effort -> empty (fast lane).
        2. Filter EXTENDED_MODIFIERS by ``min_effort`` (effort >= spec) AND
           by ``applicable_phases`` (phase in spec).
        3. Drop PERSONA when ``persona_role`` is None (template would have
           an unfilled ``{role}`` slot).
        4. Mutex on THINKING_GROUP — pick the highest tier whose
           ``min_effort`` <= effort. (Selection of one canonical thinking
           trigger; the others are skipped.)
        5. Stable order by ``_CATEGORY_ORDER`` then by name.

    Returns:
        Tuple of ``ModifierSpec`` in stable emission order.
    """
    if effort == EffortLevel.MINIMAL:
        return ()

    effort_rank = _EFFORT_RANK[effort]

    # Step 2: filter by effort + phase.
    pool: list[ModifierSpec] = []
    for spec in EXTENDED_MODIFIERS:
        if _EFFORT_RANK[spec.min_effort] > effort_rank:
            continue
        if phase not in spec.applicable_phases:
            continue
        pool.append(spec)

    # Step 3: PERSONA needs a role.
    if persona_role is None:
        pool = [m for m in pool if m.name != "PERSONA"]

    # Step 4: thinking-trigger mutex — keep at most one.
    thinking = [m for m in pool if m.name in THINKING_GROUP]
    if thinking:
        # Pick the highest-tier trigger this effort qualifies for.
        thinking.sort(key=lambda m: _EFFORT_RANK[m.min_effort], reverse=True)
        chosen_thinking = thinking[0]
        pool = [
            m
            for m in pool
            if m.name not in THINKING_GROUP or m is chosen_thinking
        ]

    # Step 5: stable order — category index, then name (ASCII order).
    pool.sort(key=lambda m: (_category_rank(m.category), m.name))
    return tuple(pool)


# ---------------------------------------------------------------------------
# 7. generate_session_nonce — kernel helper (NOTE-)
# ---------------------------------------------------------------------------


def generate_session_nonce() -> str:
    """Return a 64-bit hex nonce suitable for the session-bound envelope.

    Format: ``[a-f0-9]{16}``. Generated via :func:`secrets.token_hex` so the
    bytes pass cryptographic-grade randomness checks. The kernel scopes one
    nonce per ``_AmplifierCore`` instance and never persists it.
    """
    return secrets.token_hex(8)


# ---------------------------------------------------------------------------
# 8. inject_modifiers — wraps in <system-reminder id="amp:NONCE">…
# ---------------------------------------------------------------------------

_NONCE_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_]{3,64}$")


def inject_modifiers(
    prompt: str,
    modifiers: Iterable[ModifierSpec],
    *,
    persona_role: str | None = None,
    session_nonce: str | None = None,
) -> str:
    """Prepend a ``<system-reminder>`` block carrying the chosen modifiers.

    Output shape (per .5):

    .. code-block:: text

        <system-reminder id="amp:NONCE">
        Apply the following amplification modifiers to your reasoning and output:
        <modifier text 1>
        <modifier text 2>
        ...
        </system-reminder id="amp:NONCE">

        <prompt>

    The close tag's ``id`` matches the open tag's. The kernel records the
    nonce, scans the next iteration's ``prev_output`` for it, and rejects
    iterations where the model echoes our envelope (smuggling protection,
    NOTE-).

    Args:
        prompt: original user-side prompt (already anchor-injected).
        modifiers: ordered iterable from :func:`select_modifiers`.
        persona_role: optional role string for the PERSONA template. The
            value is neutralized via :func:`_neutralize_xml` before
            substitution (CRIT Flaw 3 defense-in-depth).
        session_nonce: per-session id from kernel; format ``[A-Za-z0-9_]{4,64}``.
            If ``None``, a PID-derived fallback nonce is used and a WARNING
            is logged. Production callers MUST pass a real nonce.

    Returns:
        Prompt with the rendered modifier envelope prepended.
    """
    mods = tuple(modifiers)
    if not mods:
        return prompt

    rendered: list[str] = []
    for m in mods:
        text = m.text
        if m.name == "PERSONA" and persona_role is not None:
            # CRIT Flaw 3 mitigation: neutralize the role string here even
            # though the kernel SHOULD have already done so. Defense in depth.
            safe_role = _neutralize_xml(persona_role)
            text = text.format(role=safe_role)
        rendered.append(text)

    block_body = "\n".join(rendered)

    # Nonce defaulting.
    if session_nonce is None:
        LOG.warning(
            "inject_modifiers called without session_nonce — using PID "
            "fallback. Kernel callers MUST pass a per-session nonce in "
            "production."
        )
        session_nonce = f"pidfallback{os.getpid():x}"

    # Validate nonce shape — fail closed on malformed input.
    if not _NONCE_RE.fullmatch(session_nonce):
        LOG.warning(
            "inject_modifiers received malformed session_nonce; replacing "
            "with PID fallback."
        )
        session_nonce = f"pidfallback{os.getpid():x}"

    open_tag = f'<system-reminder id="amp:{session_nonce}">'
    close_tag = f'</system-reminder id="amp:{session_nonce}">'


    return "\n".join(
        [
            open_tag,
            "Apply the following amplification modifiers to your reasoning and output:",
            block_body,
            close_tag,
            "",  # blank-line separator before original prompt
            prompt,
        ]
    )


# ---------------------------------------------------------------------------
# 9. Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AUDIT",
    "CORE_MODIFIERS",
    "CRIT",
    "EXTENDED_MODIFIERS",
    "FINISH",
    "GHOST",
    "L99",
    "MODIFIER_REGISTRY",
    "OODA",
    "PERSONA",
    "SKEPTIC",
    "THINKING_GROUP",
    "THINK_4K",
    "THINK_HARD_10K",
    "ULTRATHINK_31K",
    "WORSTCASE",
    "ModifierSpec",
    "_neutralize_xml",
    "generate_session_nonce",
    "inject_modifiers",
    "select_modifiers",
]
