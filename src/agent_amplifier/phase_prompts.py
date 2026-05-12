# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Phase-aware prompt templates and parsers (IP-7).

Per . Different system-prompt fragment per iteration phase.
Mechanizes Self-Refine (arXiv 2303.17651) and Reflexion (arXiv 2303.11366)
at the runtime hook level.


    * ``PHASE_PROMPTS`` and ``_REQUIRED_SLOTS`` wrapped in
      ``MappingProxyType``.
    * Sentinel constants (``AWAITING_EVALUATION_SENTINEL``, etc.) shared by
      prompt builders AND parser functions — single source of truth
.
    * Per-phase f-string builders ``_PHASE_PROMPT_BUILDERS`` replace the V1
      ``format_map(_StrictDict)`` pattern.
    * ``get_phase_prompt`` validates required + extra slots, then applies
      ``_neutralize_xml`` to every value before dispatching to the builder
.

Anti-drift rule (per .4): module-level state MUST be
immutable. Sentinel constants are defined exactly once here and imported
by templates, parsers, and the kernel.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from types import MappingProxyType

from agent_amplifier.semantic_modifiers import _neutralize_xml
from agent_amplifier.types import PhaseIndex

# ---------------------------------------------------------------------------
# 1. Sentinel constants — shared between prompt builders and parsers.
# ---------------------------------------------------------------------------

#: EXPLORE-phase terminator. Pure ASCII (NOTE- invariant).
AWAITING_EVALUATION_SENTINEL: str = "AWAITING-EVALUATION"

#: EVALUATE-phase pick prefix. Parser strips the colon; line may be
#: backtick-wrapped.
CHOSEN_SENTINEL: str = "CHOSEN:"

#: VERIFY-phase pass marker.
STATUS_PASS_SENTINEL: str = "STATUS: PASS"

#: VERIFY-phase issue-count regex. ``re.MULTILINE`` so the pattern matches
#: when the sentinel appears mid-block.
STATUS_ISSUES_RE: re.Pattern[str] = re.compile(
    r"^STATUS: ISSUES (\d+)$", re.MULTILINE
)

#: REFINE-phase terminal completion marker.
REFINE_DONE_SENTINEL: str = "REFINE-DONE"


# ---------------------------------------------------------------------------
# 2. Per-phase f-string builders
# ---------------------------------------------------------------------------


def _build_explore(*, anchor: str) -> str:
    return (
        "PHASE: EXPLORE (iteration 0).\n"
        f'Cast a wide net. Consider AT LEAST 3 distinct approaches to the user\'s request: "{anchor}".\n'
        "For each approach, list: (a) one-sentence description, (b) primary tradeoff, (c) reversibility.\n"
        "Do NOT commit to any approach yet. Do NOT write code. Do NOT call write/edit tools.\n"
        "Output format: numbered list of options with tradeoffs. End with the literal line "
        f"`{AWAITING_EVALUATION_SENTINEL}` so the next phase can detect handoff."
    )


def _build_evaluate(*, anchor: str, prev_output: str) -> str:
    return (
        "PHASE: EVALUATE (iteration 1).\n"
        f'User goal (verbatim, do NOT reinterpret): "{anchor}".\n'
        "You produced these options in EXPLORE:\n"
        f"{prev_output}\n\n"
        "Critique each option against the goal. Apply L99 — pick exactly ONE. Justify in <=120 words.\n"
        f"End with the literal line `{CHOSEN_SENTINEL} <one-sentence summary of pick>`. The next phase parses this line."
    )


def _build_execute(*, anchor: str, chosen: str) -> str:
    return (
        "PHASE: EXECUTE (iteration 2).\n"
        f'User goal (verbatim): "{anchor}".\n'
        f"Chosen approach: {chosen}\n"
        "FINISH the task. Implement the chosen approach. Tests first if applicable.\n"
        "Project standards from CLAUDE.md and any active skill apply.\n"
        "Do NOT re-debate the approach. Do NOT explain what you would do — do it."
    )


def _build_verify(*, anchor: str, prev_output: str) -> str:
    return (
        "PHASE: VERIFY (iteration 3).\n"
        f'User goal (verbatim): "{anchor}".\n'
        "Implementation under review:\n"
        f"{prev_output}\n\n"
        "AUDIT mode. Apply CRIT: identify exactly 3 specific flaws a senior reviewer would catch.\n"
        "Check four axes: (1) correctness vs. goal, (2) edge cases & error handling, "
        "(3) security (input validation, secrets, injection), (4) performance/complexity.\n"
        "Reading is not verification — if the code can be executed, run it.\n"
        f"Output format: section per axis, then a final line `{STATUS_PASS_SENTINEL}` "
        "or `STATUS: ISSUES <count>` (matched by parser regex)."
    )


def _build_refine(*, anchor: str, issues: str) -> str:
    return (
        "PHASE: REFINE (iteration 4 — terminal).\n"
        f'User goal (verbatim): "{anchor}".\n'
        "Issues from VERIFY:\n"
        f"{issues}\n\n"
        "Fix MINIMALLY. Targeted changes only. Do NOT over-engineer. Do NOT add new features.\n"
        "Do NOT re-architect. Address only the listed issues.\n"
        "Output format: brief summary of changes, then the corrected artifact. "
        f"End with the literal line `{REFINE_DONE_SENTINEL}` so the kernel can detect terminal completion."
    )


# ---------------------------------------------------------------------------
# 3. Backing dicts (private) and frozen public views
# ---------------------------------------------------------------------------

_PHASE_PROMPT_BUILDERS_RAW: dict[PhaseIndex, Callable[..., str]] = {
    PhaseIndex.EXPLORE: _build_explore,
    PhaseIndex.EVALUATE: _build_evaluate,
    PhaseIndex.EXECUTE: _build_execute,
    PhaseIndex.VERIFY: _build_verify,
    PhaseIndex.REFINE: _build_refine,
}
_PHASE_PROMPT_BUILDERS: Mapping[PhaseIndex, Callable[..., str]] = (
    MappingProxyType(_PHASE_PROMPT_BUILDERS_RAW)
)


# Public V1-compatible PHASE_PROMPTS view — populated by calling each
# builder with literal ``{slot}`` placeholders. Tests inspect templates
# as strings (substring presence checks); they do NOT call ``.format()``.
_PHASE_PROMPTS_RAW: dict[PhaseIndex, str] = {
    PhaseIndex.EXPLORE: _build_explore(anchor="{anchor}"),
    PhaseIndex.EVALUATE: _build_evaluate(
        anchor="{anchor}", prev_output="{prev_output}"
    ),
    PhaseIndex.EXECUTE: _build_execute(
        anchor="{anchor}", chosen="{chosen}"
    ),
    PhaseIndex.VERIFY: _build_verify(
        anchor="{anchor}", prev_output="{prev_output}"
    ),
    PhaseIndex.REFINE: _build_refine(anchor="{anchor}", issues="{issues}"),
}
PHASE_PROMPTS: Mapping[PhaseIndex, str] = MappingProxyType(_PHASE_PROMPTS_RAW)


_REQUIRED_SLOTS_RAW: dict[PhaseIndex, frozenset[str]] = {
    PhaseIndex.EXPLORE: frozenset({"anchor"}),
    PhaseIndex.EVALUATE: frozenset({"anchor", "prev_output"}),
    PhaseIndex.EXECUTE: frozenset({"anchor", "chosen"}),
    PhaseIndex.VERIFY: frozenset({"anchor", "prev_output"}),
    PhaseIndex.REFINE: frozenset({"anchor", "issues"}),
}
_REQUIRED_SLOTS: Mapping[PhaseIndex, frozenset[str]] = MappingProxyType(
    _REQUIRED_SLOTS_RAW
)


# ---------------------------------------------------------------------------
# 4. Public API — get_phase_prompt + advance_phase + required_slots
# ---------------------------------------------------------------------------


def get_phase_prompt(phase: PhaseIndex, context: Mapping[str, str]) -> str:
    """Resolve the phase template with required context slots.

    Steps (per .4):
        1. Validate ``phase`` is a known builder key.
        2. Validate required keys are present (KeyError on missing).
        3. Validate no extra keys (TypeError — fail-fast on contract drift).
        4. Apply :func:`_neutralize_xml` to EVERY value (defense in depth
           against tag smuggling —  / B-08).
        5. Dispatch to the per-phase builder.

    Args:
        phase: PhaseIndex member.
        context: mapping carrying exactly the required slots for ``phase``.

    Raises:
        ValueError: ``phase`` not a known builder key.
        KeyError: a required slot is missing.
        TypeError: ``context`` carries unknown slots (contract violation).

    Returns:
        Resolved prompt string.
    """
    if phase not in _PHASE_PROMPT_BUILDERS:
        raise ValueError(f"phase_prompts: unknown phase {phase!r}")

    required = _REQUIRED_SLOTS[phase]
    provided = set(context.keys())
    missing = required - provided
    extra = provided - required

    if missing:
        raise KeyError(
            f"phase_prompts: missing required slot(s) {sorted(missing)} "
            f"for phase {phase.name}"
        )
    if extra:
        # Strict-by-default. The kernel shouldn't pass extra keys; if it
        # does we fail fast so a typo doesn't silently no-op.
        raise TypeError(
            f"phase_prompts: unexpected slot(s) {sorted(extra)} "
            f"for phase {phase.name} (allowed: {sorted(required)})"
        )


    neutralized = {k: _neutralize_xml(str(v)) for k, v in context.items()}

    builder = _PHASE_PROMPT_BUILDERS[phase]
    return builder(**neutralized)


def advance_phase(current: PhaseIndex) -> PhaseIndex:
    """Increment phase. Caps at REFINE (4) — REFINE is terminal."""
    next_value = min(int(current) + 1, int(PhaseIndex.REFINE))
    return PhaseIndex(next_value)


def required_slots(phase: PhaseIndex) -> frozenset[str]:
    """Return the set of slot names a given phase requires."""
    return _REQUIRED_SLOTS[phase]


# ---------------------------------------------------------------------------
# 5. Parser functions — share sentinel constants with builders
# ---------------------------------------------------------------------------


def detect_explore_done(text: object) -> bool:
    """True iff the text ends an EXPLORE phase.

    Robust to trailing whitespace and optional quoting. The sentinel may
    appear inside backticks or quotes — substring detection is sufficient.

    Parameter is typed ``object`` because the kernel may pass a value from
    an adapter that hasn't been runtime-checked yet.
    """
    if not isinstance(text, str):
        return False
    return AWAITING_EVALUATION_SENTINEL in text


def parse_evaluate_chosen(text: object) -> str | None:
    """Return the chosen-approach summary parsed from EVALUATE output.

    Looks for ``CHOSEN: <text>`` on a line (optionally backtick-wrapped).
    Returns the text after the colon, stripped. ``None`` if no match.
    """
    if not isinstance(text, str):
        return None
    for line in text.splitlines():
        stripped = line.strip().strip("`").strip()
        if stripped.startswith(CHOSEN_SENTINEL):
            tail = stripped[len(CHOSEN_SENTINEL) :].strip()
            return tail or None
    return None


def parse_verify_status(text: object) -> tuple[bool, int]:
    """Return ``(passed, issue_count)`` parsed from VERIFY output.

    Returns:
        ``(True, 0)``  — STATUS: PASS detected.
        ``(False, n)`` — STATUS: ISSUES <n> detected.
        ``(False, -1)`` — neither sentinel found (caller decides escalation).
    """
    if not isinstance(text, str):
        return (False, -1)
    if STATUS_PASS_SENTINEL in text:
        return (True, 0)
    m = STATUS_ISSUES_RE.search(text)
    if m:
        try:
            return (False, int(m.group(1)))
        except (TypeError, ValueError):  # pragma: no cover — defensive
            return (False, -1)
    return (False, -1)


def parse_refine_done(text: object) -> bool:
    """True iff REFINE phase emitted the terminal sentinel."""
    if not isinstance(text, str):
        return False
    return REFINE_DONE_SENTINEL in text


# ---------------------------------------------------------------------------
# 6. Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "AWAITING_EVALUATION_SENTINEL",
    "CHOSEN_SENTINEL",
    "PHASE_PROMPTS",
    "REFINE_DONE_SENTINEL",
    "STATUS_ISSUES_RE",
    "STATUS_PASS_SENTINEL",
    "advance_phase",
    "detect_explore_done",
    "get_phase_prompt",
    "parse_evaluate_chosen",
    "parse_refine_done",
    "parse_verify_status",
    "required_slots",
]
