# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Single-turn envelope builder for the Claude Code adapter.

Claude Code fires the ``UserPromptSubmit`` injection point exactly once per
user turn. The kernel-path multi-iteration loop (EXPLORE -> EVALUATE ->
EXECUTE -> VERIFY -> REFINE across separate turns) cannot run here because
there is no "next turn" to escalate to from a hook. This module produces an
envelope that compresses the same five-stage loop into a SINGLE response,
using:

* XML phase staging tags (``<plan>``, ``<execute>``, ``<reflection>``,
  ``<refine>``, ``<final_answer>``) so the model produces all stages in one
  reply. Standard tag names — parser-portable.
* Stage-wise persona escalation (LEVEL_0 generalist for ``<plan>``,
  LEVEL_2 maintainer for ``<execute>``, LEVEL_3 distinguished + AI safety
  reviewer for ``<reflection>``). Same 4-level ladder as the multi-iteration
  path, applied within one response.
* Adaptive thinking + tier-routed effort, configured on the Anthropic
  Messages API request body itself (see
  :mod:`agent_amplifier._internal.foundry_payload`).
* Subagent dispatch directive for ``MAX`` complexity tier — the envelope
  asks Claude to invoke the Task tool with ``subagent_type="general-purpose"``
  and pass the inline envelope verbatim as the subagent's prompt. If the
  Task tool is unavailable the envelope says fall back to inline execution.

Two public builders:

* :func:`build_inline_envelope` — used for ``LOW``, ``MEDIUM``, ``HIGH`` tiers
  (and as the fallback inside the subagent envelope itself).
* :func:`build_subagent_envelope` — used for the ``MAX`` tier.

Both return the full ``<system-reminder>...</system-reminder>`` block. The
caller prepends this to the user's prompt with two newlines of separation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from string import Template

__all__ = [
    "build_inline_envelope",
    "build_subagent_envelope",
    "modifier_line",
    "stable_envelope_id",
]


# ---------------------------------------------------------------------------
# Modifier line — leverages Claude Code's hardcoded thinking-trigger phrases.
# ---------------------------------------------------------------------------


_MODIFIER_BY_TIER: Mapping[str, str] = {
    "MINIMAL": "",
    "LOW": "",
    "MEDIUM": "think hard about this task before producing the deliverable.",
    "HIGH": "ultrathink — this is a hard task; use deep reasoning.",
    "MAX": "ultrathink — this is critical complexity; dispatch via subagent.",
}


def modifier_line(tier: str) -> str:
    """Return the one-line thinking modifier for the given complexity tier."""
    return _MODIFIER_BY_TIER.get(tier, "")


# ---------------------------------------------------------------------------
# Envelope id — stable hash of the user query.
# ---------------------------------------------------------------------------


def stable_envelope_id(query: str) -> str:
    """Return a 16-char hex id derived from the query.

    Same query produces the same id every call — keeps the envelope
    idempotent and makes log-grep across turns easy.
    """
    return hashlib.blake2b(query.encode("utf-8"), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Templates — string.Template so XML literal braces survive.
# ---------------------------------------------------------------------------


_INLINE_TEMPLATE = Template(
    """<system-reminder id="amp:$eid">
$modifier

You will handle this engineering task in ONE response. There is no next turn
for this task; complete the work in this response.

Use tools (Read, Grep, Edit, Write, Bash, sequential-thinking) freely; each
tool call is one iteration step. Emit your response using these stages:

<plan>
  Brief: at most 200 words. Identify at least two candidate approaches and
  pick one. State the tradeoff for the picked approach.
  Voice: $persona_plan
</plan>

<execute>
  Produce the deliverable the user asked for. Use tools as needed.
  Voice: $persona_execute
</execute>

<reflection>
  Adversarial self-review of <execute>. Identify at least one specific flaw
  OR explicitly state "No flaws found." Find the things a senior code
  reviewer would catch.
  Voice: $persona_reflection
</reflection>

<refine>
  Only if <reflection> found flaws. Fix them. Otherwise emit
  <refine>SKIP</refine>.
</refine>

<final_answer>
  Clean, deliverable-only restatement of the answer. This is what the user
  reads. Make it complete and direct — no meta-commentary about phases or
  process.
</final_answer>

Hard rules:
  - The deliverable MUST appear in <final_answer>. Never leave it implicit.
  - Finish the work in this response; do not emit a deferred-handoff marker.
  - If the user's ask is trivial (under 30 words to answer fully), emit only
    <final_answer> and skip the other tags.
</system-reminder>"""
)


_SUBAGENT_TEMPLATE = Template(
    """<system-reminder id="amp:$eid">
$modifier

This task has been classified as MAX complexity. To keep context clean and
run a thorough audit pass, dispatch a subagent.

DISPATCH:
  Use the Task tool with:
    subagent_type: "general-purpose"
    description: an eight-word summary of the user's task
    prompt: the user's original prompt PLUS the inline phase-staged envelope
            below, passed verbatim.

INLINE ENVELOPE FOR THE SUBAGENT (pass verbatim as part of its prompt):
----- begin inline envelope -----
$inline_envelope
----- end inline envelope -----

After the subagent returns, extract the contents of its <final_answer> block
and present that to the user as your reply. Do NOT re-summarise; do NOT add
meta-commentary; the subagent's <final_answer> IS the deliverable.

Fallback: if the Task tool is unavailable in this session, run the inline
envelope yourself.
</system-reminder>"""
)


# ---------------------------------------------------------------------------
# Public builders.
# ---------------------------------------------------------------------------


def build_inline_envelope(
    *,
    query: str,
    tier: str,
    personas: Mapping[str, str],
) -> str:
    """Build the inline XML phase-staged envelope.

    Parameters
    ----------
    query
        The user's prompt; used only for the stable envelope id.
    tier
        ``effort_router`` complexity name (``MINIMAL``, ``LOW``,
        ``MEDIUM``, ``HIGH``, ``MAX``). Determines the modifier line.
    personas
        Mapping with keys ``"plan"``, ``"execute"``, ``"reflection"``
        whose values are the role-description strings to inject as
        ``Voice:`` lines under each stage. See
        :func:`agent_amplifier.personas.compose_single_turn_personas`.

    Returns
    -------
    str
        The full ``<system-reminder>...</system-reminder>`` block.
    """
    return _INLINE_TEMPLATE.substitute(
        eid=stable_envelope_id(query),
        modifier=modifier_line(tier),
        persona_plan=personas["plan"],
        persona_execute=personas["execute"],
        persona_reflection=personas["reflection"],
    )


def build_subagent_envelope(
    *,
    query: str,
    tier: str,
    personas: Mapping[str, str],
) -> str:
    """Build the subagent-dispatch envelope (MAX tier).

    The subagent receives the inline envelope (from
    :func:`build_inline_envelope`) verbatim as part of its prompt, and runs
    the full phase loop in an isolated context. The parent turn extracts
    the subagent's ``<final_answer>`` and surfaces it.
    """
    inline = build_inline_envelope(query=query, tier=tier, personas=personas)
    return _SUBAGENT_TEMPLATE.substitute(
        eid=stable_envelope_id(query),
        modifier=modifier_line(tier),
        inline_envelope=inline,
    )
