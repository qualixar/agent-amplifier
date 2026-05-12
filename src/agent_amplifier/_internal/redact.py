# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Agent Amplifier — secret/PII redactor (.5).

Single-source masking utility for logs, errors, and SLM payloads.
Idempotent. Stdlib-only.

Patterns covered (extend cautiously — false positives mask real text):
    * OpenAI API keys: ``sk-...``
    * Anthropic API keys: ``sk-ant-...``
    * GitHub PATs: ``ghp_...``
    * Bearer / token / api_key headers (generic)
    * Email addresses

Usage::

    from agent_amplifier._internal.redact import redact
    safe = redact(user_query)

Returned format for matches: ``[REDACTED:TYPE]`` (no last-4 — last-4 was
considered but rejected: leaks too much for short tokens).
"""

from __future__ import annotations

import re

# Order matters: anthropic must be checked BEFORE openai because the
# anthropic prefix ``sk-ant-`` would otherwise be partially consumed by the
# openai regex's ``sk-`` prefix. We anchor anthropic with ``sk-ant-`` to make
# them disjoint, but still order matches for safety.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
        "[REDACTED:ANTHROPIC_KEY]",
    ),
    (
        re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
        "[REDACTED:OPENAI_KEY]",
    ),
    (
        re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
        "[REDACTED:GITHUB_PAT]",
    ),
    (
        # STAGE-5C-001: broadened from .2 — original regex
        # required `[:=]`; real-world `Authorization: Bearer <token>` uses
        # whitespace. Allow EITHER `[:=]` OR `\s+`.
        re.compile(
            r"\b(?:bearer|token|api[_-]?key)\s*(?:[:=]\s*|\s+)"
            r"[A-Za-z0-9_\-\.]{12,}",
            re.IGNORECASE,
        ),
        "[REDACTED:GENERIC_TOKEN]",
    ),
    (
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED:EMAIL]",
    ),
)


def redact(text: str) -> str:
    """Return ``text`` with known secret/PII patterns masked.

    Idempotent: ``redact(redact(x)) == redact(x)`` for all ``str`` x.
    Pure: no I/O, no shared mutable state.
    Total: never raises for ``str`` input. Non-``str`` input raises ``TypeError``.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"redact expects str, got {type(text).__name__}"
        )
    for rx, repl in _PATTERNS:
        text = rx.sub(repl, text)
    return text


__all__ = ["redact"]
