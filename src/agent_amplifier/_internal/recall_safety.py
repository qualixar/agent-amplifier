# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Universal recall-safety helpers (V2.1, decision H-1, STAGE-5C-005).

These helpers apply prompt-injection defense to ANY text that crosses the
trust boundary into the LLM context window — regardless of which adapter or
callback produced the recall. Every recalled text passes through
``apply_recall_safety()`` before the kernel exposes it to a phase prompt.

Layers (from STAGE-5C-003 layers 3-4, lifted out of slm_bridge.py V2.0):
- Layer 3 (cap): ``MAX_RECALLED_TEXT_BYTES`` truncation before exposure.
- Layer 4 (observability): ``detect_smuggling_signals`` returns names of
  suspicious sequences for LOG.warning. NEVER raises.

Plus universal text neutralization (look-alike chars, fake system-reminder
tags), formerly in slm_bridge V2.0 §1.10.

This module has ZERO dependencies beyond stdlib. It does NOT know what an
adapter is, what SLM is, or what the kernel is. It is pure text-in / text-out.
"""
from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MAX_RECALLED_TEXT_BYTES: Final[int] = 8192
"""Layer-3 hard cap. Apply BEFORE neutralize so neutralize works on bounded input."""


# ---------------------------------------------------------------------------
# Neutralization (look-alike + fake system-reminder)
# ---------------------------------------------------------------------------

# Lookalike replacement table — extends V2.0 slm_bridge §1.10 set.
# U+2039/U+203A are the high-impact ones (visual confusion with < / >).
# B2 adds fullwidth (U+FF1C/E) and mathematical angle (U+27E8/9)
# variants because SEC-02 demonstrated they bypass the previous map.
_LOOKALIKE_MAP: Final[dict[str, str]] = {
    "‹": "<",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    "›": ">",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    "˂": "<",  # MODIFIER LETTER LEFT ARROWHEAD
    "˃": ">",  # MODIFIER LETTER RIGHT ARROWHEAD
    "〈": "<",  # LEFT-POINTING ANGLE BRACKET (deprecated CJK)
    "〉": ">",  # RIGHT-POINTING ANGLE BRACKET (deprecated CJK)
    "＜": "<",  # FULLWIDTH LESS-THAN SIGN (U+FF1C) — B2
    "＞": ">",  # FULLWIDTH GREATER-THAN SIGN (U+FF1E) — B2
    "⟨": "<",  # MATHEMATICAL LEFT ANGLE BRACKET (U+27E8) — B2
    "⟩": ">",  # MATHEMATICAL RIGHT ANGLE BRACKET (U+27E9) — B2
}

# Zero-width characters used by smuggling vectors to insert hidden code points
# inside otherwise-benign ASCII tags. We strip them BEFORE the lookalike pass
# so a payload like ``<​system-reminder>`` cannot evade the regex.
# B2: SEC-02 enumerated these as the most common vectors.
_ZERO_WIDTH_CHARS: Final[tuple[str, ...]] = (
    "​",  # ZERO WIDTH SPACE
    "‌",  # ZERO WIDTH NON-JOINER
    "‍",  # ZERO WIDTH JOINER
    "﻿",  # ZERO WIDTH NO-BREAK SPACE / BOM
)

# Fake/forged control tags.  We rewrite the angle brackets to brackets so the
# resulting text cannot satisfy the LLM's tag consumer.
# extends the previous V2.0 system-reminder-only pass with ``tool-call`` and
# ``function-call`` because SEC-02 already enumerated them as
# tag-smuggling vectors but the previous neutralizer only flagged them via
# observability without rewriting.
#
# finding (2026-05-10): 's fix omitted
# ``tool-use`` / ``tool_use``, which IS the canonical Anthropic Messages API
# tool-use content-block tag. Memory recall content (e.g. CLAUDE.md) is
# user-controlled in the v1.0.0 Claude Code memory adapter, so an attacker
# with PR access to a project's CLAUDE.md could inject ``<tool_use>`` blocks
# that the model might treat as legitimate tool invocations. Closed inline
# during verification.
_FORGED_TAG_RE: Final[re.Pattern[str]] = re.compile(
    r"<\s*(/?\s*(?:system[-_]?reminder|tool[-_]?call|tool[-_]?use|function[-_]?call)\s*)>",
    flags=re.IGNORECASE,
)


def _strip_zero_width(text: str) -> str:
    """Remove zero-width code points from ``text`` (B2).

    Zero-width characters are invisible but still parsed by tokenizers, so
    attackers use them to fragment otherwise-blocked tag sequences. Stripping
    them in a dedicated pre-pass keeps the lookalike regex authoritative on
    a clean ASCII-or-ASCII-lookalike string.

    Idempotent: ``_strip_zero_width(_strip_zero_width(x)) == _strip_zero_width(x)``.
    Pure: no I/O.
    """
    if not text:
        return text
    out = text
    for ch in _ZERO_WIDTH_CHARS:
        if ch in out:
            out = out.replace(ch, "")
    return out


def neutralize_xml(text: str) -> str:
    """Replace look-alike chars + fake system-reminder tags with safe ASCII.

    Idempotent: applying twice is identical to applying once.

    B2: zero-width characters are stripped FIRST so they cannot
    fragment a sequence the regex would otherwise match.

    Source: V2.0 slm_bridge §1.10 (extracted, generalized) + STAGE-5C-003 layer 1-2.
    """
    if not text:
        return text
    out = _strip_zero_width(text)
    for src, dst in _LOOKALIKE_MAP.items():
        if src in out:
            out = out.replace(src, dst)
    # rewrite forged control tags (system-reminder,
    # tool-call, function-call) to bracketed inert text so the kernel
    # cannot pass them through to the model.
    out = _FORGED_TAG_RE.sub(r"[\1]", out)
    return out


# ---------------------------------------------------------------------------
# Smuggling-signal detection (observability layer-4)
# ---------------------------------------------------------------------------

# Compiled once at module load.
_SIGNAL_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "ignore-instruction": re.compile(
        r"\b(ignore|disregard)\s+(?:all\s+)?(?:previous|prior|above|earlier)\b",
        re.I,
    ),
    "system-reminder-fake": re.compile(
        r"<\s*/?\s*system[-_]?reminder\s*>", re.I
    ),
    "long-base64": re.compile(r"[A-Za-z0-9+/=]{120,}"),
    # B2: extended langle/rangle classes to cover fullwidth (U+FF1C/E)
    # and mathematical angle (U+27E8/9) variants surfaced by SEC-02.
    "lookalike-langle": re.compile(r"[‹˂〈＜⟨]"),
    "lookalike-rangle": re.compile(r"[›˃〉＞⟩]"),
    "tool-call-fake": re.compile(r"<\s*tool[-_]?call\s*>", re.I),
    # <tool_use> is the canonical Anthropic Messages API tool-use
    # content-block tag; user-controlled memory recall must not pass it through.
    "tool-use-fake": re.compile(r"<\s*tool[-_]?use\s*>", re.I),
    "function-call-fake": re.compile(r"<\s*function[-_]?call\s*>", re.I),
    # B2: any zero-width code point IS a smuggling signal — there
    # is no benign use case in recalled text. Pattern is char-class for speed.
    "zero-width": re.compile(r"[​‌‍﻿]"),
}


def detect_smuggling_signals(text: str) -> list[str]:
    """Return list of signal names found in ``text``. Pure function, no I/O.

    Each signal name is a stable string for log filters and metrics. Ordering
    is the iteration order of ``_SIGNAL_PATTERNS`` (insertion order).
    """
    if not text:
        return []
    return [name for name, pat in _SIGNAL_PATTERNS.items() if pat.search(text)]


# ---------------------------------------------------------------------------
# Convenience: cap → neutralize → detect, returns (safe_text, signals)
# ---------------------------------------------------------------------------


def _cap_to_bytes(text: str, byte_limit: int) -> str:
    """Cap ``text`` to at most ``byte_limit`` UTF-8 bytes.

    the constant is named ``MAX_RECALLED_TEXT_BYTES``
    so the cap MUST be measured in bytes.  Using ``text[:N]`` counts
    code points, so an 8192-character emoji string is 32768 bytes — 4×
    the documented budget.  This helper encodes, slices on a UTF-8
    boundary using ``errors="ignore"`` on decode, and returns a string
    whose UTF-8 byte length is provably ``<= byte_limit``.
    """
    if not text:
        return text
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text
    # ``errors="ignore"`` drops the partial multi-byte sequence at the
    # slice boundary so we never emit invalid UTF-8.
    return encoded[:byte_limit].decode("utf-8", errors="ignore")


def apply_recall_safety(text: str) -> tuple[str, list[str]]:
    """Apply layer-3 (cap), layer-1/2 (neutralize), layer-4 (detect signals).

    Returns ``(safe_text, signals)`` where ``signals`` is a possibly-empty list
    of signal names. Caller is expected to ``LOG.warning`` if signals is
    non-empty but MUST NOT raise.

    The cap is applied FIRST (before neutralize) so we don't waste work
    neutralizing 1 MB strings.

    cap is measured in UTF-8 bytes via ``_cap_to_bytes``,
    not Python ``str`` indexing — so a multi-byte emoji string cannot exceed
    the documented byte budget.

    B2: signal detection runs on the UNION of pre- and post-neutralize
    text so observability sees zero-width / lookalike attempts even after the
    neutralizer has stripped them. Signals are deduplicated, preserving the
    insertion order of ``_SIGNAL_PATTERNS``.
    """
    if not text:
        return "", []
    capped = _cap_to_bytes(text, MAX_RECALLED_TEXT_BYTES)
    safe = neutralize_xml(capped)
    pre_signals = detect_smuggling_signals(capped)
    post_signals = detect_smuggling_signals(safe)
    seen: set[str] = set()
    signals: list[str] = []
    for name in (*pre_signals, *post_signals):
        if name not in seen:
            seen.add(name)
            signals.append(name)
    return safe, signals


__all__ = [
    "MAX_RECALLED_TEXT_BYTES",
    "apply_recall_safety",
    "detect_smuggling_signals",
    "neutralize_xml",
]
