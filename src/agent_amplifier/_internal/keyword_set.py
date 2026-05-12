# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Shared keyword extraction. Internal API. Used by convergence + goal_anchor.

Constants and helpers live here so the two consumer modules cannot drift
apart (DECISIONS-LOCKED C-8 + .5 +  + ).

This module is import-cycle-free and has no internal dependencies beyond
stdlib.
"""

from __future__ import annotations

import re
from typing import Final

# Hard cap on text size analyzed (.2 / Sec F-07). Anything
# beyond this is truncated BEFORE regex execution to bound CPU cost
# regardless of caller input.
MAX_OUTPUT_CHARS_FOR_ANALYSIS: Final[int] = 256_000

# ASCII-only regex per DECISIONS-LOCKED C-7 / .4.
# `IGNORECASE` eliminates the need for a per-call ``text.lower()`` allocation
# (.3 / Perf F-04). Tokens must start with a letter and have
# total length >= 2 (the trailing class repeats `{1,}` after the leading
# letter — minimum length is therefore 2).
_KEYWORD_RE: Final[re.Pattern[str]] = re.compile(
    r"[a-z][a-z0-9_]{1,}", re.IGNORECASE
)

# Frozen at module import — never mutated. Public so tests + downstream
# modules can introspect.
STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "is",
        "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "this", "that", "it", "its", "with",
        "for", "by", "at", "as", "from", "but", "if", "then", "so",
        "you", "your", "we", "our", "they", "their", "i", "me", "my",
    }
)


def keyword_set(text: str | None) -> frozenset[str]:
    """Extract a deterministic ``frozenset`` of lower-cased keywords.

    Properties (verified by hypothesis tests in tests/test_internal_keyword_set.py):
        * ``keyword_set(None) == keyword_set("") == frozenset()``
        * idempotent: tokens of the keyword-set's text are a subset of the
          original keyword set (lower-casing + stop-word filtering only
          shrinks the set)
        * bounded CPU: ``O(min(len(text), MAX_OUTPUT_CHARS_FOR_ANALYSIS))``
        * never raises on any ``str`` input
    """
    if not text:
        return frozenset()

    if len(text) > MAX_OUTPUT_CHARS_FOR_ANALYSIS:
        text = text[:MAX_OUTPUT_CHARS_FOR_ANALYSIS]
    # IGNORECASE eliminates the per-call .lower() allocation.
    tokens = _KEYWORD_RE.findall(text)
    if not tokens:
        return frozenset()
    # Lower at set-construction time, only once per token, only for tokens
    # that passed the regex. (Cheaper than text.lower() over the full string.)
    return frozenset(t.lower() for t in tokens if t.lower() not in STOPWORDS)


__all__ = [
    "MAX_OUTPUT_CHARS_FOR_ANALYSIS",
    "STOPWORDS",
    "keyword_set",
]
