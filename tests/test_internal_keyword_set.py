# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier._internal.keyword_set``.

Spec source: .5 + §4.7 (per-locked C-8 +  + ).

Properties asserted:
  * ``keyword_set(None) == keyword_set("") == frozenset()``
  * idempotent: tokens of a tokenset string are a subset of original tokens
  * bounded CPU: ``MAX_OUTPUT_CHARS_FOR_ANALYSIS`` truncates input
  * never raises on any ``str`` input
  * stopwords are filtered
  * casefolded — ``frozenset`` members are lowercase
  * deterministic — same input ⇒ same output
"""

from __future__ import annotations

import re
import time

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_amplifier._internal.keyword_set import (
    _KEYWORD_RE,
    MAX_OUTPUT_CHARS_FOR_ANALYSIS,
    STOPWORDS,
    keyword_set,
)

# ---------------------------------------------------------------------------
# Empty / None handling
# ---------------------------------------------------------------------------


def test_none_returns_empty_frozenset() -> None:
    assert keyword_set(None) == frozenset()


def test_empty_string_returns_empty_frozenset() -> None:
    assert keyword_set("") == frozenset()


def test_returns_frozenset_type() -> None:
    out = keyword_set("hello world")
    assert isinstance(out, frozenset)


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------


def test_all_stopwords_yields_empty() -> None:
    text = "the a an and or of to in on is the"
    assert keyword_set(text) == frozenset()


def test_all_non_stopwords_retained() -> None:
    text = "refactor authentication module"
    assert keyword_set(text) == frozenset({"refactor", "authentication", "module"})


def test_stopwords_filtered_from_mixed() -> None:
    out = keyword_set("the quick brown fox")
    assert "the" not in out
    assert {"quick", "brown", "fox"} <= out


def test_stopwords_constant_is_frozenset() -> None:
    assert isinstance(STOPWORDS, frozenset)
    # Mutating attempt must raise. frozenset has no `add` method.
    assert not hasattr(STOPWORDS, "add")


# ---------------------------------------------------------------------------
# Casefolding + ASCII tokenization
# ---------------------------------------------------------------------------


def test_case_insensitive() -> None:
    assert keyword_set("REFACTOR Refactor refactor") == frozenset({"refactor"})


def test_punctuation_only_yields_empty() -> None:
    assert keyword_set("!!! ??? ... ,,,") == frozenset()


def test_unicode_codepoints_dropped_by_ascii_regex() -> None:
    # ASCII regex (DECISIONS-LOCKED C-7 / .4) — non-ASCII letters
    # are excluded; only the ASCII tokens survive.
    out = keyword_set("auth ёлка café résumé")
    assert "auth" in out
    # The ASCII tail of the Latin-extended words ("caf", "r", "sum") are
    # legitimate matches because the regex requires len >= 2 (`{1,}` after a
    # leading letter). We don't over-assert which tail letters survive — the
    # contract is: ASCII-only, never raise, never produce non-ASCII tokens.
    for tok in out:
        assert tok.isascii(), f"non-ASCII leaked: {tok!r}"


def test_underscore_token_admitted() -> None:
    assert "snake_case" in keyword_set("This is snake_case naming")


def test_digits_after_letter_admitted() -> None:
    assert "x86" in keyword_set("Optimize x86 path")


def test_token_must_start_with_letter() -> None:
    # Per .4: regex `[a-z][a-z0-9_]{1,}` — leading char is alpha.
    # `123abc` should NOT yield "123abc" as a token (the `123` is rejected,
    # then `abc` is matched as its own token of length 3).
    out = keyword_set("123abc")
    assert "123abc" not in out
    assert "abc" in out


def test_minimum_token_length_two() -> None:
    # Pattern requires {1,} chars AFTER the leading letter, so length >= 2.
    assert keyword_set("a b c") == frozenset()
    assert keyword_set("ab cd") == frozenset({"ab", "cd"})


# ---------------------------------------------------------------------------
# CPU bound
# ---------------------------------------------------------------------------


@pytest.mark.perf
def test_keyword_set_under_50ms_for_10mb() -> None:
    """.5 + §4.7: 10 MB single-char input returns deterministically <50 ms.

    Note: 10 M ``a`` chars truncates to ``MAX_OUTPUT_CHARS_FOR_ANALYSIS`` (256 K),
    and the regex ``[a-z][a-z0-9_]{1,}`` is greedy — it consumes the entire
    truncated run as a SINGLE 256 K-char token (length >= 2 satisfied). Result
    is ``frozenset({"a"*256_000})``. The size of the result is bounded; the
    runtime is bounded; both are deterministic.
    """
    start = time.monotonic()
    out = keyword_set("a" * 10_000_000)
    elapsed_ms = (time.monotonic() - start) * 1000
    # Single token because the regex eats the run greedily after truncation.
    assert len(out) == 1
    (only,) = out
    assert only == "a" * MAX_OUTPUT_CHARS_FOR_ANALYSIS
    assert elapsed_ms < 50, f"keyword_set took {elapsed_ms:.2f} ms (>50 ms cap)"


def test_input_larger_than_cap_is_truncated() -> None:
    # Construct text with a unique token at the very end — past the cap, it
    # should be dropped by truncation.
    head = "abc " * (MAX_OUTPUT_CHARS_FOR_ANALYSIS // 4)
    tail_marker = "zztail"
    text = head + tail_marker
    # Sanity: text is longer than the cap.
    assert len(text) > MAX_OUTPUT_CHARS_FOR_ANALYSIS
    out = keyword_set(text)
    assert "abc" in out
    assert tail_marker not in out


def test_max_output_chars_constant_value() -> None:
    # Pin the constant — bumping it should be a deliberate decision.
    assert MAX_OUTPUT_CHARS_FOR_ANALYSIS == 256_000


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------


# Use printable ASCII only — non-ASCII codepoints are out-of-scope for V1
# (DECISIONS-LOCKED C-7).
_ASCII_TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=2000
)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_ASCII_TEXT)
def test_keyword_set_idempotent(text: str) -> None:
    """.6: tokens of the tokenset are a subset of the original."""
    once = keyword_set(text)
    # Round-trip the keyword set as space-separated text — its keyword_set
    # should be a subset (lowercasing + stop-word filtering only shrinks).
    twice = keyword_set(" ".join(once))
    assert twice <= once


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_ASCII_TEXT)
def test_keyword_set_never_raises(text: str) -> None:
    keyword_set(text)            # must not raise


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_ASCII_TEXT)
def test_keyword_set_returns_lowercase(text: str) -> None:
    out = keyword_set(text)
    for tok in out:
        assert tok == tok.lower(), f"non-lowercase token: {tok!r}"


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_ASCII_TEXT)
def test_keyword_set_excludes_stopwords(text: str) -> None:
    out = keyword_set(text)
    assert out.isdisjoint(STOPWORDS)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_ASCII_TEXT)
def test_keyword_set_deterministic(text: str) -> None:
    assert keyword_set(text) == keyword_set(text)


# ---------------------------------------------------------------------------
# Module-level constants integrity
# ---------------------------------------------------------------------------


def test_keyword_re_compiled_pattern() -> None:
    assert isinstance(_KEYWORD_RE, re.Pattern)
    # Pin the source pattern — accidental relaxation = silent semantics shift.
    assert _KEYWORD_RE.pattern == r"[a-z][a-z0-9_]{1,}"
    assert _KEYWORD_RE.flags & re.IGNORECASE


def test_stopwords_contains_canonical_set() -> None:
    # A representative sample — exhaustive enumeration is brittle.
    for w in ("the", "a", "an", "and", "or", "of", "to", "in", "on"):
        assert w in STOPWORDS
