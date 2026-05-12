# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Hypothesis property tests for Cluster C (.7 / ).

Properties verified:
    1. Jaccard symmetry        — j(a,b) == j(b,a)
    2. Jaccard idempotence     — j(s,s) == 1 for any keyword set s
    3. Jaccard boundedness     — j(a,b) in [0, 1]
    4. Damping monotonicity    — strictly decreasing in iteration while
                                  above the numerical floor; non-increasing
                                  at/below the floor
    5. Keyword-set idempotence — keyword_set(" ".join(s)) is a subset of s
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from agent_amplifier._internal.keyword_set import keyword_set
from agent_amplifier.convergence import ConvergenceDetector

# Restrict to non-surrogate characters so encode("utf-8") cannot raise.
_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=2000,
)


@given(a=_text, b=_text)
def test_jaccard_symmetric(a: str, b: str) -> None:
    cd = ConvergenceDetector()
    s_a = keyword_set(a)
    s_b = keyword_set(b)
    assert cd._jaccard(s_a, s_b) == cd._jaccard(s_b, s_a)


@given(a=_text)
def test_jaccard_idempotent(a: str) -> None:
    cd = ConvergenceDetector()
    s = keyword_set(a)
    assert cd._jaccard(s, s) == 1.0


@given(a=_text, b=_text)
def test_jaccard_bounded(a: str, b: str) -> None:
    cd = ConvergenceDetector()
    j = cd._jaccard(keyword_set(a), keyword_set(b))
    assert 0.0 <= j <= 1.0


@given(t=st.integers(min_value=0, max_value=99))
def test_damping_factor_strictly_decreasing_in_iteration_within_clamp_range(
    t: int,
) -> None:
    """Strictly decreasing while above the numerical floor; non-increasing
    once at the floor (the ``exp(-exp(...))`` underflow is bounded by
    ``eps = 1e-12`` to keep callers from seeing exact zero)."""
    cd = ConvergenceDetector()
    eps = 1e-12
    a = cd.damping_factor(t)
    b = cd.damping_factor(t + 1)
    if a > eps:
        assert b < a
    else:
        assert b <= a


@given(a=_text)
def test_keyword_set_idempotent(a: str) -> None:
    """``keyword_set`` of the joined tokens is a subset of the original.

    Tokens that survived the regex + 2-char-min + stop-word filter once will
    survive again; lower-casing is fixpoint after the first pass. The set
    can shrink (never grow) because tokens like ``"a"`` are filtered out by
    the regex anyway — this property is therefore a containment, not
    strict equality.
    """
    s1 = keyword_set(a)
    s2 = keyword_set(" ".join(s1))
    assert s2 <= s1
