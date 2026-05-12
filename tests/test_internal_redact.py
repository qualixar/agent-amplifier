# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier._internal.redact``.

Spec source: .

Properties asserted:
    * Each pattern in isolation masks correctly.
    * Composition: multiple secrets in one string all masked.
    * Idempotence (hypothesis): ``redact(redact(t)) == redact(t)``.
    * Total: never raises for ``str`` input; raises ``TypeError`` for non-str.
    * Bounded output: redacted output length is bounded — no exponential blowup.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_amplifier._internal.redact import _PATTERNS, redact

# ---------------------------------------------------------------------------
# Per-pattern unit tests ( §4.3)
# ---------------------------------------------------------------------------


def test_openai_key_masked() -> None:
    assert (
        redact("My key is sk-proj-AbCdEf1234567890ghijkl")
        == "My key is [REDACTED:OPENAI_KEY]"
    )


def test_anthropic_key_masked_and_not_double_masked() -> None:
    out = redact("Anthropic: sk-ant-api03-AbCdEf1234567890")
    assert out == "Anthropic: [REDACTED:ANTHROPIC_KEY]"
    # Critical: the anthropic prefix `sk-ant-` must NOT be left exposed by an
    # earlier OpenAI substitution.
    assert "sk-" not in out


def test_github_pat_masked() -> None:
    assert redact("PAT ghp_" + "A" * 40) == "PAT [REDACTED:GITHUB_PAT]"


def test_generic_bearer_masked() -> None:
    assert (
        redact("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9")
        == "Authorization: [REDACTED:GENERIC_TOKEN]"
    )


def test_api_key_param_masked() -> None:
    assert "[REDACTED:GENERIC_TOKEN]" in redact("api_key=abcdef1234567890")


def test_api_dash_key_param_masked() -> None:
    assert "[REDACTED:GENERIC_TOKEN]" in redact("api-key=abcdef1234567890")


def test_token_param_masked() -> None:
    assert "[REDACTED:GENERIC_TOKEN]" in redact("token: abcdef1234567890")


def test_email_masked() -> None:
    assert (
        redact("Contact varun@qualixar.com please")
        == "Contact [REDACTED:EMAIL] please"
    )


# ---------------------------------------------------------------------------
# Composition + idempotence ( §4.3 +  contract)
# ---------------------------------------------------------------------------


def test_idempotent_compound() -> None:
    s = (
        "OpenAI sk-proj-x12345678901234567890 + "
        "Anthropic sk-ant-api03-y12345678901234 + "
        "email varun@qualixar.com"
    )
    once = redact(s)
    twice = redact(once)
    assert once == twice
    assert "[REDACTED:OPENAI_KEY]" in once
    assert "[REDACTED:ANTHROPIC_KEY]" in once
    assert "[REDACTED:EMAIL]" in once


def test_no_op_on_clean_text() -> None:
    s = "This is a perfectly normal sentence with no secrets."
    assert redact(s) == s


def test_empty_string_is_unchanged() -> None:
    assert redact("") == ""


def test_redact_returns_str_type() -> None:
    out = redact("Plain text.")
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# Pattern ordering — anthropic must be matched before openai
# ---------------------------------------------------------------------------


def test_anthropic_before_openai_in_pattern_order() -> None:
    """Pattern ordering is load-bearing — anthropic check FIRST.

    Otherwise the openai regex `\\bsk-...` matches 'sk-ant-...' and
    redacts to ANTHROPIC_KEY only after we've already done the openai
    sub. The current implementation guards both directions: anthropic
    is anchored on `sk-ant-` so its match is disjoint from openai's
    `sk-` prefix when openai runs first — but we still pin order.
    """
    first_pattern = _PATTERNS[0][0].pattern
    assert "sk-ant-" in first_pattern
    second_pattern = _PATTERNS[1][0].pattern
    assert "sk-" in second_pattern
    assert "ant-" not in second_pattern


# ---------------------------------------------------------------------------
# Type-error for non-string input
# ---------------------------------------------------------------------------


def test_non_string_raises_type_error() -> None:
    with pytest.raises(TypeError):
        redact(42)            # type: ignore[arg-type]


def test_non_string_bytes_raises_type_error() -> None:
    with pytest.raises(TypeError):
        redact(b"sk-proj-x12345678901234567890")  # type: ignore[arg-type]


def test_non_string_none_raises_type_error() -> None:
    with pytest.raises(TypeError):
        redact(None)            # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hypothesis property tests (idempotence + boundedness)
# ---------------------------------------------------------------------------


_PRINTABLE = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    max_size=2_000,
)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_PRINTABLE)
def test_redact_idempotent(text: str) -> None:
    once = redact(text)
    twice = redact(once)
    assert once == twice


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_PRINTABLE)
def test_redact_bounded_output(text: str) -> None:
    """Length of output is bounded — no exponential blowup."""
    out = redact(text)
    # Each replacement substitutes ≥ 1 char with ≤ ~30 chars (`[REDACTED:...]`).
    # Coarse upper bound: 30x input length is generous and catches blowups.
    assert len(out) <= max(30, 30 * len(text))


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_PRINTABLE)
def test_redact_never_raises(text: str) -> None:
    redact(text)            # must not raise


# ---------------------------------------------------------------------------
# Defense-in-depth: redacted text is itself idempotent under nested redacts
# ---------------------------------------------------------------------------


def test_redacted_marker_string_is_stable() -> None:
    """Edge case — a literal '[REDACTED:OPENAI_KEY]' in input is preserved."""
    s = "Already redacted: [REDACTED:OPENAI_KEY] foo"
    assert redact(s) == s
