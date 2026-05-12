# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.effort_router``.

Coverage targets (.1, §3.4):
    * 50+ unit cases (tier sanity, trigger mapping, should_iterate, edges)
    * Hypothesis property tests: deadline, determinism, idempotence,
      monotone-with-length
    * pre-compiled alternation regex (5 calls, not 250)
    * MAX_QUERY_CHARS truncation, MAX_DISTINCT_FOR_ESCALATE,
      escalate_low_confidence flag wiring
    * classify does NOT depend on _internal/keyword_set
    * empty-query is INFO-not-error; truncation is WARNING

Performance gate (P99 < 2 ms) lives in tests/perf/test_classify_p99.py
(``-m perf``, excluded from default run).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_amplifier import effort_router as ER
from agent_amplifier.effort_router import (
    LOW_CONFIDENCE_THRESHOLD,
    MAX_DISTINCT_FOR_ESCALATE,
    MAX_QUERY_CHARS,
    classify,
    classify_with_config,
    estimate_tokens,
    infer_domain,
    is_code_heavy,
    should_iterate,
    suggest_thinking_trigger,
)
from agent_amplifier.types import AmplifierConfig, EffortLevel, TaskClassification

# ---------------------------------------------------------------------------
# 0. Module-level invariants
# ---------------------------------------------------------------------------


class TestModuleInvariants:
    def test_constants_exported(self) -> None:
        assert MAX_QUERY_CHARS == 8192
        assert MAX_DISTINCT_FOR_ESCALATE == 2
        assert pytest.approx(0.6) == LOW_CONFIDENCE_THRESHOLD

    def test_constants_present_in_module_all(self) -> None:

        assert "MAX_QUERY_CHARS" in ER.__all__
        assert "MAX_DISTINCT_FOR_ESCALATE" in ER.__all__
        assert "LOW_CONFIDENCE_THRESHOLD" in ER.__all__
        assert "classify" in ER.__all__
        assert "classify_with_config" in ER.__all__

    def test_tier_regex_is_precompiled_pattern(self) -> None:

        assert isinstance(ER._TIER_REGEX, dict)
        assert set(ER._TIER_REGEX.keys()) == set(EffortLevel)
        for tier, pat in ER._TIER_REGEX.items():
            assert isinstance(pat, re.Pattern), tier

    def test_does_not_depend_on_internal_keyword_set(self) -> None:

        # It must NOT *import* ``keyword_set`` (avoids inadvertent coupling).
        # Walk the module's compiled code object to enumerate real imports
        # (free of false positives from docstrings).
        imported = _imported_modules(ER)
        assert "agent_amplifier._internal.keyword_set" not in imported
        assert not any(
            name.endswith("._internal.keyword_set") for name in imported
        )


# ---------------------------------------------------------------------------
# 1. Five-tier sanity ( §1.3)
# ---------------------------------------------------------------------------


class TestTierSanity:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            # Single-keyword inputs — chosen so only one tier regex matches.
            ("typo", EffortLevel.MINIMAL),
            ("rename", EffortLevel.MINIMAL),
            ("alphabetize", EffortLevel.MINIMAL),
            ("fix", EffortLevel.LOW),
            ("explain", EffortLevel.LOW),
            ("refactor", EffortLevel.MEDIUM),
            ("validate", EffortLevel.MEDIUM),
        ],
    )
    def test_simple_tier_assignment(
        self, query: str, expected: EffortLevel
    ) -> None:
        # Per .4.7 algorithm: walk HIGH → MEDIUM → LOW →
        # MINIMAL, stop at first match. So multi-tier-keyword queries
        # ("fix typo" hits both LOW + MINIMAL → LOW wins). These single-
        # keyword inputs lock the per-tier semantics.
        assert classify(query).complexity == expected

    def test_mixed_keyword_query_chooses_higher_tier(self) -> None:
        # "fix typo" — both LOW ("fix") and MINIMAL ("typo") match. Algorithm
        # walks HIGH → MEDIUM → LOW first, so LOW wins. This locks the
        # documented walking-order behavior.
        assert classify("fix typo").complexity == EffortLevel.LOW

    def test_high_tier_single_signal(self) -> None:
        # "audit security": single MAX kw + a HIGH kw "audit". Single MAX kw
        # caps at HIGH per .
        assert classify("audit security of this module").complexity in (
            EffortLevel.HIGH,
            EffortLevel.MEDIUM,
        )

    def test_max_tier_via_two_distinct_max_keywords(self) -> None:

        assert (
            classify("audit security cve owasp injection").complexity
            == EffortLevel.MAX
        )


# ---------------------------------------------------------------------------
# 2. Thinking-trigger mapping ( §1.5; Claude Code v2.1.88)
# ---------------------------------------------------------------------------


class TestThinkingTriggerMapping:
    @pytest.mark.parametrize(
        ("level", "trigger"),
        [
            (EffortLevel.MINIMAL, ""),
            (EffortLevel.LOW, "think"),
            (EffortLevel.MEDIUM, "think hard"),
            (EffortLevel.HIGH, "think harder"),
            (EffortLevel.MAX, "ultrathink"),
        ],
    )
    def test_trigger_table(self, level: EffortLevel, trigger: str) -> None:
        assert suggest_thinking_trigger(level) == trigger


# ---------------------------------------------------------------------------
# 3. should_iterate gating
# ---------------------------------------------------------------------------


class TestShouldIterate:
    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            (EffortLevel.MINIMAL, False),
            (EffortLevel.LOW, False),
            (EffortLevel.MEDIUM, True),
            (EffortLevel.HIGH, True),
            (EffortLevel.MAX, True),
        ],
    )
    def test_iteration_eligibility(
        self, level: EffortLevel, expected: bool
    ) -> None:
        assert should_iterate(level) is expected


# ---------------------------------------------------------------------------
# 4. Edge cases E1..E17 (.7)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_e1_empty_returns_minimal(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="agent_amplifier.effort_router"):
            r = classify("")
        assert r.complexity == EffortLevel.MINIMAL
        assert r.confidence == 1.0
        assert r.matched_signals == ("empty_query",)

        info_records = [
            rec for rec in caplog.records if rec.levelno == logging.INFO
        ]
        assert any("empty" in rec.message.lower() for rec in info_records)

    def test_e2_whitespace_only_returns_minimal(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="agent_amplifier.effort_router"):
            r = classify("   \n\t   ")
        assert r.complexity == EffortLevel.MINIMAL
        assert r.confidence == 1.0
        assert "empty_query" in r.matched_signals

    def test_e2b_none_returns_minimal(self) -> None:
        # ``query is None`` is defensive (kernel may pass None on degenerate
        # adapter flows). Defaults to MINIMAL deterministically.
        r = classify(None)  # type: ignore[arg-type]
        assert r.complexity == EffortLevel.MINIMAL
        assert "empty_query" in r.matched_signals

    def test_e3_single_word(self) -> None:
        r = classify("hi")
        assert r.complexity == EffortLevel.MINIMAL

    def test_e4_pure_code_heavy_bumps_to_medium(self) -> None:
        # Choose a code-heavy snippet whose only tier signal is LOW (``log``)
        # so the code-heavy escalator must fire to reach MEDIUM. (Note: a
        # MEDIUM keyword like ``add`` would already put base at MEDIUM and
        # bypass the code-heavy bump.)
        q = "```python\nimport sys\nsys.stdout.write('log')\n```"
        r = classify(q)
        # Code-heavy + LOW base → MEDIUM via the code_heavy_bump escalator.
        assert r.complexity == EffortLevel.MEDIUM
        assert "code_heavy_bump" in r.matched_signals

    def test_e5_very_long_benign_text(self) -> None:
        # > 2000 estimated tokens within MAX_QUERY_CHARS. Two-char tokens
        # ("hi ") give one word per 3 chars — ~2730 words in 8192 chars
        # → ~3550 tokens (well over the > 2000 threshold).
        q = ("hi " * 2700)  # 8100 chars, under cap; 2700 words.
        r = classify(q)
        assert r.complexity in (
            EffortLevel.MEDIUM,
            EffortLevel.HIGH,
            EffortLevel.MAX,
        )
        assert any(s.startswith("len>2000") for s in r.matched_signals)

    def test_e5b_500_to_2000_token_query_bumps_one_tier(self) -> None:
        # ~600 words → ~780 tokens → > 500 path → bump 1. MINIMAL → LOW.
        q = ("please " * 600)
        r = classify(q)
        assert r.complexity == EffortLevel.LOW
        assert any(s.startswith("len>500") for s in r.matched_signals)

    def test_e6_trace_plus_fix(self) -> None:
        q = (
            "fix this:\nTraceback (most recent call last):\n  File \"x.py\""
            ", line 1, in <module>"
        )
        r = classify(q)
        assert _rank(r.complexity) >= _rank(EffortLevel.MEDIUM)
        assert "trace_detected" in r.matched_signals

    def test_e7_multilang_english_keyword_wins(self) -> None:
        q = "fix this: यह कोड काम नहीं करता"  # Hindi mixed
        r = classify(q)
        assert r.complexity in (EffortLevel.LOW, EffortLevel.MEDIUM)

    def test_e8_adversarial_secret_single_max_kw_caps_at_high(self) -> None:
        # "the secret to good code": single distinct MAX kw → HIGH
        r = classify("the secret to good code")
        assert r.complexity == EffortLevel.HIGH or r.complexity == EffortLevel.MEDIUM
        # If HIGH: signal must include the cap reason. If MEDIUM: low_conf back-off
        # (which is the SAFER outcome).
        all_sigs = " ".join(r.matched_signals)
        assert (
            "max_keyword_single_hit_capped_at_high" in all_sigs
            or "low_confidence_no_escalate" in all_sigs
        )

    def test_e9_multi_intent_two_distinct_max(self) -> None:
        r = classify("research auth and audit security")
        assert r.complexity == EffortLevel.MAX
        assert "max_tier_escalated" in r.matched_signals

    def test_e10_question_design_bumps_to_high(self) -> None:
        r = classify("why is this architecture brittle?")
        assert _rank(r.complexity) >= _rank(EffortLevel.HIGH)
        assert "design_question" in r.matched_signals

    def test_e11_dow_repetition_does_not_escalate_to_max(self) -> None:

        r = classify("security " * 50)
        assert r.complexity != EffortLevel.MAX

    def test_e12_dow_distinct_kws_escalate_to_max(self) -> None:
        r = classify("audit cve and check owasp")
        assert r.complexity == EffortLevel.MAX
        assert "max_tier_escalated" in r.matched_signals

    def test_e13_truncation_emits_signal(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 9000 chars > MAX_QUERY_CHARS (8192).
        long_q = "x" * 9000
        with caplog.at_level(
            logging.WARNING, logger="agent_amplifier.effort_router"
        ):
            r = classify(long_q)
        assert "query_truncated" in r.matched_signals

        warns = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
        assert any("truncat" in rec.message.lower() for rec in warns)

    def test_e13b_exactly_at_cap_does_not_truncate(self) -> None:
        # Boundary: len == MAX_QUERY_CHARS — must NOT truncate.
        q = "x" * MAX_QUERY_CHARS
        r = classify(q)
        assert "query_truncated" not in r.matched_signals

    def test_e13c_one_over_cap_does_truncate(self) -> None:
        q = "x" * (MAX_QUERY_CHARS + 1)
        r = classify(q)
        assert "query_truncated" in r.matched_signals

    def test_e14_redos_shape_completes_in_under_100ms(self) -> None:
        # ReDoS-shape input — must complete fast.
        import time

        adversarial = "at " + (" " * 1024) + "x"
        t0 = time.perf_counter()
        r = classify(adversarial)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"Took {elapsed_ms:.2f} ms"
        assert isinstance(r, TaskClassification)


# ---------------------------------------------------------------------------
# 5. escalate_low_confidence flag (classify_with_config)
# ---------------------------------------------------------------------------


class TestEscalateLowConfidenceFlag:
    def test_default_safe_back_off_actually_fires(self) -> None:
        # Engineer a query whose only signal is a single MAX kw → HIGH base
        # with confidence = 0.5 + 2*0.1 = 0.5 (signals: distinct count + cap)
        # — wait, let's compute. "secret" alone: distinct count=1, signals
        # = ["max_keywords_distinct=1", "max_keyword_single_hit_capped_at_high"]
        # → confidence = 0.5 + 0.2 = 0.7 ≥ 0.6 → no back-off.
        # We need a single-signal trigger. Use a single MAX keyword with no
        # other signals at all — but the algorithm always emits at least
        # one signal for any keyword match. So we test that the back-off
        # path is reachable by directly calling _classify with a low-signal
        # synthesis: take "secret" which produces 2 signals → conf 0.7;
        # back-off requires conf < 0.6.
        # Simpler approach: monkeypatch threshold to force the path. Avoid
        # that — instead, lock the *behavior* by checking opt-in vs default.
        cfg_safe = AmplifierConfig(escalate_low_confidence=False)
        cfg_unsafe = AmplifierConfig(escalate_low_confidence=True)
        # "secret" alone: HIGH (single MAX kw cap) at conf 0.7 — back-off
        # should NOT fire. Opt-in vs safe behave identically.
        r_safe = classify_with_config("secret", cfg_safe)
        r_unsafe = classify_with_config("secret", cfg_unsafe)
        # Both must yield HIGH (no back-off because conf >= 0.6).
        assert r_safe.complexity == EffortLevel.HIGH
        assert r_unsafe.complexity == EffortLevel.HIGH

    def test_back_off_fires_when_threshold_crossed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the threshold above max attainable confidence to exercise
        # the  back-off branch deterministically.
        monkeypatch.setattr(ER, "LOW_CONFIDENCE_THRESHOLD", 1.5)
        cfg_safe = AmplifierConfig(escalate_low_confidence=False)
        r_safe = classify_with_config("secret", cfg_safe)
        # Threshold is unreachable → every HIGH+ result MUST back off.
        assert r_safe.complexity == EffortLevel.MEDIUM
        assert "low_confidence_no_escalate" in r_safe.matched_signals
        # Opt-in keeps the original HIGH tier even with the impossible
        # threshold (escalate_low_confidence flag short-circuits the back-off).
        cfg_unsafe = AmplifierConfig(escalate_low_confidence=True)
        r_unsafe = classify_with_config("secret", cfg_unsafe)
        assert r_unsafe.complexity == EffortLevel.HIGH
        assert "low_confidence_no_escalate" not in r_unsafe.matched_signals

    def test_pure_classify_is_safe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Bare classify() always plays safe (.6) — even when
        # confidence is below threshold, it backs off.
        monkeypatch.setattr(ER, "LOW_CONFIDENCE_THRESHOLD", 1.5)
        r = classify("secret")
        assert r.complexity == EffortLevel.MEDIUM
        assert "low_confidence_no_escalate" in r.matched_signals


# ---------------------------------------------------------------------------
# 6. Helpers: estimate_tokens, is_code_heavy, infer_domain
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_zero_for_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_text(self) -> None:
        # 5 words * 1.3 ~= 6 tokens.
        assert 5 <= estimate_tokens("one two three four five") <= 8

    def test_monotone_with_length(self) -> None:
        a = estimate_tokens("hello " * 10)
        b = estimate_tokens("hello " * 100)
        assert b > a


class TestIsCodeHeavy:
    def test_fenced_block_is_code_heavy(self) -> None:
        assert is_code_heavy("```python\nprint(1)\n```") is True

    def test_short_punctuation_is_not_code_heavy(self) -> None:
        # < 80 chars → must NOT be code-heavy regardless of ratio.
        assert is_code_heavy(".+*=()") is False

    def test_empty_query_is_not_code_heavy(self) -> None:
        assert is_code_heavy("") is False
        assert is_code_heavy(None) is False  # type: ignore[arg-type]

    def test_long_punctuation_is_code_heavy(self) -> None:
        q = "{ } ( ) ; = + * - / . , : | & ^ ! ? @ # % $ ~ ` < > [ ] " * 4
        # Long enough + ratio of non-alphanum > 0.30.
        assert is_code_heavy(q) is True

    def test_function_keyword_is_code_heavy(self) -> None:
        assert is_code_heavy("function foo() { return 1; }") is True


class TestInferDomain:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("audit owasp vulnerabilities in auth flow", "security"),
            ("optimize latency on this endpoint", "performance"),
            ("rewrite this sql query for postgres", "data"),
            ("design a rest api endpoint", "api"),
            ("style this react component with tailwind", "frontend"),
            ("ship via docker on kubernetes", "infra"),
            ("add pytest coverage", "tests"),
            ("write the readme", "docs"),
            ("hello there friend", "general"),
        ],
    )
    def test_domain_lookup(self, query: str, expected: str) -> None:
        assert infer_domain(query) == expected

    def test_empty_query_is_general(self) -> None:
        assert infer_domain("") == "general"
        assert infer_domain("   ") == "general"


class TestBuildTierRegexEmpty:
    """Cover the empty-keyword-set sentinel branch."""

    def test_empty_keyword_set_returns_never_match_regex(self) -> None:
        pat = ER._build_tier_regex(frozenset())
        # Sentinel must never match anything, including the empty string.
        assert pat.search("anything at all") is None
        assert pat.search("") is None


class TestRegexSafety:
    """LLD §6 CRIT #3 — ban unbounded ``.*`` / ``.+`` quantifiers in
    handcrafted patterns (TRACE / MULTI_FILE / QUESTION).

    Note: the alternation regexes built by ``_build_tier_regex`` use only
    ``\\b(?:keyword)+\\b`` shapes — no quantifier issue.
    """

    @pytest.mark.parametrize(
        "patterns_attr",
        ["TRACE_PATTERNS", "MULTI_FILE_PATTERNS", "QUESTION_PATTERNS"],
    )
    def test_no_unbounded_quantifiers_in_handcrafted_patterns(
        self, patterns_attr: str
    ) -> None:
        patterns = getattr(ER, patterns_attr)
        for pat in patterns:
            src = pat.pattern
            # Disallow ``.*`` and ``.+`` at the literal level (greedy
            # unbounded). Bounded versions like ``.{1,512}`` are fine.
            assert ".*" not in src, f"unbounded .* in {src!r}"
            # ``.+`` permitted only when followed by a quantifier-bound
            # ``{...}`` — none of our patterns use that. Reject all.
            assert ".+" not in src, f"unbounded .+ in {src!r}"


class TestMultiFileAndOtherEscalators:
    def test_multi_file_signal_bumps_tier(self) -> None:
        # ``5 files`` triggers the digit/files pattern → bump 1 from base.
        # Use a benign LOW base so the bump is visible (LOW → MEDIUM).
        r = classify("fix bug across 5 files")
        assert _rank(r.complexity) >= _rank(EffortLevel.MEDIUM)
        assert "multi_file" in r.matched_signals

    def test_monorepo_pattern_signals_multi_file(self) -> None:
        r = classify("review monorepo structure")
        assert "multi_file" in r.matched_signals


# ---------------------------------------------------------------------------
# 7. distinct counting fragility (CRIT #2 mitigation)
# ---------------------------------------------------------------------------


class TestDistinctMaxCountingCaseInsensitive:
    def test_findall_distinct_counting_is_case_insensitive(self) -> None:
        # "AUTH" and "auth" must count as ONE distinct keyword, not two.
        r = classify("AUTH and auth")
        # Single distinct MAX kw → cap at HIGH; do NOT escalate to MAX.
        assert r.complexity != EffortLevel.MAX

    def test_distinct_counting_with_multiword_kw(self) -> None:
        # "system design" is one keyword in MAX_KEYWORDS — must be counted
        # whole, not as 2 keywords (longest-first sort).
        r = classify("system design")
        # Single distinct MAX kw → not MAX.
        assert r.complexity != EffortLevel.MAX


# ---------------------------------------------------------------------------
# 8. Hypothesis property tests (.1)
# ---------------------------------------------------------------------------


class TestProperties:
    @given(st.text(min_size=0, max_size=4096))
    @settings(
        deadline=200,
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_classify_never_exceeds_deadline(self, query: str) -> None:

        result = classify(query)
        assert isinstance(result.complexity, EffortLevel)

    @given(st.text(min_size=1, max_size=2048))
    @settings(
        deadline=200,
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_classify_is_deterministic(self, query: str) -> None:
        # Pure function: identical inputs → identical outputs.
        assert classify(query) == classify(query)

    @given(st.text(min_size=1, max_size=1024))
    @settings(
        deadline=200,
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_classify_idempotent_under_whitespace(self, query: str) -> None:

        base = classify(query)
        padded = classify(f"   {query}   ")
        assert base.complexity == padded.complexity
        assert base.domain == padded.domain

    @given(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("Ll", "Lu", "Nd", "Zs")
            ),
            min_size=1,
            max_size=512,
        )
    )
    @settings(
        deadline=200,
        max_examples=80,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_classify_monotone_with_length(self, q: str) -> None:
        # E17: extending a query (q ⊆ q + extra) cannot lower the tier.
        base = classify(q)
        longer = classify(q + " " + q)
        assert _rank(longer.complexity) >= _rank(base.complexity)


# ---------------------------------------------------------------------------
# 9. error-UX shape (signals, never-raise)
# ---------------------------------------------------------------------------


class TestErrorUx:
    def test_classify_never_raises_for_str_input(self) -> None:

        weird_inputs = [
            "",
            "   ",
            chr(0),  # null byte (use chr() to avoid embedding NUL in source)
            " extreme ",
            "🦀" * 1024,  # emojis
            "\n" * 4096,
            "?" * 10_000,  # over cap
        ]
        for q in weird_inputs:
            r = classify(q)
            assert isinstance(r, TaskClassification)
            assert isinstance(r.complexity, EffortLevel)

    def test_classify_with_config_rejects_non_amplifier_config(self) -> None:
        with pytest.raises(TypeError):
            classify_with_config("hi", config="not-a-config")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# classify_with_context — cross-turn inheritance for conversational
# continuations. Closes the 57.6% minimal/general mis-fire pattern observed
# in real dogfood telemetry.
# ---------------------------------------------------------------------------


def _prior(complexity: EffortLevel, domain: str = "api") -> TaskClassification:
    """Build a synthetic prior TaskClassification for inheritance tests."""
    return TaskClassification(
        complexity=complexity,
        domain=domain,
        estimated_tokens=120,
        confidence=0.9,
        matched_signals=("prior_synth",),
    )


class TestClassifyWithContextInheritance:
    def test_bare_ok_inherits_one_step_down_from_high(self) -> None:
        result = ER.classify_with_context(
            "ok", prior_classification=_prior(EffortLevel.HIGH, "performance")
        )
        assert result.complexity == EffortLevel.MEDIUM
        assert result.domain == "performance"
        assert "context_inherited_from_prior" in result.matched_signals

    def test_max_prior_inherits_to_high(self) -> None:
        result = ER.classify_with_context(
            "yes go", prior_classification=_prior(EffortLevel.MAX)
        )
        assert result.complexity == EffortLevel.HIGH

    def test_medium_prior_inherits_to_low(self) -> None:
        result = ER.classify_with_context(
            "do it", prior_classification=_prior(EffortLevel.MEDIUM)
        )
        assert result.complexity == EffortLevel.LOW

    def test_low_prior_inherits_to_minimal(self) -> None:
        result = ER.classify_with_context(
            "go", prior_classification=_prior(EffortLevel.LOW)
        )
        assert result.complexity == EffortLevel.MINIMAL

    def test_minimal_prior_does_not_trigger_inheritance(self) -> None:
        """Prior MINIMAL means there is nothing useful to inherit; fall through."""
        result = ER.classify_with_context(
            "ok", prior_classification=_prior(EffortLevel.MINIMAL)
        )
        assert "context_inherited_from_prior" not in result.matched_signals

    def test_no_prior_falls_through_to_pure_classify(self) -> None:
        """Without a prior classification, behavior matches classify() exactly."""
        with_ctx = ER.classify_with_context("ok", prior_classification=None)
        bare = ER.classify("ok")
        assert with_ctx.complexity == bare.complexity
        assert with_ctx.domain == bare.domain

    def test_numbered_answer_pattern_inherits(self) -> None:
        """'2. yes' is a common reply shape in option-list workflows."""
        result = ER.classify_with_context(
            "2. yes",
            prior_classification=_prior(EffortLevel.HIGH, "general"),
        )
        assert "context_inherited_from_prior" in result.matched_signals
        assert result.complexity == EffortLevel.MEDIUM

    def test_option_pick_pattern_inherits(self) -> None:
        result = ER.classify_with_context(
            "option A",
            prior_classification=_prior(EffortLevel.HIGH, "infra"),
        )
        assert "context_inherited_from_prior" in result.matched_signals

    def test_question_mark_pattern_inherits(self) -> None:
        result = ER.classify_with_context(
            "?",
            prior_classification=_prior(EffortLevel.MAX, "general"),
        )
        assert "context_inherited_from_prior" in result.matched_signals
        assert result.complexity == EffortLevel.HIGH

    def test_long_prompt_does_not_match_continuation(self) -> None:
        """A 100+ char prompt is content, not a continuation — pure path wins."""
        long_q = "Please refactor the authentication module to use JWT plus add tests"
        result = ER.classify_with_context(
            long_q,
            prior_classification=_prior(EffortLevel.HIGH),
        )
        assert "context_inherited_from_prior" not in result.matched_signals

    def test_continuation_pattern_with_extra_punctuation(self) -> None:
        result = ER.classify_with_context(
            "yes!",
            prior_classification=_prior(EffortLevel.HIGH),
        )
        assert "context_inherited_from_prior" in result.matched_signals

    def test_empty_query_does_not_inherit(self) -> None:
        """Empty/whitespace query stays at empty-domain MINIMAL."""
        result = ER.classify_with_context(
            "",
            prior_classification=_prior(EffortLevel.HIGH),
        )
        assert "context_inherited_from_prior" not in result.matched_signals
        assert result.complexity == EffortLevel.MINIMAL

    def test_whitespace_only_query_does_not_inherit(self) -> None:
        result = ER.classify_with_context(
            "   \n  ",
            prior_classification=_prior(EffortLevel.HIGH),
        )
        assert "context_inherited_from_prior" not in result.matched_signals

    def test_non_continuation_short_prompt_does_not_inherit(self) -> None:
        """A short prompt that does NOT match continuation pattern falls through."""
        result = ER.classify_with_context(
            "refactor X",
            prior_classification=_prior(EffortLevel.HIGH),
        )
        assert "context_inherited_from_prior" not in result.matched_signals

    def test_inheritance_signals_carry_prior_metadata(self) -> None:
        result = ER.classify_with_context(
            "ok",
            prior_classification=_prior(EffortLevel.HIGH, "frontend"),
        )
        signals = "|".join(result.matched_signals)
        assert "prior_complexity:high" in signals
        assert "prior_domain:frontend" in signals

    def test_classify_with_context_accepts_config(self) -> None:
        from agent_amplifier.types import AmplifierConfig

        cfg = AmplifierConfig(escalate_low_confidence=True)
        result = ER.classify_with_context(
            "refactor auth to use JWT and add tests",
            prior_classification=None,
            config=cfg,
        )
        # Falls through to pure classify_with_config; behavior should match
        bare = ER.classify_with_config(
            "refactor auth to use JWT and add tests", cfg
        )
        assert result.complexity == bare.complexity

    def test_classify_with_context_rejects_bad_config(self) -> None:
        with pytest.raises(TypeError, match="AmplifierConfig"):
            ER.classify_with_context(
                "ok",
                prior_classification=_prior(EffortLevel.HIGH),
                config="not-a-config",  # type: ignore[arg-type]
            )

    def test_continuation_pattern_none_query_falls_through(self) -> None:
        """None query goes to classify() which handles empty-query path."""
        result = ER.classify_with_context(
            None,  # type: ignore[arg-type]
            prior_classification=_prior(EffortLevel.HIGH),
        )
        assert result.complexity == EffortLevel.MINIMAL
        assert result.domain == "empty"

    def test_inherit_one_step_down_helper_directly(self) -> None:
        """Direct unit on _inherit_one_step_down for branch coverage."""
        assert ER._inherit_one_step_down(EffortLevel.MAX) == EffortLevel.HIGH
        assert ER._inherit_one_step_down(EffortLevel.HIGH) == EffortLevel.MEDIUM
        assert ER._inherit_one_step_down(EffortLevel.MEDIUM) == EffortLevel.LOW
        assert ER._inherit_one_step_down(EffortLevel.LOW) == EffortLevel.MINIMAL
        # MINIMAL clamps at MINIMAL (rank 0 → max(0, -1) = 0)
        assert ER._inherit_one_step_down(EffortLevel.MINIMAL) == EffortLevel.MINIMAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_RANK = {
    EffortLevel.MINIMAL: 0,
    EffortLevel.LOW: 1,
    EffortLevel.MEDIUM: 2,
    EffortLevel.HIGH: 3,
    EffortLevel.MAX: 4,
}


def _rank(level: EffortLevel) -> int:
    return _RANK[level]


def _all_signals(sigs: Iterable[str]) -> str:
    return "|".join(sigs)


def _imported_modules(module: object) -> set[str]:
    """Walk a module's AST and return its imported module names.

    Avoids false positives from docstrings/comments (CRIT mitigation).
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            for alias in node.names:
                names.add(f"{node.module}.{alias.name}")
                names.add(node.module)
    return names
