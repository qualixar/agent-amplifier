# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.goal_anchor`` (IP-3 Goal Anchor Protocol).

Coverage targets per  V2.0 §4.6:
    line   >= 95 %
    branch >= 90 %

Findings traced explicitly:
    *   test_capture_caps_at_8KB,
              test_capture_strips_control_characters,
              test_capture_replaces_double_quotes,
              test_capture_handles_terminal_escape_sequences
    *   test_measure_drift_with_precomputed_kw_skips_internal_tokenization
    *   test_classify_drift_below_warn_returns_on_track,
              test_classify_drift_in_drifting_band,
              test_classify_drift_at_or_above_alert
    *  test_inject_old_anchor_logs_warning_via_log_not_warnings_warn
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from unittest.mock import patch

import pytest

from agent_amplifier.goal_anchor import (
    ANCHOR_MAX_AGE_SECONDS,
    DEFAULT_REINJECTION_INTERVAL,
    DRIFT_ALERT_THRESHOLD,
    DRIFT_WARN_THRESHOLD,
    INJECTION_TOKEN_COST_ESTIMATE,
    MAX_ANCHOR_ESCAPED_CHARS,
    DriftLevel,
    GoalAnchor,
    GoalAnchorService,
    _escape_for_template,
)

# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestServiceConstruction:
    def test_default_construction_is_safe(self) -> None:
        svc = GoalAnchorService()
        assert svc is not None

    def test_reinjection_interval_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="reinjection_interval"):
            GoalAnchorService(reinjection_interval=0)

    def test_warn_must_be_strictly_below_alert(self) -> None:
        with pytest.raises(ValueError, match="thresholds"):
            GoalAnchorService(warn_threshold=0.7, alert_threshold=0.5)

    def test_warn_can_equal_zero(self) -> None:
        # 0.0 <= warn < alert <= 1.0 — boundary inclusive at 0.
        svc = GoalAnchorService(warn_threshold=0.0, alert_threshold=0.5)
        assert svc is not None

    def test_alert_can_equal_one(self) -> None:
        svc = GoalAnchorService(warn_threshold=0.5, alert_threshold=1.0)
        assert svc is not None

    def test_alert_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="thresholds"):
            GoalAnchorService(warn_threshold=0.5, alert_threshold=1.5)

    def test_max_anchor_age_default(self) -> None:
        # Sanity: constant matches DECISIONS-LOCKED C-6 (1 hour).
        assert ANCHOR_MAX_AGE_SECONDS == 3600.0


# ---------------------------------------------------------------------------
# capture()
# ---------------------------------------------------------------------------


class TestCapture:
    def test_capture_basic_request(self) -> None:
        svc = GoalAnchorService()
        a = svc.capture("Refactor the auth module")
        assert "Refactor" in a.text
        assert {"refactor", "auth", "module"} <= a.keyword_set
        assert a.char_count == len(a.text)
        assert a.token_estimate >= 1

    def test_capture_strips_leading_trailing_whitespace(self) -> None:
        svc = GoalAnchorService()
        a = svc.capture("  hello  ")
        assert a.text == "hello"

    def test_capture_none_treated_as_empty(self) -> None:
        svc = GoalAnchorService()
        a = svc.capture(None)  # type: ignore[arg-type]
        assert a.text == ""
        assert a.keyword_set == frozenset()
        assert a.token_estimate == 1  # max(1, 0 // 4)

    def test_capture_caps_at_8KB(self) -> None:
        # long input truncated to MAX_ANCHOR_ESCAPED_CHARS.
        svc = GoalAnchorService()
        long_text = "abcd " * 50_000  # ~250 KB
        a = svc.capture(long_text)
        assert len(a.text) <= MAX_ANCHOR_ESCAPED_CHARS

    def test_capture_strips_control_characters(self) -> None:
        # null byte + bell + bs all dropped.
        svc = GoalAnchorService()
        a = svc.capture("\x00\x01evil\x07")
        assert a.text == "evil"

    def test_capture_replaces_double_quotes(self) -> None:
        # straight double-quote replaced with curly to keep template safe.
        svc = GoalAnchorService()
        a = svc.capture('Refactor "auth" module')
        assert '"' not in a.text
        assert "“" in a.text

    def test_capture_handles_terminal_escape_sequences(self) -> None:
        # ESC (0x1B) is non-printable, dropped along with surrounding CSI bytes.
        svc = GoalAnchorService()
        a = svc.capture("Hello \x1b[31mRED\x1b[0m world")
        assert "\x1b" not in a.text
        # Surviving printable text retains "Hello", "RED", "world" content.
        assert "Hello" in a.text and "world" in a.text

    def test_capture_collapses_newlines_to_spaces(self) -> None:
        svc = GoalAnchorService()
        a = svc.capture("line1\nline2\rline3")
        assert "\n" not in a.text and "\r" not in a.text

    def test_capture_token_estimate_is_at_least_one(self) -> None:
        svc = GoalAnchorService()
        a = svc.capture("a")
        assert a.token_estimate == 1

    def test_capture_records_monotonic_and_iso_timestamps(self) -> None:
        svc = GoalAnchorService()
        before = time.monotonic()
        a = svc.capture("hello")
        after = time.monotonic()
        assert before <= a.captured_at_monotonic <= after
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", a.captured_at_wall_iso)


# ---------------------------------------------------------------------------
# inject()
# ---------------------------------------------------------------------------


class TestInject:
    def _make_anchor(self) -> GoalAnchor:
        return GoalAnchorService().capture("Refactor the auth module")

    def test_inject_at_interval_boundary(self) -> None:
        svc = GoalAnchorService(reinjection_interval=5)
        a = self._make_anchor()
        out = svc.inject("ctx", a, tool_call_count=5)
        assert "GOAL ANCHOR" in out
        assert "Refactor" in out
        assert "ctx" in out

    def test_inject_skipped_off_interval(self) -> None:
        svc = GoalAnchorService(reinjection_interval=5)
        a = self._make_anchor()
        out = svc.inject("ctx", a, tool_call_count=4)
        assert out == "ctx"

    def test_inject_force_overrides_interval(self) -> None:
        svc = GoalAnchorService(reinjection_interval=5)
        a = self._make_anchor()
        out = svc.inject("ctx", a, tool_call_count=1, force=True)
        assert "GOAL ANCHOR" in out

    def test_inject_skipped_when_count_zero(self) -> None:
        svc = GoalAnchorService(reinjection_interval=5)
        a = self._make_anchor()
        # tool_call_count == 0 must not trigger (modulo 0 would otherwise match).
        out = svc.inject("ctx", a, tool_call_count=0)
        assert out == "ctx"

    def test_inject_with_empty_anchor_returns_context_unchanged(self) -> None:
        svc = GoalAnchorService()
        empty = svc.capture("")
        out = svc.inject("ctx", empty, tool_call_count=5)
        assert out == "ctx"

    def test_inject_with_none_context_handled(self) -> None:
        svc = GoalAnchorService(reinjection_interval=1)
        a = self._make_anchor()
        out = svc.inject(None, a, tool_call_count=1)  # type: ignore[arg-type]
        assert "GOAL ANCHOR" in out

    def test_inject_zero_or_negative_interval_treated_as_one(self) -> None:
        svc = GoalAnchorService(reinjection_interval=5)
        a = self._make_anchor()
        out = svc.inject("ctx", a, tool_call_count=1, interval=0)
        assert "GOAL ANCHOR" in out

    def test_inject_old_anchor_logs_warning_via_log_not_warnings_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # must emit via logging.warning, never via warnings.warn.
        svc = GoalAnchorService(max_anchor_age_s=0.0)
        a = self._make_anchor()
        with caplog.at_level(logging.WARNING, logger="agent_amplifier.goal_anchor"):
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                out = svc.inject("ctx", a, tool_call_count=1)
            assert wlist == []  # no warnings.warn fired
        assert any("anchor" in r.message.lower() for r in caplog.records)
        assert "GOAL ANCHOR" in out  # still injected (warn-and-inject)


# ---------------------------------------------------------------------------
# measure_drift()
# ---------------------------------------------------------------------------


class TestMeasureDrift:
    def test_drift_zero_for_aligned_output(self) -> None:
        svc = GoalAnchorService()
        a = svc.capture("refactor authentication module quickly")
        # Output uses the SAME informative tokens (2-char filter excludes 'a','to').
        d = svc.measure_drift("authentication module refactor quickly", a)
        assert d == 0.0

    def test_drift_one_for_disjoint_output(self) -> None:
        svc = GoalAnchorService()
        a = svc.capture("refactor authentication module")
        d = svc.measure_drift("baking cookies recipe", a)
        assert d == 1.0

    def test_drift_with_empty_anchor(self) -> None:
        svc = GoalAnchorService()
        empty = svc.capture("")
        assert svc.measure_drift("anything", empty) == 0.0

    def test_drift_with_empty_output(self) -> None:
        svc = GoalAnchorService()
        a = svc.capture("refactor auth")
        assert svc.measure_drift("", a) == 1.0

    def test_drift_both_empty(self) -> None:
        svc = GoalAnchorService()
        empty = svc.capture("")
        assert svc.measure_drift("", empty) == 0.0

    def test_drift_partial_overlap_in_unit_interval(self) -> None:
        svc = GoalAnchorService()
        a = svc.capture("alpha beta gamma")
        d = svc.measure_drift("alpha delta epsilon", a)
        assert 0.0 < d < 1.0

    def test_measure_drift_with_precomputed_kw_skips_internal_tokenization(
        self,
    ) -> None:
        # kernel passes precomputed_kw; module MUST NOT re-tokenize.
        svc = GoalAnchorService()
        a = svc.capture("alpha beta gamma")
        precomputed = frozenset({"alpha", "delta", "epsilon"})
        with patch(
            "agent_amplifier.goal_anchor._keyword_set"
        ) as mocked:
            d = svc.measure_drift("UNUSED", a, precomputed_kw=precomputed)
            assert mocked.call_count == 0
        # Drift = 1 - jaccard({alpha,beta,gamma},{alpha,delta,epsilon})
        # = 1 - 1/5 = 0.8
        assert abs(d - 0.8) < 1e-9

    def test_measure_drift_with_empty_precomputed_kw_returns_one(self) -> None:
        svc = GoalAnchorService()
        a = svc.capture("alpha beta")
        d = svc.measure_drift("ignored", a, precomputed_kw=frozenset())
        assert d == 1.0


# ---------------------------------------------------------------------------
# classify_drift() —  STABLE PUBLIC API
# ---------------------------------------------------------------------------


class TestClassifyDrift:
    def test_classify_drift_below_warn_returns_on_track(self) -> None:
        svc = GoalAnchorService()
        assert svc.classify_drift(0.0) is DriftLevel.ON_TRACK
        assert svc.classify_drift(DRIFT_WARN_THRESHOLD - 0.001) is DriftLevel.ON_TRACK

    def test_classify_drift_in_drifting_band(self) -> None:
        svc = GoalAnchorService()
        assert svc.classify_drift(DRIFT_WARN_THRESHOLD) is DriftLevel.DRIFTING
        assert svc.classify_drift(0.6) is DriftLevel.DRIFTING

    def test_classify_drift_at_or_above_alert(self) -> None:
        svc = GoalAnchorService()
        assert svc.classify_drift(DRIFT_ALERT_THRESHOLD) is DriftLevel.DRIFTED
        assert svc.classify_drift(0.99) is DriftLevel.DRIFTED


# ---------------------------------------------------------------------------
# estimated_injection_tokens()
# ---------------------------------------------------------------------------


class TestEstimatedInjectionTokens:
    def test_estimated_includes_template_overhead(self) -> None:
        svc = GoalAnchorService()
        a = svc.capture("hello world")
        n = svc.estimated_injection_tokens(a)
        assert n == a.token_estimate + INJECTION_TOKEN_COST_ESTIMATE


# ---------------------------------------------------------------------------
# _escape_for_template helper ()
# ---------------------------------------------------------------------------


class TestEscape:
    def test_empty_input(self) -> None:
        assert _escape_for_template("") == ""

    def test_truncates_to_cap(self) -> None:
        s = "x" * (MAX_ANCHOR_ESCAPED_CHARS + 100)
        out = _escape_for_template(s)
        assert len(out) == MAX_ANCHOR_ESCAPED_CHARS

    def test_strips_null_byte(self) -> None:
        assert "\x00" not in _escape_for_template("a\x00b")

    def test_default_reinjection_interval_constant(self) -> None:
        assert DEFAULT_REINJECTION_INTERVAL == 5
