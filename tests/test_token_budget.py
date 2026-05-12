# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.token_budget`` (IP-10 Cost-Bounded Amplification).

Coverage targets per .6:
    line   >= 95 %
    branch >= 90 %

Findings traced explicitly:
    *    test_track_iteration_with_prefix_charges_prefix_once,
               test_no_lru_cache_on_instance_method
    *    test_track_concurrent_does_not_double_count_or_skip_warning
    *   test_logs_warning_via_log_not_warnings_warn_when_tiktoken_missing
    *   test_observability_callback_fires_at_70_80_90_each_once,
               test_observability_callback_fires_on_budget_hit_at_100,
               test_observability_callback_failure_is_swallowed,
               test_warning_text_includes_actionable_hint
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any
from unittest.mock import patch

import pytest

from agent_amplifier.token_budget import (
    _BASE_BUDGET_BY_EFFORT,
    _WARN_RATIO_THRESHOLDS,
    BudgetExhaustedSignal,
    BudgetReport,
    TokenBudgetController,
)
from agent_amplifier.types import AmplifierEvent, BudgetMode, EffortLevel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Recorder:
    """Drop-in observability_callback replacement for tests."""

    def __init__(self) -> None:
        self.events: list[tuple[AmplifierEvent, dict[str, Any]]] = []

    def __call__(
        self, event: AmplifierEvent, payload: dict[str, Any]
    ) -> None:
        self.events.append((event, payload))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self) -> None:
        c = TokenBudgetController()
        assert c.is_exhausted() is False
        r = c.report()
        assert r.allocated == 0
        assert r.used == 0

    def test_invalid_track_value_rejected(self) -> None:
        c = TokenBudgetController()
        with pytest.raises(ValueError, match="tokens_used"):
            c.track(-1)

    def test_warn_ratio_thresholds_constant(self) -> None:

        assert _WARN_RATIO_THRESHOLDS == (0.70, 0.80, 0.90, 1.00)


# ---------------------------------------------------------------------------
# allocate()
# ---------------------------------------------------------------------------


class TestAllocate:
    def test_minimal_mode_uses_base(self) -> None:
        c = TokenBudgetController(BudgetMode.MINIMAL)
        n = c.allocate(EffortLevel.MEDIUM)
        assert n == _BASE_BUDGET_BY_EFFORT[EffortLevel.MEDIUM]

    def test_balanced_mode_doubles_base(self) -> None:
        c = TokenBudgetController(BudgetMode.BALANCED)
        n = c.allocate(EffortLevel.MEDIUM)
        assert n == 2 * _BASE_BUDGET_BY_EFFORT[EffortLevel.MEDIUM]

    def test_unlimited_mode_returns_maxsize(self) -> None:
        c = TokenBudgetController(BudgetMode.UNLIMITED)
        assert c.allocate(EffortLevel.MAX) == sys.maxsize

    def test_auto_mode_uses_effort_multiplier(self) -> None:
        c = TokenBudgetController(BudgetMode.AUTO)
        # AUTO + MEDIUM: base * (1 + 0.5 * 2.0) = base * 2.0
        assert c.allocate(EffortLevel.MEDIUM) == int(
            _BASE_BUDGET_BY_EFFORT[EffortLevel.MEDIUM] * 2.0
        )

    def test_max_override_wins(self) -> None:
        c = TokenBudgetController(
            BudgetMode.MINIMAL, max_tokens_override=12345
        )
        assert c.allocate(EffortLevel.HIGH) == 12345

    def test_allocate_is_idempotent(self) -> None:
        c = TokenBudgetController(BudgetMode.AUTO)
        first = c.allocate(EffortLevel.LOW)
        second = c.allocate(EffortLevel.MAX)  # ignored
        assert first == second


# ---------------------------------------------------------------------------
# track() / track_text()
# ---------------------------------------------------------------------------


class TestTrack:
    def test_track_increments_used(self) -> None:
        c = TokenBudgetController(BudgetMode.AUTO)
        c.allocate(EffortLevel.MEDIUM)
        c.track(100)
        c.track(200)
        assert c.report().used == 300

    def test_track_before_allocate_defaults(self) -> None:
        c = TokenBudgetController(BudgetMode.AUTO)
        c.track(50)
        r = c.report()
        assert r.allocated > 0
        assert r.used == 50

    def test_track_text_uses_char_div_4_fallback(self) -> None:
        c = TokenBudgetController(BudgetMode.AUTO)
        c.allocate(EffortLevel.MEDIUM)
        text = "x" * 400
        n = c.track_text(text)
        assert n == 100

    def test_track_text_empty_returns_zero(self) -> None:
        c = TokenBudgetController(BudgetMode.AUTO)
        c.allocate(EffortLevel.MEDIUM)
        assert c.track_text("") == 0

    def test_track_iteration_with_prefix_charges_prefix_once(self) -> None:

        c = TokenBudgetController(
            BudgetMode.AUTO, injected_prefix_tokens=120
        )
        c.allocate(EffortLevel.MEDIUM)
        text = "x" * 400  # variable -> 100 tokens
        total = c.track_iteration_with_prefix(text)
        assert total == 100 + 120
        assert c.report().used == 100 + 120

    def test_track_iteration_with_prefix_zero_prefix(self) -> None:
        c = TokenBudgetController(
            BudgetMode.AUTO, injected_prefix_tokens=0
        )
        c.allocate(EffortLevel.MEDIUM)
        n = c.track_iteration_with_prefix("x" * 400)
        assert n == 100  # only variable

    def test_track_iteration_with_prefix_empty_and_zero_prefix(self) -> None:
        # Closes the ``total > 0`` False branch.
        c = TokenBudgetController(
            BudgetMode.AUTO, injected_prefix_tokens=0
        )
        c.allocate(EffortLevel.MEDIUM)
        assert c.track_iteration_with_prefix("") == 0
        assert c.report().used == 0

    def test_track_iteration_with_prefix_is_atomic_single_threshold_event(
        self,
    ) -> None:
        # CRIT-1 — the prefix + variable charge must land in ONE track()
        # call so threshold crossings are attributed to one iteration.
        rec = _Recorder()
        c = TokenBudgetController(
            BudgetMode.MINIMAL,
            max_tokens_override=100,
            observability_callback=rec,
            injected_prefix_tokens=50,
        )
        c.allocate(EffortLevel.LOW)
        # variable = 50 (200 chars / 4), prefix = 50, total = 100 -> 100 %
        c.track_iteration_with_prefix("x" * 200)
        # 70/80/90/100 all crossed in a single accumulation; per-level
        # dedup says one event each.
        events = [e for e, _ in rec.events]
        assert events.count(AmplifierEvent.ON_BUDGET_LOW) == 3
        assert events.count(AmplifierEvent.ON_BUDGET_HIT) == 1

    def test_no_lru_cache_on_instance_method(self) -> None:

        c = TokenBudgetController()
        # ``functools.lru_cache`` decorates with ``__wrapped__``; the
        # decorated bound method has a ``cache_info`` attribute. Plain
        # methods do not.
        assert not hasattr(c.track_text, "cache_info")
        assert not hasattr(c.track, "cache_info")


# ---------------------------------------------------------------------------
# remaining / should_stop_for_budget / is_exhausted
# ---------------------------------------------------------------------------


class TestStopConditions:
    def test_remaining_unlimited(self) -> None:
        c = TokenBudgetController(BudgetMode.UNLIMITED)
        c.allocate(EffortLevel.MEDIUM)
        c.track(10**9)
        assert c.remaining() == sys.maxsize

    def test_remaining_floors_at_zero(self) -> None:
        c = TokenBudgetController(
            BudgetMode.MINIMAL, max_tokens_override=100
        )
        c.allocate(EffortLevel.LOW)
        c.track(150)  # over-spend
        assert c.remaining() == 0

    def test_should_stop_unlimited_never(self) -> None:
        c = TokenBudgetController(BudgetMode.UNLIMITED)
        c.allocate(EffortLevel.MEDIUM)
        c.track(10**9)
        assert c.should_stop_for_budget() is False

    def test_should_stop_limited(self) -> None:
        c = TokenBudgetController(
            BudgetMode.MINIMAL, max_tokens_override=100
        )
        c.allocate(EffortLevel.LOW)
        c.track(50)
        assert c.should_stop_for_budget() is False
        c.track(50)  # hits 100
        assert c.should_stop_for_budget() is True

    def test_is_exhausted_flips_on_full_use(self) -> None:
        c = TokenBudgetController(
            BudgetMode.MINIMAL, max_tokens_override=100
        )
        c.allocate(EffortLevel.LOW)
        assert c.is_exhausted() is False
        c.track(100)
        assert c.is_exhausted() is True


# ---------------------------------------------------------------------------
# mark_iteration / report / reset
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_mark_iteration_increments(self) -> None:
        c = TokenBudgetController(BudgetMode.AUTO)
        c.mark_iteration()
        c.mark_iteration()
        assert c.report().iterations_completed == 2

    def test_report_contains_expected_fields(self) -> None:
        c = TokenBudgetController(BudgetMode.AUTO)
        c.allocate(EffortLevel.MEDIUM)
        c.track(100)
        r = c.report()
        assert isinstance(r, BudgetReport)
        assert r.mode == BudgetMode.AUTO
        assert r.allocated > 0
        assert r.used == 100
        assert r.iterations_completed == 0

    def test_reset_clears_state(self) -> None:
        c = TokenBudgetController(
            BudgetMode.MINIMAL, max_tokens_override=100
        )
        c.allocate(EffortLevel.LOW)
        c.track(100)
        c.reset()
        r = c.report()
        assert r.used == 0
        assert r.allocated == 0
        assert r.exhausted is False
        assert r.warnings_emitted == 0


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestObservability:
    def test_observability_callback_fires_at_70_80_90_each_once(self) -> None:

        rec = _Recorder()
        c = TokenBudgetController(
            BudgetMode.MINIMAL,
            max_tokens_override=100,
            observability_callback=rec,
        )
        c.allocate(EffortLevel.LOW)
        c.track(70)  # 70 %
        c.track(10)  # 80 %
        c.track(10)  # 90 %
        events = [e for e, _ in rec.events]
        assert events.count(AmplifierEvent.ON_BUDGET_LOW) == 3

    def test_observability_callback_fires_on_budget_hit_at_100(self) -> None:

        rec = _Recorder()
        c = TokenBudgetController(
            BudgetMode.MINIMAL,
            max_tokens_override=100,
            observability_callback=rec,
        )
        c.allocate(EffortLevel.LOW)
        c.track(100)
        # We expect 70/80/90/100 to all fire at once when we land at 100 %
        # —  says NO fire-once-on-the-warning-sequence guard, only
        # per-level dedup.
        events = [e for e, _ in rec.events]
        assert events.count(AmplifierEvent.ON_BUDGET_HIT) == 1
        assert events.count(AmplifierEvent.ON_BUDGET_LOW) == 3

    def test_observability_callback_each_level_fires_only_once(self) -> None:
        rec = _Recorder()
        c = TokenBudgetController(
            BudgetMode.MINIMAL,
            max_tokens_override=100,
            observability_callback=rec,
        )
        c.allocate(EffortLevel.LOW)
        c.track(70)
        c.track(0)  # no new crossing
        c.track(70)  # 140 % — clamped at 100 %; 80/90/100 each once
        events = [e for e, _ in rec.events]
        # 70 from first track, then 80/90/100 from the over-spend.
        assert events.count(AmplifierEvent.ON_BUDGET_LOW) == 3
        assert events.count(AmplifierEvent.ON_BUDGET_HIT) == 1

    def test_observability_callback_failure_is_swallowed(self) -> None:

        def boom(_event: AmplifierEvent, _payload: dict[str, Any]) -> None:
            raise RuntimeError("simulated callback failure")

        c = TokenBudgetController(
            BudgetMode.MINIMAL,
            max_tokens_override=100,
            observability_callback=boom,
        )
        c.allocate(EffortLevel.LOW)
        # Should not raise.
        c.track(100)
        # Internal counter still incremented.
        assert c.is_exhausted() is True

    def test_payload_contains_signal(self) -> None:
        rec = _Recorder()
        c = TokenBudgetController(
            BudgetMode.MINIMAL,
            max_tokens_override=100,
            observability_callback=rec,
        )
        c.allocate(EffortLevel.LOW)
        c.track(100)
        last_event, last_payload = rec.events[-1]
        assert last_event is AmplifierEvent.ON_BUDGET_HIT
        assert "signal" in last_payload
        sig = last_payload["signal"]
        assert isinstance(sig, BudgetExhaustedSignal)
        assert sig.allocated == 100
        assert sig.used >= 100

    def test_warning_text_includes_actionable_hint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:

        c = TokenBudgetController(
            BudgetMode.MINIMAL, max_tokens_override=100
        )
        c.allocate(EffortLevel.LOW)
        with caplog.at_level(
            logging.WARNING, logger="agent_amplifier.token_budget"
        ):
            c.track(100)
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "AGENT_AMP_BUDGET=unlimited" in joined
        assert "BudgetMode.UNLIMITED" in joined

    def test_logs_warning_via_log_not_warnings_warn_when_tiktoken_missing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:

        with patch.dict(sys.modules, {"tiktoken": None}):
            with caplog.at_level(
                logging.WARNING, logger="agent_amplifier.token_budget"
            ):
                c = TokenBudgetController(
                    BudgetMode.AUTO, use_tiktoken=True
                )
                # Side effect: fallback log is emitted by __init__.
            assert any(
                "tiktoken" in r.getMessage().lower() for r in caplog.records
            )
            # And tiktoken usage flag flipped off.
            assert c.report().use_tiktoken is False

    def test_track_text_uses_tiktoken_encoder_when_available(self) -> None:
        # Drive the tiktoken-installed branch via a stub encoder.
        c = TokenBudgetController(BudgetMode.AUTO)
        c.allocate(EffortLevel.MEDIUM)

        class _StubEnc:
            def encode(self, text: str) -> list[int]:
                # Map every 2 chars to one token.
                return [0] * max(1, len(text) // 2)

        c._tiktoken_encoder = _StubEnc()  # type: ignore[assignment]
        n = c.track_text("xxxxxx")  # 6 chars -> 3 tokens
        assert n == 3
        assert c.report().used == 3

    def test_track_text_falls_back_when_tiktoken_encode_raises(self) -> None:
        c = TokenBudgetController(BudgetMode.AUTO)
        c.allocate(EffortLevel.MEDIUM)

        class _BoomEnc:
            def encode(self, text: str) -> list[int]:
                raise RuntimeError("encode boom")

        c._tiktoken_encoder = _BoomEnc()  # type: ignore[assignment]
        n = c.track_text("x" * 400)  # fallback char/4 -> 100
        assert n == 100

    def test_init_uses_real_tiktoken_when_imported(self) -> None:
        # Drive the import-success branch in __init__ via a stub module.
        fake = type(
            "FakeMod",
            (),
            {
                "get_encoding": staticmethod(
                    lambda name: type(
                        "Enc", (), {"encode": lambda self, t: [0] * len(t)}
                    )()
                )
            },
        )
        with patch.dict(sys.modules, {"tiktoken": fake}):
            c = TokenBudgetController(
                BudgetMode.AUTO, use_tiktoken=True
            )
        assert c.report().use_tiktoken is True


# ---------------------------------------------------------------------------
# Tokenizer-aware encoder selection ()
# ---------------------------------------------------------------------------


class TestEncodingSelection:
    """``_select_encoding_name`` chooses the right tiktoken encoding for
    each model family — modern frontier models get ``o200k_base``, legacy
    GPT-3.5/4 get ``cl100k_base``, unknown defaults to legacy for back-compat."""

    def test_none_model_defaults_to_legacy(self) -> None:
        from agent_amplifier.token_budget import _select_encoding_name

        assert _select_encoding_name(None) == "cl100k_base"

    def test_empty_model_defaults_to_legacy(self) -> None:
        from agent_amplifier.token_budget import _select_encoding_name

        assert _select_encoding_name("") == "cl100k_base"

    def test_gpt_3_5_picks_legacy(self) -> None:
        from agent_amplifier.token_budget import _select_encoding_name

        assert _select_encoding_name("gpt-3.5-turbo") == "cl100k_base"

    def test_legacy_gpt_4_picks_legacy(self) -> None:
        from agent_amplifier.token_budget import _select_encoding_name

        assert _select_encoding_name("gpt-4") == "cl100k_base"
        assert _select_encoding_name("gpt-4-32k") == "cl100k_base"

    def test_gpt_4o_picks_modern(self) -> None:
        from agent_amplifier.token_budget import _select_encoding_name

        assert _select_encoding_name("gpt-4o") == "o200k_base"
        assert _select_encoding_name("gpt-4o-mini") == "o200k_base"

    def test_gpt_4_turbo_picks_modern(self) -> None:
        from agent_amplifier.token_budget import _select_encoding_name

        assert _select_encoding_name("gpt-4-turbo") == "o200k_base"

    def test_gpt_5_picks_modern(self) -> None:
        from agent_amplifier.token_budget import _select_encoding_name

        assert _select_encoding_name("gpt-5") == "o200k_base"

    def test_claude_picks_modern(self) -> None:
        from agent_amplifier.token_budget import _select_encoding_name

        assert _select_encoding_name("claude-opus-4-7") == "o200k_base"
        assert _select_encoding_name("claude-3-5-sonnet") == "o200k_base"

    def test_legacy_davinci_picks_legacy(self) -> None:
        from agent_amplifier.token_budget import _select_encoding_name

        assert _select_encoding_name("text-davinci-003") == "cl100k_base"
        assert _select_encoding_name("davinci-002") == "cl100k_base"

    def test_unknown_model_picks_modern_default(self) -> None:
        """An unknown model name (e.g. a new Anthropic release we haven't
        catalogued) falls through to ``o200k_base`` — the modern public
        proxy — rather than the legacy encoder."""
        from agent_amplifier.token_budget import _select_encoding_name

        assert _select_encoding_name("unknown-future-model") == "o200k_base"

    def test_init_passes_model_to_encoder_selection(self) -> None:
        """Constructor with ``model="claude-opus-4-7"`` requests
        ``o200k_base`` from tiktoken."""
        captured: list[str] = []

        def _capture(name: str) -> Any:
            captured.append(name)

            class _Enc:
                def encode(self, t: str) -> list[int]:
                    return [0] * len(t)

            return _Enc()

        fake = type("FakeMod", (), {"get_encoding": staticmethod(_capture)})
        with patch.dict(sys.modules, {"tiktoken": fake}):
            c = TokenBudgetController(
                BudgetMode.AUTO,
                use_tiktoken=True,
                model="claude-opus-4-7",
            )
        assert captured == ["o200k_base"]
        assert c.report().encoding_name == "o200k_base"

    def test_init_legacy_model_requests_cl100k(self) -> None:
        captured: list[str] = []

        def _capture(name: str) -> Any:
            captured.append(name)

            class _Enc:
                def encode(self, t: str) -> list[int]:
                    return [0] * len(t)

            return _Enc()

        fake = type("FakeMod", (), {"get_encoding": staticmethod(_capture)})
        with patch.dict(sys.modules, {"tiktoken": fake}):
            c = TokenBudgetController(
                BudgetMode.AUTO,
                use_tiktoken=True,
                model="gpt-3.5-turbo",
            )
        assert captured == ["cl100k_base"]
        assert c.report().encoding_name == "cl100k_base"

    def test_encoding_name_none_when_tiktoken_disabled(self) -> None:
        """When use_tiktoken=False, encoding_name remains ``None``."""
        c = TokenBudgetController(BudgetMode.AUTO, use_tiktoken=False)
        assert c.report().encoding_name is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_track_concurrent_does_not_double_count_or_skip_warning(
        self,
    ) -> None:

        rec = _Recorder()
        c = TokenBudgetController(
            BudgetMode.MINIMAL,
            max_tokens_override=400,
            observability_callback=rec,
        )
        c.allocate(EffortLevel.LOW)

        n_threads = 4
        n_ops = 100

        def worker() -> None:
            for _ in range(n_ops):
                c.track(1)

        threads = [
            threading.Thread(target=worker) for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        r = c.report()
        assert r.used == n_threads * n_ops
        assert r.exhausted is True
        # Each ladder level fired exactly once across the race.
        events = [e for e, _ in rec.events]
        assert events.count(AmplifierEvent.ON_BUDGET_LOW) == 3
        assert events.count(AmplifierEvent.ON_BUDGET_HIT) == 1
