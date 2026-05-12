# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""TDD tests for model_router — RED phase."""

from __future__ import annotations

import os
from unittest.mock import patch

from agent_amplifier.types import EffortLevel


class TestModelSuggestionShape:
    def test_suggestion_has_tier(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        router = ModelRouter()
        suggestion = router.suggest(EffortLevel.HIGH)
        assert hasattr(suggestion, "tier")
        assert isinstance(suggestion.tier, str)

    def test_suggestion_has_display(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        suggestion = ModelRouter().suggest(EffortLevel.MEDIUM)
        assert hasattr(suggestion, "display")
        assert isinstance(suggestion.display, str)
        assert len(suggestion.display) > 0

    def test_suggestion_has_reason(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        suggestion = ModelRouter().suggest(EffortLevel.LOW)
        assert hasattr(suggestion, "reason")
        assert isinstance(suggestion.reason, str)

    def test_suggestion_is_frozen(self) -> None:
        from agent_amplifier.model_router import ModelSuggestion

        s = ModelSuggestion(tier="opus", display="d", reason="r")
        try:
            s.tier = "haiku"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


class TestDefaultRouting:
    def test_minimal_routes_to_haiku(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        assert ModelRouter().suggest(EffortLevel.MINIMAL).tier == "haiku"

    def test_low_routes_to_haiku(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        assert ModelRouter().suggest(EffortLevel.LOW).tier == "haiku"

    def test_medium_routes_to_sonnet(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        assert ModelRouter().suggest(EffortLevel.MEDIUM).tier == "sonnet"

    def test_high_routes_to_opus(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        assert ModelRouter().suggest(EffortLevel.HIGH).tier == "opus"

    def test_max_routes_to_opus(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        assert ModelRouter().suggest(EffortLevel.MAX).tier == "opus"

    def test_string_complexity_accepted(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        assert ModelRouter().suggest("HIGH").tier == "opus"

    def test_string_case_insensitive(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        assert ModelRouter().suggest("high").tier == "opus"

    def test_unknown_string_defaults_to_sonnet(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        assert ModelRouter().suggest("UNKNOWN_TIER").tier == "sonnet"


class TestDisabled:
    def test_disabled_returns_none_tier(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        router = ModelRouter(enabled=False)
        suggestion = router.suggest(EffortLevel.MAX)
        assert suggestion.tier == "none"

    def test_disabled_reason_explains(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        suggestion = ModelRouter(enabled=False).suggest(EffortLevel.HIGH)
        assert "disabled" in suggestion.reason.lower()

    def test_enabled_property(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        assert ModelRouter(enabled=True).enabled is True
        assert ModelRouter(enabled=False).enabled is False


class TestCustomMap:
    def test_custom_map_overrides_default(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        router = ModelRouter(model_map={"HIGH": "haiku"})
        assert router.suggest(EffortLevel.HIGH).tier == "haiku"

    def test_model_map_property_returns_copy(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        router = ModelRouter()
        m = router.model_map
        m["HIGH"] = "changed"
        assert router.suggest(EffortLevel.HIGH).tier == "opus"


class TestEnvOverride:
    def test_env_var_overrides_map(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        with patch.dict(os.environ, {"AGENT_AMP_MODEL_MAP": '{"HIGH": "haiku"}'}):
            router = ModelRouter()
            s = router.suggest(EffortLevel.HIGH)
            assert s.tier == "haiku"
            assert s.overridden is True

    def test_env_var_invalid_json_ignored(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        with patch.dict(os.environ, {"AGENT_AMP_MODEL_MAP": "not json"}):
            router = ModelRouter()
            assert router.suggest(EffortLevel.HIGH).tier == "opus"

    def test_env_var_empty_ignored(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        with patch.dict(os.environ, {"AGENT_AMP_MODEL_MAP": ""}):
            router = ModelRouter()
            assert router.suggest(EffortLevel.HIGH).tier == "opus"

    def test_env_var_non_dict_ignored(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        with patch.dict(os.environ, {"AGENT_AMP_MODEL_MAP": '["list"]'}):
            router = ModelRouter()
            assert router.suggest(EffortLevel.HIGH).tier == "opus"


class TestDomainInReason:
    def test_domain_appears_in_reason(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        s = ModelRouter().suggest(EffortLevel.HIGH, domain="frontend")
        assert "frontend" in s.reason

    def test_general_domain_omitted_from_reason(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        s = ModelRouter().suggest(EffortLevel.HIGH, domain="general")
        assert "domain=" not in s.reason


class TestReportOutput:
    def test_suggest_for_report_returns_dict(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        d = ModelRouter().suggest_for_report(EffortLevel.HIGH)
        assert isinstance(d, dict)
        assert "tier" in d
        assert "display" in d
        assert "reason" in d
        assert "overridden" in d

    def test_suggest_for_report_matches_suggest(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        router = ModelRouter()
        s = router.suggest(EffortLevel.MEDIUM, domain="backend")
        d = router.suggest_for_report(EffortLevel.MEDIUM, domain="backend")
        assert d["tier"] == s.tier
        assert d["display"] == s.display
        assert d["reason"] == s.reason


class TestFactory:
    def test_create_router_returns_model_router(self) -> None:
        from agent_amplifier.model_router import ModelRouter, create_router

        router = create_router()
        assert isinstance(router, ModelRouter)

    def test_create_router_forwards_args(self) -> None:
        from agent_amplifier.model_router import create_router

        router = create_router(enabled=False, model_map={"LOW": "opus"})
        assert router.enabled is False
        assert router.suggest("LOW").tier == "none"


class TestDefaultMapClassVar:
    def test_default_map_exposed(self) -> None:
        from agent_amplifier.model_router import ModelRouter

        assert "HIGH" in ModelRouter.DEFAULT_MAP
        assert ModelRouter.DEFAULT_MAP["HIGH"] == "opus"
