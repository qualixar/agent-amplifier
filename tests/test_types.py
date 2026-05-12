# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.types``.

Spec source: , §3.1.

Public API surface frozen here:
    AmplifierEvent, EffortLevel, PhaseIndex, ConvergenceState, BudgetMode,
    TaskClassification, QualityScore, AmplifierConfig, ObservabilityCallback.
"""

from __future__ import annotations

import dataclasses
import json
from types import MappingProxyType
from typing import Any

import pytest

from agent_amplifier.types import (
    _ALLOWED_CONFIG_FIELDS,
    _ALLOWED_ROUTERS,
    _ALLOWED_SELECTORS,
    AmplifierConfig,
    AmplifierEvent,
    BudgetMode,
    ConvergenceState,
    EffortLevel,
    ObservabilityCallback,
    PhaseIndex,
    QualityScore,
    TaskClassification,
)

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_public_surface() -> None:
    import agent_amplifier.types as types_module

    expected = {
        "AmplifierEvent",
        "EffortLevel",
        "PhaseIndex",
        "ConvergenceState",
        "BudgetMode",
        "TaskClassification",
        "QualityScore",
        "AmplifierConfig",
        "ObservabilityCallback",
    }
    assert expected.issubset(set(types_module.__all__))
    # Private names are not in __all__.
    assert "_ALLOWED_CONFIG_FIELDS" not in types_module.__all__


# ---------------------------------------------------------------------------
# AmplifierEvent
# ---------------------------------------------------------------------------


def test_amplifier_event_has_on_budget_low() -> None:
    assert AmplifierEvent.ON_BUDGET_LOW.value == "on_budget_low"


def test_amplifier_event_has_on_budget_hit() -> None:
    assert AmplifierEvent.ON_BUDGET_HIT.value == "on_budget_hit"


def test_amplifier_event_canonical_members() -> None:
    """Pin the full member list — adding/removing values is a deliberate op."""
    expected = {
        "BEFORE_STEP", "AFTER_STEP", "ON_ITERATION", "ON_CONVERGE",
        "ON_DRIFT", "ON_BUDGET_LOW", "ON_BUDGET_HIT",
    }
    assert {m.name for m in AmplifierEvent} == expected


def test_amplifier_event_str_mixin_serializable() -> None:
    # str-mixin enums dump as their string value via json.dumps.
    assert json.dumps(AmplifierEvent.BEFORE_STEP) == '"before_step"'


# ---------------------------------------------------------------------------
# Other enums
# ---------------------------------------------------------------------------


def test_effort_level_canonical_values() -> None:
    assert {m.value for m in EffortLevel} == {
        "minimal", "low", "medium", "high", "max",
    }


def test_phase_index_is_int_enum() -> None:
    assert int(PhaseIndex.EXPLORE) == 0
    assert PhaseIndex.REFINE > PhaseIndex.EXPLORE
    assert int(PhaseIndex.EVALUATE) == PhaseIndex.EXPLORE + 1


def test_convergence_state_canonical_values() -> None:
    assert {m.value for m in ConvergenceState} == {
        "improving", "stagnant", "oscillating", "converged",
    }


def test_budget_mode_canonical_values() -> None:
    assert {m.value for m in BudgetMode} == {
        "auto", "minimal", "balanced", "unlimited",
    }


# ---------------------------------------------------------------------------
# TaskClassification
# ---------------------------------------------------------------------------


def test_task_classification_defaults() -> None:
    tc = TaskClassification(
        complexity=EffortLevel.LOW, domain="x", estimated_tokens=100
    )
    assert tc.confidence == 0.5
    assert tc.matched_signals == ()


def test_task_classification_to_dict_round_trip() -> None:
    tc = TaskClassification(
        complexity=EffortLevel.HIGH,
        domain="security",
        estimated_tokens=2000,
        confidence=0.9,
        matched_signals=("audit", "secure"),
    )
    d = tc.to_dict()
    assert d == {
        "complexity": "high",
        "domain": "security",
        "estimated_tokens": 2000,
        "confidence": 0.9,
        "matched_signals": ["audit", "secure"],
    }


def test_task_classification_rejects_negative_tokens() -> None:
    with pytest.raises(ValueError, match=r"estimated_tokens"):
        TaskClassification(
            complexity=EffortLevel.LOW, domain="x", estimated_tokens=-1
        )


def test_task_classification_rejects_empty_domain() -> None:
    with pytest.raises(ValueError, match=r"domain"):
        TaskClassification(
            complexity=EffortLevel.LOW, domain="", estimated_tokens=10
        )


def test_task_classification_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match=r"confidence"):
        TaskClassification(
            complexity=EffortLevel.LOW,
            domain="x",
            estimated_tokens=10,
            confidence=1.1,
        )
    with pytest.raises(ValueError, match=r"confidence"):
        TaskClassification(
            complexity=EffortLevel.LOW,
            domain="x",
            estimated_tokens=10,
            confidence=-0.1,
        )


def test_task_classification_is_frozen() -> None:
    tc = TaskClassification(
        complexity=EffortLevel.LOW, domain="x", estimated_tokens=10
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        tc.domain = "y"            # type: ignore[misc]


def test_task_classification_has_slots() -> None:
    tc = TaskClassification(
        complexity=EffortLevel.LOW, domain="x", estimated_tokens=10
    )
    assert not hasattr(tc, "__dict__")


# ---------------------------------------------------------------------------
# QualityScore
# ---------------------------------------------------------------------------


def test_quality_score_valid() -> None:
    q = QualityScore(score=0.8, delta_from_previous=0.1, iteration=2)
    d = q.to_dict()
    assert d == {"score": 0.8, "delta_from_previous": 0.1, "iteration": 2}


def test_quality_score_rejects_score_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"score"):
        QualityScore(score=1.5, delta_from_previous=0.0, iteration=0)


def test_quality_score_rejects_delta_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"delta_from_previous"):
        QualityScore(score=0.5, delta_from_previous=2.0, iteration=0)


def test_quality_score_rejects_negative_iteration() -> None:
    with pytest.raises(ValueError, match=r"iteration"):
        QualityScore(score=0.5, delta_from_previous=0.0, iteration=-1)


def test_quality_score_is_frozen() -> None:
    q = QualityScore(score=0.5, delta_from_previous=0.0, iteration=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.score = 0.9            # type: ignore[misc]


# ---------------------------------------------------------------------------
# AmplifierConfig — defaults + validation
# ---------------------------------------------------------------------------


def test_amplifier_config_defaults() -> None:
    cfg = AmplifierConfig()
    assert cfg.max_iterations == 4
    assert cfg.convergence_threshold == 0.95
    assert cfg.budget_mode is BudgetMode.AUTO
    assert cfg.goal_reinjection_interval == 5
    assert cfg.effort_router == "heuristic"
    assert cfg.tool_selector == "heuristic"
    assert cfg.observability_callback is None
    assert cfg.escalate_low_confidence is False


def test_amplifier_config_default_observability_callback_is_none() -> None:
    assert AmplifierConfig().observability_callback is None


def test_amplifier_config_default_escalate_low_confidence_is_false() -> None:
    assert AmplifierConfig().escalate_low_confidence is False


def test_amplifier_config_accepts_callable_callback() -> None:
    cb: ObservabilityCallback = lambda e, p: None  # noqa: E731
    cfg = AmplifierConfig(observability_callback=cb)
    assert cfg.observability_callback is cb


def test_amplifier_config_accepts_class_with_call() -> None:
    class Sink:
        def __call__(self, e: AmplifierEvent, p: dict[str, object]) -> None:
            pass

    sink = Sink()
    cfg = AmplifierConfig(observability_callback=sink)
    assert cfg.observability_callback is sink


@pytest.mark.parametrize("bad", [42, "lambda", object(), 3.14, [1]])
def test_amplifier_config_rejects_non_callable_callback(bad: object) -> None:
    with pytest.raises(TypeError, match=r"observability_callback must be callable"):
        AmplifierConfig(observability_callback=bad)            # type: ignore[arg-type]


def test_amplifier_config_rejects_string_for_escalate_low_confidence() -> None:
    with pytest.raises(TypeError, match=r"escalate_low_confidence must be bool"):
        AmplifierConfig(escalate_low_confidence="yes")        # type: ignore[arg-type]


def test_amplifier_config_rejects_int_for_escalate_low_confidence() -> None:
    # Bool is a subclass of int — 1 IS technically a bool-ish value, but
    # ``isinstance(1, bool)`` is False, so plain ints are rejected.
    with pytest.raises(TypeError, match=r"escalate_low_confidence must be bool"):
        AmplifierConfig(escalate_low_confidence=1)            # type: ignore[arg-type]


def test_amplifier_config_rejects_string_budget_mode() -> None:
    """A-4 strict — no auto-coerce in __post_init__."""
    with pytest.raises(TypeError, match=r"budget_mode must be a BudgetMode enum"):
        AmplifierConfig(budget_mode="auto")            # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, 11, -1, 100])
def test_amplifier_config_rejects_bad_max_iterations(bad: int) -> None:
    with pytest.raises(ValueError, match=r"max_iterations"):
        AmplifierConfig(max_iterations=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_amplifier_config_rejects_bad_convergence_threshold(bad: float) -> None:
    with pytest.raises(ValueError, match=r"convergence_threshold"):
        AmplifierConfig(convergence_threshold=bad)


def test_amplifier_config_rejects_zero_goal_reinjection_interval() -> None:
    with pytest.raises(ValueError, match=r"goal_reinjection_interval"):
        AmplifierConfig(goal_reinjection_interval=0)


def test_amplifier_config_rejects_unknown_router() -> None:
    with pytest.raises(ValueError, match=r"effort_router"):
        AmplifierConfig(effort_router="ml")


def test_amplifier_config_rejects_unknown_selector() -> None:
    with pytest.raises(ValueError, match=r"tool_selector"):
        AmplifierConfig(tool_selector="ml")


# ---------------------------------------------------------------------------
# AmplifierConfig.recall_limit (/ H6)
#
# .6.M binds `recall_limit` to the kernel's memory plane
# call: `_resolve_recall(query, limit=config.recall_limit)`. Default 3.
# Validation domain [1, 100] — chosen to allow batching (kernel-side cap is
# 8 KB per chunk via `MAX_RECALLED_TEXT_BYTES`, so 100 chunks ≈ 800 KB worst
# case, which is still bounded). Outside that range the user is almost
# certainly mis-configuring.
# ---------------------------------------------------------------------------


def test_amplifier_config_recall_limit_default_is_three() -> None:
    cfg = AmplifierConfig()
    assert cfg.recall_limit == 3


@pytest.mark.parametrize("good", [1, 2, 3, 5, 10, 50, 100])
def test_amplifier_config_recall_limit_accepts_valid_range(good: int) -> None:
    cfg = AmplifierConfig(recall_limit=good)
    assert cfg.recall_limit == good


def test_amplifier_config_recall_limit_rejects_zero() -> None:
    with pytest.raises(ValueError, match=r"recall_limit must be in \[1, 100\]"):
        AmplifierConfig(recall_limit=0)


def test_amplifier_config_recall_limit_rejects_negative() -> None:
    with pytest.raises(ValueError, match=r"recall_limit must be in \[1, 100\]"):
        AmplifierConfig(recall_limit=-1)


def test_amplifier_config_recall_limit_rejects_too_large() -> None:
    with pytest.raises(ValueError, match=r"recall_limit must be in \[1, 100\]"):
        AmplifierConfig(recall_limit=101)


def test_amplifier_config_recall_limit_rejects_extreme_value() -> None:
    with pytest.raises(ValueError, match=r"recall_limit must be in \[1, 100\]"):
        AmplifierConfig(recall_limit=10_000)


def test_amplifier_config_recall_limit_rejects_non_int() -> None:
    """Bool subclasses int but is rejected; floats and strings rejected too."""
    with pytest.raises(TypeError, match=r"recall_limit must be int"):
        AmplifierConfig(recall_limit=3.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"recall_limit must be int"):
        AmplifierConfig(recall_limit="3")  # type: ignore[arg-type]
    # bool is technically a subclass of int — refuse to silently coerce
    # ``True`` to 1 (defensive against TOML deserialization quirks).
    with pytest.raises(TypeError, match=r"recall_limit must be int"):
        AmplifierConfig(recall_limit=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AmplifierConfig — frozen + slots
# ---------------------------------------------------------------------------


def test_amplifier_config_is_frozen() -> None:
    cfg = AmplifierConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.max_iterations = 5            # type: ignore[misc]


def test_amplifier_config_callback_field_is_immutable_on_frozen() -> None:
    cfg = AmplifierConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.observability_callback = lambda e, p: None            # type: ignore[misc]


def test_amplifier_config_has_slots() -> None:
    cfg = AmplifierConfig()
    assert not hasattr(cfg, "__dict__")


# ---------------------------------------------------------------------------
# AmplifierConfig.to_dict —  sentinel rule
# ---------------------------------------------------------------------------


def test_to_dict_callable_emits_sentinel() -> None:
    cfg = AmplifierConfig(observability_callback=lambda e, p: None)
    assert cfg.to_dict()["observability_callback"] == "<callable>"


def test_to_dict_none_emits_none() -> None:
    cfg = AmplifierConfig()
    assert cfg.to_dict()["observability_callback"] is None


def test_to_dict_no_repr_leak() -> None:
    """No closure or function repr leaks into serialized form."""
    secret = "REDACT_ME_SECRET_LAMBDA_BODY"
    cfg = AmplifierConfig(observability_callback=lambda e, p: secret)
    rendered = json.dumps(cfg.to_dict())
    assert "lambda" not in rendered
    assert "<function" not in rendered
    assert secret not in rendered


def test_to_dict_callable_round_trip_via_validate_config_raises() -> None:
    """The sentinel string is a one-way breadcrumb (.5).

    Passing it back through any reconstructor must NOT yield a callable.
    Here we instantiate AmplifierConfig directly with the sentinel string —
    that fails the callable check with TypeError.
    """
    with pytest.raises(TypeError, match=r"observability_callback must be callable"):
        AmplifierConfig(observability_callback="<callable>")            # type: ignore[arg-type]


def test_to_dict_complete_keys() -> None:
    d = AmplifierConfig().to_dict()
    assert set(d) == {
        "max_iterations",
        "convergence_threshold",
        "budget_mode",
        "goal_reinjection_interval",
        "effort_router",
        "tool_selector",
        "observability_callback",
        "escalate_low_confidence",
        "recall_limit",
        "disabled_ips",
        "ip_order",
        "persona",
    }


def test_to_dict_includes_recall_limit_default() -> None:
    d = AmplifierConfig().to_dict()
    assert d["recall_limit"] == 3


def test_to_dict_includes_recall_limit_override() -> None:
    d = AmplifierConfig(recall_limit=7).to_dict()
    assert d["recall_limit"] == 7


def test_to_dict_includes_dashboard_ip_state() -> None:
    d = AmplifierConfig(disabled_ips=("kernel",), ip_order=("kernel",)).to_dict()
    assert d["disabled_ips"] == ["kernel"]
    assert d["ip_order"] == ["kernel"]


def test_disabled_ips_must_be_tuple_of_non_empty_str() -> None:
    """Covers types.py validator branch (line 480) — wrong-type rejection."""
    import pytest

    with pytest.raises(TypeError, match="disabled_ips must be a tuple"):
        AmplifierConfig(disabled_ips=["kernel"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="disabled_ips must be a tuple"):
        AmplifierConfig(disabled_ips=("",))
    with pytest.raises(TypeError, match="disabled_ips must be a tuple"):
        AmplifierConfig(disabled_ips=(123,))  # type: ignore[arg-type]


def test_ip_order_must_be_tuple_of_non_empty_str() -> None:
    """Covers types.py validator branch (line 484) — wrong-type rejection."""
    import pytest

    with pytest.raises(TypeError, match="ip_order must be a tuple"):
        AmplifierConfig(ip_order=["kernel"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ip_order must be a tuple"):
        AmplifierConfig(ip_order=("",))
    with pytest.raises(TypeError, match="ip_order must be a tuple"):
        AmplifierConfig(ip_order=(123,))  # type: ignore[arg-type]


def test_to_dict_budget_mode_is_string() -> None:
    d = AmplifierConfig().to_dict()
    assert d["budget_mode"] == "auto"
    assert isinstance(d["budget_mode"], str)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_allowed_routers_is_tuple() -> None:
    assert isinstance(_ALLOWED_ROUTERS, tuple)
    assert "heuristic" in _ALLOWED_ROUTERS


def test_allowed_selectors_is_tuple() -> None:
    assert isinstance(_ALLOWED_SELECTORS, tuple)
    assert "heuristic" in _ALLOWED_SELECTORS


def test_allowed_config_fields_is_mappingproxy() -> None:
    """the allowed-fields constant must reject direct assignment."""
    assert isinstance(_ALLOWED_CONFIG_FIELDS, MappingProxyType)
    with pytest.raises(TypeError):
        _ALLOWED_CONFIG_FIELDS["foo"] = int            # type: ignore[index]


def test_allowed_config_fields_canonical_keys() -> None:
    expected = {
        "max_iterations",
        "convergence_threshold",
        "budget_mode",
        "goal_reinjection_interval",
        "effort_router",
        "tool_selector",
        "observability_callback",
        "escalate_low_confidence",
        "recall_limit",
        "disabled_ips",
        "ip_order",
        "persona",
    }
    assert set(_ALLOWED_CONFIG_FIELDS) == expected


def test_allowed_config_fields_recall_limit_typed_as_int() -> None:
    """validate the type registry advertises int for recall_limit.

    The config layer keys off this type when accepting TOML values, so a
    drift here would silently re-permit booleans (``True`` → 1) which the
    dataclass-level check refuses.
    """
    assert _ALLOWED_CONFIG_FIELDS["recall_limit"] is int


# ---------------------------------------------------------------------------
# V2.1 — RecalledPattern + Outcome (.5.5)
# ---------------------------------------------------------------------------


def test_recalled_pattern_defaults() -> None:
    from agent_amplifier.types import RecalledPattern

    p = RecalledPattern(text="hello")
    assert p.text == "hello"
    assert p.score == 0.0
    assert p.tags == ()
    assert p.source == ""
    assert p.metadata == {}


def test_recalled_pattern_rich_construction() -> None:
    from agent_amplifier.types import RecalledPattern

    p = RecalledPattern(
        text="hi",
        score=0.42,
        tags=("project-rule",),
        source="cursor:.cursor/rules/python.mdc",
        metadata={"foo": "bar"},
    )
    assert p.score == 0.42
    assert p.tags == ("project-rule",)
    assert p.source.startswith("cursor:")
    assert p.metadata == {"foo": "bar"}


def test_recalled_pattern_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    from agent_amplifier.types import RecalledPattern

    p = RecalledPattern(text="hi")
    with pytest.raises(FrozenInstanceError):
        p.text = "bye"  # type: ignore[misc]


def test_recalled_pattern_metadata_immutable() -> None:
    """H2: metadata is wrapped in MappingProxyType — mutation raises.

    Without this guarantee, ``dataclasses.replace(pat, text=safe)`` shares
    the same dict reference across kernel-internal copies and user code,
    enabling spooky-action-at-a-distance bugs. The fix freezes metadata
    in ``__post_init__`` regardless of what the caller passed in.
    """
    from types import MappingProxyType

    from agent_amplifier.types import RecalledPattern

    p = RecalledPattern(text="hi", metadata={"k": "v"})
    assert isinstance(p.metadata, MappingProxyType)
    assert p.metadata["k"] == "v"
    with pytest.raises(TypeError):
        p.metadata["k"] = "evil"  # type: ignore[index]
    with pytest.raises(TypeError):
        p.metadata["new"] = "x"  # type: ignore[index]


def test_recalled_pattern_metadata_decoupled_from_source_dict() -> None:
    """H2: mutating the dict passed to RecalledPattern does NOT leak
    into the constructed pattern (defensive copy semantics).
    """
    from agent_amplifier.types import RecalledPattern

    src: dict[str, Any] = {"k": "v"}
    p = RecalledPattern(text="hi", metadata=src)
    src["k"] = "mutated"
    src["new"] = "added"
    assert p.metadata["k"] == "v"
    assert "new" not in p.metadata


def test_recalled_pattern_replace_preserves_metadata_immutability() -> None:
    """H2: dataclasses.replace produces a new pattern whose metadata
    is also frozen — no mutable leak through the replace path.
    """
    import dataclasses
    from types import MappingProxyType

    from agent_amplifier.types import RecalledPattern

    p = RecalledPattern(text="hi", metadata={"k": "v"})
    p2 = dataclasses.replace(p, text="bye")
    assert isinstance(p2.metadata, MappingProxyType)
    assert p2.metadata["k"] == "v"
    with pytest.raises(TypeError):
        p2.metadata["k"] = "evil"  # type: ignore[index]


def test_outcome_basic_construction() -> None:
    from agent_amplifier.types import EffortLevel, Outcome

    o = Outcome(
        query="q",
        effort=EffortLevel.MEDIUM,
        iterations=3,
        quality=0.75,
        converged=True,
        tokens_used=4096,
    )
    assert o.query == "q"
    assert o.effort is EffortLevel.MEDIUM
    assert o.iterations == 3
    assert o.quality == 0.75
    assert o.converged is True
    assert o.tokens_used == 4096


def test_outcome_to_dict_round_trips_fields() -> None:
    from agent_amplifier.types import EffortLevel, Outcome

    o = Outcome(
        query="x",
        effort=EffortLevel.HIGH,
        iterations=1,
        quality=1.0,
    )
    d = o.to_dict()
    assert d["effort"] == "high"
    assert d["iterations"] == 1
    assert d["quality"] == 1.0
    assert d["converged"] is False
    assert d["tokens_used"] == 0


def test_outcome_rejects_negative_iterations() -> None:
    from agent_amplifier.types import EffortLevel, Outcome

    with pytest.raises(ValueError, match="iterations must be >= 0"):
        Outcome(
            query="q",
            effort=EffortLevel.LOW,
            iterations=-1,
            quality=0.0,
        )


def test_outcome_rejects_quality_out_of_range() -> None:
    from agent_amplifier.types import EffortLevel, Outcome

    with pytest.raises(ValueError, match=r"quality must be in \[0,1\]"):
        Outcome(
            query="q",
            effort=EffortLevel.LOW,
            iterations=0,
            quality=1.5,
        )


def test_outcome_rejects_negative_tokens_used() -> None:
    from agent_amplifier.types import EffortLevel, Outcome

    with pytest.raises(ValueError, match="tokens_used must be >= 0"):
        Outcome(
            query="q",
            effort=EffortLevel.LOW,
            iterations=0,
            quality=0.5,
            tokens_used=-1,
        )


# ---------------------------------------------------------------------------
# — RecalledPattern boundary validation
# ---------------------------------------------------------------------------


def test_recalled_pattern_rejects_non_str_text() -> None:
    """text must be str, not int."""
    from agent_amplifier.types import RecalledPattern

    with pytest.raises(TypeError, match="text must be str"):
        RecalledPattern(text=12345)  # type: ignore[arg-type]


def test_recalled_pattern_rejects_score_out_of_range() -> None:
    """score must be in [0,1]."""
    from agent_amplifier.types import RecalledPattern

    with pytest.raises(ValueError, match=r"score must be in \[0,1\]"):
        RecalledPattern(text="x", score=2.0)
    with pytest.raises(ValueError, match=r"score must be in \[0,1\]"):
        RecalledPattern(text="x", score=-0.1)


def test_recalled_pattern_rejects_non_numeric_score() -> None:
    """score must be a number, bool excluded."""
    from agent_amplifier.types import RecalledPattern

    with pytest.raises(TypeError, match="score must be a number"):
        RecalledPattern(text="x", score="0.5")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="score must be a number"):
        RecalledPattern(text="x", score=True)  # type: ignore[arg-type]


def test_recalled_pattern_rejects_list_tags() -> None:
    """tags must be tuple, not list."""
    from agent_amplifier.types import RecalledPattern

    with pytest.raises(TypeError, match="tags must be a tuple"):
        RecalledPattern(text="x", tags=["a"])  # type: ignore[arg-type]


def test_recalled_pattern_rejects_non_str_tag_element() -> None:
    """each tag element must be str."""
    from agent_amplifier.types import RecalledPattern

    with pytest.raises(TypeError, match="tags must be a tuple"):
        RecalledPattern(text="x", tags=("a", 1))  # type: ignore[arg-type]


def test_recalled_pattern_rejects_non_str_source() -> None:
    """source must be str."""
    from agent_amplifier.types import RecalledPattern

    with pytest.raises(TypeError, match="source must be str"):
        RecalledPattern(text="x", source=42)  # type: ignore[arg-type]


def test_recalled_pattern_rejects_non_mapping_metadata() -> None:
    """metadata must be a Mapping."""
    from agent_amplifier.types import RecalledPattern

    with pytest.raises(TypeError, match="metadata must be a Mapping"):
        RecalledPattern(text="x", metadata="not-a-mapping")  # type: ignore[arg-type]


def test_recalled_pattern_accepts_int_score() -> None:
    """Integer 0 / 1 are common literals for score; coerce-friendly."""
    from agent_amplifier.types import RecalledPattern

    # No error for int score in valid range.
    p = RecalledPattern(text="x", score=1)
    assert p.score == 1


# ---------------------------------------------------------------------------
# — Outcome boundary validation
# ---------------------------------------------------------------------------


def test_outcome_rejects_non_str_query() -> None:
    """query must be str."""
    from agent_amplifier.types import EffortLevel, Outcome

    with pytest.raises(TypeError, match="query must be str"):
        Outcome(
            query=123,  # type: ignore[arg-type]
            effort=EffortLevel.LOW,
            iterations=0,
            quality=0.5,
        )


def test_outcome_rejects_non_effort_effort() -> None:
    """effort must be EffortLevel."""
    from agent_amplifier.types import Outcome

    with pytest.raises(TypeError, match="effort must be EffortLevel"):
        Outcome(
            query="q",
            effort="low",  # type: ignore[arg-type]
            iterations=0,
            quality=0.5,
        )


def test_outcome_rejects_bool_iterations() -> None:
    """iterations must be int (bool subclass excluded)."""
    from agent_amplifier.types import EffortLevel, Outcome

    with pytest.raises(TypeError, match="iterations must be int"):
        Outcome(
            query="q",
            effort=EffortLevel.LOW,
            iterations=True,  # type: ignore[arg-type]
            quality=0.5,
        )


def test_outcome_rejects_str_quality() -> None:
    """quality must be a number."""
    from agent_amplifier.types import EffortLevel, Outcome

    with pytest.raises(TypeError, match="quality must be a number"):
        Outcome(
            query="q",
            effort=EffortLevel.LOW,
            iterations=0,
            quality="0.5",  # type: ignore[arg-type]
        )


def test_outcome_rejects_non_bool_converged() -> None:
    """converged must be bool, not str."""
    from agent_amplifier.types import EffortLevel, Outcome

    with pytest.raises(TypeError, match="converged must be bool"):
        Outcome(
            query="q",
            effort=EffortLevel.LOW,
            iterations=0,
            quality=0.5,
            converged="yes",  # type: ignore[arg-type]
        )


def test_outcome_rejects_bool_tokens_used() -> None:
    """tokens_used must be int (bool subclass excluded)."""
    from agent_amplifier.types import EffortLevel, Outcome

    with pytest.raises(TypeError, match="tokens_used must be int"):
        Outcome(
            query="q",
            effort=EffortLevel.LOW,
            iterations=0,
            quality=0.5,
            tokens_used=True,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# — StepEnvelope.extras / .budget immutability
# ---------------------------------------------------------------------------


def test_step_envelope_extras_is_immutable() -> None:
    """env.extras must reject mutation."""
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        env = amp.before_step("hi")
        with pytest.raises(TypeError):
            env.extras["new_key"] = "should-not-stick"  # type: ignore[index]
    finally:
        amp.close()


def test_step_envelope_budget_is_immutable() -> None:
    """env.budget must reject mutation."""
    from agent_amplifier.kernel import AgentAmplifier

    amp = AgentAmplifier()
    try:
        env = amp.before_step("hi")
        with pytest.raises(TypeError):
            env.budget["allocated"] = 999_999_999  # type: ignore[index]
    finally:
        amp.close()


def test_step_envelope_idempotent_freeze_when_passed_mapping_proxy() -> None:
    """passing an already-frozen MappingProxyType must NOT
    re-wrap it.  Idempotency proof — constructor short-circuits.
    """
    from types import MappingProxyType

    from agent_amplifier.kernel import StepEnvelope
    from agent_amplifier.types import EffortLevel, TaskClassification

    cls = TaskClassification(
        complexity=EffortLevel.LOW,
        domain="general",
        estimated_tokens=100,
        confidence=1.0,
    )
    pre_budget = MappingProxyType({"mode": "auto", "allocated": 1000})
    pre_extras = MappingProxyType({"key": "val"})
    env = StepEnvelope(
        classification=cls,
        thinking_trigger=None,
        phase="EXPLORE",
        persona="default",
        recommended_tools=(),
        recommended_groups=(),
        iteration=0,
        step_id=0,
        envelope="env",
        budget=pre_budget,
        recalled_patterns=(),
        suggested_model=None,
        extras=pre_extras,
    )
    # Same identity — not re-wrapped.
    assert env.budget is pre_budget
    assert env.extras is pre_extras
