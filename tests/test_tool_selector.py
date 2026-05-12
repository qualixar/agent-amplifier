# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.tool_selector`` (IP-11, MoE routing).

Coverage targets (.2):
    * 25+ unit cases (10 routing, 4 veto, 4 integration, 2 perf, 2
      adversarial, 3 property)
    *  + B-10: pre-compiled regex per group; ``TOOL_GROUPS`` is a
      ``MappingProxyType``.
    * non-dependency on ``_internal/keyword_set``.
"""

from __future__ import annotations

import re
from types import MappingProxyType

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_amplifier import tool_selector as TS
from agent_amplifier.tool_selector import (
    SHARED_TOOLS,
    TOOL_GROUPS,
    classify_tools,
    recommend_tools,
    should_call,
)

# ---------------------------------------------------------------------------
# 0. Module-level invariants (B-10 + )
# ---------------------------------------------------------------------------


class TestModuleInvariants:
    def test_tool_groups_is_mapping_proxy(self) -> None:
        # B-10 / Test F-02 — ``TOOL_GROUPS`` must be immutable at the proxy
        # level so monkey-patches in tests don't bleed across the suite.
        assert isinstance(TOOL_GROUPS, MappingProxyType)
        with pytest.raises(TypeError):
            TOOL_GROUPS["evil"] = ()  # type: ignore[index]

    def test_shared_tools_is_tuple(self) -> None:
        assert isinstance(SHARED_TOOLS, tuple)
        assert all(isinstance(t, str) for t in SHARED_TOOLS)

    def test_group_regex_is_precompiled(self) -> None:

        assert hasattr(TS, "_GROUP_REGEX")
        for name in TOOL_GROUPS:
            pat = TS._GROUP_REGEX[name]
            assert isinstance(pat, re.Pattern)

    def test_does_not_depend_on_internal_keyword_set(self) -> None:
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(TS))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    imports.add(f"{node.module}.{alias.name}")
                    imports.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
        assert "agent_amplifier._internal.keyword_set" not in imports

    def test_module_all_exports(self) -> None:
        for name in (
            "SHARED_TOOLS",
            "TOOL_GROUPS",
            "classify_tools",
            "recommend_tools",
            "should_call",
        ):
            assert name in TS.__all__


# ---------------------------------------------------------------------------
# 1. classify_tools — group routing (10 unit cases)
# ---------------------------------------------------------------------------


class TestClassifyTools:
    @pytest.mark.parametrize(
        ("query", "expected_groups"),
        [
            ("research arxiv papers on RAG", {"research"}),
            ("compare two libraries", {"research"}),
            ("generate an image of a logo", {"media"}),
            ("create a slides deck", {"media"}),
            ("query the postgres database", {"data"}),
            ("read this excel spreadsheet", {"data"}),
            ("publish a release on npm", {"deploy"}),
            ("ship this PR", {"deploy"}),
            ("send this gmail to the team", {"comms"}),
            ("schedule meeting on calendar", {"comms"}),
        ],
    )
    def test_single_group_routing(
        self, query: str, expected_groups: set[str]
    ) -> None:
        groups = set(classify_tools(query))
        assert expected_groups.issubset(groups), (
            f"{query!r} → {groups}, expected ⊇ {expected_groups}"
        )

    def test_multi_group_query(self) -> None:
        # "research and publish" → research + deploy
        groups = set(classify_tools("research best practices and publish"))
        assert {"research", "deploy"}.issubset(groups)

    def test_empty_query_returns_empty(self) -> None:
        assert classify_tools("") == []

    def test_whitespace_query_returns_empty(self) -> None:
        assert classify_tools("   \n\t   ") == []

    def test_no_keyword_match_returns_empty(self) -> None:
        # "hello world" — no group keywords → no groups.
        assert classify_tools("hello world") == []


# ---------------------------------------------------------------------------
# 2. recommend_tools — composition (4 integration cases)
# ---------------------------------------------------------------------------


class TestRecommendTools:
    def test_includes_shared_tools_when_available(self) -> None:
        available = ["Read", "Write", "Edit", "Bash", "WebSearch"]
        rec = recommend_tools("research the docs", available)
        # SHARED_TOOLS that are available must appear before group tools.
        for shared in SHARED_TOOLS:
            if shared in available:
                assert shared in rec

    def test_filters_to_available_only(self) -> None:
        # ``WebSearch`` requested, but not in available list.
        available = ["Read", "Write", "Bash"]
        rec = recommend_tools("research papers", available)
        assert all(t in available for t in rec)
        assert "WebSearch" not in rec

    def test_dedupes_preserving_order(self) -> None:
        # SHARED + a group tool — no duplicates, SHARED first.
        available = [*SHARED_TOOLS, "WebSearch"]
        rec = recommend_tools("research", available)
        assert len(rec) == len(set(rec))
        # Shared tools precede the routed tools.
        ws_idx = rec.index("WebSearch")
        for shared in SHARED_TOOLS:
            if shared in rec:
                assert rec.index(shared) < ws_idx

    def test_empty_query_returns_only_available_shared(self) -> None:
        # No groups match → only SHARED_TOOLS appear.
        available = ["Read", "WebSearch"]
        rec = recommend_tools("", available)
        # Only ``Read`` is shared AND available.
        assert "Read" in rec
        assert "WebSearch" not in rec


# ---------------------------------------------------------------------------
# 3. should_call — per-tool veto (4 cases)
# ---------------------------------------------------------------------------


class TestShouldCall:
    def test_shared_tool_always_allowed(self) -> None:
        # SHARED tools are NEVER vetoed.
        for shared in SHARED_TOOLS:
            assert should_call(shared, "anything") is True

    def test_routed_tool_in_matching_group_allowed(self) -> None:
        # Research query + research-group tool → allowed.
        assert should_call("WebSearch", "research papers on this topic") is True

    def test_routed_tool_in_unmatched_group_vetoed(self) -> None:
        # Media query + research-group tool → vetoed.
        assert should_call("WebSearch", "render an image") is False

    def test_unknown_tool_default_allow(self) -> None:
        # Unknown (not in any group, not in SHARED) → default-allow.
        # Locked B-3 / LLD §2.5: never veto unfamiliar tools.
        assert should_call("UnknownTool42", "research") is True


# ---------------------------------------------------------------------------
# 4. Property-based (3 cases)
# ---------------------------------------------------------------------------


class TestProperties:
    @given(st.text(min_size=0, max_size=512))
    @settings(
        deadline=200,
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_classify_tools_is_deterministic(self, query: str) -> None:
        assert classify_tools(query) == classify_tools(query)

    @given(
        st.text(min_size=0, max_size=512),
        st.lists(
            st.sampled_from(
                [*SHARED_TOOLS, "WebSearch", "WebFetch", "UnknownX"]
            ),
            min_size=0,
            max_size=10,
            unique=True,
        ),
    )
    @settings(
        deadline=200,
        max_examples=80,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_recommend_tools_subset_of_available(
        self, query: str, available: list[str]
    ) -> None:
        rec = recommend_tools(query, available)
        assert set(rec).issubset(set(available))

    @given(st.text(min_size=1, max_size=256))
    @settings(
        deadline=200,
        max_examples=80,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_classify_tools_returns_subset_of_known_groups(
        self, query: str
    ) -> None:
        groups = classify_tools(query)
        assert set(groups).issubset(set(TOOL_GROUPS.keys()))


# ---------------------------------------------------------------------------
# 5. Adversarial (2 cases)
# ---------------------------------------------------------------------------


class TestBuildGroupRegexEmpty:
    """Cover the empty-keyword-set sentinel branch."""

    def test_empty_keyword_set_returns_never_match_regex(self) -> None:
        pat = TS._build_group_regex(frozenset())
        assert pat.search("anything at all") is None
        assert pat.search("") is None


class TestAdversarial:
    def test_long_input_does_not_blow_up(self) -> None:
        # Pathological large input — must complete in reasonable time.
        import time

        q = "research " * 10_000
        t0 = time.perf_counter()
        groups = classify_tools(q)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 200, f"{elapsed_ms:.2f} ms"
        assert "research" in groups

    def test_unicode_does_not_break(self) -> None:
        groups = classify_tools("research 论文 papers")
        assert "research" in groups
