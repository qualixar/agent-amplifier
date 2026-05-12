# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Agent Amplifier — Intelligent Tool Selector (IP-11, MoE routing).

Mixture-of-Experts inspired router: given a free-text query, return the
list of *group names* that match the query. Combined with the always-on
:data:`SHARED_TOOLS`, these groups produce a focused, ordered tool
recommendation — shrinking the active tool surface so context isn't
bloated by irrelevant descriptions.

Theoretical anchor:
    * **DeepSeekMoE** — arXiv 2401.06066. Two-tier expert structure:
      shared experts (always active) + routed experts (top-K per token).
    * Tool-shortlisting heuristic — keyword-driven group selection is a
      well-known agent practice; we route a small, ordered shortlist
      instead of exposing every tool for every query.

V2.0 fixes:
    * **** — Pre-compiled alternation regex per group; same
      longest-first sort + word-boundary protection as effort_router.
    * **B-10** — :data:`TOOL_GROUPS` is wrapped in
      :class:`types.MappingProxyType` so direct mutation raises
      ``TypeError``.
    * **** — Like effort_router, this module uses regex alternation
      not tokenization, and deliberately does NOT depend on
      ``agent_amplifier._internal.keyword_set``.

Non-dependency on effort_router:
    The ``_build_tier_regex`` helper has the same shape as the one in
    effort_router. We *internalize* an equivalent helper here rather than
    depend on a private (leading-underscore) symbol from a sibling module
    — keeps the import graph cycle-free and the public API minimal.

Performance budget (.8):
    * Module-load: < 30 ms.
    * Per-call P99: < 1 ms.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final

# ---------------------------------------------------------------------------
# 1. Tool groups (frozen V1; B-3 + B-10)
# ---------------------------------------------------------------------------

#: Always-on toolset. Never vetoed by :func:`should_call`.
SHARED_TOOLS: Final[tuple[str, ...]] = (
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
)


_RAW_TOOL_GROUPS: dict[str, tuple[str, ...]] = {
    "research": (
        "WebSearch",
        "WebFetch",
        "mcp__superlocalmemory__recall",
        "mcp__semantic-scholar__search",
        "mcp__context7__resolve-library-id",
        "mcp__context7__get-library-docs",
        "mcp__microsoft-docs__microsoft_docs_search",
        "mcp__microsoft-docs__microsoft_docs_fetch",
        "mcp__github__search_repositories",
        "mcp__zotero__search",
    ),
    "media": (
        "mcp__gemini__gemini-generate-image",
        "mcp__gemini__gemini-generate-video",
        "mcp__gamma__create_gamma",
        "mcp__fal__minimax_video",
        "mcp__pencil__draw",
    ),
    "data": (
        "mcp__duckdb__query",
        "mcp__sqlite__query",
        "mcp__excel__read",
        "mcp__excel__write",
        "mcp__powerpoint__create",
    ),
    "deploy": (
        "mcp__github__create_pull_request",
        "mcp__github__push",
        "mcp__github__create_release",
        "mcp__npm__publish",
        "mcp__pypi__upload",
    ),
    "comms": (
        "mcp__google-workspace__gmail_send",
        "mcp__google-workspace__calendar_create",
        "mcp__slack__post_message",
    ),
}

#: Read-only mapping of group name → ordered tuple of tool identifiers.
#: Wrapped in :class:`MappingProxyType` so external mutation raises
#: ``TypeError`` (B-10 anti-drift).
TOOL_GROUPS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    _RAW_TOOL_GROUPS
)


# ---------------------------------------------------------------------------
# 2. Routing keywords + pre-compiled regex
# ---------------------------------------------------------------------------

GROUP_KEYWORDS: Final[dict[str, frozenset[str]]] = {
    "research": frozenset(
        {
            "research", "find out", "what is", "compare", "literature",
            "paper", "papers", "arxiv", "documentation", "docs",
            "library", "framework", "best practice", "best practices",
            "state of the art", "sota",
        }
    ),
    "media": frozenset(
        {
            "image", "picture", "diagram", "chart", "video", "render",
            "generate visual", "thumbnail", "logo", "presentation",
            "slides",
        }
    ),
    "data": frozenset(
        {
            "database", "sql", "query", "table", "spreadsheet", "csv",
            "excel", "duckdb", "sqlite", "postgres", "rows", "columns",
            "join",
        }
    ),
    "deploy": frozenset(
        {
            "deploy", "publish", "release", "ship", "pull request", "pr",
            "merge", "tag", "version bump", "npm publish", "npm", "pypi",
        }
    ),
    "comms": frozenset(
        {
            "email", "gmail", "calendar", "schedule meeting", "slack",
            "send message", "notify", "announce",
        }
    ),
}


def _build_group_regex(kw_set: Iterable[str]) -> re.Pattern[str]:
    """Build a case-insensitive longest-first alternation regex.

    Mirrors ``effort_router._build_tier_regex`` semantics — duplicated
    on purpose so this module has no dependency on a private symbol of
    its sibling.
    """
    materialized = list(kw_set)
    if not materialized:
        return re.compile(r"(?!x)x")  # never-matches sentinel
    escaped = sorted(
        (re.escape(k) for k in materialized), key=len, reverse=True
    )
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


_GROUP_REGEX: Final[Mapping[str, re.Pattern[str]]] = MappingProxyType(
    {name: _build_group_regex(kws) for name, kws in GROUP_KEYWORDS.items()}
)

# Reverse map: tool → group name. Built once. ``None`` means SHARED tool.
_TOOL_TO_GROUP: Final[Mapping[str, str]] = MappingProxyType(
    {tool: group for group, tools in TOOL_GROUPS.items() for tool in tools}
)


# ---------------------------------------------------------------------------
# 3. Public API
# ---------------------------------------------------------------------------


def classify_tools(query: str) -> list[str]:
    """Return the list of group names whose keyword regex matches ``query``.

    SHARED_TOOLS are NOT included in the return value — they are appended
    by :func:`recommend_tools`.

    Performance: P99 < 1 ms (.8).
    """
    if not query or not query.strip():
        return []
    return [
        name for name, pat in _GROUP_REGEX.items() if pat.search(query)
    ]


def recommend_tools(query: str, available_tools: list[str]) -> list[str]:
    """Compose the final ordered tool recommendation.

    Steps (.5):
        1. groups = :func:`classify_tools` (query)
        2. ordered = SHARED_TOOLS ++ tools from each matched group
        3. dedupe preserving first occurrence
        4. intersect with ``available_tools`` (preserve step-3 order)
    """
    groups = classify_tools(query)
    available_set = set(available_tools)

    ordered: list[str] = []
    seen: set[str] = set()

    for shared in SHARED_TOOLS:
        if shared in available_set and shared not in seen:
            ordered.append(shared)
            seen.add(shared)

    for group in groups:
        for tool in TOOL_GROUPS[group]:
            if tool in available_set and tool not in seen:
                ordered.append(tool)
                seen.add(tool)

    return ordered


def should_call(tool_name: str, query: str) -> bool:
    """Per-tool veto for hook adapters (PreToolUse).

    Returns True (allow) by default. Returns False ONLY if BOTH:
        (a) ``tool_name`` is a routed (non-shared) MCP tool, AND
        (b) the group containing ``tool_name`` was NOT in
            :func:`classify_tools` (query).

    Default-allow protects unfamiliar tools — we never veto SHARED_TOOLS
    (locked B-3 + LLD §2.5).
    """
    if tool_name in SHARED_TOOLS:
        return True
    group = _TOOL_TO_GROUP.get(tool_name)
    if group is None:
        # Unknown tool — default-allow.
        return True
    return group in classify_tools(query)


__all__ = [
    "GROUP_KEYWORDS",
    "SHARED_TOOLS",
    "TOOL_GROUPS",
    "classify_tools",
    "recommend_tools",
    "should_call",
]
