# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier._internal.ctx_schema``.

Spec source: .15 — adapter-context boundary
validation. Bad shape → ``ValueError`` (caught by the kernel's outer
try/except per locked E-7 graceful degradation).
"""

from __future__ import annotations

import pytest

from agent_amplifier._internal.ctx_schema import (
    _ALLOWED_KEYS,
    _TYPED,
    validate_context,
)

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_none_returns_empty_dict() -> None:
    out = validate_context(None)
    assert out == {}
    assert isinstance(out, dict)


def test_empty_dict_passes_through() -> None:
    out = validate_context({})
    assert out == {}


def test_minimal_valid_ctx() -> None:
    ctx: dict[str, object] = {"available_tools": ["bash", "edit"]}
    out = validate_context(ctx)
    assert out == ctx
    # Defensive copy — mutating the result must not affect the input.
    assert out is not ctx


def test_full_valid_ctx() -> None:
    ctx: dict[str, object] = {
        "available_tools": ["bash"],
        "prev_output": "hello world",
        "chosen": "step",
        "issues": "",
        "amp_tokens_used": 100,
        "amp_recalled_patterns": [],
    }
    out = validate_context(ctx)
    assert out == ctx


# ---------------------------------------------------------------------------
# Defensive copy
# ---------------------------------------------------------------------------


def test_returns_new_dict_object() -> None:
    ctx: dict[str, object] = {"available_tools": ["x"]}
    out = validate_context(ctx)
    assert out is not ctx
    out["extra"] = "leak"
    assert "extra" not in ctx


# ---------------------------------------------------------------------------
# Type-check failures (.15 _TYPED)
# ---------------------------------------------------------------------------


def test_available_tools_must_be_list() -> None:
    with pytest.raises(ValueError, match=r"available_tools.*list"):
        validate_context({"available_tools": "bash"})


def test_available_tools_int_rejected() -> None:
    with pytest.raises(ValueError, match=r"available_tools.*list"):
        validate_context({"available_tools": 42})


def test_prev_output_must_be_str() -> None:
    with pytest.raises(ValueError, match=r"prev_output.*str"):
        validate_context({"prev_output": 42})


def test_chosen_must_be_str() -> None:
    with pytest.raises(ValueError, match=r"chosen.*str"):
        validate_context({"chosen": ["bash"]})


def test_issues_must_be_str() -> None:
    with pytest.raises(ValueError, match=r"issues.*str"):
        validate_context({"issues": []})


def test_amp_tokens_used_must_be_int() -> None:
    with pytest.raises(ValueError, match=r"amp_tokens_used.*int"):
        validate_context({"amp_tokens_used": "100"})


# ---------------------------------------------------------------------------
# Top-level shape failures
# ---------------------------------------------------------------------------


def test_non_dict_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"context must be dict or None"):
        validate_context("not a dict")            # type: ignore[arg-type]


def test_list_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"context must be dict or None"):
        validate_context([1, 2, 3])            # type: ignore[arg-type]


def test_int_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"context must be dict or None"):
        validate_context(42)            # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Forward-compat: unknown keys pass through
# ---------------------------------------------------------------------------


def test_unknown_key_passes_through() -> None:
    """.15: unknown keys are passed through (forward compat)."""
    ctx: dict[str, object] = {"available_tools": [], "future_field": "ok"}
    out = validate_context(ctx)
    assert out["future_field"] == "ok"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_deeply_nested_list_in_available_tools_accepted() -> None:
    """``list`` is the only required check — element types are not enforced."""
    ctx: dict[str, object] = {"available_tools": [["nested"]]}
    out = validate_context(ctx)
    assert out == ctx


def test_partial_ctx_validates_only_present_keys() -> None:
    """Absent keys are not validated — caller may omit any/all."""
    ctx: dict[str, object] = {"prev_output": "ok"}
    out = validate_context(ctx)
    assert out == ctx


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_allowed_keys_is_set() -> None:
    assert isinstance(_ALLOWED_KEYS, set)
    expected = {
        "available_tools",
        "prev_output",
        "chosen",
        "issues",
        "amp_tokens_used",
        "amp_recalled_patterns",
    }
    assert expected == _ALLOWED_KEYS


def test_typed_keys_subset_of_allowed() -> None:
    # _TYPED exposes only those keys whose type is enforced — must be
    # a subset of the allowed key set.
    assert set(_TYPED).issubset(_ALLOWED_KEYS)
