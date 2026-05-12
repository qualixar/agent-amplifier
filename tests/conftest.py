# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Project-wide pytest fixtures.

Per .5:
    1. autouse `_assert_globals_unchanged` snapshots module-level constants
       and asserts equality on teardown — catches accidental mutation of
       MappingProxyType-wrapped dicts via their backing dict.
    2. `_reset_warnings` restores the warnings filter list around each test
       so an in-test `warnings.simplefilter(...)` cannot leak into the next
       test under `pytest-randomly`.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Generator
from typing import Any

import pytest
from hypothesis import HealthCheck, settings

# Hypothesis profiles (closes methodology gap MED-16):
#   default     — interactive use, 100 examples, fast
#   ci          — CI runs, 200 examples, allow slow data fixtures
#   ci_parallel — adversarial concurrency fuzz, 200 examples (parallel handled
#                 by xdist; hypothesis's `derandomize=True` + larger budget
#                 catches state-machine flakes before they reach prod)
# Select profile via HYPOTHESIS_PROFILE env var (defaults to "default").
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci_parallel",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(autouse=True)
def _reset_warnings() -> Generator[None, None, None]:
    """Snapshot + restore the warnings filter list around every test."""
    saved = warnings.filters[:]
    try:
        yield
    finally:
        warnings.filters[:] = saved


@pytest.fixture(autouse=True)
def _assert_globals_unchanged() -> Generator[None, None, None]:
    """Snapshot module-level constants; assert they remain unchanged.

    This guards against:
      * A test mutating the backing dict of a `MappingProxyType` constant
        (proxy reflects mutation — silent drift).
      * A test reassigning a module-level tuple/frozenset.

    Per .4 anti-drift rule + CRIT Flaw 3 mitigation: we also
    snapshot the IDENTITY of compiled-regex tuples so reassignment is caught
    even when value-equality is preserved.
    """
    # Import inside the fixture so test collection works even if a target
    # module is being introduced and not yet importable.
    snapshots: dict[str, Any] = {}
    try:
        from agent_amplifier import types as _types

        snapshots["types._ALLOWED_ROUTERS"] = _types._ALLOWED_ROUTERS
        snapshots["types._ALLOWED_SELECTORS"] = _types._ALLOWED_SELECTORS
        # MappingProxyType: snapshot a plain-dict copy for equality check.
        snapshots["types._ALLOWED_CONFIG_FIELDS"] = dict(
            _types._ALLOWED_CONFIG_FIELDS
        )
        snapshots["types._ALLOWED_CONFIG_FIELDS_ID"] = id(
            _types._ALLOWED_CONFIG_FIELDS
        )
    except Exception:
        # types.py not importable yet; skip those snapshots.
        pass

    try:
        from agent_amplifier._internal import redact as _redact

        snapshots["redact._PATTERNS"] = _redact._PATTERNS
        snapshots["redact._PATTERNS_id"] = id(_redact._PATTERNS)
    except Exception:
        pass

    try:
        from agent_amplifier._internal import keyword_set as _kw

        snapshots["keyword_set.STOPWORDS"] = _kw.STOPWORDS
        snapshots["keyword_set._KEYWORD_RE_id"] = id(_kw._KEYWORD_RE)
        snapshots[
            "keyword_set.MAX_OUTPUT_CHARS_FOR_ANALYSIS"
        ] = _kw.MAX_OUTPUT_CHARS_FOR_ANALYSIS
    except Exception:
        pass

    try:
        from agent_amplifier._internal import ctx_schema as _cs

        snapshots["ctx_schema._ALLOWED_KEYS"] = set(_cs._ALLOWED_KEYS)
        snapshots["ctx_schema._TYPED"] = dict(_cs._TYPED)
    except Exception:
        pass

    yield

    # Re-import lazily; skip checks for any module that wasn't snapshotted.
    if "types._ALLOWED_ROUTERS" in snapshots:
        from agent_amplifier import types as _types

        assert (
            snapshots["types._ALLOWED_ROUTERS"] == _types._ALLOWED_ROUTERS
        ), "types._ALLOWED_ROUTERS mutated during test"
        assert (
            snapshots["types._ALLOWED_SELECTORS"]
            == _types._ALLOWED_SELECTORS
        ), "types._ALLOWED_SELECTORS mutated during test"
        assert (
            dict(_types._ALLOWED_CONFIG_FIELDS)
            == snapshots["types._ALLOWED_CONFIG_FIELDS"]
        ), "types._ALLOWED_CONFIG_FIELDS backing-dict mutated during test"
        # Identity check — catches reassignment even if equal-by-value.
        assert (
            id(_types._ALLOWED_CONFIG_FIELDS)
            == snapshots["types._ALLOWED_CONFIG_FIELDS_ID"]
        ), "types._ALLOWED_CONFIG_FIELDS rebound during test"

    if "redact._PATTERNS" in snapshots:
        from agent_amplifier._internal import redact as _redact

        assert (
            snapshots["redact._PATTERNS"] == _redact._PATTERNS
        ), "redact._PATTERNS mutated"
        # CRIT Flaw 3 mitigation — id stability blocks tuple-rebind sneak.
        assert (
            id(_redact._PATTERNS) == snapshots["redact._PATTERNS_id"]
        ), "redact._PATTERNS rebound during test"

    if "keyword_set.STOPWORDS" in snapshots:
        from agent_amplifier._internal import keyword_set as _kw

        assert (
            snapshots["keyword_set.STOPWORDS"] == _kw.STOPWORDS
        ), "keyword_set.STOPWORDS mutated"
        assert (
            id(_kw._KEYWORD_RE) == snapshots["keyword_set._KEYWORD_RE_id"]
        ), "keyword_set._KEYWORD_RE rebound during test"
        assert (
            snapshots["keyword_set.MAX_OUTPUT_CHARS_FOR_ANALYSIS"]
            == _kw.MAX_OUTPUT_CHARS_FOR_ANALYSIS
        ), "keyword_set.MAX_OUTPUT_CHARS_FOR_ANALYSIS mutated"

    if "ctx_schema._ALLOWED_KEYS" in snapshots:
        from agent_amplifier._internal import ctx_schema as _cs

        assert (
            set(_cs._ALLOWED_KEYS) == snapshots["ctx_schema._ALLOWED_KEYS"]
        ), "ctx_schema._ALLOWED_KEYS mutated"
        assert (
            dict(_cs._TYPED) == snapshots["ctx_schema._TYPED"]
        ), "ctx_schema._TYPED mutated"
