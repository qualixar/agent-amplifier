# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.adapter_base`` (,
/16/22).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Concrete test adapters
# ---------------------------------------------------------------------------


def _make_concrete_adapter(name: str = "fakefw"):
    """Helper: create a concrete subclass with given framework_name."""
    from agent_amplifier.adapter_base import AdapterBase

    class _A(AdapterBase):
        framework_name = name
        version = "1.2.3"

        def install(self) -> None:
            self._mark_installed()

        def uninstall(self) -> None:
            self._mark_uninstalled()

        def on_before_step(self, context: dict[str, Any]) -> dict[str, Any]:
            return {**context, "before_called": True}

        def on_after_step(
            self, context: dict[str, Any], result: dict[str, Any] | str
        ) -> dict[str, Any]:
            return {"action": "continue"}

    return _A


# ---------------------------------------------------------------------------
# A. ABC + framework_name regex (cases 1-7) —
# ---------------------------------------------------------------------------


def test_a1_cannot_instantiate_abstract_base() -> None:
    from agent_amplifier.adapter_base import AdapterBase

    with pytest.raises(TypeError):
        AdapterBase(kernel=None)  # type: ignore[abstract,arg-type]


def test_a2_empty_framework_name_rejected() -> None:
    from agent_amplifier.adapter_base import AdapterBase

    class Bad(AdapterBase):
        framework_name = ""

        def install(self) -> None: ...
        def uninstall(self) -> None: ...
        def on_before_step(self, context: dict[str, Any]) -> dict[str, Any]:
            return context

        def on_after_step(self, context, result):  # type: ignore[no-untyped-def]
            return {}

    with pytest.raises(TypeError, match="framework_name"):
        Bad(kernel=None)  # type: ignore[arg-type]


def test_a3_space_in_framework_name_rejected() -> None:
    A = _make_concrete_adapter(name="Has Space")
    with pytest.raises(TypeError, match="framework_name"):
        A(kernel=None)  # type: ignore[arg-type]


def test_a4_canonical_framework_name_accepted() -> None:
    A = _make_concrete_adapter(name="claude_code")
    inst = A(kernel=None)  # type: ignore[arg-type]
    assert inst.framework_name == "claude_code"


def test_a5_uppercase_framework_name_rejected() -> None:
    A = _make_concrete_adapter(name="A_b")
    with pytest.raises(TypeError):
        A(kernel=None)  # type: ignore[arg-type]


def test_a6_too_short_rejected() -> None:
    """Min 3 chars per regex ``^[a-z][a-z0-9_]{2,31}$`` => 1 letter + 2 chars = 3."""
    A = _make_concrete_adapter(name="ab")  # only 2 chars
    with pytest.raises(TypeError):
        A(kernel=None)  # type: ignore[arg-type]


def test_a7_too_long_rejected() -> None:
    A = _make_concrete_adapter(name="a" + "b" * 32)  # 33 total
    with pytest.raises(TypeError):
        A(kernel=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# B. Lifecycle (cases 8-12)
# ---------------------------------------------------------------------------


def test_b8_double_install_raises() -> None:
    from agent_amplifier.adapter_base import AdapterAlreadyInstalledError

    A = _make_concrete_adapter()
    a = A(kernel=None)  # type: ignore[arg-type]
    a.install()
    with pytest.raises(AdapterAlreadyInstalledError):
        a.install()


def test_b9_uninstall_before_install_raises() -> None:
    from agent_amplifier.adapter_base import AdapterNotInstalledError

    A = _make_concrete_adapter()
    a = A(kernel=None)  # type: ignore[arg-type]
    with pytest.raises(AdapterNotInstalledError):
        a.uninstall()


def test_b10_is_installed_default_tracks_flag() -> None:
    A = _make_concrete_adapter()
    a = A(kernel=None)  # type: ignore[arg-type]
    assert a.is_installed() is False
    a.install()
    assert a.is_installed() is True
    a.uninstall()
    assert a.is_installed() is False


def test_b11_on_before_step_returns_dict() -> None:
    A = _make_concrete_adapter()
    a = A(kernel=None)  # type: ignore[arg-type]
    out = a.on_before_step({"x": 1})
    assert isinstance(out, dict) and out["before_called"] is True


def test_b12_repr_includes_metadata() -> None:
    A = _make_concrete_adapter(name="claude_code")
    a = A(kernel=None)  # type: ignore[arg-type]
    r = repr(a)
    assert "claude_code" in r and "False" in r


# ---------------------------------------------------------------------------
# C. Async siblings (cases 13-14) —
# ---------------------------------------------------------------------------


def test_c13_aon_before_step_default_bridges_to_sync() -> None:
    """Default ``aon_before_step`` runs ``on_before_step`` via anyio."""
    A = _make_concrete_adapter()
    a = A(kernel=None)  # type: ignore[arg-type]

    async def _go() -> dict[str, Any]:
        return await a.aon_before_step({"x": 1})

    out = asyncio.run(_go())
    assert out["before_called"] is True


def test_c14_overridden_aon_called_directly() -> None:
    """Subclass that overrides aon_* must NOT round-trip through the thread."""
    from agent_amplifier.adapter_base import AdapterBase

    class _Async(AdapterBase):
        framework_name = "asyncfw"

        def install(self) -> None:
            self._mark_installed()

        def uninstall(self) -> None:
            self._mark_uninstalled()

        def on_before_step(self, context):  # type: ignore[no-untyped-def]
            return {"sync": True}

        def on_after_step(self, context, result):  # type: ignore[no-untyped-def]
            return {"action": "continue"}

        async def aon_before_step(self, context):  # type: ignore[no-untyped-def]
            return {"async_native": True}

    a = _Async(kernel=None)  # type: ignore[arg-type]

    async def _go() -> dict[str, Any]:
        return await a.aon_before_step({})

    out = asyncio.run(_go())
    assert out == {"async_native": True}


# ---------------------------------------------------------------------------
# D. detect() (cases 15-16) —
# ---------------------------------------------------------------------------


def test_d15_default_detect_uses_find_spec() -> None:
    """Default detect() returns importlib.util.find_spec result."""
    A_present = _make_concrete_adapter(name="logging")  # stdlib, always present
    A_absent = _make_concrete_adapter(name="abcdef_no_such_module_xyz")
    assert A_present.detect() is True
    assert A_absent.detect() is False


def test_d16_overridden_detect_honored() -> None:
    from agent_amplifier.adapter_base import AdapterBase

    class _A(AdapterBase):
        framework_name = "fakefw"

        @classmethod
        def detect(cls) -> bool:
            return True

        def install(self) -> None: ...
        def uninstall(self) -> None: ...
        def on_before_step(self, context):  # type: ignore[no-untyped-def]
            return context

        def on_after_step(self, context, result):  # type: ignore[no-untyped-def]
            return {}

    assert _A.detect() is True


def test_default_detect_handles_blank_framework_name_gracefully() -> None:
    """A class that hasn't overridden framework_name shouldn't crash detect()."""
    from agent_amplifier.adapter_base import AdapterBase

    assert AdapterBase.detect() is False


# ---------------------------------------------------------------------------
# V2.1 — memory plane defaults (.5.1)
# ---------------------------------------------------------------------------


def test_default_memory_recall_returns_empty_list() -> None:
    """V2.1: AdapterBase.default_memory_recall returns [] by default."""
    from agent_amplifier.adapter_base import AdapterBase

    class _MemAdapter(AdapterBase):
        framework_name = "mem_default_a"

        def install(self) -> None:
            self._mark_installed()

        def uninstall(self) -> None:
            self._mark_uninstalled()

        def on_before_step(self, context):  # type: ignore[no-untyped-def]
            return context

        def on_after_step(self, context, result):  # type: ignore[no-untyped-def]
            return {"action": "continue"}

    inst = _MemAdapter(kernel=None)
    assert inst.default_memory_recall("anything", 5) == []


def test_default_memory_remember_returns_none_no_op() -> None:
    """V2.1: AdapterBase.default_memory_remember is a no-op default."""
    from agent_amplifier.adapter_base import AdapterBase
    from agent_amplifier.types import EffortLevel, Outcome

    class _MemAdapter(AdapterBase):
        framework_name = "mem_default_b"

        def install(self) -> None:
            self._mark_installed()

        def uninstall(self) -> None:
            self._mark_uninstalled()

        def on_before_step(self, context):  # type: ignore[no-untyped-def]
            return context

        def on_after_step(self, context, result):  # type: ignore[no-untyped-def]
            return {"action": "continue"}

    inst = _MemAdapter(kernel=None)
    out = inst.default_memory_remember(
        Outcome(
            query="q",
            effort=EffortLevel.LOW,
            iterations=1,
            quality=0.5,
        )
    )
    assert out is None
