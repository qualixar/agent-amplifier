# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Cluster E — final coverage gap closures (STAGE-5C-COV).

This module targets the last sub-1% partial branches and lines that the
existing ``*_branches.py`` suites couldn't reach without monkeypatching at
specific seams. Goal: project-wide 100% line + 100% branch coverage.

Targeted gaps (from `pytest --cov --cov-report=term-missing`):

  * adapter_base.py 123-124  — find_spec raises (ValueError path)
  * adapter_base.py 189-191  — aon_after_step default impl bridges to sync
  * bench.py 210->212        — matplotlib path with without_pass=None
  * bench.py 212->214        — matplotlib path with with_pass=None
  * cli.py 243->242          — install loop iterates past first non-match
  * cli.py 265->264          — uninstall loop iterates past first non-match
  * config.py 191            — /etc/agent-amplifier exists path
  * kernel.py 428->432       — anchor set during the unlock window
  * slm_bridge.py 156->163   — existing key + non-posix
  * slm_bridge.py 425        — slm subprocess returns nonzero
"""

from __future__ import annotations

import asyncio
import gc
from pathlib import Path
from typing import Any

import pytest

from agent_amplifier.adapter_base import AdapterBase

# ---------------------------------------------------------------------------
# adapter_base.py — find_spec raising + async after-step bridge
# ---------------------------------------------------------------------------


def _make_concrete_adapter_for_async(name: str = "asyncbase") -> type[AdapterBase]:
    """Concrete adapter that ONLY implements the sync hooks; default
    ``aon_after_step`` should bridge to ``on_after_step`` via anyio."""

    class _A(AdapterBase):
        framework_name = name
        version = "0.0.1"

        def install(self) -> None:
            self._mark_installed()

        def uninstall(self) -> None:
            self._mark_uninstalled()

        def on_before_step(
            self, context: dict[str, Any]
        ) -> dict[str, Any]:
            return context

        def on_after_step(
            self,
            context: dict[str, Any],
            result: dict[str, Any] | str,
        ) -> dict[str, Any]:
            # Encode the bridge proof in the return so the test asserts it.
            return {"action": "continue", "via_sync": True, "result": result}

    return _A


def test_default_aon_after_step_bridges_to_sync() -> None:
    """Cover lines 189-191: default ``aon_after_step`` runs ``on_after_step``
    via ``anyio.to_thread.run_sync``."""
    A = _make_concrete_adapter_for_async()
    a = A(kernel=None)  # type: ignore[arg-type]

    async def _go() -> dict[str, Any]:
        return await a.aon_after_step({"k": 1}, {"r": 2})

    out = asyncio.run(_go())
    assert out["action"] == "continue"
    assert out["via_sync"] is True
    assert out["result"] == {"r": 2}


def test_detect_swallows_value_error_from_find_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover lines 123-124: when ``importlib.util.find_spec`` raises,
    ``detect()`` returns False without propagating."""
    import importlib.util as _iu

    from agent_amplifier import adapter_base

    class _A(AdapterBase):
        framework_name = "synthetic_module_xyz"
        version = "1.0.0"

        def install(self) -> None: ...
        def uninstall(self) -> None: ...

        def on_before_step(
            self, context: dict[str, Any]
        ) -> dict[str, Any]:
            return context

        def on_after_step(
            self,
            context: dict[str, Any],
            result: dict[str, Any] | str,
        ) -> dict[str, Any]:
            return {"action": "continue"}

    def _raise_value(name: str) -> Any:
        raise ValueError("badly-formed module spec")

    monkeypatch.setattr(_iu, "find_spec", _raise_value)
    # adapter_base imports importlib.util at top; same module object.
    monkeypatch.setattr(adapter_base.importlib.util, "find_spec", _raise_value)

    assert _A.detect() is False


# ---------------------------------------------------------------------------
# bench.py — matplotlib path with one-of-two None values
# ---------------------------------------------------------------------------


def test_matplotlib_export_chart_with_only_with_amp(
    tmp_path: Path,
) -> None:
    """Cover branch 210->212 — matplotlib path skips the ``without_pass``
    bar when it is None."""
    pytest.importorskip("matplotlib")
    from agent_amplifier import bench

    out = tmp_path / "with_only.svg"
    bench._export_chart(
        str(out),
        with_pass=7,
        with_tokens=19000,
        without_pass=None,
        without_tokens=None,
        n=10,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_matplotlib_export_chart_with_only_without_amp(
    tmp_path: Path,
) -> None:
    """Cover branch 212->214 — matplotlib path skips the ``with_pass`` bar
    when it is None."""
    pytest.importorskip("matplotlib")
    from agent_amplifier import bench

    out = tmp_path / "without_only.svg"
    bench._export_chart(
        str(out),
        with_pass=None,
        with_tokens=None,
        without_pass=4,
        without_tokens=28000,
        n=10,
    )
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# cli.py — loop iteration past first non-matching adapter (243->242, 265->264)
# ---------------------------------------------------------------------------


def _make_cli_adapter(name: str) -> type[AdapterBase]:
    """Minimal adapter for CLI loop tests.

    ``install``/``uninstall`` are intentionally side-effect-free: the CLI
    creates a fresh instance for each subcommand call, so calling
    ``_mark_uninstalled`` on a never-installed instance would raise. We
    only need the dispatch to succeed for the loop-iteration branch test.
    """

    class _Fake(AdapterBase):
        framework_name = name
        version = "9.9.9"
        # legacy CLI loop tests assert
        # ``installed: <name>`` semantics, so opt into the persistent path.
        INSTALL_PERSISTENT = True

        @classmethod
        def detect(cls) -> bool:
            return True

        def install(self) -> None:
            return None

        def uninstall(self) -> None:
            return None

        def on_before_step(
            self, context: dict[str, Any]
        ) -> dict[str, Any]:
            return context

        def on_after_step(
            self,
            context: dict[str, Any],
            result: dict[str, Any] | str,
        ) -> dict[str, Any]:
            return {"action": "continue"}

    return _Fake


@pytest.fixture(autouse=True)
def _gc_cli_adapters() -> Any:
    yield
    gc.collect()


def test_install_target_skips_first_adapter_to_match_second(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cover branch 243->242 — the install loop iterates past the first
    adapter (whose name does NOT match) and matches on the second."""
    from agent_amplifier.cli import main

    keep_first = _make_cli_adapter(name="cluster_e_skip_first")  # noqa: F841
    keep_target = _make_cli_adapter(name="cluster_e_match_second")  # noqa: F841

    rc = main(["install", "cluster_e_match_second"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "installed: cluster_e_match_second" in out
    # The first adapter should NOT be referenced in the success line.
    assert "installed: cluster_e_skip_first" not in out


def test_uninstall_target_skips_first_adapter_to_match_second(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cover branch 265->264 — the uninstall loop iterates past the first
    adapter (whose name does NOT match) and matches on the second."""
    from agent_amplifier.cli import main

    keep_first = _make_cli_adapter(name="cluster_e_uninst_skip_first")  # noqa: F841
    keep_target = _make_cli_adapter(  # noqa: F841
        name="cluster_e_uninst_match_second"
    )

    rc = main(["uninstall", "cluster_e_uninst_match_second"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "uninstalled: cluster_e_uninst_match_second" in out


# ---------------------------------------------------------------------------
# config.py — /etc/agent-amplifier resolution branch (line 191)
# ---------------------------------------------------------------------------


def test_allowed_roots_appends_etc_when_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cover line 191 — when /etc/agent-amplifier exists, it is appended to
    the allowed-roots list.

    Strategy: subclass ``pathlib.Path`` to wrap ``/etc/agent-amplifier`` so
    that ``.exists()`` returns True and ``.resolve()`` returns a real tmp
    directory. We monkeypatch the module-level ``Path`` reference with a
    callable proxy that intercepts ONLY the ``/etc/agent-amplifier``
    construction; all other ``Path(...)`` and ``Path.home()`` calls flow
    through to the real ``pathlib.Path``.
    """
    from agent_amplifier import config as cfg_mod

    fake_etc = tmp_path / "fake-etc-agent-amplifier"
    fake_etc.mkdir()
    real_path_cls = cfg_mod.Path

    class _PathProxy:
        """Callable + classmethod proxy.

        - ``_PathProxy(arg)`` returns a real ``Path`` for everything except
          ``/etc/agent-amplifier``, where we return a fake whose ``exists``
          and ``resolve`` point at ``fake_etc``.
        - ``_PathProxy.home()`` and other classmethods delegate to the real
          ``Path``.
        """

        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            if args and args[0] == "/etc/agent-amplifier":
                # Make a real Path object to a real (existing) directory
                # so `.exists()` is True and `.resolve()` succeeds.
                return real_path_cls(fake_etc)
            return real_path_cls(*args, **kwargs)

        @classmethod
        def home(cls) -> Any:
            return real_path_cls.home()

    monkeypatch.setattr(cfg_mod, "Path", _PathProxy)
    roots = cfg_mod._allowed_roots()
    # Two roots: home + /etc/agent-amplifier (fake → fake_etc).
    assert len(roots) == 2
    assert any(str(r).endswith("fake-etc-agent-amplifier") for r in roots)


# ---------------------------------------------------------------------------
# kernel.py — anchor double-check sees value already set (branch 428->432)
# ---------------------------------------------------------------------------


def test_anchor_double_check_sees_value_set_during_unlock_window() -> None:
    """Cover branch 428->432: ``need_anchor`` was True (computed under lock),
    then ``_goal.capture`` ran outside the lock, then re-acquired the lock,
    then re-checked — but a concurrent path set the anchor in the unlock
    window so the if-False branch fires.

    We force this by injecting a ``GoalAnchorService.capture`` whose side
    effect is to set ``core._state.anchor`` BEFORE returning, so the
    re-check finds it non-None.
    """
    from agent_amplifier.goal_anchor import GoalAnchorService
    from agent_amplifier.kernel import _AmplifierCore

    core = _AmplifierCore()

    real_svc = GoalAnchorService()
    sentinel_anchor = real_svc.capture("sentinel anchor for race window")

    class _RacingService:
        """Wraps the real ``GoalAnchorService`` but ``capture`` has a
        side-effect that mimics another concurrent path winning the
        anchor-install race."""

        def capture(self, query: str) -> Any:
            # Simulate a concurrent path winning: install the anchor BEFORE
            # we return. The kernel's re-check at line 428 will see the
            # state has changed and skip the assignment (branch 428->432).
            core._state.anchor = sentinel_anchor
            # Return a DIFFERENT anchor so the test can assert the re-check
            # guarded against overwrite.
            return real_svc.capture("racing capture's own result")

        # Forward all the other GoalAnchorService methods that the kernel's
        # ``_before_step_inner`` and ``_after_step`` invoke. Without these,
        # the kernel's outer ``except Exception`` swallows the AttributeError
        # and the branch instrumentation may not register the inner code.
        def inject(self, *args: Any, **kwargs: Any) -> Any:
            return real_svc.inject(*args, **kwargs)

        def measure_drift(self, *args: Any, **kwargs: Any) -> Any:
            return real_svc.measure_drift(*args, **kwargs)

        def classify_drift(self, *args: Any, **kwargs: Any) -> Any:
            return real_svc.classify_drift(*args, **kwargs)

    core._goal = _RacingService()  # type: ignore[assignment]

    async def _go() -> Any:
        return await core.before_step("trigger anchor capture", {})

    asyncio.run(_go())
    # Race winner's anchor is the one that survived (re-check skipped overwrite).
    assert core._state.anchor is sentinel_anchor


# ---------------------------------------------------------------------------
# V2.1 NOTE: slm_bridge.py was moved to examples/slm_provider.py and is no
# longer part of core. The session-key + recall-blocking tests that used
# to live here are now the responsibility of the examples module's own
# integration tests (TBD). They are intentionally NOT carried into core.
# ---------------------------------------------------------------------------
