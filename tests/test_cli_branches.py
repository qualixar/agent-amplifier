# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""CLI branch coverage (.py).

These tests target the install/uninstall/config/bench/list/status branches
that the smoke-tests in test_cli.py do not exercise. Anti-rationalization:
"adapter-doesn't-exist-yet is not an excuse — test the 'adapter not found'
branch."

Strategy: declare fake AdapterBase subclasses inside each test, then revert
``__subclasses__`` so they don't pollute neighbouring tests. ``__subclasses__``
is read-only on the C side, so we use the standard pattern of garbage-
collecting the class refs after the test (pytest's per-test isolation +
``gc.collect()``).
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import pytest

from agent_amplifier.adapter_base import AdapterBase

# ---------------------------------------------------------------------------
# Helpers — fake adapters, isolated per test
# ---------------------------------------------------------------------------


def _make_adapter(
    *,
    name: str,
    detect_value: bool = True,
    detect_raises: bool = False,
    install_raises: BaseException | None = None,
    uninstall_raises: BaseException | None = None,
    is_installed_value: bool = False,
    is_installed_raises: bool = False,
    install_persistent: bool = True,
) -> type[AdapterBase]:
    """Build a one-off AdapterBase subclass for a single CLI test.

    ``install_persistent`` defaults to True for the
    legacy CLI tests that assert the ``installed: <name>`` message.  Tests
    targeting the ``ready: <name>`` (marker-only) path pass False.
    """

    class _Fake(AdapterBase):
        framework_name = name
        version = "9.9.9"
        INSTALL_PERSISTENT = install_persistent

        @classmethod
        def detect(cls) -> bool:
            if detect_raises:
                raise RuntimeError("boom")
            return detect_value

        def install(self) -> None:
            if install_raises is not None:
                raise install_raises

        def uninstall(self) -> None:
            if uninstall_raises is not None:
                raise uninstall_raises

        def is_installed(self) -> bool:
            if is_installed_raises:
                raise RuntimeError("is_installed boom")
            return is_installed_value

        def on_before_step(self, context: dict[str, Any]) -> dict[str, Any]:
            return context

        def on_after_step(
            self,
            context: dict[str, Any],
            result: dict[str, Any] | str,
        ) -> dict[str, Any]:
            return {"action": "continue"}

    return _Fake


@pytest.fixture(autouse=True)
def _gc_after() -> Any:
    """Ensure per-test fake adapter classes don't bleed into siblings."""
    yield
    gc.collect()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_with_adapter_prints_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    keep = _make_adapter(name="cli_test_listed_a", detect_value=True)  # noqa: F841
    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cli_test_listed_a" in out
    assert "detected=True" in out


def test_list_handles_detect_exception(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    keep = _make_adapter(name="cli_test_listed_b", detect_raises=True)  # noqa: F841
    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    # detect() raised → CLI swallows → reports detected=False.
    assert "cli_test_listed_b" in out
    assert "detected=False" in out


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_prints_installed_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    keep_a = _make_adapter(name="cli_test_status_a", is_installed_value=True)  # noqa: F841
    keep_b = _make_adapter(name="cli_test_status_b", is_installed_value=False)  # noqa: F841
    rc = main(["status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cli_test_status_a" in out
    assert "INSTALLED" in out
    # cli_test_status_b is NOT printed (not installed).
    assert "cli_test_status_b" not in out


def test_status_swallows_is_installed_exception(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    keep = _make_adapter(name="cli_test_status_c", is_installed_raises=True)  # noqa: F841
    rc = main(["status"])
    assert rc == 0
    # Exception path → treated as not-installed, so nothing printed for it.
    assert "cli_test_status_c" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def test_install_target_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    keep = _make_adapter(name="cli_test_install_ok")  # noqa: F841
    rc = main(["install", "cli_test_install_ok"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "installed: cli_test_install_ok" in out


def test_install_already_installed_returns_three(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    keep = _make_adapter(  # noqa: F841
        name="cli_test_install_dup",
        install_raises=RuntimeError("already installed bro"),
    )
    rc = main(["install", "cli_test_install_dup"])
    err = capsys.readouterr().err
    assert rc == 3
    assert "already installed" in err


def test_install_permission_denied_returns_four(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    keep = _make_adapter(  # noqa: F841
        name="cli_test_install_perm",
        install_raises=PermissionError("nope"),
    )
    rc = main(["install", "cli_test_install_perm"])
    err = capsys.readouterr().err
    assert rc == 4
    assert "permission denied" in err


def test_install_generic_failure_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    keep = _make_adapter(  # noqa: F841
        name="cli_test_install_fail",
        install_raises=RuntimeError("disk on fire"),
    )
    rc = main(["install", "cli_test_install_fail"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "install failed" in err


def test_install_no_target_no_auto_returns_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    rc = main(["install"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "requires <target>" in err


def test_install_auto_with_adapter_installs(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_amplifier import cli

    keep_yes = _make_adapter(name="cli_test_auto_yes", detect_value=True)
    keep_no = _make_adapter(name="cli_test_auto_no", detect_value=False)
    # ships bundled adapters; isolate this test from them so the
    # behavior contract under test (detect-true installs, detect-false skips)
    # is not contaminated by host-side state on the dev machine.
    monkeypatch.setattr(cli, "_enumerate_adapters", lambda: [keep_yes, keep_no])
    rc = cli.main(["install", "--auto"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "installed: cli_test_auto_yes" in out
    assert "installed 1 adapter" in out
    # not-detected adapter is NOT installed.
    assert "installed: cli_test_auto_no" not in out


def test_install_auto_swallows_install_exception(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_amplifier import cli

    keep = _make_adapter(
        name="cli_test_auto_fail",
        detect_value=True,
        install_raises=RuntimeError("kapow"),
    )
    monkeypatch.setattr(cli, "_enumerate_adapters", lambda: [keep])
    rc = cli.main(["install", "--auto"])
    captured = capsys.readouterr()
    # --auto should keep going even if one fails; final rc is 0.
    assert rc == 0
    # The FAILED line is on stderr.
    assert "FAILED" in captured.err or "FAILED" in captured.out


# ---------------------------------------------------------------------------
# uninstall
# ---------------------------------------------------------------------------


def test_uninstall_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    keep = _make_adapter(name="cli_test_uninst_ok")  # noqa: F841
    rc = main(["uninstall", "cli_test_uninst_ok"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "uninstalled: cli_test_uninst_ok" in out


def test_uninstall_unknown_target_returns_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    rc = main(["uninstall", "absolutely_not_a_real_adapter"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "unknown target" in err


def test_uninstall_failure_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    keep = _make_adapter(  # noqa: F841
        name="cli_test_uninst_fail",
        uninstall_raises=RuntimeError("permissions hell"),
    )
    rc = main(["uninstall", "cli_test_uninst_fail"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "uninstall failed" in err


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_path_prints_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main
    from agent_amplifier.config import USER_CONFIG_PATH

    rc = main(["config", "path"])
    out = capsys.readouterr().out
    assert rc == 0
    assert str(USER_CONFIG_PATH) in out


def test_config_set_without_kv_returns_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    rc = main(["config", "set"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "key=value" in err


def test_config_set_without_equals_returns_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    rc = main(["config", "set", "no_equals_sign"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "key=value" in err


def test_config_set_with_kv_returns_non_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``config set`` is not implemented in V1, so it
    MUST exit non-zero and print to stderr.  This protects ``set -e``
    wrapper scripts from believing the config was changed."""
    from agent_amplifier.cli import main

    rc = main(["config", "set", "max_iterations=5"])
    err = capsys.readouterr().err
    assert rc != 0
    assert "not implemented" in err


def test_config_show_handles_load_error(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_amplifier import cli

    def boom() -> Any:
        raise RuntimeError("config corrupted")

    monkeypatch.setattr(cli, "load_config", boom)
    rc = cli.main(["config", "show"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "config error" in err


# ---------------------------------------------------------------------------
# bench (just exercise the CLI plumb-through; bench logic tested elsewhere)
# ---------------------------------------------------------------------------


def test_bench_subcommand_routes_to_run_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier.cli import main

    rc = main(["bench", "--task", "swe-bench-lite-mini", "--with-amp"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Task" in out
    assert "With amplifier" in out


# ---------------------------------------------------------------------------
# verbose / quiet flags exercise the logging branch (line 130/134)
# ---------------------------------------------------------------------------


def test_verbose_flag_sets_debug_level() -> None:
    from agent_amplifier.cli import main

    rc = main(["-v", "list"])
    assert rc == 0


def test_quiet_flag_sets_warning_level() -> None:
    from agent_amplifier.cli import main

    rc = main(["-q", "list"])
    assert rc == 0


# ---------------------------------------------------------------------------
# doctor — slm-not-found vs slm-present + anyio-version error path
# ---------------------------------------------------------------------------


def test_doctor_when_slm_present(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cover line 211 (slm path printed when shutil.which finds slm)."""
    fake_slm = tmp_path / "slm"
    fake_slm.write_text("#!/bin/sh\necho 0.0.1\n")
    fake_slm.chmod(0o755)

    from agent_amplifier import cli

    monkeypatch.setattr(cli.shutil, "which", lambda name: str(fake_slm))
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert str(fake_slm) in out


def test_doctor_when_slm_missing(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the slm-not-installed install hint (MED-2).

    Post the doctor message is "not installed (pip install ...)";
    the prior "not found" wording was V2.0-era when SLM was the only
    cross-session memory mechanism mentioned. We accept either spelling
    so this test does not regress on a future tweak that drops back to
    "not found" — the load-bearing assertion is the install hint.
    """
    from agent_amplifier import cli

    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    # MED-2: doctor demoted SLM to "third-party memory providers
    # (optional)" and the absent-SLM hint reads "not installed". We accept
    # either wording so a future tweak doesn't regress this test.
    assert "not installed" in out or "not found" in out
    assert "pip install superlocalmemory" in out


def test_doctor_anyio_version_falls_back_when_unavailable(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the except-branch in _anyio_version (lines 321-322)."""
    import importlib.metadata as _m

    def boom(*args: Any, **kwargs: Any) -> str:
        raise _m.PackageNotFoundError("anyio")

    monkeypatch.setattr(_m, "version", boom)
    from agent_amplifier import cli

    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "anyio" in out
    assert "missing" in out


def test_unknown_subcommand_via_argparse_exits_two() -> None:
    """Already covered in test_cli.py but include a SystemExit assertion."""
    from agent_amplifier.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["frobnicate_now"])
    # argparse rejects unknown sub at parse-time → SystemExit(2).
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# argparse Namespace direct dispatch — covers line 130, 134
# (when args.cmd is install/bench, ensure dispatch returns the right rc)
# ---------------------------------------------------------------------------


def test_main_dispatch_uninstall_branch_via_namespace() -> None:
    """Cover the args.cmd == 'uninstall' dispatch line (130)."""
    from agent_amplifier.cli import main

    # Unknown target → returns 2; the dispatch path itself is what we cover.
    rc = main(["uninstall", "no_such_thing"])
    assert rc == 2


def test_main_dispatch_config_branch_via_namespace() -> None:
    """Cover the args.cmd == 'config' dispatch line (132)."""
    from agent_amplifier.cli import main

    rc = main(["config", "path"])
    assert rc == 0


def test_main_dispatch_bench_branch_via_namespace() -> None:
    """Cover the args.cmd == 'bench' dispatch line (134)."""
    from agent_amplifier.cli import main

    rc = main(
        [
            "bench",
            "--task",
            "swe-bench-lite-mini",
            "--without-amp",
        ]
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# bench passes argparse.Namespace into run_cli — also closes line 306-308
# already covered by test_bench_subcommand_routes_to_run_cli, but explicitly
# call _cmd_bench to lock the surface.
# ---------------------------------------------------------------------------


def test_cmd_bench_directly_returns_zero() -> None:
    from agent_amplifier import cli

    args = argparse.Namespace(
        task="swe-bench-lite-mini",
        model="sonnet",
        with_amp=True,
        without_amp=False,
        compare=False,
        export_svg=None,
    )
    rc = cli._cmd_bench(args)
    assert rc == 0


def test_cmd_config_unreachable_branch_returns_two() -> None:
    """Cover the defensive ``return 2`` on line 302 by passing an unknown op
    through ``_cmd_config`` directly (argparse normally guards via choices)."""
    from agent_amplifier import cli

    args = argparse.Namespace(op="frobnicate_op_unreachable", kv=None)
    rc = cli._cmd_config(args)
    assert rc == 2


# ---------------------------------------------------------------------------
# V2.1 — Cover the empty-adapter branches in _cmd_list / _cmd_status /
# _cmd_install --auto. These lines exist whenever no concrete AdapterBase
# subclass is loadable. We force the empty list by monkey-patching
# ``_enumerate_adapters`` to return [].
# ---------------------------------------------------------------------------


def test_list_empty_adapters_prints_v1_message(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_amplifier import cli

    monkeypatch.setattr(cli, "_enumerate_adapters", lambda: [])
    rc = cli.main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no bundled adapters" in out


def test_status_empty_adapters_prints_v1_message(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_amplifier import cli

    monkeypatch.setattr(cli, "_enumerate_adapters", lambda: [])
    rc = cli.main(["status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no adapters bundled" in out


def test_install_auto_empty_adapters_prints_nothing_to_install(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_amplifier import cli

    monkeypatch.setattr(cli, "_enumerate_adapters", lambda: [])
    rc = cli.main(["install", "--auto"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nothing to install" in out
