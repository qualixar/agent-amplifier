# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.adapters.claude_code.installer``.

Coverage targets: 100% line + 100% branch on installer.py.

Critical safety contract:
    * NO test ever writes to ``~/.claude/settings.json``. Every test passes
      an explicit ``settings_path`` inside ``tmp_path``.
    * Helper monkeypatches ``_DEFAULT_SETTINGS_PATH`` so even bugs that ignore
      the explicit path argument still hit ``tmp_path``, never the user's HOME.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from agent_amplifier.adapters.claude_code import installer as _ins


@pytest.fixture(autouse=True)
def _redirect_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect _DEFAULT_SETTINGS_PATH into tmp so any unspecified-path call
    can never hit the user's real HOME."""
    redirect = tmp_path / "_default" / "settings.json"
    redirect.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_ins, "_DEFAULT_SETTINGS_PATH", redirect)
    return redirect


@pytest.fixture
def settings_path(tmp_path: Path) -> Path:
    return tmp_path / "settings.json"


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# build_command
# ---------------------------------------------------------------------------


def test_build_command_with_arg() -> None:
    cmd = _ins.build_command("/usr/bin/python", module="x.y", arg="Foo")
    assert cmd == "/usr/bin/python -m x.y Foo"
    assert "x.y" in cmd  # marker is the module path


def test_build_command_no_arg() -> None:
    cmd = _ins.build_command("/usr/bin/python", module="x.y", arg="")
    assert cmd == "/usr/bin/python -m x.y"


def test_build_command_defaults_to_sys_executable() -> None:
    import sys as _sys
    cmd = _ins.build_command(module="x.y", arg="A")
    assert cmd.startswith(_sys.executable)


# ---------------------------------------------------------------------------
# install — happy paths
# ---------------------------------------------------------------------------


def test_install_into_missing_settings_creates_file(
    settings_path: Path,
) -> None:
    assert not settings_path.exists()
    res = _ins.install(settings_path, python_executable="/p/python")
    assert settings_path.exists()
    data = _read_json(settings_path)
    for ev in ("UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop", "PreCompact"):
        assert ev in data["hooks"]
        assert len(data["hooks"][ev]) == 1
    # Backup is None because the file did not exist before.
    assert res["backup_path"] is None
    assert res["verified"] is True
    assert sorted(res["added_events"]) == [
        "PostToolUse", "PreCompact", "PreToolUse", "Stop", "UserPromptSubmit",
    ]


def test_install_preserves_unrelated_keys(settings_path: Path) -> None:
    settings_path.write_text(json.dumps({
        "preferred_model": "sonnet",
        "hooks": {
            "PreToolUse": [
                {"hooks": [{"type": "command", "command": "/usr/local/bin/foo"}]}
            ],
            "Other": [{"hooks": [{"type": "command", "command": "x"}]}],
        },
    }), encoding="utf-8")
    _ins.install(settings_path, python_executable="/p/python")
    data = _read_json(settings_path)
    # Unrelated top-level key preserved.
    assert data["preferred_model"] == "sonnet"
    # Unrelated hook event preserved.
    assert data["hooks"]["Other"][0]["hooks"][0]["command"] == "x"
    # Pre-existing PreToolUse user entry preserved AND amp entry added.
    pre = data["hooks"]["PreToolUse"]
    assert any(g["hooks"][0]["command"] == "/usr/local/bin/foo" for g in pre)
    assert any(_ins._AMP_MARKER in g["hooks"][0]["command"] for g in pre)


def test_install_idempotent(settings_path: Path) -> None:
    res1 = _ins.install(settings_path, python_executable="/p/python")
    assert len(res1["added_events"]) == 5
    res2 = _ins.install(settings_path, python_executable="/p/python")
    assert res2["added_events"] == []
    assert sorted(res2["already_present"]) == [
        "PostToolUse", "PreCompact", "PreToolUse", "Stop", "UserPromptSubmit",
    ]


def test_install_backup_taken_when_file_existed(settings_path: Path) -> None:
    settings_path.write_text("{}", encoding="utf-8")
    res = _ins.install(settings_path, python_executable="/p/python")
    assert res["backup_path"] is not None
    bak = Path(res["backup_path"])
    assert bak.exists()
    assert bak.read_text() == "{}"


def test_install_uses_default_path_when_none(
    _redirect_default: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    res = _ins.install(python_executable="/p/python")
    assert res["settings_path"] == str(_redirect_default)
    assert _redirect_default.exists()


def test_install_command_has_marker(settings_path: Path) -> None:
    _ins.install(settings_path, python_executable="/p/python")
    data = _read_json(settings_path)
    cmd = data["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert _ins._AMP_MARKER in cmd


# ---------------------------------------------------------------------------
# Idempotency via marker — different shapes of an existing entry
# ---------------------------------------------------------------------------


def test_install_recognizes_marker_in_command_string(settings_path: Path) -> None:
    """A pre-existing entry whose command string contains the marker (but no
    explicit `_amp_marker` field) should still suppress re-add."""
    settings_path.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{
                    "type": "command",
                    "command": (
                        "/old/python -m agent_amplifier.adapters.claude_code"
                        ".hooks UserPromptSubmit"
                    ),
                }]}
            ],
        },
    }), encoding="utf-8")
    res = _ins.install(settings_path, python_executable="/new/python")
    assert "UserPromptSubmit" in res["already_present"]


def test_install_skips_non_dict_inner_hooks_entry(settings_path: Path) -> None:
    """If a hook group contains a non-dict 'hooks' inner element, it should
    NOT crash the marker scan."""
    settings_path.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": ["malformed-string-not-a-dict"]},
                {"hooks": [{"command": "x"}]},  # no _amp_marker, no marker in command
            ],
        },
    }), encoding="utf-8")
    res = _ins.install(settings_path, python_executable="/p/python")
    assert "UserPromptSubmit" in res["added_events"]


def test_install_skips_non_dict_group(settings_path: Path) -> None:
    """A non-dict element in the event array must not crash detection."""
    settings_path.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                "not-a-dict-group",
                {"hooks": [{"type": "command", "command": "x"}]},
            ],
        },
    }), encoding="utf-8")
    res = _ins.install(settings_path, python_executable="/p/python")
    assert "UserPromptSubmit" in res["added_events"]


def test_install_skips_group_with_non_list_hooks(settings_path: Path) -> None:
    settings_path.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": "not-a-list"},
                {"hooks": [{"type": "command", "command": "x"}]},
            ],
        },
    }), encoding="utf-8")
    res = _ins.install(settings_path, python_executable="/p/python")
    assert "UserPromptSubmit" in res["added_events"]


# ---------------------------------------------------------------------------
# Malformed settings rejection
# ---------------------------------------------------------------------------


def test_install_rejects_invalid_json(settings_path: Path) -> None:
    settings_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(_ins.MalformedSettingsError, match="not valid JSON"):
        _ins.install(settings_path, python_executable="/p/python")


def test_install_rejects_non_object_root(settings_path: Path) -> None:
    settings_path.write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(_ins.MalformedSettingsError, match="must be a JSON object"):
        _ins.install(settings_path, python_executable="/p/python")


def test_install_rejects_non_dict_hooks(settings_path: Path) -> None:
    settings_path.write_text(json.dumps({"hooks": [1, 2]}), encoding="utf-8")
    with pytest.raises(_ins.MalformedSettingsError, match="hooks` must be"):
        _ins.install(settings_path, python_executable="/p/python")


def test_install_rejects_non_array_event(settings_path: Path) -> None:
    settings_path.write_text(json.dumps({
        "hooks": {"UserPromptSubmit": "not-array"}
    }), encoding="utf-8")
    with pytest.raises(_ins.MalformedSettingsError, match="must be an array"):
        _ins.install(settings_path, python_executable="/p/python")


def test_install_empty_string_settings(settings_path: Path) -> None:
    """Empty string body is treated as no-content (start fresh)."""
    settings_path.write_text("   \n", encoding="utf-8")
    res = _ins.install(settings_path, python_executable="/p/python")
    assert len(res["added_events"]) == 5


# ---------------------------------------------------------------------------
# Verify failure → restore from backup
# ---------------------------------------------------------------------------


def test_install_verify_failure_restores_backup(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_path.write_text(json.dumps({"existing": "data"}), encoding="utf-8")
    monkeypatch.setattr(_ins, "_verify_install", lambda p: False)
    with pytest.raises(_ins.VerifyFailedError, match="restored from"):
        _ins.install(settings_path, python_executable="/p/python")
    # File restored to pre-install state.
    assert _read_json(settings_path) == {"existing": "data"}


def test_install_verify_failure_no_backup_path(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify fails AND no backup taken (file didn't exist) → no-backup error."""
    assert not settings_path.exists()
    monkeypatch.setattr(_ins, "_verify_install", lambda p: False)
    with pytest.raises(_ins.VerifyFailedError, match="no backup existed"):
        _ins.install(settings_path, python_executable="/p/python")


def test_verify_install_negative_branches(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct exercise of verify's failure branches."""
    # Branch 1: malformed file → verify returns False.
    settings_path.write_text("{not json", encoding="utf-8")
    assert _ins._verify_install(settings_path) is False
    # Branch 2: hooks not dict → False.
    settings_path.write_text(json.dumps({"hooks": "x"}), encoding="utf-8")
    # _read_settings would raise; suppress that for direct verify call.
    monkeypatch.setattr(
        _ins, "_read_settings", lambda p: {"hooks": "x"}
    )
    assert _ins._verify_install(settings_path) is False


# ---------------------------------------------------------------------------
# Atomic-write internals
# ---------------------------------------------------------------------------


def test_atomic_write_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "settings.json"
    _ins._atomic_write(target, {"hooks": {}})
    assert target.exists()
    assert _read_json(target) == {"hooks": {}}


def test_atomic_write_preserves_mode(settings_path: Path) -> None:
    settings_path.write_text("{}", encoding="utf-8")
    settings_path.chmod(0o640)
    _ins._atomic_write(settings_path, {"hooks": {}})
    mode = stat.S_IMODE(settings_path.stat().st_mode)
    assert mode == 0o640


def test_atomic_write_cleans_up_temp_on_failure(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **k: object) -> object:
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        _ins._atomic_write(settings_path, {"hooks": {}})
    leftovers = list(settings_path.parent.glob(f".{settings_path.name}.amp-*.tmp"))
    assert leftovers == []


def test_atomic_write_temp_unlink_failure_swallowed(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the cleanup unlink also fails, the original error must still propagate."""
    real_replace = os.replace
    def boom_replace(*a: object, **k: object) -> object:
        raise OSError("fail rename")
    real_unlink = Path.unlink
    def boom_unlink(self, *a: object, **k: object) -> object:
        raise OSError("fail unlink")
    monkeypatch.setattr(os, "replace", boom_replace)
    monkeypatch.setattr(Path, "unlink", boom_unlink)
    with pytest.raises(OSError, match="fail rename"):
        _ins._atomic_write(settings_path, {"hooks": {}})
    # Restore for cleanup.
    monkeypatch.setattr(Path, "unlink", real_unlink)
    monkeypatch.setattr(os, "replace", real_replace)


def test_atomic_write_copystat_oserror_swallowed(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When copystat fails, the write proceeds with default perms."""
    settings_path.write_text("{}", encoding="utf-8")

    import shutil as _shutil
    def boom_copystat(*a: object, **k: object) -> object:
        raise OSError("nope")
    monkeypatch.setattr(_shutil, "copystat", boom_copystat)
    monkeypatch.setattr(_ins.shutil, "copystat", boom_copystat)
    _ins._atomic_write(settings_path, {"hooks": {}})
    assert _read_json(settings_path) == {"hooks": {}}


def test_install_atomic_write_failure_propagates(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the atomic write itself fails (not verify), propagate the error."""
    settings_path.write_text("{}", encoding="utf-8")

    def boom(*a: object, **k: object) -> object:
        raise OSError("simulated write fail")

    monkeypatch.setattr(_ins, "_atomic_write", boom)
    with pytest.raises(OSError, match="simulated write fail"):
        _ins.install(settings_path, python_executable="/p/python")


def test_verify_install_returns_false_when_event_missing(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct verify exercise: hooks dict missing one of the events."""
    monkeypatch.setattr(
        _ins, "_read_settings",
        lambda p: {"hooks": {"UserPromptSubmit": []}},  # other 3 missing
    )
    assert _ins._verify_install(settings_path) is False
