# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.adapters.claude_code.uninstaller``.

Coverage targets: 100% line + 100% branch on uninstaller.py.

Critical safety contract: NO test ever writes to ``~/.claude/settings.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_amplifier.adapters.claude_code import installer as _ins
from agent_amplifier.adapters.claude_code import uninstaller as _un


@pytest.fixture(autouse=True)
def _redirect_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
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
# install/uninstall round-trip
# ---------------------------------------------------------------------------


def test_uninstall_removes_amp_entries_only(settings_path: Path) -> None:
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
    res = _un.uninstall(settings_path)
    assert res["verified"] is True
    assert sorted(res["removed_events"]) == [
        "PostToolUse", "PreCompact", "PreToolUse", "Stop", "UserPromptSubmit",
    ]
    data = _read_json(settings_path)
    # Top-level non-amp key preserved.
    assert data["preferred_model"] == "sonnet"
    # User's PreToolUse entry preserved (only amp's row removed).
    pre = data["hooks"]["PreToolUse"]
    assert len(pre) == 1
    assert pre[0]["hooks"][0]["command"] == "/usr/local/bin/foo"
    # Other event untouched.
    assert "Other" in data["hooks"]


def test_uninstall_drops_empty_event_array(settings_path: Path) -> None:
    """When amp's row was the only entry for an event, drop the event key."""
    settings_path.write_text(json.dumps({"hooks": {}}), encoding="utf-8")
    _ins.install(settings_path, python_executable="/p/python")
    _un.uninstall(settings_path)
    data = _read_json(settings_path)
    # All four amp-only events removed AND hooks dict dropped.
    assert "hooks" not in data


def test_uninstall_when_settings_missing(settings_path: Path) -> None:
    assert not settings_path.exists()
    res = _un.uninstall(settings_path)
    assert res["verified"] is True
    assert res["removed_events"] == []
    assert res["backup_path"] is None


def test_uninstall_when_no_hooks_dict(settings_path: Path) -> None:
    settings_path.write_text(json.dumps({"preferred_model": "x"}), encoding="utf-8")
    res = _un.uninstall(settings_path)
    assert res["verified"] is True
    assert res["removed_events"] == []


def test_uninstall_when_no_amp_entries(settings_path: Path) -> None:
    """User has hooks but no amp entries → no-op, no backup."""
    settings_path.write_text(json.dumps({
        "hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "x"}]}]}
    }), encoding="utf-8")
    res = _un.uninstall(settings_path)
    assert res["verified"] is True
    assert res["removed_events"] == []
    assert res["backup_path"] is None


def test_uninstall_skips_non_list_event_value(settings_path: Path) -> None:
    """If a hook event's value is not a list, skip it gracefully."""
    settings_path.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": "garbage",
            "PreToolUse": [
                {"hooks": [{
                    "type": "command",
                    "command": "x",
                    "_amp_marker": _ins._AMP_MARKER,
                }]}
            ],
        }
    }), encoding="utf-8")
    res = _un.uninstall(settings_path)
    assert "PreToolUse" in res["removed_events"]
    # Non-list event value left untouched (no exception).
    data = _read_json(settings_path)
    assert data["hooks"]["UserPromptSubmit"] == "garbage"


def test_uninstall_uses_default_path_when_none(_redirect_default: Path) -> None:
    res = _un.uninstall()
    assert res["settings_path"] == str(_redirect_default)


def test_uninstall_idempotent(settings_path: Path) -> None:
    _ins.install(settings_path, python_executable="/p/python")
    _un.uninstall(settings_path)
    res2 = _un.uninstall(settings_path)
    assert res2["removed_events"] == []
    assert res2["verified"] is True


# ---------------------------------------------------------------------------
# Marker recognition
# ---------------------------------------------------------------------------


def test_is_amp_entry_marker_in_command_string(settings_path: Path) -> None:
    """Detection works when only the command-string marker is present."""
    settings_path.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{"type": "command", "command":
                    "/x -m agent_amplifier.adapters.claude_code.hooks U"}]
            }]
        }
    }), encoding="utf-8")
    res = _un.uninstall(settings_path)
    assert "UserPromptSubmit" in res["removed_events"]


def test_is_amp_entry_rejects_non_dict_inner(settings_path: Path) -> None:
    """A hook group whose inner ``hooks`` array contains non-dict entries is
    not recognized as ours."""
    settings_path.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [{"hooks": ["string-not-dict"]}]
        }
    }), encoding="utf-8")
    res = _un.uninstall(settings_path)
    assert res["removed_events"] == []
    assert _read_json(settings_path)["hooks"]["UserPromptSubmit"][0]["hooks"][0] == "string-not-dict"


def test_is_amp_entry_rejects_non_list_inner(settings_path: Path) -> None:
    settings_path.write_text(json.dumps({
        "hooks": {"UserPromptSubmit": [{"hooks": "x"}]}
    }), encoding="utf-8")
    res = _un.uninstall(settings_path)
    assert res["removed_events"] == []


def test_is_amp_entry_rejects_non_dict_group() -> None:
    assert _un._is_amp_entry("not-a-dict") is False


# ---------------------------------------------------------------------------
# Verify failure → restore
# ---------------------------------------------------------------------------


def test_uninstall_verify_failure_restores_backup(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ins.install(settings_path, python_executable="/p/python")
    monkeypatch.setattr(_un, "_verify_uninstall", lambda p: False)
    with pytest.raises(_ins.VerifyFailedError, match="restored from"):
        _un.uninstall(settings_path)
    # File restored to post-install (pre-uninstall) state.
    data = _read_json(settings_path)
    assert "hooks" in data
    assert _ins._has_amp_entry(data["hooks"]["UserPromptSubmit"])


def test_uninstall_verify_negative_branches(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct exercise of _verify_uninstall failure branches."""
    settings_path.write_text("{not json", encoding="utf-8")
    assert _un._verify_uninstall(settings_path) is False

    # hooks not dict → True (no amp entries possible).
    monkeypatch.setattr(_un, "_read_settings", lambda p: {"hooks": "x"})
    assert _un._verify_uninstall(settings_path) is True


def test_uninstall_skips_non_list_event_in_verify(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When verify scans events, non-list values are skipped."""
    monkeypatch.setattr(
        _un, "_read_settings",
        lambda p: {"hooks": {"UserPromptSubmit": "junk"}},
    )
    assert _un._verify_uninstall(settings_path) is True


def test_uninstall_atomic_write_failure_propagates(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ins.install(settings_path, python_executable="/p/python")

    def boom(*a: object, **k: object) -> object:
        raise OSError("disk full")

    monkeypatch.setattr(_un, "_atomic_write", boom)
    with pytest.raises(OSError, match="disk full"):
        _un.uninstall(settings_path)


def test_uninstall_verify_failure_no_backup_path(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If somehow verify fails when no backup was taken (synthetic), surface
    the no-backup branch of the error message."""
    settings_path.write_text(json.dumps({
        "hooks": {"UserPromptSubmit": [{
            "hooks": [{"type": "command", "command": "x", "_amp_marker": _ins._AMP_MARKER}]
        }]}
    }), encoding="utf-8")
    monkeypatch.setattr(_un, "_make_backup", lambda p: None)
    monkeypatch.setattr(_un, "_verify_uninstall", lambda p: False)
    with pytest.raises(_ins.VerifyFailedError, match="no backup existed"):
        _un.uninstall(settings_path)


def test_verify_uninstall_returns_false_when_amp_still_present(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hits the True branch of any(...) inside _verify_uninstall."""
    monkeypatch.setattr(
        _un, "_read_settings",
        lambda p: {"hooks": {"UserPromptSubmit": [
            {"hooks": [{"_amp_marker": _ins._AMP_MARKER, "command": "x"}]}
        ]}},
    )
    assert _un._verify_uninstall(settings_path) is False
