# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Settings.json installer for the Claude Code hook adapter (day-0).

8-step surgical add protocol (H-3):

    1. Resolve settings.json path (default ``~/.claude/settings.json``;
       overridable for tests).
    2. Read existing JSON. If file is missing, treat as empty ``{}``.
    3. Validate that the hooks key (when present) is a dict — refuse to
       proceed against a malformed file.
    4. Build the five hook entries (UserPromptSubmit, PreToolUse,
       PostToolUse, Stop, PreCompact) carrying a stable marker so uninstall can match
       them later WITHOUT relying on path-equality (Python paths drift).
    5. Surgical-merge: for each event, append our entry only if no entry
       with the same marker is already present. Idempotent — re-running
       installer.install() is a no-op.
    6. Take a timestamped backup of the existing file (only if it exists)
       BEFORE writing. Backup name: ``settings.json.amp-bak.<UTC>``.
    7. Write the merged JSON to a temp file in the same directory. ``os.replace``
       to atomically swap onto settings.json.
    8. Re-open and re-parse settings.json to verify the merge survived
       (round-trip check). On verify failure, restore from backup and raise.

The 8-step protocol exists because ``~/.claude/settings.json`` is non-trivial
to reverse — the March 24 + April 25 incidents both involved settings.json
being clobbered. We never use the Write tool on this file; only the
atomic-temp + os.replace pattern.

The installer is **idempotent**: running it twice is a no-op (every event
already has our entry). It is **reversible**: the uninstaller (sibling
module) removes ONLY our marker'd entries, leaves the rest intact.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Final

LOG = logging.getLogger("agent_amplifier.adapters.claude_code.installer")

# ---------------------------------------------------------------------------
# Constants — the 5 hook events we register, the marker, and the entry shape
# ---------------------------------------------------------------------------

# Stable marker baked into every hook entry so uninstall can identify our
# rows even if the user's Python path or virtualenv changes.
_AMP_MARKER: Final[str] = "agent_amplifier.adapters.claude_code"

# The 5 hook events we register, mapped to their entry-point module:argument.
# PreCompact (CC 2.1.105+) is observe-only in v1.0; the active-deferral
# variant ships in v1.0.1.
_HOOK_TARGETS: Final[tuple[tuple[str, str, str], ...]] = (
    # (event_name, module_path, arg)
    ("UserPromptSubmit", "agent_amplifier.adapters.claude_code.hooks", "UserPromptSubmit"),
    ("PreToolUse",       "agent_amplifier.adapters.claude_code.hooks", "PreToolUse"),
    ("PostToolUse",      "agent_amplifier.adapters.claude_code.hooks", "PostToolUse"),
    ("Stop",             "agent_amplifier.adapters.claude_code.stop_hook", ""),
    ("PreCompact",       "agent_amplifier.adapters.claude_code.hooks", "PreCompact"),
)

_DEFAULT_SETTINGS_PATH: Final[Path] = Path.home() / ".claude" / "settings.json"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InstallerError(RuntimeError):
    """Base for installer failures."""


class MalformedSettingsError(InstallerError):
    """settings.json contains a non-dict ``hooks`` value or is unparseable."""


class VerifyFailedError(InstallerError):
    """Post-write re-read failed to find our marker — file restored from backup."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_command(python_executable: str | None = None, *, module: str, arg: str) -> str:
    """Build the shell command string Claude Code will execute for a hook.

    The command always carries the marker as a literal substring so uninstall
    can grep for it. The shape is::

        <python> -m <module> [<arg>]

    With ``arg`` omitted for the Stop hook (the module's ``main`` is the
    entry point, no event-name positional argument).
    """
    py = shlex.quote(python_executable or sys.executable)
    if arg:
        return f"{py} -m {module} {arg}"
    return f"{py} -m {module}"


def _entry_for(event: str, command: str) -> dict[str, Any]:
    """Build one hook-entry group in Claude Code's settings.json shape.

    Claude Code's schema (as of 2026-05-10):

        "hooks": {
            "<EventName>": [
                {"hooks": [{"type": "command", "command": "..."}]},
                ...
            ]
        }

    Each top-level array entry is a "group" with its own optional matcher
    + inner ``hooks`` array of commands. We use a single command per group
    (one row per event) so uninstall is a clean array-filter.
    """
    return {
        "hooks": [
            {
                "type": "command",
                "command": command,
                # Carry the marker as an explicit field so users + uninstaller
                # can identify amp rows at a glance. Claude Code ignores
                # unknown fields. NOTE: also fine that the command itself
                # contains the marker (via the module path) — we match
                # either way.
                "_amp_marker": _AMP_MARKER,
            }
        ]
    }


def _has_amp_entry(group_list: list[Any]) -> bool:
    """True if any group in ``group_list`` already carries our marker.

    Robust to both the explicit ``_amp_marker`` field and the marker
    appearing inside the ``command`` string.
    """
    for group in group_list:
        if not isinstance(group, dict):
            continue
        inner = group.get("hooks")
        if not isinstance(inner, list):
            continue
        for h in inner:
            if not isinstance(h, dict):
                continue
            if h.get("_amp_marker") == _AMP_MARKER:
                return True
            cmd = h.get("command")
            if isinstance(cmd, str) and _AMP_MARKER in cmd:
                return True
    return False


def _make_backup(settings_path: Path) -> Path | None:
    """Copy settings.json → settings.json.amp-bak.<UTC>. Returns the bak path.

    Returns ``None`` when settings.json does not exist (fresh install).
    """
    if not settings_path.exists():
        return None
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    bak = settings_path.with_name(f"{settings_path.name}.amp-bak.{ts}")
    shutil.copy2(settings_path, bak)
    LOG.info("backup: %s → %s", settings_path, bak)
    return bak


def _atomic_write(settings_path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as JSON to ``settings_path`` atomically.

    Uses ``mkstemp`` in the same directory as settings.json so the
    ``os.replace`` is a same-filesystem rename (atomic on POSIX).
    Permission bits + ownership are preserved by ``shutil.copystat`` from
    the original file when one exists.
    """
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{settings_path.name}.amp-",
        suffix=".tmp",
        dir=str(settings_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        # Preserve mode/ownership when the original exists.
        if settings_path.exists():
            try:
                shutil.copystat(settings_path, tmp_path)
            except OSError:  # pragma: no cover - best-effort metadata
                LOG.debug("copystat failed; continuing with default perms")
        os.replace(tmp_path, settings_path)
    except Exception:
        # Clean up the temp file if rename never happened.
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def _read_settings(settings_path: Path) -> dict[str, Any]:
    """Read settings.json. Empty/missing returns ``{}``. Malformed raises.

    Step 3 of the protocol: if ``hooks`` exists and is not a dict, we
    refuse to proceed — better to surface than silently overwrite.
    """
    if not settings_path.exists():
        return {}
    text = settings_path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedSettingsError(
            f"settings.json at {settings_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise MalformedSettingsError(
            f"settings.json root must be a JSON object, got {type(data).__name__}"
        )
    if "hooks" in data and not isinstance(data["hooks"], dict):
        raise MalformedSettingsError(
            f"settings.json `hooks` must be a JSON object, got "
            f"{type(data['hooks']).__name__}"
        )
    return data


def _verify_install(settings_path: Path) -> bool:
    """Re-read settings.json and confirm every event carries our marker."""
    try:
        data = _read_settings(settings_path)
    except MalformedSettingsError:
        return False
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return False
    return all(
        _has_amp_entry(hooks.get(event, []) or [])
        for event, _mod, _arg in _HOOK_TARGETS
    )


def install(
    settings_path: Path | str | None = None,
    *,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Install amp's hook entries into settings.json. Idempotent.

    Parameters
    ----------
    settings_path:
        Override the default ``~/.claude/settings.json`` (used by tests + the
        scratch-install smoke step).
    python_executable:
        Override ``sys.executable`` (used by tests). Real installs use
        ``sys.executable`` so the hook command points at the venv that has
        agent_amplifier installed.

    Returns
    -------
    dict with ``settings_path``, ``backup_path`` (or None), ``added_events``
    (list of events freshly added this call), ``already_present`` (list of
    events that already had our entry — idempotent path), and ``verified``
    boolean.

    Raises
    ------
    MalformedSettingsError
        If the existing settings.json is unparseable or has a non-dict
        ``hooks`` key. We refuse to proceed rather than silently overwrite.
    VerifyFailedError
        If the post-write re-read does not find our marker on every event.
        File is restored from backup before raising.
    PermissionError
        If we cannot read or write settings.json.
    """
    sp = Path(settings_path) if settings_path else _DEFAULT_SETTINGS_PATH

    # --- Steps 1-3: read + validate
    data = _read_settings(sp)
    hooks = data.setdefault("hooks", {})
    # ``setdefault`` may have just inserted an empty dict — that is fine.
    if not isinstance(hooks, dict):  # pragma: no cover - guarded by _read_settings
        raise MalformedSettingsError("hooks key was not a dict after read+default")

    # --- Step 4-5: surgical merge per event
    added: list[str] = []
    already: list[str] = []
    for event, module, arg in _HOOK_TARGETS:
        existing = hooks.setdefault(event, [])
        if not isinstance(existing, list):
            raise MalformedSettingsError(
                f"settings.json hooks[{event!r}] must be an array, got "
                f"{type(existing).__name__}"
            )
        if _has_amp_entry(existing):
            already.append(event)
            continue
        cmd = build_command(python_executable, module=module, arg=arg)
        existing.append(_entry_for(event, cmd))
        added.append(event)

    # If nothing changed, short-circuit before doing any disk I/O.
    if not added:
        LOG.info(
            "agent-amp install: already installed on every event (idempotent no-op)"
        )
        return {
            "settings_path": str(sp),
            "backup_path": None,
            "added_events": [],
            "already_present": already,
            "verified": _verify_install(sp),
        }

    # --- Step 6: backup
    backup = _make_backup(sp)

    # --- Step 7: atomic write
    try:
        _atomic_write(sp, data)
    except Exception:
        # If backup exists, leave it behind so the user can recover.
        # Re-raise so the CLI surfaces the failure.
        raise

    # --- Step 8: verify
    ok = _verify_install(sp)
    if not ok:
        # Restore from backup if we made one.
        if backup is not None:
            shutil.copy2(backup, sp)
        raise VerifyFailedError(
            f"post-write verify failed; restored from {backup}"
            if backup is not None
            else "post-write verify failed; no backup existed (settings.json may be empty)"
        )

    return {
        "settings_path": str(sp),
        "backup_path": str(backup) if backup else None,
        "added_events": added,
        "already_present": already,
        "verified": True,
    }


__all__ = [
    "InstallerError",
    "MalformedSettingsError",
    "VerifyFailedError",
    "build_command",
    "install",
]
