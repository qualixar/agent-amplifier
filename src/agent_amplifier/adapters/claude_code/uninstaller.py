# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Symmetric uninstaller for the Claude Code hook adapter (day-0).

Removes ONLY entries that carry the amp marker (either the ``_amp_marker``
field or the marker substring inside the ``command`` string). Never touches
other hooks. Same atomic-temp + os.replace pattern as installer.py — we do
not Write the user's settings.json directly.

Idempotent: running uninstall when nothing is installed is a no-op.
Reversible: a timestamped ``.amp-bak.<UTC>`` is taken BEFORE every write,
so a misbehaving uninstall can be reversed by ``cp <bak> settings.json``.

The `removed_events` list returned mirrors `added_events` from installer's
return shape so the CLI / tests can confirm symmetry.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_amplifier.adapters.claude_code.installer import (
    _AMP_MARKER,
    _HOOK_TARGETS,
    MalformedSettingsError,
    VerifyFailedError,
    _atomic_write,
    _make_backup,
    _read_settings,
)

LOG = logging.getLogger("agent_amplifier.adapters.claude_code.uninstaller")


def _is_amp_entry(group: Any) -> bool:
    """True if this hook-group entry carries the amp marker.

    Mirror of installer._has_amp_entry but operating on a single group
    rather than a list of groups (so we can use it inside list-comprehension
    filters).
    """
    if not isinstance(group, dict):
        return False
    inner = group.get("hooks")
    if not isinstance(inner, list):
        return False
    for h in inner:
        if not isinstance(h, dict):
            continue
        if h.get("_amp_marker") == _AMP_MARKER:
            return True
        cmd = h.get("command")
        if isinstance(cmd, str) and _AMP_MARKER in cmd:
            return True
    return False


def _verify_uninstall(settings_path: Path) -> bool:
    """Re-read settings.json and confirm no event carries our marker."""
    try:
        data = _read_settings(settings_path)
    except MalformedSettingsError:
        return False
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        # No hooks dict means no amp entries possible.
        return True
    for event, _mod, _arg in _HOOK_TARGETS:
        groups = hooks.get(event, [])
        if not isinstance(groups, list):
            continue
        if any(_is_amp_entry(g) for g in groups):
            return False
    return True


def uninstall(
    settings_path: Path | str | None = None,
) -> dict[str, Any]:
    """Remove amp's hook entries from settings.json. Idempotent.

    Returns
    -------
    dict with ``settings_path``, ``backup_path`` (or None), ``removed_events``
    (events from which at least one amp entry was removed), and ``verified``
    boolean.

    Raises
    ------
    MalformedSettingsError
        If settings.json is unparseable.
    VerifyFailedError
        If the post-write re-read still shows amp markers (file restored).
    """
    import shutil

    from agent_amplifier.adapters.claude_code.installer import (
        _DEFAULT_SETTINGS_PATH,
    )

    sp = Path(settings_path) if settings_path else _DEFAULT_SETTINGS_PATH

    if not sp.exists():
        # Nothing to do — idempotent path.
        return {
            "settings_path": str(sp),
            "backup_path": None,
            "removed_events": [],
            "verified": True,
        }

    data = _read_settings(sp)
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        # No hooks at all means no amp entries to remove.
        return {
            "settings_path": str(sp),
            "backup_path": None,
            "removed_events": [],
            "verified": True,
        }

    removed: list[str] = []
    changed = False
    for event, _mod, _arg in _HOOK_TARGETS:
        groups = hooks.get(event)
        if not isinstance(groups, list):
            continue
        kept = [g for g in groups if not _is_amp_entry(g)]
        if len(kept) != len(groups):
            removed.append(event)
            changed = True
        if kept:
            hooks[event] = kept
        else:
            # Drop empty event arrays so we leave settings.json minimal.
            del hooks[event]

    # If hooks is now empty, drop it too (fully restore pre-install shape).
    if not hooks:
        data.pop("hooks", None)

    if not changed:
        return {
            "settings_path": str(sp),
            "backup_path": None,
            "removed_events": [],
            "verified": _verify_uninstall(sp),
        }

    backup = _make_backup(sp)
    try:
        _atomic_write(sp, data)
    except Exception:
        raise

    ok = _verify_uninstall(sp)
    if not ok:
        if backup is not None:
            shutil.copy2(backup, sp)
        raise VerifyFailedError(
            f"post-uninstall verify still finds amp markers; restored from {backup}"
            if backup is not None
            else "post-uninstall verify failed; no backup existed"
        )

    return {
        "settings_path": str(sp),
        "backup_path": str(backup) if backup else None,
        "removed_events": removed,
        "verified": True,
    }


__all__ = [
    "uninstall",
]
