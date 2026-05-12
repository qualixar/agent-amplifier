# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Claude Code adapter package — day-0 (per H-2 + H-3 + H-4).

Public surface (preserves import path
``from agent_amplifier.adapters.claude_code import ClaudeCodeAdapter``):

    * :class:`ClaudeCodeAdapter` — file-based memory adapter (legacy, unchanged
      behavior; flips ``INSTALL_PERSISTENT`` to True now that the package owns
      the hook installer too)
    * :class:`StateStore` — SQLite WAL state store for hook adapter
      (multi-Claude-session-safe; per-user-turn schema per H-4)

Hook handlers (script entry points wired into ``~/.claude/settings.json``):
    * :mod:`hooks` — ``UserPromptSubmit`` + ``PreToolUse`` + ``PostToolUse``
    * :mod:`stop_hook` — ``Stop``

Installer (CLI wiring for ``agent-amp install/uninstall claude-code``):
    * :mod:`installer` — 8-step surgical settings.json add (timestamped .bak,
      idempotent, scratch-smoke-first, atomic rename, post-write verify)
    * :mod:`uninstaller` — symmetric removal

The hook→kernel mapping is per-user-turn (H-4):
    UserPromptSubmit → kernel.before_step (amplification injection)
    PreToolUse       → state.record_event (logging only)
    PostToolUse      → state.record_event (logging only)
    Stop             → state.write_outcome (no kernel.after_step in v1.0.0)
"""
from __future__ import annotations

from agent_amplifier.adapters.claude_code.installer import (
    InstallerError,
    MalformedSettingsError,
    VerifyFailedError,
)
from agent_amplifier.adapters.claude_code.memory import (
    _PER_CHUNK_BYTES,
    ClaudeCodeAdapter,
)
from agent_amplifier.adapters.claude_code.state import StateStore

__all__ = [
    "_PER_CHUNK_BYTES",
    "ClaudeCodeAdapter",
    "InstallerError",
    "MalformedSettingsError",
    "StateStore",
    "VerifyFailedError",
]
