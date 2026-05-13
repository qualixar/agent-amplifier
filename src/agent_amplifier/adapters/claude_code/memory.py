# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""ClaudeCodeAdapter — file-based memory binding for Anthropic's Claude Code CLI.

.1 row 1; research:

Memory locations read (in priority order):
    1. ``./CLAUDE.md``           — project-level instructions (checked into git)
    2. ``./MEMORY.md``           — amplifier convention: short handoff index
       (~50 lines), maintained alongside CLAUDE.md by hosts that follow this
       pattern.
    3. ``~/.claude/CLAUDE.md``   — user-global instructions

Memory write target:
    * ``./CLAUDE.md`` only — and only if it already exists. We never
      auto-create user files (anti-surprise). Failure is fire-and-forget
      WARNING per .5.1.

Out of scope (intentional):
    * ``@imports`` resolution. Claude CLI walks imports itself; the adapter
      surfaces only the literal file content. A V1.1 enhancement could walk
      them. Documented in research file gotchas.
    * ``~/.claude/projects/<hash>/`` transcript JSONL. Out of scope for
      V1 — recall_safety would need richer parsing.
    * The MCP-backed ``mcp__superlocalmemory__*`` channel. SLM is a separate
      memory plane (see ``examples/slm_provider.py``).

The kernel applies ``recall_safety.apply_recall_safety`` to every returned
chunk; we DO NOT call it from the adapter. We DO cap chunks at
``_PER_CHUNK_BYTES`` so a multi-MB CLAUDE.md cannot dominate one recall
batch — the kernel's 8 KB cap is per-chunk; this adapter cap is a courtesy
ceiling so the kernel doesn't truncate a useful section's leading bytes.
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from agent_amplifier._internal.redact import redact
from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.adapters._path_safety import (
    safe_open_append,
    safe_read_text,
)
from agent_amplifier.types import RecalledPattern

if TYPE_CHECKING:  # pragma: no cover
    from agent_amplifier.types import Outcome

LOG = logging.getLogger("agent_amplifier.adapters.claude_code")

# Per-chunk soft cap. Half the kernel's MAX_RECALLED_TEXT_BYTES so callers
# get two sibling chunks rather than one truncated 8 KB wall.
_PER_CHUNK_BYTES: int = 4096

# H2 split — naive, deliberately. The contract is "naive keyword rank for V1";
# the universal memory plane handles smarter ranking later.
_H2_SPLIT_RE: re.Pattern[str] = re.compile(r"^## ", flags=re.MULTILINE)


class ClaudeCodeAdapter(AdapterBase):
    """Adapter for Anthropic's Claude Code CLI.

    day-0 (per H-2 + H-3): ``install()`` / ``uninstall()`` now delegate
    to ``installer.install()`` / ``uninstaller.uninstall()`` which surgically
    add/remove our 4 hook entries (UserPromptSubmit, PreToolUse, PostToolUse,
    Stop) from ``~/.claude/settings.json`` via the 8-step atomic protocol.
    INSTALL_PERSISTENT flips to True so the CLI prints "installed" honestly.
    """

    framework_name: ClassVar[str] = "claude_code"
    """ABC-regex compliant slug. NOT used for source strings — see HOST_NAME."""

    HOST_NAME: ClassVar[str] = "claude-code"
    """Public source-string slug used in ``RecalledPattern.source``."""

    version: ClassVar[str] = "1.0.0"

    is_single_iteration: ClassVar[bool] = True
    """Claude Code fires the ``UserPromptSubmit`` injection point exactly once
    per user turn. The kernel routes this adapter through the structured
    single-turn envelope (XML phase staging + adaptive thinking + optional
    subagent dispatch for high-complexity tasks) so the full Agent Amplifier
    value lands inside that single host turn."""

    INSTALL_PERSISTENT: ClassVar[bool] = True
    """day-0: flipped True. ``install()`` writes hook entries into
    ``~/.claude/settings.json``; the CLI prints "installed: claude_code"
    instead of the file-only "ready" message."""

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls) -> bool:
        """True if Claude Code is installed on this host.

        Heuristics (any one is sufficient):
            * ``~/.claude/settings.json`` exists (CLI installed)
            * ``CLAUDE_CODE`` environment variable is set
        """
        try:
            settings = Path.home() / ".claude" / "settings.json"
            if settings.exists():
                return True
        except OSError:
            # Path.home() can theoretically raise on broken HOME;
            # fall through to env-var check.
            pass
        return bool(os.environ.get("CLAUDE_CODE"))

    # ------------------------------------------------------------------
    # required AdapterBase abstract methods — day-0:
    # delegate to installer/uninstaller for the hook side-effects.
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Install hook entries into ``~/.claude/settings.json`` (delegates).

        The 8-step surgical protocol lives in ``installer.install``. After
        delegation succeeds we still set the in-process ``_installed`` flag
        so ``is_installed()`` returns True until ``uninstall()`` is called.
        """
        # Local import to avoid a module-init cycle: installer imports
        # nothing from memory, but tests sometimes monkeypatch installer
        # at import time and we want a fresh resolution per call.
        from agent_amplifier.adapters.claude_code import installer as _installer

        _installer.install()
        self._mark_installed()

    def uninstall(self) -> None:
        """Remove hook entries from ``~/.claude/settings.json`` (delegates)."""
        from agent_amplifier.adapters.claude_code import uninstaller as _uninstaller

        _uninstaller.uninstall()
        self._mark_uninstalled()

    def on_before_step(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover - identity
        return context

    def on_after_step(
        self,
        context: dict[str, Any],
        result: dict[str, Any] | str,
    ) -> dict[str, Any]:  # pragma: no cover - identity
        return {"action": "continue"}

    # ------------------------------------------------------------------
    # memory plane (.5.1)
    # ------------------------------------------------------------------

    def default_memory_recall(
        self, query: str, limit: int = 3
    ) -> list[RecalledPattern]:
        """Read CLAUDE.md / MEMORY.md / ~/.claude/CLAUDE.md and rank by query.

        B3: each read goes through ``safe_read_text`` which refuses
        symlinks and paths that resolve outside the expected root.
        """
        candidates = self._candidate_paths_with_roots()
        out: list[RecalledPattern] = []
        q_lower = (query or "").lower()
        for path, root in candidates:
            try:
                if not path.is_file():
                    continue
            except OSError as exc:  # pragma: no cover - extremely rare
                LOG.warning(
                    "claude-code recall: stat %s failed: %r",
                    redact(str(path)),
                    redact(repr(exc)),
                )
                continue
            text = safe_read_text(path, root)
            if text is None:
                LOG.warning(
                    "claude-code recall: refused unsafe path %s",
                    redact(str(path)),
                )
                continue
            chunks = self._rank_chunks(text, q_lower)
            for chunk in chunks:
                out.append(
                    RecalledPattern(
                        text=chunk[:_PER_CHUNK_BYTES],
                        source=f"{self.HOST_NAME}:{path}",
                    )
                )
                if len(out) >= limit:
                    return out
        return out[:limit]

    def default_memory_remember(self, outcome: Outcome) -> None:
        """Append a brief outcome summary to ``./MEMORY.md``, auto-creating it.

        H-5 Mode 3 fallback (non-SLM users): write target is ``MEMORY.md``
        (amp's convention file from this adapter's docstring), NEVER
        ``CLAUDE.md``. Auto-create if missing so the closed-loop pattern
        works for users who haven't created MEMORY.md yet.

        B4: append goes through ``safe_open_append`` which uses
        ``O_NOFOLLOW`` on POSIX so a symlink swapped in between the
        ``is_file()`` check and the open raises ``OSError`` instead of
        redirecting writes (SEC-04 attack vector).
        """
        cwd = Path.cwd()
        path = cwd / "MEMORY.md"
        # Auto-create when missing — closed-loop pattern requires the file
        # to exist on the next turn for ClaudeCodeAdapter recall to read it.
        if not path.exists():
            try:
                path.write_text(
                    "# Agent Amplifier — Project Memory\n"
                    "\n"
                    "This file is appended to by Agent Amplifier on each user turn.\n"
                    "Delete to reset; edit freely; safe to commit if you want\n"
                    "amp's session history tracked across machines.\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                LOG.warning(
                    "claude-code remember: cannot create MEMORY.md at %s: %r",
                    redact(str(path)),
                    redact(repr(exc)),
                )
                return
        if not path.is_file():
            # Path resolved to something that is NOT a regular file
            # (directory, symlink target chain ending elsewhere). Refuse.
            return
        snippet = (outcome.query or "")[:100]
        block = (
            f"\n## Amplifier note ({date.today().isoformat()})\n"
            f"{snippet} -> quality={outcome.quality:.2f}\n"
        )
        # pass cwd as allowed_root so the parent
        # chain is validated — a symlinked project dir cannot redirect
        # the append even when the final filename is a regular file.
        fh = safe_open_append(path, allowed_root=cwd)
        if fh is None:
            LOG.warning(
                "claude-code remember: refused unsafe append target %s",
                redact(str(path)),
            )
            return
        try:
            fh.write(block)
        except OSError as exc:
            LOG.warning(
                "claude-code remember: append to %s failed: %r",
                redact(str(path)),
                redact(repr(exc)),
            )
        finally:
            fh.close()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _candidate_paths() -> tuple[Path, ...]:
        """Resolve the read-priority list. Errors collapse to skip-this-source."""
        return tuple(p for p, _root in ClaudeCodeAdapter._candidate_paths_with_roots())

    @staticmethod
    def _candidate_paths_with_roots() -> tuple[tuple[Path, Path], ...]:
        """Same as ``_candidate_paths`` but each entry pairs the path with the
        allowed root the resolved path MUST live inside (B3).

        * Project files (CLAUDE.md / MEMORY.md) → allowed root = CWD
        * User-global file (~/.claude/CLAUDE.md) → allowed root = ~/.claude
        """
        out: list[tuple[Path, Path]] = []
        cwd = Path.cwd()
        out.append((cwd / "CLAUDE.md", cwd))
        out.append((cwd / "MEMORY.md", cwd))
        # Path.home() can theoretically raise on a broken HOME; silently skip.
        with contextlib.suppress(OSError):  # pragma: no cover - rare
            user_root = Path.home() / ".claude"
            out.append((user_root / "CLAUDE.md", user_root))
        return tuple(out)

    @staticmethod
    def _rank_chunks(text: str, q_lower: str) -> list[str]:
        """Split on H2 headings, keep sections whose lower-case form matches.

        If the text has NO ``## `` headings, the entire body is returned as
        one chunk (still subject to keyword filter unless query is empty).
        Empty query keeps every chunk so a "show me everything" recall path
        works.
        """
        if not text:
            return []
        sections = _H2_SPLIT_RE.split(text)
        # The first split element is the prologue (text before the first H2).
        # Keep it iff it has content; subsequent elements are headings+body.
        chunks = [s for s in sections if s.strip()]
        if not q_lower:
            return chunks
        return [s for s in chunks if q_lower in s.lower()]
