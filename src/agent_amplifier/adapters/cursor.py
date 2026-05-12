# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""CursorAdapter — file-based memory binding for Cursor IDE.

.1 row 2; research:

Memory locations read:
    1. ``./.cursor/rules/*.mdc``  — current standard (Cursor v0.45+)
    2. ``./.cursorrules``         — legacy (still read; Cursor docs recommend
                                     migrating to ``.mdc``)

User Rules (Cursor Settings UI) live in an opaque app-config DB and are
intentionally out of scope.

MDC format (= **M**arkdown **D**escriptor **C**onfig):
    ::

        ---
        description: One-line summary
        globs: src/**/*.py
        alwaysApply: true
        ---
        # body markdown ...

Frontmatter is YAML-ish but we DO NOT take a PyYAML dependency. We parse
``key: value`` lines manually inside the ``---`` fenced block. Lists like
``globs: ["a", "b"]`` come through as the raw string; that's fine — the
adapter does not interpret globs, only surfaces them as tags.

Memory write:
    * Writes a new ``.cursor/rules/agent-amplifier-<date>.mdc`` ONLY if
      ``.cursor/rules/`` already exists. We never auto-create the directory
      (anti-surprise; .5.1 fire-and-forget contract).
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from agent_amplifier._internal.redact import redact
from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.adapters._path_safety import (
    safe_open_write,
    safe_read_text,
)
from agent_amplifier.types import RecalledPattern

if TYPE_CHECKING:  # pragma: no cover
    from agent_amplifier.types import Outcome

LOG = logging.getLogger("agent_amplifier.adapters.cursor")

_PER_CHUNK_BYTES: int = 4096
"""Per-chunk soft cap; kernel re-applies its own 8 KB cap."""

_FRONTMATTER_FENCE: str = "---"
"""MDC frontmatter delimiter."""


class CursorAdapter(AdapterBase):
    """Adapter for Cursor IDE."""

    framework_name: ClassVar[str] = "cursor"
    HOST_NAME: ClassVar[str] = "cursor"
    version: ClassVar[str] = "1.0.0"

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls) -> bool:
        """True iff a Cursor rules file is present in CWD."""
        cwd = Path.cwd()
        rules_dir = cwd / ".cursor" / "rules"
        try:
            if rules_dir.is_dir():
                # Any *.mdc file present?
                for _ in rules_dir.glob("*.mdc"):
                    return True
        except OSError:  # pragma: no cover - extremely rare
            pass
        return (cwd / ".cursorrules").is_file()

    # ------------------------------------------------------------------
    # required AdapterBase abstract methods
    # ------------------------------------------------------------------

    def install(self) -> None:  # pragma: no cover
        self._mark_installed()

    def uninstall(self) -> None:  # pragma: no cover
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
        """Read MDC files (current) and ``.cursorrules`` (legacy fallback).

        B3: every read goes through ``safe_read_text`` whose root
        is the project ``./.cursor`` directory (or CWD for the legacy
        ``.cursorrules`` file). Symlinks and out-of-tree resolutions return
        ``None`` and are skipped with a WARNING.
        """
        out: list[RecalledPattern] = []
        q_lower = (query or "").lower()
        cwd = Path.cwd()

        cursor_root = cwd / ".cursor"
        rules_dir = cursor_root / "rules"
        if rules_dir.is_dir():
            try:
                mdc_files = sorted(rules_dir.glob("*.mdc"))
            except OSError as exc:  # pragma: no cover - listing should not fail
                LOG.warning(
                    "cursor recall: cannot list %s: %r",
                    redact(str(rules_dir)),
                    redact(repr(exc)),
                )
                mdc_files = []
            for path in mdc_files:
                pattern = self._parse_mdc(path, q_lower, cursor_root)
                if pattern is not None:
                    out.append(pattern)
                    if len(out) >= limit:
                        return out

        # Legacy fallback when no MDC produced any hit
        if not out:
            legacy = cwd / ".cursorrules"
            if legacy.is_file():
                body = safe_read_text(legacy, cwd)
                if body is None:
                    LOG.warning(
                        "cursor recall: refused unsafe path %s",
                        redact(str(legacy)),
                    )
                    body = ""
                if body and (not q_lower or q_lower in body.lower()):
                    out.append(
                        RecalledPattern(
                            text=body[:_PER_CHUNK_BYTES],
                            tags=("legacy",),
                            source=f"{self.HOST_NAME}:{legacy}",
                        )
                    )
        return out[:limit]

    def default_memory_remember(self, outcome: Outcome) -> None:
        """Write a new MDC note into existing ``.cursor/rules/``.

        No-op when the directory does not exist (anti-surprise).

        B4: refuses to overwrite a symlinked target — a pre-staged
        symlink at ``.cursor/rules/agent-amplifier-<today>.mdc`` could
        otherwise redirect the write. POSIX ``O_NOFOLLOW`` raises ``ELOOP``
        in that case; we catch and log.
        """
        cwd = Path.cwd()
        rules_dir = cwd / ".cursor" / "rules"
        if not rules_dir.is_dir():
            return
        today = date.today().isoformat()
        target = rules_dir / f"agent-amplifier-{today}.mdc"
        snippet = (outcome.query or "")[:100]
        body = (
            f"---\n"
            f"description: Amplifier note from {today}\n"
            f"alwaysApply: false\n"
            f"---\n"
            f"\n"
            f"# Amplifier outcome ({today})\n"
            f"{snippet} -> quality={outcome.quality:.2f}\n"
        )
        # safe_open_write validates the parent
        # chain (cwd -> .cursor -> rules) so a symlinked .cursor or
        # rules dir cannot redirect the write, AND uses O_NOFOLLOW on
        # POSIX so the final segment is symlink-safe.
        fh = safe_open_write(target, allowed_root=cwd)
        if fh is None:
            LOG.warning(
                "cursor remember: refused unsafe write target %s",
                redact(str(target)),
            )
            return
        try:
            fh.write(body)
        except OSError as exc:  # pragma: no cover - rare post-open failure
            LOG.warning(
                "cursor remember: write to %s failed: %r",
                redact(str(target)),
                redact(repr(exc)),
            )
        finally:
            fh.close()

    # ------------------------------------------------------------------
    # internals — naive MDC parser (no PyYAML dep)
    # ------------------------------------------------------------------

    def _parse_mdc(
        self, path: Path, q_lower: str, allowed_root: Path
    ) -> RecalledPattern | None:
        """Parse one ``.mdc`` file into a ``RecalledPattern`` or ``None``.

        Returns ``None`` when (a) the path is a symlink / outside
        ``allowed_root`` (B3), (b) the file cannot be read, OR
        (c) keyword filter says skip. Malformed frontmatter is treated as
        "no metadata" and the entire file is returned as the body chunk.

        MED-5: ``globs`` is normalized to ``list[str]`` and
        surfaced via ``RecalledPattern.metadata["globs"]`` so the kernel
        (or a future scoped-recall layer) can filter by current-file
        context. Both single-string (``globs: src/**/*.py``) and list
        (``globs: ["src/**/*.py", "tests/**/*.py"]``) forms are accepted.
        """
        raw = safe_read_text(path, allowed_root)
        if raw is None:
            LOG.warning(
                "cursor recall: refused unsafe path %s", redact(str(path))
            )
            return None

        meta, body = self._split_frontmatter(raw)
        description = meta.get("description", "")
        always_apply = self._parse_bool(meta.get("alwaysApply", ""))
        globs_list = self._parse_globs(meta.get("globs", ""))

        # Keyword filter — alwaysApply rules ALWAYS pass (Cursor semantics:
        # they're injected on every prompt anyway, so they should always
        # surface in recall too).
        haystack = f"{description}\n{body}".lower()
        if not always_apply and q_lower and q_lower not in haystack:
            return None

        tags: tuple[str, ...] = ()
        if always_apply:
            tags = (*tags, "project-rule")
        if globs_list:
            tags = (*tags, "scoped")

        # MappingProxyType is applied by RecalledPattern.__post_init__; we
        # pass a plain dict + the dataclass freezes it.
        metadata: dict[str, Any] = {}
        if globs_list:
            metadata["globs"] = globs_list

        return RecalledPattern(
            text=body[:_PER_CHUNK_BYTES] if body else raw[:_PER_CHUNK_BYTES],
            tags=tags,
            source=f"{self.HOST_NAME}:{path}",
            metadata=metadata,
        )

    @staticmethod
    def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
        """Split MDC into ``(metadata, body)``.

        Returns ``({}, raw)`` when the file does not start with the
        frontmatter fence. Tolerates trailing whitespace on the fence line.

        MED-5: the ``globs`` value is preserved with its outer
        brackets intact so the list form (``globs: ["a", "b"]``) survives
        the quote-stripping pass. ``_parse_globs`` then normalizes to
        ``list[str]`` regardless of which form was used.
        """
        if not raw.startswith(_FRONTMATTER_FENCE):
            return {}, raw
        lines = raw.splitlines(keepends=True)
        # First line is the opening fence
        meta: dict[str, str] = {}
        end_idx = -1
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == _FRONTMATTER_FENCE:
                end_idx = idx
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" in stripped:
                key, _, val = stripped.partition(":")
                key_clean = key.strip()
                raw_val = val.strip()
                # MED-5: preserve list-form values verbatim. A
                # leading "[" + trailing "]" is the YAML flow-list shape;
                # do NOT strip outer quotes from list elements at this
                # layer. The dedicated _parse_globs helper handles both
                # single-string and list shapes.
                if raw_val.startswith("[") and raw_val.endswith("]"):
                    meta[key_clean] = raw_val
                else:
                    meta[key_clean] = raw_val.strip('"').strip("'")
        if end_idx < 0:
            # Malformed frontmatter (no closing fence) — treat as plain body
            return {}, raw
        body = "".join(lines[end_idx + 1 :])
        # Drop a single leading blank line for cleanliness
        if body.startswith("\n"):
            body = body[1:]
        return meta, body

    @staticmethod
    def _parse_bool(val: str) -> bool:
        """Parse ``"true"`` / ``"True"`` / ``"yes"`` / ``"1"`` as ``True``.

        Anything else is ``False``. Strict on purpose — silent coercion of
        empty strings to ``False`` is the desired V1 behavior.
        """
        return val.strip().lower() in {"true", "yes", "1"}

    @staticmethod
    def _parse_globs(val: str) -> list[str]:
        """Normalize a frontmatter ``globs`` value to ``list[str]``.

        MED-5. Accepts:

            globs: src/**/*.py                  -> ["src/**/*.py"]
            globs: "src/**/*.py"                -> ["src/**/*.py"]
            globs: ["src/**/*.py", "tests/**"]  -> ["src/**/*.py", "tests/**"]
            globs: []                           -> []
            globs:                              -> []

        We deliberately avoid pulling in PyYAML; this naive splitter is
        good enough for the documented Cursor MDC formats. Any glob value
        that fails to parse degrades to a single-element list of the raw
        string — never raises, always returns a ``list[str]``.
        """
        s = val.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            # Split on commas; strip whitespace and outer quotes.
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
            return [p for p in parts if p]
        # Single-string form — maybe still wrapped in quotes.
        return [s.strip('"').strip("'")]
