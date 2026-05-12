# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""GitHubCopilotAdapter — file-based memory binding for GitHub Copilot.

.1 row 3; research:

Memory locations read:
    1. ``./.github/copilot-instructions.md``         — repo-wide, auto-loaded
    2. ``./.github/instructions/*.instructions.md``  — path-scoped (with YAML
                                                       frontmatter ``applyTo:``)

MED-6: scoped ``*.instructions.md`` files may carry an ``applyTo:``
key in YAML frontmatter (Copilot path-glob convention). We parse it and
surface it via ``RecalledPattern.metadata["apply_to"]``. The kernel
intentionally does NOT filter recalls by current-file context yet — over
recall is safe for V1; the metadata simply preserves the signal so a future
scoped-recall layer can use it.

Out of scope (intentional):
    * Personal IDE-level instructions (VSCode setting
      ``github.copilot.chat.codeGeneration.instructions``) — opaque settings DB.
    * Copilot's internal "agentic memory" tool store — not a file API.

Memory write target:
    * ``./.github/copilot-instructions.md`` — appends only when the file
      already exists. We never auto-create ``./.github/`` (anti-surprise).
"""
from __future__ import annotations

import logging
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

LOG = logging.getLogger("agent_amplifier.adapters.github_copilot")

_PER_CHUNK_BYTES: int = 4096

_H2_SPLIT_RE: re.Pattern[str] = re.compile(r"^## ", flags=re.MULTILINE)

_FRONTMATTER_FENCE: str = "---"
"""YAML frontmatter delimiter used by Copilot ``*.instructions.md``."""

_APPLY_TO_KEY: str = "applyTo"
"""Frontmatter key whose value is a Copilot path-glob (single string).

MED-6. Cf. Microsoft docs:
``https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot``.
Only single-string form is documented for ``applyTo`` (unlike Cursor
``globs:`` which accepts both single + list); we still tolerate a list
because the parser is generous and the metadata layer is downstream."""


class GitHubCopilotAdapter(AdapterBase):
    """Adapter for GitHub Copilot Workspace + Chat."""

    framework_name: ClassVar[str] = "github_copilot"
    HOST_NAME: ClassVar[str] = "github-copilot"
    version: ClassVar[str] = "1.0.0"

    REPO_INSTRUCTIONS: ClassVar[str] = ".github/copilot-instructions.md"
    SCOPED_DIR: ClassVar[str] = ".github/instructions"
    SCOPED_GLOB: ClassVar[str] = "*.instructions.md"

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls) -> bool:
        """True iff a Copilot instructions file is present in CWD."""
        cwd = Path.cwd()
        if (cwd / cls.REPO_INSTRUCTIONS).is_file():
            return True
        scoped = cwd / cls.SCOPED_DIR
        if scoped.is_dir():
            try:
                for _ in scoped.glob(cls.SCOPED_GLOB):
                    return True
            except OSError:  # pragma: no cover - extremely rare
                pass
        return False

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
        """Read repo instructions + scoped ``*.instructions.md``.

        B3: every read goes through ``safe_read_text`` whose root is
        ``./.github``; symlinks and out-of-tree resolutions are skipped.
        """
        out: list[RecalledPattern] = []
        q_lower = (query or "").lower()
        cwd = Path.cwd()
        gh_root = cwd / ".github"

        # 1. Repo-wide instructions
        repo_path = cwd / self.REPO_INSTRUCTIONS
        if repo_path.is_file():
            for chunk in self._read_and_rank(repo_path, q_lower, gh_root):
                out.append(
                    RecalledPattern(
                        text=chunk[:_PER_CHUNK_BYTES],
                        source=f"{self.HOST_NAME}:{repo_path}",
                    )
                )
                if len(out) >= limit:
                    return out

        # 2. Path-scoped instructions. MED-6: parse frontmatter
        # to extract the Copilot ``applyTo:`` glob and surface it via
        # ``RecalledPattern.metadata["apply_to"]``. We do NOT filter by
        # current-file context yet (kernel doesn't expose that); the
        # metadata preserves the signal for future scoped-recall logic.
        scoped_dir = cwd / self.SCOPED_DIR
        if scoped_dir.is_dir():
            try:
                scoped_files = sorted(scoped_dir.glob(self.SCOPED_GLOB))
            except OSError as exc:  # pragma: no cover
                LOG.warning(
                    "github-copilot recall: cannot list %s: %r",
                    redact(str(scoped_dir)),
                    redact(repr(exc)),
                )
                scoped_files = []
            for path in scoped_files:
                meta, body_text = self._read_with_frontmatter(path, gh_root)
                if body_text is None:
                    continue
                # Extra metadata for the kernel/observability layer. Only
                # populate if applyTo is present so adapters that ship
                # frontmatter-less files keep an empty metadata dict.
                pattern_metadata: dict[str, Any] = {}
                apply_to = self._parse_apply_to(meta.get(_APPLY_TO_KEY, ""))
                if apply_to:
                    pattern_metadata["apply_to"] = apply_to
                for chunk in self._rank_chunks(body_text, q_lower):
                    out.append(
                        RecalledPattern(
                            text=chunk[:_PER_CHUNK_BYTES],
                            tags=("scoped",),
                            source=f"{self.HOST_NAME}:{path}",
                            metadata=pattern_metadata,
                        )
                    )
                    if len(out) >= limit:
                        return out
        return out[:limit]

    def default_memory_remember(self, outcome: Outcome) -> None:
        """Append outcome summary to ``.github/copilot-instructions.md`` if present.

        B4: append goes through ``safe_open_append`` (POSIX
        ``O_NOFOLLOW``) so a symlink swapped in for the target raises
        ``OSError`` instead of redirecting writes (SEC-04).
        """
        cwd = Path.cwd()
        path = cwd / self.REPO_INSTRUCTIONS
        if not path.is_file():
            return
        snippet = (outcome.query or "")[:100]
        block = (
            f"\n## Amplifier note ({date.today().isoformat()})\n"
            f"{snippet} -> quality={outcome.quality:.2f}\n"
        )
        # parent-chain symlink defense via allowed_root.
        fh = safe_open_append(path, allowed_root=cwd)
        if fh is None:
            LOG.warning(
                "github-copilot remember: refused unsafe append target %s",
                redact(str(path)),
            )
            return
        try:
            fh.write(block)
        except OSError as exc:
            LOG.warning(
                "github-copilot remember: append to %s failed: %r",
                redact(str(path)),
                redact(repr(exc)),
            )
        finally:
            fh.close()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _read_and_rank(
        path: Path, q_lower: str, allowed_root: Path
    ) -> list[str]:
        """Read ``path`` and return H2-split chunks filtered by ``q_lower``.

        B3: read goes through ``safe_read_text`` so symlinks /
        out-of-tree resolutions are refused.
        """
        text = safe_read_text(path, allowed_root)
        if text is None:
            LOG.warning(
                "github-copilot recall: refused unsafe path %s",
                redact(str(path)),
            )
            return []
        if not text:
            return []
        return GitHubCopilotAdapter._rank_chunks(text, q_lower)

    @staticmethod
    def _rank_chunks(text: str, q_lower: str) -> list[str]:
        """Split ``text`` on H2 sections and filter by ``q_lower``.

        Pulled out of ``_read_and_rank`` in MED-6 so the
        path-scoped recall path can apply frontmatter parsing once and
        then rank the body separately.
        """
        if not text:
            return []
        sections = _H2_SPLIT_RE.split(text)
        chunks = [s for s in sections if s.strip()]
        if not q_lower:
            return chunks
        return [s for s in chunks if q_lower in s.lower()]

    @staticmethod
    def _read_with_frontmatter(
        path: Path, allowed_root: Path
    ) -> tuple[dict[str, str], str | None]:
        """Read ``path`` and split off any YAML-style frontmatter.

        MED-6. Returns ``(meta, body)``:

        - ``meta`` is the parsed frontmatter dict (empty if no fence).
        - ``body`` is the post-fence content, OR the entire file when
          there is no frontmatter. ``None`` when the file is unsafe
          to read (symlink / out-of-tree).

        We deliberately do not depend on PyYAML — the Copilot frontmatter
        keys we care about (``applyTo``) are simple ``key: value`` lines.
        """
        text = safe_read_text(path, allowed_root)
        if text is None:
            LOG.warning(
                "github-copilot recall: refused unsafe path %s",
                redact(str(path)),
            )
            return {}, None
        if not text:
            return {}, ""
        if not text.startswith(_FRONTMATTER_FENCE):
            return {}, text
        meta: dict[str, str] = {}
        lines = text.splitlines(keepends=True)
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
                meta[key.strip()] = val.strip().strip('"').strip("'")
        if end_idx < 0:
            # Malformed frontmatter — treat as plain body.
            return {}, text
        body = "".join(lines[end_idx + 1 :])
        if body.startswith("\n"):
            body = body[1:]
        return meta, body

    @staticmethod
    def _parse_apply_to(val: str) -> str:
        """Normalize the ``applyTo:`` frontmatter value to a glob string.

        MED-6. Copilot's documented form is single-string
        (``applyTo: "**/*.ts"``) so we return the raw stripped value.
        Empty / whitespace-only / unset returns ``""``.
        """
        return val.strip().strip('"').strip("'")
