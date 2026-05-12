# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""SLMAdapter — composed-pipeline (Mode 2) adapter for SuperLocalMemory.

Bridges the amplifier's kernel ``memory_recall`` callback to SLM's
``slm session-context`` CLI command. The kernel's memory plane therefore
returns SLM-recalled facts when SLM is installed; this is the difference
between Mode 1 (adjacent) and Mode 2 (composed).

Why ``slm session-context`` and not ``slm recall``:
    * ``slm recall`` runs the full 4-channel + spreading-activation retrieval
      pipeline. Production-quality but ~2-9s on a typical machine — too slow
      for amp's UserPromptSubmit hook, which has a sub-300ms budget so the
      user does not perceive latency before the model starts thinking.
    * ``slm session-context`` is SLM's pre-computed query-aware recall path
      designed exactly for hooks. Returns formatted markdown in ~70-100ms.

The adapter is an intentionally thin shim so amp's ``superlocalmemory``
dependency stays loosely coupled — amp shells out to the CLI rather than
importing SLM internals. SLM API changes between v3.4 and v4 will not
break amp; only the CLI contract matters. Pin range
``superlocalmemory>=3.4,<4`` as an OPTIONAL extras dependency.

Fail-open: if ``slm`` is missing, errors out, or returns malformed output,
``default_memory_recall`` returns ``[]`` — the kernel falls through to
its no-memory path. The amplifier keeps working.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from typing import TYPE_CHECKING, Any, ClassVar

from agent_amplifier._internal.redact import redact
from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.types import RecalledPattern

if TYPE_CHECKING:  # pragma: no cover
    from agent_amplifier.types import Outcome

LOG = logging.getLogger("agent_amplifier.adapters.slm")

# Per-chunk soft cap. Same shape as the Claude Code adapter so cross-adapter
# memory plane behavior is uniform.
_PER_CHUNK_BYTES: int = 4096

# H2 split — same shape as ClaudeCodeAdapter._rank_chunks.
_H2_SPLIT_RE: re.Pattern[str] = re.compile(r"^## ", flags=re.MULTILINE)

# CLI invocation budget. ``slm session-context`` is sub-100ms in practice;
# the timeout exists as a safety net so a hung SLM process can never block
# amp's UserPromptSubmit hook.
_CLI_TIMEOUT_SECONDS: float = 5.0
_REMEMBER_TIMEOUT_SECONDS: float = 10.0


def detect_slm() -> bool:
    """Return True iff the ``slm`` CLI is on PATH.

    Used by hosts (the Claude Code adapter) to decide whether to wire
    SLMAdapter as the ``memory_recall`` source. NEVER raises.
    """
    try:
        return shutil.which("slm") is not None
    except OSError:  # pragma: no cover - extremely defensive
        return False


def _rank_chunks(text: str, q_lower: str) -> list[str]:
    """Split markdown on H2 headings; keep sections matching the query.

    Mirrors ClaudeCodeAdapter._rank_chunks so the kernel sees a consistent
    chunk shape regardless of which adapter ran.
    """
    if not text:
        return []
    sections = _H2_SPLIT_RE.split(text)
    chunks = [s for s in sections if s.strip()]
    if not q_lower:
        return chunks
    return [s for s in chunks if q_lower in s.lower()]


class SLMAdapter(AdapterBase):
    """Adapter that surfaces SLM's session-context recall to amp's kernel.

    Concrete responsibilities:

    * ``default_memory_recall(query, limit)`` — runs ``slm session-context
      "<query>"`` and chunks the markdown response.
    * ``default_memory_remember(outcome)`` — runs ``slm remember "<summary>"``
      with tags so future amp turns can recall this outcome via Mode 2.

    Lifecycle methods (``install`` / ``uninstall``) are no-ops because SLM
    integration is purely process-local — there is no persistent registration
    on the host beyond the existence of the ``slm`` binary on PATH.
    """

    framework_name: ClassVar[str] = "slm"
    """ABC-regex compliant slug. Free-tier adapter — same convention as
    ClaudeCodeAdapter's ``claude_code``."""

    HOST_NAME: ClassVar[str] = "superlocalmemory"
    """Public source-string slug used in ``RecalledPattern.source``."""

    version: ClassVar[str] = "1.0.0"

    INSTALL_PERSISTENT: ClassVar[bool] = False
    """SLMAdapter writes nothing to disk on install — it only relies on the
    ``slm`` CLI being on PATH. The CLI's "ready: slm" message is honest."""

    @classmethod
    def detect(cls) -> bool:
        """True iff the ``slm`` CLI is callable on this host."""
        return detect_slm()

    # ------------------------------------------------------------------
    # required AdapterBase abstract methods (no-op for marker-only)
    # ------------------------------------------------------------------

    def install(self) -> None:
        self._mark_installed()

    def uninstall(self) -> None:
        self._mark_uninstalled()

    def on_before_step(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover - identity, exercised by base
        return context

    def on_after_step(
        self,
        context: dict[str, Any],
        result: dict[str, Any] | str,
    ) -> dict[str, Any]:  # pragma: no cover - identity
        return {"action": "continue"}

    # ------------------------------------------------------------------
    # memory plane (Mode 2)
    # ------------------------------------------------------------------

    def default_memory_recall(
        self, query: str, limit: int = 3
    ) -> list[RecalledPattern]:
        """Run ``slm session-context "<query>"`` and return chunked recalls.

        MUST NOT raise. On any failure (binary missing, timeout, non-zero
        exit, malformed output) returns ``[]`` and logs at WARNING.
        """
        if not detect_slm():
            return []
        q = (query or "").strip()
        try:
            proc = subprocess.run(
                ["slm", "session-context", q] if q else ["slm", "session-context"],
                capture_output=True,
                text=True,
                timeout=_CLI_TIMEOUT_SECONDS,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            LOG.warning(
                "slm session-context failed: %s", redact(repr(exc))
            )
            return []
        if proc.returncode != 0:
            LOG.warning(
                "slm session-context exit=%d stderr=%s",
                proc.returncode,
                redact(proc.stderr.strip()[:200]),
            )
            return []
        text = proc.stdout
        chunks = _rank_chunks(text, q.lower())
        out: list[RecalledPattern] = []
        for chunk in chunks:
            out.append(
                RecalledPattern(
                    text=chunk[:_PER_CHUNK_BYTES],
                    source=f"{self.HOST_NAME}:session-context",
                )
            )
            if len(out) >= limit:
                break
        return out[:limit]

    def default_memory_remember(self, outcome: Outcome) -> None:
        """Persist amp's outcome summary into SLM via ``slm remember``.

        Mode 3 closed-loop write-back. Fire-and-forget, fail-open. Subject
        to ``_REMEMBER_TIMEOUT_SECONDS`` so a hung SLM cannot block Stop.
        """
        if not detect_slm():
            return
        try:
            content = self._format_remember_content(outcome)
        except Exception as exc:  # pragma: no cover - guarded by Outcome dataclass
            LOG.warning("slm remember formatter failed: %s", redact(repr(exc)))
            return
        if not content.strip():
            return
        try:
            proc = subprocess.run(
                ["slm", "remember", content],
                capture_output=True,
                text=True,
                timeout=_REMEMBER_TIMEOUT_SECONDS,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            LOG.warning("slm remember failed: %s", redact(repr(exc)))
            return
        if proc.returncode != 0:
            LOG.warning(
                "slm remember exit=%d stderr=%s",
                proc.returncode,
                redact(proc.stderr.strip()[:200]),
            )

    @staticmethod
    def _format_remember_content(outcome: Outcome) -> str:
        """Build the 1-line summary that amp persists into SLM.

        Schema (deliberately compact):
            ``[amp] turn outcome: <effort>/<query-prefix> →
              quality=<q> converged=<c> iters=<n> tokens=<t>``

        Never includes raw user content beyond the redacted query prefix.
        """
        q = (outcome.query or "")[:120]
        effort = getattr(outcome.effort, "value", str(outcome.effort))
        return (
            f"[amp] turn outcome: effort={effort} query={redact(q)!r} "
            f"quality={outcome.quality:.2f} "
            f"converged={'yes' if outcome.converged else 'no'} "
            f"iters={outcome.iterations} "
            f"tokens={outcome.tokens_used}"
        )
