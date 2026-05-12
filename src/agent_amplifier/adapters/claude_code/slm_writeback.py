# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Mode-3 closed-loop writeback (per H-5).

Stop hook calls :func:`write_outcome_to_slm` after persisting the per-turn
outcome to SQLite. The write goes to SLM so future amp turns (in this or
another Claude Code session) can recall it via Mode 2's SLMAdapter.

Decoupled from kernel:
    The kernel object does not survive across Claude Code hook subprocesses,
    so we cannot route this through ``kernel.finalize`` + ``memory_remember``
    callbacks. Instead, the stop hook constructs a 1-line summary directly
    and shells out to ``slm remember``. Same end state — SLM has a row
    representing this turn — without the cross-process kernel resurrection
    cost.

Fail-open: if SLM is missing, fails, or times out, the function returns
silently. The user's Claude Code session is never blocked by SLM
unavailability.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

from agent_amplifier._internal.redact import redact

LOG = logging.getLogger("agent_amplifier.adapters.claude_code.slm_writeback")

# Latency budget for SLM remember. Stop hook is non-user-blocking but we
# still cap so a runaway SLM cannot keep the subprocess alive forever.
_REMEMBER_TIMEOUT_SECONDS: float = 10.0


def _slm_available() -> bool:
    try:
        return shutil.which("slm") is not None
    except OSError:  # pragma: no cover - extremely defensive
        return False


def _format_summary(
    session_id: str,
    turn_id: int,
    envelope: dict[str, Any],
    tool_calls: int,
    tool_results: int,
    duration_ms: int,
    converged: bool,
) -> str:
    """1-line amp-tagged summary persisted into SLM as a memory.

    The schema is deliberately compact and stable across releases — SLM
    indexes content by entity + lexical channels; redundant fields would
    only inflate the index without improving recall.
    """
    complexity = envelope.get("classification_complexity") or "unknown"
    domain = envelope.get("classification_domain") or "general"
    persona = envelope.get("persona") or "default"
    phase = envelope.get("phase") or "unknown"
    prompt_redacted = (envelope.get("user_prompt_redacted") or "")[:160]
    return (
        f"[amp] turn-outcome session={redact(session_id)} turn={turn_id} "
        f"complexity={complexity} domain={domain} "
        f"persona={persona[:80]} phase={phase} "
        f"tools={tool_calls}/{tool_results} duration_ms={duration_ms} "
        f"converged={'yes' if converged else 'no'} "
        f"prompt={redact(prompt_redacted)!r}"
    )


def write_outcome_to_slm(
    session_id: str,
    turn_id: int,
    *,
    envelope: dict[str, Any],
    tool_calls: int,
    tool_results: int,
    duration_ms: int,
    converged: bool,
) -> bool:
    """Write a per-turn outcome summary into SLM.

    Returns ``True`` if SLM accepted the write (best-effort), ``False`` on
    any failure path (SLM missing, timeout, non-zero exit). Caller MUST
    NOT depend on the return value for correctness — this is fire-and-forget
    closed-loop instrumentation.
    """
    if not _slm_available():
        return False
    summary = _format_summary(
        session_id,
        turn_id,
        envelope,
        tool_calls,
        tool_results,
        duration_ms,
        converged,
    )
    try:
        proc = subprocess.run(
            ["slm", "remember", summary],
            capture_output=True,
            text=True,
            timeout=_REMEMBER_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        LOG.warning("slm remember failed: %s", redact(repr(exc)))
        return False
    if proc.returncode != 0:
        LOG.warning(
            "slm remember exit=%d stderr=%s",
            proc.returncode,
            redact(proc.stderr.strip()[:200]),
        )
        return False
    return True


__all__ = [
    "write_outcome_to_slm",
]
