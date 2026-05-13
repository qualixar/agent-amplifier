# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Stop hook entry point (day-0, per H-4 corrigendum).

Per the H-4 corrigendum (DECISIONS-LOCKED.md, 2026-05-10 evening), the Stop
hook does NOT call ``kernel.after_step`` or ``kernel.finalize`` in v1.0.0.
The kernel object cannot survive across hook subprocesses; recreating it
just to record per-turn outcome metrics is wasted disk I/O. Instead, this
handler:

    1. Reads the latest envelope row for this session from SQLite.
    2. Computes summary metrics directly from the ``events`` rows written
       by ``PreToolUse`` / ``PostToolUse`` between the last envelope and now
       (tool-call count, approximate duration).
    3. Writes one ``outcomes`` row.
    4. Logs an ``[amp]`` summary line to stderr (visible in user terminal).
    5. Marks the session row's ``closed_at`` (defensive — Claude Code may
       fire Stop multiple times if the user resumes; ``upsert_session`` at
       the next UserPromptSubmit reopens it).

Convergence/drift comparison happens at the NEXT user turn's
``UserPromptSubmit``, which reads the prior outcome row from SQLite via
``StateStore.latest_envelope`` + a sibling future ``latest_outcome`` reader.
For v1.0.0 day-0 we just persist the data; the kernel-side comparison is
already implemented in ``kernel.before_step`` and reads from the same DB.

Failure semantics: same fail-open contract as ``hooks.py``. Any exception
is caught, logged to stderr with ``[amp]`` prefix, and we exit 0 so the
user's Claude Code session is never blocked by amplifier internals.
"""
from __future__ import annotations

import contextlib
import json
import logging
import shutil
import sqlite3
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent_amplifier._internal.keyword_set import keyword_set
from agent_amplifier._internal.redact import redact
from agent_amplifier.adapters.claude_code.state import StateStore
from agent_amplifier.adapters.claude_code.transcript import (
    final_assistant_message,
)

LOG = logging.getLogger("agent_amplifier.adapters.claude_code.stop_hook")

_AMP_TAG: str = "[amp]"


# ---------------------------------------------------------------------------
# Helpers (same fail-open shape as hooks.py — kept module-local to avoid a
# new shared module and the audit cycle that would attach to it)
# ---------------------------------------------------------------------------


def _read_event() -> dict[str, Any]:
    """Read JSON from stdin. Empty dict on any failure (defensive)."""
    try:
        raw = sys.stdin.read()
    except Exception as exc:  # pragma: no cover - stdin should always read
        LOG.warning("failed to read stdin: %s", exc)
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        out = json.loads(raw)
        if isinstance(out, dict):
            return out
        return {}
    except (json.JSONDecodeError, ValueError) as exc:
        LOG.warning("malformed Stop event JSON: %s", exc)
        return {}


def _amp_log(msg: str) -> None:
    print(f"{_AMP_TAG} {msg}", file=sys.stderr, flush=True)


def _resolve_session_id(event: dict[str, Any]) -> str:
    """session_id from event, else env, else ppid sentinel."""
    import os

    sid = event.get("session_id") or os.environ.get("CLAUDE_SESSION_ID", "")
    if not sid:
        sid = f"pid-{os.getppid()}"
    return str(sid)


def _resolve_cwd(event: dict[str, Any]) -> str:
    """cwd from event, else current process cwd. Mirrors hooks._resolve_cwd."""
    import os

    cwd = event.get("cwd")
    if not cwd:
        cwd = os.getcwd()
    return str(cwd)


def _compute_turn_tokens(
    store: StateStore, session_id: str, turn_id: int, project_cwd: str
) -> int:
    """Return per-turn tokens delta read from Claude Code's transcript JSONL.

    IP-10 v2 / Option C: replaces the v1.0 hardcoded ``tokens_used=0``.
    Reads the cumulative session total from the transcript, subtracts the
    sum of prior outcome rows (turn_id < this), and clamps at zero
    (defensive against backfill, malformed transcripts, or session reuse).

    Fail-open: every error path collapses to 0 so the Stop hook never
    aborts because of a transcript glitch.
    """
    from pathlib import Path

    from agent_amplifier.adapters.claude_code.transcript import tokens_for_session

    try:
        cumulative = tokens_for_session(session_id, Path(project_cwd))
    except Exception:
        return 0
    try:
        prior = store.prior_tokens_for_session(session_id, turn_id)
    except Exception:
        prior = 0
    return max(0, cumulative - prior)


def _compute_quality_score_tier1(
    *,
    envelope: Mapping[str, Any],
    session_id: str,
    project_cwd: str,
) -> float | None:
    """F1A Tier 1 — Jaccard similarity between envelope goal text and the
    last assistant message's text content from the Claude Code transcript.

    Returns a value in ``[0, 1]`` when both sides yield a non-empty keyword
    set; ``None`` when the transcript is missing, empty, or text-free
    (e.g. tool-use-only turn).

    Implementation re-uses ``agent_amplifier._internal.keyword_set`` — the
    same extractor used by ``convergence.py``, so quality scoring shares
    tokenization semantics with the convergence detector.

    Fail-open: any exception → ``None``. The Stop hook MUST NOT fail
    because quality scoring stumbled.
    """
    try:
        goal_parts: list[str] = []
        prompt = envelope.get("user_prompt_redacted")
        if isinstance(prompt, str):
            goal_parts.append(prompt)
        env_text = envelope.get("envelope_text")
        if isinstance(env_text, str):
            goal_parts.append(env_text)
        if not goal_parts:
            return None
        goal_text = " ".join(goal_parts)
        final = final_assistant_message(session_id, Path(project_cwd))
        if final is None:
            return None
        goal_kw = keyword_set(goal_text)
        out_kw = keyword_set(final)
        if not goal_kw or not out_kw:
            return None
        union = goal_kw | out_kw
        if not union:  # pragma: no cover - guarded by truthiness above
            return None
        return len(goal_kw & out_kw) / len(union)
    except Exception as exc:  # pragma: no cover - fail-open defense
        LOG.warning("quality_score Tier1 failed (fail-open): %s", exc)
        return None


def _store() -> StateStore:
    return StateStore()


# ---------------------------------------------------------------------------
# Per-turn metric computation
# ---------------------------------------------------------------------------


def _events_for_turn(
    store: StateStore, session_id: str, turn_id: int
) -> tuple[int, int, int]:
    """Return ``(tool_call_count, post_count, duration_ms)`` for a turn.

    ``tool_call_count`` = number of ``PreToolUse`` events.
    ``post_count`` = number of ``PostToolUse`` events (used to detect
    in-flight tool calls — pre > post means at least one tool was still
    running at Stop, which we surface in the summary).
    ``duration_ms`` = milliseconds between the earliest and latest event
    in this turn. 0 if fewer than 2 events.

    A direct ``sqlite3`` query is used here (not StateStore methods) only
    because StateStore intentionally exposes a narrow surface; opening one
    connection for three aggregates avoids three round-trips and keeps the
    Stop hook within the sub-50ms target.
    """
    pre = 0
    post = 0
    first_ts: float | None = None
    last_ts: float | None = None
    with contextlib.closing(
        sqlite3.connect(str(store.db_path), timeout=5.0)
    ) as conn:
        cur = conn.execute(
            """
            SELECT event_type, timestamp FROM events
             WHERE session_id = ? AND turn_id = ?
            """,
            (session_id, turn_id),
        )
        for ev_type, ts in cur.fetchall():
            ts_f = float(ts)
            if first_ts is None or ts_f < first_ts:
                first_ts = ts_f
            if last_ts is None or ts_f > last_ts:
                last_ts = ts_f
            if ev_type == "PreToolUse":
                pre += 1
            elif ev_type == "PostToolUse":
                post += 1
    if first_ts is not None and last_ts is not None and last_ts > first_ts:
        duration_ms = int((last_ts - first_ts) * 1000)
    else:
        duration_ms = 0
    return pre, post, duration_ms


def _infer_stop_reason(
    pre_count: int, post_count: int, event_stop_reason: object
) -> str:
    """Derive ``stop_reason`` from observed turn shape.

    Claude Code's ``Stop`` event does not currently expose a structured
    ``stop_reason`` field in the public hook spec; ``event_stop_reason`` is
    almost always ``None`` in practice. Without inference, every outcome
    row would record ``"unknown"`` and the ``agent-amp report`` dashboard
    would be unable to distinguish a real completion from a model that
    bailed silently.

    Inference rules (deterministic, defensive):

    * If the host did pass a non-empty ``stop_reason`` in the event, we
      trust it verbatim — host wins over inference.
    * ``pre_count > post_count`` → ``"in_flight"`` (tool calls were started
      but not yet completed when ``Stop`` fired; rare but real).
    * ``pre_count == 0 and post_count == 0`` → ``"empty"`` (model produced
      a text-only response with no tool use — e.g. a refusal or a short
      direct answer).
    * Otherwise → ``"complete"`` (the happy path — at least one tool call
      and its result were recorded, and tools are not in flight).

    The ``"abandoned"`` reason is set by the sweep mechanism on a separate
    code path (``UserPromptSubmit`` retroactively marks orphaned envelopes),
    not here. Likewise, ``"compact"`` cannot be inferred from the events
    table at v1.0 because ``PreCompact`` events are observed-only and not
    yet rowed into ``events``.
    """
    if isinstance(event_stop_reason, str) and event_stop_reason.strip():
        return event_stop_reason
    if pre_count > post_count:
        return "in_flight"
    if pre_count == 0 and post_count == 0:
        return "empty"
    return "complete"


# ---------------------------------------------------------------------------
# Stop entry point
# ---------------------------------------------------------------------------


def on_stop() -> int:
    """Handle a Claude Code ``Stop`` event. Fail-open: always exits 0."""
    try:
        _on_stop_impl()
    except Exception as exc:
        LOG.warning("Stop hook failed: %s", exc)
        _amp_log(f"hook Stop error (fail-open): {redact(str(exc))}")
    return 0


def _on_stop_impl() -> None:
    event = _read_event()
    session_id = _resolve_session_id(event)
    store = _store()

    last_envelope = store.latest_envelope(session_id)
    if last_envelope is None:
        # No envelope means no UserPromptSubmit ever fired for this session
        # (or the row was GC'd). Nothing to summarize.
        _amp_log(f"stop session={redact(session_id)} no-envelope")
        store.close_session(session_id)
        return

    turn_id = int(last_envelope["turn_id"])
    pre_count, post_count, duration_ms = _events_for_turn(
        store, session_id, turn_id
    )
    in_flight = max(0, pre_count - post_count)

    # H-4: per-user-turn iteration. v1.0.0 records ``iterations_completed=1``
    # because we treat the whole user turn as one amplification cycle. The
    # field is reserved for future per-iteration scoring (V1.1 candidate).
    converged = in_flight == 0
    # IP-10 v2 / Option C: compute real per-turn token delta from the
    # Claude Code transcript JSONL instead of the v1.0 hardcoded 0.
    project_cwd = _resolve_cwd(event)
    tokens_used = _compute_turn_tokens(store, session_id, turn_id, project_cwd)

    # v1.1 F1A: layered quality metric. Tier 1 (Jaccard) is always cheap and
    # deterministic. Tier 2 (local embedding) and Tier 3 (trajectory delta)
    # are wired in F1B / F1C commits — this commit lands the schema, the
    # transcript reader, and the Tier 1 baseline so v1.1 dashboards can show
    # a real ``quality_score`` from day 0 of the upgrade.
    quality_score = _compute_quality_score_tier1(
        envelope=last_envelope,
        session_id=session_id,
        project_cwd=project_cwd,
    )

    store.write_outcome(
        session_id,
        turn_id,
        iterations_completed=1,
        # In v1.0.0 we cannot prove convergence from hook data alone; we
        # record the turn outcome conservatively as "completed", and let
        # the next turn's before_step compare for plateau-style drift.
        converged=converged,
        drift_at_end=0.0,
        tokens_used=tokens_used,
        duration_ms=duration_ms,
        amplification_enabled=True,
        quality_estimate=None,
        # F1A new fields:
        completed=converged,  # explicit mirror under the renamed column
        quality_score=quality_score,
        convergence_state=None,  # filled in F1C per-session pool
        finalize_report={
            "tool_calls": pre_count,
            "tool_results": post_count,
            "in_flight_at_stop": in_flight,
            "envelope_turn_id": turn_id,
            "stop_reason": _infer_stop_reason(
                pre_count, post_count, event.get("stop_reason")
            ),
            "wallclock_at_stop": time.time(),
        },
    )

    store.close_session(session_id)

    # H-5 Mode 3 closed-loop write-back: SLM if available (faster + richer
    # recall surface), else MEMORY.md (host-native fallback so non-SLM
    # users still close the loop). Fail-open by contract — any failure is
    # logged but never blocks Stop completion.
    target = "none"
    if _slm_available_for_writeback():
        ok = _maybe_writeback_to_slm(
            session_id, turn_id, last_envelope,
            pre_count=pre_count, post_count=post_count,
            duration_ms=duration_ms, converged=converged,
        )
        target = "slm" if ok else "slm-failed"
    else:
        ok = _maybe_writeback_to_memory_md(
            last_envelope,
            pre_count=pre_count, post_count=post_count,
            duration_ms=duration_ms, converged=converged,
        )
        target = "memory.md" if ok else "memory.md-failed"

    _amp_log(
        f"stop session={redact(session_id)} turn={turn_id} "
        f"tools={pre_count}/{post_count} duration_ms={duration_ms} "
        f"in_flight={in_flight} writeback={target}"
    )


def _slm_available_for_writeback() -> bool:
    """Cheap PATH lookup. Same primitive ``SLMAdapter.detect`` uses, kept
    local so stop_hook does not depend on the SLMAdapter class for a
    boolean it can answer in two lines."""
    try:
        return shutil.which("slm") is not None
    except OSError:  # pragma: no cover - extremely defensive
        return False


def _maybe_writeback_to_slm(
    session_id: str,
    turn_id: int,
    envelope: dict[str, Any],
    *,
    pre_count: int,
    post_count: int,
    duration_ms: int,
    converged: bool,
) -> bool:
    """Mode-3 wiring: shell out to ``slm remember`` with the turn summary.

    Wraps the import + call in a try/except so any failure (slm CLI gone,
    import error, network failure) cannot block the Stop hook's main path.
    """
    try:
        from agent_amplifier.adapters.claude_code.slm_writeback import (
            write_outcome_to_slm,
        )
    except ImportError:  # pragma: no cover - same package, always importable
        return False
    try:
        return write_outcome_to_slm(
            session_id, turn_id,
            envelope=envelope,
            tool_calls=pre_count, tool_results=post_count,
            duration_ms=duration_ms, converged=converged,
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("slm writeback failed: %s", exc)
        return False


def _maybe_writeback_to_memory_md(
    envelope: dict[str, Any],
    *,
    pre_count: int,
    post_count: int,
    duration_ms: int,
    converged: bool,
) -> bool:
    """Non-SLM Mode 3 fallback: append the turn outcome to ``./MEMORY.md``.

    Constructs a synthetic ``Outcome`` from envelope + counts and routes it
    through ``ClaudeCodeAdapter.default_memory_remember``, which auto-creates
    MEMORY.md on first call and appends thereafter. NEVER touches CLAUDE.md.

    Returns True iff the append succeeded (best-effort signal for the
    ``[amp] stop ... slm_written=...`` log line).
    """
    try:
        from agent_amplifier.adapters.claude_code.memory import (
            ClaudeCodeAdapter,
        )
        from agent_amplifier.types import EffortLevel, Outcome
    except ImportError:  # pragma: no cover - same package
        return False

    # Map classification_complexity to EffortLevel; unknowns fall back to
    # MEDIUM so the synthetic Outcome is well-typed.
    complexity = (envelope.get("classification_complexity") or "").lower()
    effort_map = {
        "minimal": EffortLevel.MINIMAL,
        "low": EffortLevel.LOW,
        "medium": EffortLevel.MEDIUM,
        "high": EffortLevel.HIGH,
        "max": EffortLevel.MAX,
    }
    effort = effort_map.get(complexity, EffortLevel.MEDIUM)
    # Synthesize a minimal Outcome — quality estimated from convergence,
    # tokens unknown at hook level (kernel-only metric in v1.0.0).
    quality = 0.85 if converged else 0.5
    query = envelope.get("user_prompt_redacted") or ""
    outcome = Outcome(
        query=query,
        effort=effort,
        iterations=1,
        quality=quality,
        converged=converged,
        tokens_used=0,
    )
    try:
        adapter = ClaudeCodeAdapter(kernel=None)
        adapter.default_memory_remember(outcome)
        return True
    except Exception as exc:  # pragma: no cover - default_memory_remember is
        # already fail-open; this is belt-and-suspenders for the synthetic
        # Outcome construction path.
        LOG.warning("memory.md writeback failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point: ``python -m agent_amplifier.adapters.claude_code.stop_hook``.

    The settings.json hook command points here; argv is empty in practice.
    Reserved for forward-compat (e.g. ``--debug``) without breaking the
    hook command string.
    """
    _ = list(sys.argv[1:] if argv is None else argv)  # accept any flags silently
    return on_stop()


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())


__all__ = [
    "main",
    "on_stop",
]
