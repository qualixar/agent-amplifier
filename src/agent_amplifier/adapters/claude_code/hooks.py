# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Claude Code hook entry points (day-0, per H-4).

This module is invoked as a script from ``~/.claude/settings.json`` — every
hook event triggers a fresh subprocess that calls one of the entry points
here. Per H-4 the mapping is:

    UserPromptSubmit  → on_user_prompt_submit  (kernel.before_step + inject)
    PreToolUse        → on_pre_tool_use        (logging only)
    PostToolUse       → on_post_tool_use       (logging only)
    Stop              → handled in stop_hook.py

Each handler:
    1. Reads the Claude Code event JSON from ``sys.stdin``
    2. Performs its work (kernel call OR pure logging)
    3. Writes a hook-response JSON to ``sys.stdout`` if it needs to inject
       context, otherwise exits 0 quietly
    4. Logs ``[amp]`` lines to ``sys.stderr`` (visible in user terminal)

Failure semantics (CRITICAL): every handler is **fail-open**. If anything
goes wrong (SLM unavailable, kernel crash, JSON malformed), the hook MUST
NOT block the user's Claude Code session. We log to stderr and exit 0.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

from agent_amplifier import AgentAmplifier
from agent_amplifier._internal.redact import redact
from agent_amplifier.adapters.claude_code.state import StateStore
from agent_amplifier.model_router import ModelRouter
from agent_amplifier.types import EffortLevel, TaskClassification

LOG = logging.getLogger("agent_amplifier.adapters.claude_code.hooks")

# Logger to user's terminal stderr (visible during Claude Code session). We
# prefix every line with ``[amp]`` so the user can grep their session
# transcript later.
_AMP_TAG: str = "[amp]"

# Cap on how much of the user prompt we keep in SQLite (redacted). The
# full prompt is what the model sees; we only store a redacted prefix for
# the paper-data outcome rows.
_PROMPT_REDACT_BYTES: int = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_event() -> dict[str, Any]:
    """Read JSON event from stdin. Returns empty dict on failure."""
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
        LOG.warning("malformed hook event JSON: %s", exc)
        return {}


def _amp_log(msg: str) -> None:
    """Write a ``[amp] <msg>`` line to stderr — visible in user's terminal."""
    print(f"{_AMP_TAG} {msg}", file=sys.stderr, flush=True)


def _resolve_session_id(event: dict[str, Any]) -> str:
    """Extract session_id from Claude Code event JSON.

    Claude Code's hook event includes ``session_id`` directly. Fall back to
    ``$CLAUDE_SESSION_ID`` env var if event is missing the field (defensive).
    Final fallback: use the parent process id so we never block on a missing
    id.
    """
    sid = event.get("session_id") or os.environ.get("CLAUDE_SESSION_ID", "")
    if not sid:
        sid = f"pid-{os.getppid()}"
    return str(sid)


def _resolve_cwd(event: dict[str, Any]) -> str:
    """Extract cwd from event or fall back to actual cwd."""
    cwd = event.get("cwd")
    if not cwd:
        cwd = os.getcwd()
    return str(cwd)


def _resolve_model(event: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract (model, model_provider) from event when available.

    Claude Code typically passes the model name in the event when running
    a specific Claude model. We default to ``None`` if absent (the SQLite
    column is nullable) and ``provider="anthropic"`` when we have a model
    name that starts with ``claude-``.
    """
    model = event.get("model")
    if not model:
        return None, None
    model = str(model)
    provider = "anthropic" if model.startswith("claude") else "unknown"
    return model, provider


def _store() -> StateStore:
    """Open the default SQLite store. Idempotent; safe across processes."""
    return StateStore()


def _resolve_adapter() -> Any | None:
    """Pick the active adapter for amp's memory plane.

    Resolution order (H-5 + Varun's "all 3 modes for every user" rule):
        1. SLM detected → SLMAdapter (Mode 2 = ``slm session-context``;
           Mode 3 closed-loop write-back via ``slm remember`` happens in
           stop_hook).
        2. SLM absent → ClaudeCodeAdapter (Mode 2 = read CLAUDE.md /
           MEMORY.md / ~/.claude/CLAUDE.md; Mode 3 closed-loop write-back
           appends to ``./MEMORY.md``, auto-created if missing).
        3. Both unavailable (extremely defensive — both adapters are
           bundled in v1.0.0) → None, kernel runs without a memory plane.

    NEVER raises. Any failure path logs at WARNING and returns None so the
    UserPromptSubmit handler stays fail-open.
    """
    # Tier 1 — SLM detected. Wrap in try/except so a partial install of
    # agent_amplifier (somehow missing the slm subpackage) still falls
    # through to the Tier 2 ClaudeCodeAdapter rather than raising.
    try:
        from agent_amplifier.adapters.slm import SLMAdapter, detect_slm
        if detect_slm():
            try:
                return SLMAdapter(kernel=None)
            except Exception as exc:  # pragma: no cover - defensive
                LOG.warning("SLMAdapter construction failed: %s", exc)
    except ImportError:  # pragma: no cover - SLM adapter always bundled in v1.0.0
        pass

    # Tier 2 — fall back to host-native ClaudeCodeAdapter (CLAUDE.md +
    # MEMORY.md). This keeps IP-9 + all 3 modes active for every user
    # regardless of whether SLM is installed.
    try:
        from agent_amplifier.adapters.claude_code.memory import (
            ClaudeCodeAdapter,
        )
    except ImportError:  # pragma: no cover - same package
        return None
    try:
        return ClaudeCodeAdapter(kernel=None)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("ClaudeCodeAdapter construction failed: %s", exc)
        return None


def _safe_run(handler_name: str, fn: Any) -> int:
    """Wrap a handler in fail-open semantics.

    NEVER raise to Claude Code. Any exception → log + exit 0 (the user's
    session continues). audit will close any silent-failure paths;
    for v1.0.0 dogfood the priority is "do not break Varun's daily flow".
    """
    try:
        fn()
        return 0
    except Exception as exc:
        LOG.warning("hook %s failed: %s", handler_name, exc)
        # Mirror to stderr so the user sees it (instead of silent swallow).
        _amp_log(f"hook {handler_name} error (fail-open): {redact(str(exc))}")
        return 0


# ---------------------------------------------------------------------------
# Abandoned-envelope sweeper (covers Cmd+Q / force-quit / no-reply gaps)
# ---------------------------------------------------------------------------


def _sweep_abandoned_envelopes(store: StateStore) -> None:
    """Mark stale envelopes (no Stop within 30s) as ``abandoned`` outcomes.

    Called at the top of every UserPromptSubmit. Sweeps ALL sessions because
    Claude Code session_ids do not persist across process restarts.

    Fail-open: every failure is caught and logged. A sweep bug MUST NOT
    crater the main amplification injection path.
    """
    try:
        orphans = store.find_abandoned_envelopes()
    except Exception as exc:
        LOG.warning("abandoned-envelope sweep query failed: %s", exc)
        _amp_log(f"sweep error (fail-open): {redact(str(exc))}")
        return
    for orphan in orphans:
        try:
            duration_ms = int(
                (orphan["last_event_at"] - orphan["created_at"]) * 1000
            )
            in_flight = max(0, orphan["pre_count"] - orphan["post_count"])
            store.write_outcome(
                orphan["session_id"],
                int(orphan["turn_id"]),
                iterations_completed=1,
                converged=False,
                drift_at_end=0.0,
                tokens_used=0,
                duration_ms=duration_ms,
                amplification_enabled=True,
                quality_estimate=None,
                finalize_report={
                    "tool_calls": orphan["pre_count"],
                    "tool_results": orphan["post_count"],
                    "in_flight_at_stop": in_flight,
                    "envelope_turn_id": int(orphan["turn_id"]),
                    "stop_reason": "abandoned",
                    "wallclock_at_stop": time.time(),
                },
            )
        except Exception as exc:
            LOG.warning("abandoned-envelope write failed: %s", exc)
            _amp_log(f"sweep row error (fail-open): {redact(str(exc))}")


# ---------------------------------------------------------------------------
# UserPromptSubmit — the amplification injection point (H-4)
# ---------------------------------------------------------------------------


def on_user_prompt_submit() -> int:
    """Handle a UserPromptSubmit event: classify, build envelope, inject.

    This is the only hook that calls ``kernel.before_step``. The kernel
    is constructed fresh per turn (per H-4: each user turn is its own
    iteration; cross-turn state lives in SQLite, not in the kernel).
    """
    return _safe_run("UserPromptSubmit", _on_user_prompt_submit_impl)


def _read_prior_classification(
    store: StateStore, session_id: str
) -> TaskClassification | None:
    """Build a synthetic ``TaskClassification`` from the prior envelope row.

    The kernel uses this as a cross-turn signal so brief conversational
    continuations ("ok", "yes", "2. go") inherit complexity+domain from
    the immediately preceding turn instead of being mis-classified as
    minimal/general. Fail-open: any read error, missing row, or invalid
    complexity value returns ``None``; the kernel then falls back to
    pure-prompt classification.
    """
    try:
        row = store.latest_envelope(session_id)
    except Exception as exc:
        LOG.warning("prior-envelope read failed: %s", exc)
        return None
    if row is None:
        return None
    complexity_raw = row.get("classification_complexity")
    domain_raw = row.get("classification_domain")
    if not isinstance(complexity_raw, str) or not isinstance(domain_raw, str):
        return None
    try:
        complexity = EffortLevel(complexity_raw.lower())
    except ValueError:
        return None
    return TaskClassification(
        complexity=complexity,
        domain=domain_raw,
        estimated_tokens=0,
        confidence=1.0,
        matched_signals=("prior_envelope_row",),
    )


def _on_user_prompt_submit_impl() -> None:
    event = _read_event()
    session_id = _resolve_session_id(event)
    cwd = _resolve_cwd(event)
    model, provider = _resolve_model(event)
    user_prompt = event.get("prompt") or event.get("user_prompt") or ""
    if not user_prompt:
        # No prompt to amplify; nothing to do.
        return

    store = _store()
    # Lazy GC sweep on every UserPromptSubmit. Cheap (one indexed query).
    store.gc_old_sessions()
    # Flush any abandoned envelopes (Cmd+Q mid-response, force-quit, etc).
    # Stop hook does not fire when Claude Code is killed mid-reply, so any
    # envelope older than 30s without a matching outcome is retroactively
    # marked ``stop_reason="abandoned"`` here. Self-healing via INSERT OR
    # REPLACE if a real Stop ever lands later.
    _sweep_abandoned_envelopes(store)
    store.upsert_session(
        session_id,
        cwd,
        model=model,
        model_provider=provider,
        amplification_enabled=True,
    )
    turn_id = store.next_turn_id(session_id)

    # Cross-turn context: read the prior envelope so the kernel can
    # inherit complexity+domain when the current prompt is a brief
    # conversational continuation (bare ack, numbered answer, short
    # question). Fail-open: any read error returns ``None`` and the
    # kernel falls back to pure-prompt classification.
    prior_classification = _read_prior_classification(store, session_id)

    # Build the envelope. The kernel is fresh per turn; this is intentional
    # per H-4 (cross-turn semantics live in SQLite, not in the kernel).
    # H-5 Mode 2: SLMAdapter when SLM is detected, otherwise fall back to
    # ClaudeCodeAdapter (host-native CLAUDE.md / MEMORY.md). Every user
    # gets IP-9 active in v1.0.0 regardless of whether they installed SLM.
    adapter = _resolve_adapter()
    amp = AgentAmplifier(adapter=adapter)
    try:
        ctx: dict[str, Any] | None = (
            {"prior_classification": prior_classification}
            if prior_classification is not None
            else None
        )
        envelope = amp.before_step(user_prompt, context=ctx)
    finally:
        amp.close()

    # v1.1 F4: persist the cost-routing receipt — ModelRouter's tier
    # recommendation for this envelope's complexity. Lets the report show
    # actual per-turn model suggestions over time. ModelRouter is a pure
    # in-process classifier; no API calls.
    suggested_tier: str | None
    try:
        suggested_tier = ModelRouter().suggest(
            envelope.classification.complexity.value
        ).tier
    except Exception as exc:  # pragma: no cover - fail-open defense
        LOG.warning("ModelRouter.suggest failed (fail-open): %s", exc)
        suggested_tier = None

    # Persist the envelope for the Stop hook to read at session end.
    store.record_envelope(
        session_id,
        turn_id,
        user_prompt_redacted=redact(user_prompt[:_PROMPT_REDACT_BYTES]),
        classification_complexity=envelope.classification.complexity.value,
        classification_domain=envelope.classification.domain,
        thinking_trigger=envelope.thinking_trigger,
        persona=envelope.persona,
        phase=envelope.phase,
        envelope_text=envelope.envelope,
        suggested_model=suggested_tier,
    )

    # Inject the envelope as additional context. Claude Code's
    # UserPromptSubmit hook contract: write JSON ``{"hookSpecificOutput":
    # {"hookEventName": "UserPromptSubmit", "additionalContext": "..."}}``
    # to stdout. The injected text is appended to the user's prompt before
    # the model sees it.
    response = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                "<system-reminder>\n"
                f"[Agent Amplifier — turn {turn_id}, complexity="
                f"{envelope.classification.complexity.value}, "
                f"domain={envelope.classification.domain}]\n\n"
                f"{envelope.envelope}\n"
                "</system-reminder>"
            ),
        }
    }
    sys.stdout.write(json.dumps(response))
    sys.stdout.flush()

    _amp_log(
        f"turn={turn_id} complexity={envelope.classification.complexity.value} "
        f"trigger={envelope.thinking_trigger or 'none'} "
        f"persona={envelope.persona or 'default'}"
    )


# ---------------------------------------------------------------------------
# PreToolUse — logging only (per H-4)
# ---------------------------------------------------------------------------


def on_pre_tool_use() -> int:
    """Handle a PreToolUse event: log to SQLite. NO kernel call (per H-4)."""
    return _safe_run("PreToolUse", _on_pre_tool_use_impl)


def _on_pre_tool_use_impl() -> None:
    event = _read_event()
    session_id = _resolve_session_id(event)
    if not session_id:
        return
    tool_name = event.get("tool_name") or event.get("tool") or "unknown"
    tool_input = event.get("tool_input") or event.get("input") or {}
    store = _store()
    # Best-effort turn_id: most recent envelope for this session.
    last_envelope = store.latest_envelope(session_id)
    turn_id = last_envelope["turn_id"] if last_envelope else None
    # Cap event count per session to defend against runaway sessions.
    if store.count_events(session_id) >= 10_000:
        # Already bounded — don't keep writing.
        return
    store.record_event(
        session_id,
        event_type="PreToolUse",
        turn_id=turn_id,
        tool_name=str(tool_name),
        payload={"input_summary": redact(str(tool_input)[:500])},
    )


# ---------------------------------------------------------------------------
# PostToolUse — logging only (per H-4)
# ---------------------------------------------------------------------------


def on_post_tool_use() -> int:
    """Handle a PostToolUse event: log to SQLite. NO kernel call (per H-4)."""
    return _safe_run("PostToolUse", _on_post_tool_use_impl)


def _on_post_tool_use_impl() -> None:
    event = _read_event()
    session_id = _resolve_session_id(event)
    if not session_id:
        return
    tool_name = event.get("tool_name") or event.get("tool") or "unknown"
    tool_input = event.get("tool_input") or event.get("input") or {}
    tool_response = (
        event.get("tool_response")
        or event.get("tool_output")
        or event.get("output")
        or ""
    )
    store = _store()
    last_envelope = store.latest_envelope(session_id)
    turn_id = last_envelope["turn_id"] if last_envelope else None
    if store.count_events(session_id) >= 10_000:
        return
    store.record_event(
        session_id,
        event_type="PostToolUse",
        turn_id=turn_id,
        tool_name=str(tool_name),
        payload={
            "input_summary": redact(str(tool_input)[:500]),
            "output_summary": redact(str(tool_response)[:500]),
        },
    )


# ---------------------------------------------------------------------------
# PreCompact — observe-only (Claude Code 2.1.105+, paper data for v1.0.1)
# ---------------------------------------------------------------------------


def on_pre_compact() -> int:
    """Handle Claude Code's PreCompact event (CC 2.1.105+).

    v1.0 behavior is **observe-only**: we log the event to state.db so the
    paper-data corpus captures how often compaction fires relative to amp
    turns. The active-deferral variant (block compaction while an amp turn
    is in flight) ships in v1.0.1 once we have empirical data on whether
    amp turns commonly overlap with compaction events in the wild.

    Fail-open per the universal hook contract.
    """
    return _safe_run("PreCompact", _on_pre_compact_impl)


def _on_pre_compact_impl() -> None:
    event = _read_event()
    session_id = _resolve_session_id(event)
    if not session_id:
        return
    store = _store()
    last_envelope = store.latest_envelope(session_id)
    in_flight = last_envelope is not None
    turn_id = int(last_envelope["turn_id"]) if last_envelope else None
    # Cap event count per session to defend against runaway sessions —
    # same guard the Pre/PostToolUse handlers use.
    if store.count_events(session_id) < 10_000:
        store.record_event(
            session_id,
            event_type="PreCompact",
            turn_id=turn_id,
            tool_name=None,
            payload={
                "in_flight_amp_turn": in_flight,
                "compaction_reason": str(
                    event.get("compaction_reason") or "unknown"
                ),
            },
        )
    _amp_log(
        f"pre-compact session={redact(session_id)} "
        f"in_flight_amp_turn={in_flight} observe-only (v1.0.1 will defer)"
    )


# ---------------------------------------------------------------------------
# Script dispatch — invoked by ``~/.claude/settings.json`` hook commands
# ---------------------------------------------------------------------------


_ENTRY_POINTS: dict[str, Any] = {
    "UserPromptSubmit": on_user_prompt_submit,
    "PreToolUse": on_pre_tool_use,
    "PostToolUse": on_post_tool_use,
    "PreCompact": on_pre_compact,
}


def main(argv: list[str] | None = None) -> int:
    """CLI dispatcher: ``python -m agent_amplifier.adapters.claude_code.hooks <Event>``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "usage: python -m agent_amplifier.adapters.claude_code.hooks "
            "<UserPromptSubmit|PreToolUse|PostToolUse>",
            file=sys.stderr,
        )
        return 2
    event_name = args[0]
    fn = _ENTRY_POINTS.get(event_name)
    if fn is None:
        print(f"unknown hook event: {event_name}", file=sys.stderr)
        return 2
    return int(fn())


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())


__all__ = [
    "main",
    "on_post_tool_use",
    "on_pre_compact",
    "on_pre_tool_use",
    "on_user_prompt_submit",
]
