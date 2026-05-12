# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Agent Amplifier — runtime kernel (IP-1), V2.0 + V2.1 —  §3.

V1.1 refactor ticket (/047):
    This module is intentionally large for V1 — it owns the orchestration
    contract and we accept the size to keep the public-facing layout
    simple.  V1.1 will split it into:

    * ``kernel/_state.py`` (`_KernelState`, `_AmplifierCore` lock topology)
    * ``kernel/_envelope.py`` (`StepEnvelope`, render helpers)
    * ``kernel/_memory_plane.py`` (`_resolve_recall`, `_resolve_remember`)
    * ``kernel/__init__.py`` (re-exports preserving today's import paths)

    Splitting mid-pipeline carries collision risk; we defer to V1.1 as a
    deliberate, fully-tested refactor PR.


V2.1 (— universal memory plane, .5):
    * Removed direct ``SLMBridge`` coupling from core.
    * Added optional ``memory_recall`` / ``memory_remember`` callbacks.
    * Added optional ``adapter`` parameter so the kernel can fall back to
      ``AdapterBase.default_memory_recall`` / ``default_memory_remember``.
    * Added ``_resolve_recall`` / ``_resolve_remember`` private helpers that
      apply universal recall-safety (cap + neutralize + smuggling detect)
      regardless of which provider produced the text.


     (DistSys F-01) — anyio core + sync/async facades
     (Perf F-02 + DistSys F-02) — narrow lock; double-checked anchor capture
     (PM F-03 / E-6) — observability_callback wired through ``_emit``
     (DistSys F-03) — release lock before user callbacks; ContextVar guard
     (DistSys F-04) — ``step_id`` semantics under parallel dispatch
     (PM F-05) — KernelContractError 3-field message + AGENT_AMP_FALLBACK_PHASE
     (Sec F-10) — top-level try/except on before/after step (locked E-7)
     (Perf F-06) — ``StepEnvelope`` dataclass instead of dict reuse
     (DistSys F-08) — config snapshot per before_step + ``__setattr__`` guard
     (Sec F-11) — ``redact()`` applied to error messages and free-form logs

# ``dspy.settings.context()`` is the canonical
# configurable-per-context immutable-after-construction pattern (verified
# 2026-04-26 via WebSearch of DeepWiki + DSPy docs). Our ``__setattr__`` guard
# matches DSPy's "owner thread can configure once; everyone else uses
# context-managed thread-local overrides" pattern.

Threading:
    The core uses an ``anyio.Lock`` for serialization. The lock is held ONLY
    around regions that mutate ``_KernelState`` invariants. The lock is
    NEVER held across:
        * memory provider I/O (recall/remember callbacks),
        * sub-module pure computation,
        * building return dicts,
        * user-supplied callbacks (observability_callback, post-decision
          adapter code).

Cache-boundary order in rendered prompts:
    1. ``<system-reminder>`` block (modifiers + persona) — STABLE per phase
    2. ``PHASE: <name>`` + slot-resolved phase prompt — STABLE per phase
    3. Goal-anchored user query — DYNAMIC

Steps 1+2 are stable per ``(effort, phase, persona)`` tuple, which gives
prompt caches the largest possible cache-hit window.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import anyio

from agent_amplifier._anyio_portal import PortalHolder
from agent_amplifier._internal import recall_safety
from agent_amplifier._internal.ctx_schema import validate_context
from agent_amplifier._internal.redact import redact
from agent_amplifier.config import load_config
from agent_amplifier.convergence import ConvergenceDetector
from agent_amplifier.effort_router import (
    classify,
    classify_with_config,
    classify_with_context,
    suggest_thinking_trigger,
)
from agent_amplifier.goal_anchor import (
    DriftLevel,
    GoalAnchor,
    GoalAnchorService,
)
from agent_amplifier.model_router import ModelRouter
from agent_amplifier.persona_docs import resolve_flavor
from agent_amplifier.personas import (
    compose_persona,
    get_persona,
    get_strictness_profile,
)
from agent_amplifier.phase_prompts import advance_phase, get_phase_prompt
from agent_amplifier.semantic_modifiers import (
    generate_session_nonce,
    inject_modifiers,
    select_modifiers,
)
from agent_amplifier.token_budget import TokenBudgetController
from agent_amplifier.tool_selector import classify_tools, recommend_tools
from agent_amplifier.types import (
    AmplifierConfig,
    AmplifierEvent,
    ConvergenceState,
    EffortLevel,
    Outcome,
    PhaseIndex,
    RecalledPattern,
    TaskClassification,
)

if TYPE_CHECKING:  # pragma: no cover
    from agent_amplifier.adapter_base import AdapterBase

LOG = logging.getLogger("agent_amplifier.kernel")


_IN_KERNEL_LOCK: ContextVar[bool] = ContextVar(
    "_IN_KERNEL_LOCK", default=False
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _budget_report_to_dict(report: Any) -> dict[str, Any]:
    """Coerce a ``BudgetReport`` (slotted, frozen) into a JSON-safe dict.

    ``BudgetReport`` from token_budget.py is ``@dataclass(frozen=True,
    slots=True)`` — it has no ``__dict__``. We use ``dataclasses.asdict``
    which handles slotted dataclasses cleanly. CRIT-1 fix.

    Enum values (``mode``) are converted to strings for JSON safety.
    """
    try:
        d = asdict(report)
    except TypeError:
        # Defensive fallback for non-dataclass reports.
        return {
            "mode": getattr(getattr(report, "mode", None), "value", "?"),
            "allocated": getattr(report, "allocated", 0),
            "used": getattr(report, "used", 0),
            "remaining": getattr(report, "remaining", 0),
        }
    if "mode" in d and hasattr(d["mode"], "value"):
        d["mode"] = d["mode"].value
    return d


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KernelContractError(RuntimeError):
    """Raised when a sub-module returns a value the kernel cannot process.

    V2.0: 3-field message (what / why / fix). Both ``what`` and
    ``why`` pass through ``redact()`` before construction; ``fix`` is NOT
    redacted because it is a developer-facing instruction (no user data).
    """

    def __init__(self, *, what: str, why: str, fix: str) -> None:
        message = (
            f"{redact(what)}\n"
            f"  why: {redact(why)}\n"
            f"  fix: {fix}"
        )
        super().__init__(message)
        self.what, self.why, self.fix = what, why, fix


class KernelReentrancyError(RuntimeError):
    """raised if a user-supplied callback re-enters the kernel while
    the core lock is held.
    """


# ---------------------------------------------------------------------------
# Mutable per-session state
# ---------------------------------------------------------------------------


@dataclass
class _KernelState:
    """Mutable per-session state. Accessed under ``core._lock``."""

    iteration: int = 0
    tool_call_count: int = 0
    step_id: int = 0
    phase: PhaseIndex = PhaseIndex.EXPLORE
    last_classification: TaskClassification | None = None
    anchor: GoalAnchor | None = None
    last_drift: float = 0.0
    last_convergence: ConvergenceState = ConvergenceState.IMPROVING
    started_monotonic: float = field(default_factory=time.monotonic)
    max_tier_kill_switch: bool = False
    finalized: bool = False


# ---------------------------------------------------------------------------
# StepEnvelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepEnvelope:
    """Frozen, slot-backed return-type for ``before_step``.

    Type-safe, ~2× faster to construct than a dict, and adapter-friendly.
    Backward compat: ``to_dict()`` returns the legacy dict shape for
    adapters that haven't migrated.

    ``budget`` and ``extras`` are exposed as
    ``Mapping`` (read-only views via ``MappingProxyType``).  The class is
    frozen at the top level AND at the nested-dict level — mutating
    ``env.extras`` raises ``TypeError`` instead of silently corrupting
    the snapshot the kernel handed back.
    """

    classification: TaskClassification
    thinking_trigger: str | None
    phase: str
    persona: str
    recommended_tools: tuple[str, ...]
    recommended_groups: tuple[str, ...]
    iteration: int
    step_id: int
    envelope: str
    budget: Mapping[str, Any]
    recalled_patterns: tuple[str, ...]
    suggested_model: str | None
    extras: Mapping[str, Any]

    def __post_init__(self) -> None:
        # freeze nested mutable mappings.  We use
        # ``object.__setattr__`` because the dataclass is frozen at the
        # top-level setattr.
        if not isinstance(self.budget, MappingProxyType):
            object.__setattr__(
                self, "budget", MappingProxyType(dict(self.budget))
            )
        if not isinstance(self.extras, MappingProxyType):
            object.__setattr__(
                self, "extras", MappingProxyType(dict(self.extras))
            )

    def to_dict(self) -> dict[str, Any]:
        """Legacy dict view for adapters that haven't migrated to StepEnvelope."""
        d = dict(self.extras)
        d.update(
            {
                "amp_classification": self.classification.to_dict(),
                "amp_thinking_trigger": self.thinking_trigger,
                "amp_phase": self.phase,
                "amp_persona": self.persona,
                "amp_recommended_tools": list(self.recommended_tools),
                "amp_recommended_groups": list(self.recommended_groups),
                "amp_iteration": self.iteration,
                "amp_step_id": self.step_id,
                "amp_envelope": self.envelope,
                "amp_budget": dict(self.budget),
                "amp_recalled_patterns": list(self.recalled_patterns),
                "amp_suggested_model": self.suggested_model,
            }
        )
        return d


# ---------------------------------------------------------------------------
# _AmplifierCore (anyio-native, the truth)
# ---------------------------------------------------------------------------


class _AmplifierCore:
    """Async-native kernel. Owns all state. NOT directly instantiated by users.

    DO NOT instantiate more than one ``_AmplifierCore`` per session — the
    ``step_id`` invariant is per-instance.
    """

    # Type hints for mypy --strict (the actual assignment uses
    # object.__setattr__ to skirt our own __setattr__ guard during init).
    _config: AmplifierConfig

    def __init__(
        self,
        config: AmplifierConfig | None = None,
        *,
        adapter: AdapterBase | None = None,
        memory_recall: Callable[[str, int], list[RecalledPattern]] | None = None,
        memory_remember: Callable[[Outcome], None] | None = None,
    ) -> None:
        # Order matters: write _config to a private attribute FIRST so the
        # ``__setattr__`` guard (set later) doesn't fire on init. Use the
        # parent ``object.__setattr__`` to bypass the guard while it's
        # being installed.
        object.__setattr__(self, "_config", config or load_config())
        self._lock: anyio.Lock = anyio.Lock()
        self._state = _KernelState()
        self._session_nonce = generate_session_nonce()

        # Sub-modules — all pure-Python no-I/O at construction.
        self._goal = GoalAnchorService(
            reinjection_interval=self._config.goal_reinjection_interval,
        )
        self._convergence = ConvergenceDetector(
            converged_threshold=self._config.convergence_threshold,
            max_iterations=self._config.max_iterations,
        )
        self._budget = TokenBudgetController(
            self._config.budget_mode,
            observability_callback=self._make_observability_proxy(),
        )
        # IP-8 v2: resolve the user-selected PersonaFlavor once at session
        # start.  Unknown slugs fall back to senior-engineer (resolve_flavor
        # is fail-open).  The flavor's description overlays the role text
        # in compose_persona(); iteration-driven StrictnessProfile escalation
        # is independent and unchanged.
        from agent_amplifier.custom_personas import (
            CustomPersona as _CustomPersona,  # lazy to avoid circular
        )

        self._persona_flavor: _CustomPersona = resolve_flavor(self._config.persona)

        self._model_router = ModelRouter()

        # V2.1 memory plane (.5).
        self._adapter: AdapterBase | None = adapter
        self._memory_recall: (
            Callable[[str, int], list[RecalledPattern]] | None
        ) = memory_recall
        self._memory_remember: Callable[[Outcome], None] | None = (
            memory_remember
        )
        # H3: kernel-level idempotency dedup. The same
        # ``Outcome`` (canonical-JSON SHA-256 keyed) submitted twice is
        # only forwarded to the user's memory_remember once. This is
        # universal — any backend (SLM, file append, blob store) gets the
        # benefit. Bounded to 4096 keys; oldest-by-insertion wins.
        self._remember_dedup: set[str] = set()
        # Insertion-ordered dedup queue so we can evict the oldest hash
        # when we hit the cap without retaining unbounded memory.
        self._remember_dedup_order: list[str] = []
        self._remember_dedup_lock = threading.Lock()

        LOG.info(
            "AmplifierKernel ready: cfg=%s adapter=%s memory_recall=%s",
            redact(str(self._config.to_dict())),
            type(adapter).__name__ if adapter is not None else None,
            memory_recall is not None,
        )

    # ------------------------------------------------------------------
    # Config protection
    # ------------------------------------------------------------------

    @property
    def config(self) -> AmplifierConfig:
        return self._config

    def __setattr__(self, name: str, value: Any) -> None:

        # The first __init__ assignment uses object.__setattr__ to bypass
        # this guard — by the time any external code calls amp._core._config
        # = X, _config is already set, so the guard correctly fires.
        if name in ("_config", "config") and getattr(
            self, "_config", None
        ) is not None:
            raise AttributeError(
                "AmplifierConfig is immutable after construction. "
                "Construct a new AgentAmplifier with the new config."
            )
        super().__setattr__(name, value)

    # ------------------------------------------------------------------
    # Observability dispatch
    # ------------------------------------------------------------------

    def _make_observability_proxy(
        self,
    ) -> Callable[[AmplifierEvent, dict[str, Any]], None]:
        """Closure passed to TokenBudgetController.__init__ — never re-enters."""

        def _proxy(event: AmplifierEvent, payload: dict[str, Any]) -> None:
            self._emit(event, payload)

        return _proxy

    def _emit(
        self, event: AmplifierEvent, payload: dict[str, Any]
    ) -> None:
        """Fire the user's observability_callback. NEVER raises.

        B5: ``_IN_KERNEL_LOCK`` is now actively set at
        ``before_step`` / ``after_step`` entry to detect re-entrant
        callbacks. ``_emit`` deliberately does NOT consult the ContextVar
        — we WANT events to fire while we're inside the kernel call. The
        re-entry detection lives in the public entry points where it can
        raise ``KernelReentrancyError`` cleanly.

        Callbacks must not re-enter the kernel; if they do, the entry
        point raises. ``_emit`` invokes the user callback in its own
        try/except so a misbehaving observer never breaks the kernel.
        """
        # Always log at INFO so users can grep without subscribing.
        LOG.info(
            "amp.%s: %s", event.value, redact(str(payload))
        )
        cb = self._config.observability_callback
        if cb is None:
            return
        try:
            cb(event, payload)
        except KernelReentrancyError:
            # Callback tried to re-enter the kernel. Bubbling would crash
            # the host; instead we log + drop. The re-entry guard at the
            # entry point already prevented state corruption.
            LOG.warning(
                "observability_callback re-entered kernel and was rejected"
            )
        except Exception as e:
            LOG.warning(
                "observability_callback raised: %r", redact(repr(e))
            )

    # ------------------------------------------------------------------
    # Memory plane (V2.1, .5.2 / §3.5.3)
    # ------------------------------------------------------------------

    def _resolve_recall(
        self, query: str, limit: int = 3
    ) -> list[RecalledPattern]:
        """Resolve memory recall in priority order.

        Priority:
            1. explicit ``memory_recall`` callback (if set)
            2. adapter ``default_memory_recall`` (if adapter is set)
            3. empty list

        Each returned pattern has its ``text`` field already capped +
        neutralized via ``recall_safety.apply_recall_safety``. Smuggling
        signals are logged but never raised. Callback exceptions become
        WARNING and yield ``[]``.
        """
        fn: Callable[[str, int], list[RecalledPattern]] | None
        if self._memory_recall is not None:
            fn = self._memory_recall
        elif self._adapter is not None:
            fn = self._adapter.default_memory_recall
        else:
            fn = None
        if fn is None:
            return []
        try:
            raw = fn(query, limit)
        except Exception as e:
            LOG.warning("memory_recall failed: %r", redact(repr(e)))
            return []
        # The type signature of ``fn`` promises ``list[RecalledPattern]`` but
        # the runtime contract permits ``None`` (callback explicitly opting
        # out for this query).  ``cast`` to ``object`` so mypy permits the
        # subsequent runtime guards (None / non-iterable).
        # /023.
        from typing import cast as _cast
        raw_obj = _cast(object, raw)
        if raw_obj is None:
            return []
        try:
            raw_iter = iter(raw_obj)  # type: ignore[call-overload]
        except TypeError as e:
            LOG.warning(
                "memory_recall returned non-iterable: %r", redact(repr(e))
            )
            return []
        bounded = list(itertools.islice(raw_iter, limit))
        safe: list[RecalledPattern] = []
        for pat in bounded:
            # per-item try/except so ONE malformed
            # pattern does NOT degrade the entire before_step.  We log,
            # drop the bad item, and continue with the rest.
            try:
                if not isinstance(pat, RecalledPattern):
                    # Tolerate adapters that mistakenly return raw strings —
                    # wrap them.  The dataclass default values fill the gaps.
                    pat = RecalledPattern(text=str(pat))
                text, signals = recall_safety.apply_recall_safety(pat.text)
                if signals:
                    # H1: redact pat.source before logging — adapters
                    # sometimes embed credentials / paths that contain emails
                    # or API keys.
                    LOG.warning(
                        "memory_recall smuggling signals from %s: %s",
                        redact(pat.source) or "<unknown>",
                        signals,
                    )
                if text:
                    safe.append(dataclasses.replace(pat, text=text))
            except Exception as e:
                LOG.warning(
                    "memory_recall: dropping malformed pattern: %r",
                    redact(repr(e)),
                )
                continue
        return safe

    # H3: bound the dedup queue so a long-running session can't
    # accumulate unbounded outcome keys. ~4 KB max (4096 * sha-256 hex == 256 KB).
    _REMEMBER_DEDUP_MAX: int = 4096

    @staticmethod
    def _outcome_idempotency_key(outcome: Outcome) -> str:
        """Canonical SHA-256 of the outcome's serializable shape.

        Two outcomes are "the same" iff their ``to_dict()`` projections are
        bit-identical. Sort keys so insertion order doesn't perturb the
        hash. This is the  idempotency contract restored in H3.
        """
        canonical = json.dumps(outcome.to_dict(), sort_keys=True)
        # usedforsecurity=False — idempotency keys are NOT a security boundary;
        # a collision merely means the user's callback fires once instead of
        # twice. SHA-256 is overkill but stdlib + future-proof.
        return hashlib.sha256(
            canonical.encode("utf-8"), usedforsecurity=False
        ).hexdigest()

    def _resolve_remember(self, outcome: Outcome) -> None:
        """Fire-and-forget write. Never raises (V2.1-CHG-6).

        H3: kernel-level idempotency dedup. The same
        ``Outcome`` submitted twice will fire the user's callback only on
        the first call. Identity is computed from the canonical-JSON
        SHA-256 of ``Outcome.to_dict()``; the dedup set is bounded to
        ``_REMEMBER_DEDUP_MAX`` keys with FIFO eviction.
        """
        # Idempotency check FIRST — even if no callback is set, we treat
        # the Outcome as "consumed" so a later wiring of memory_remember
        # doesn't re-fire historical outcomes.
        key = self._outcome_idempotency_key(outcome)
        with self._remember_dedup_lock:
            if key in self._remember_dedup:
                return
            self._remember_dedup.add(key)
            self._remember_dedup_order.append(key)
            # Bound the cache to keep memory flat over long sessions.
            if len(self._remember_dedup_order) > self._REMEMBER_DEDUP_MAX:
                evicted = self._remember_dedup_order.pop(0)
                self._remember_dedup.discard(evicted)
        fn: Callable[[Outcome], None] | None
        if self._memory_remember is not None:
            fn = self._memory_remember
        elif self._adapter is not None:
            fn = self._adapter.default_memory_remember
        else:
            fn = None
        if fn is None:
            return
        try:
            fn(outcome)
        except Exception as e:
            LOG.warning("memory_remember failed: %r", redact(repr(e)))

    # ------------------------------------------------------------------
    # before_step
    # ------------------------------------------------------------------

    async def before_step(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> StepEnvelope:
        """Build the amplified context for the next agent step.

        Returns a NEW ``StepEnvelope`` (frozen). Never raises ( /
        locked E-7) — on internal failure, returns a degenerate envelope
        that mirrors the input context with no amplification applied.

        B5: re-entry from a user callback is now detected and
        rejected with ``KernelReentrancyError``. Without this guard the
        ``_IN_KERNEL_LOCK`` ContextVar set by V2.0  was dead code
        (only ``.get()`` calls existed; no one called ``.set(True)``).
        """
        if _IN_KERNEL_LOCK.get():
            raise KernelReentrancyError(
                "before_step re-entered from inside a kernel callback. "
                "User callbacks (observability_callback, memory_recall, "
                "memory_remember) MUST NOT call AgentAmplifier methods."
            )
        token = _IN_KERNEL_LOCK.set(True)
        try:
            return await self._before_step_inner(query, context)
        except KernelContractError:
            # KernelContractError is a developer-facing failure; do NOT
            # swallow at the outer boundary — bubble for tests + dev UX.
            raise
        except Exception as e:
            LOG.warning(
                "amp.before_step failed, returning unchanged: %r",
                redact(repr(e)),
            )
            return self._degenerate_envelope(query, context)
        finally:
            _IN_KERNEL_LOCK.reset(token)

    async def _before_step_inner(
        self,
        query: str,
        context: dict[str, Any] | None,
    ) -> StepEnvelope:

        # (e.g. available_tools=42) hits the outer try and degrades
        # gracefully.
        ctx = validate_context(context)


        config_snapshot = self._config

        # ---- Step 0. V2.1 memory plane recall (.6.M) ------
        # Already capped + neutralized + smuggling-signaled by
        # ``_resolve_recall``. Time budget is the callback's responsibility;
        # the kernel imposes none. Returns [] when no memory provider set.
        # H6 (QA-H02): bound is now configurable via
        # ``AmplifierConfig.recall_limit`` (default 3, range [1, 100]).
        seed_patterns = self._resolve_recall(
            query, limit=config_snapshot.recall_limit
        )
        seed_texts: tuple[str, ...] = tuple(
            p.text for p in seed_patterns if p.text
        )

        # ---- Step 1. Anchor capture (double-checked, ) ------------
        async with self._lock:
            need_anchor = self._state.anchor is None
        if need_anchor:
            anchor_obj = self._goal.capture(query)
            async with self._lock:
                # NOTE: coverage.py's branch tracker cannot record the False
                # arc through the trailing ``async with`` exit; the racing
                # scenario IS exercised by
                # ``test_anchor_double_check_sees_value_set_during_unlock_window``
                # but the (428→432) arc never gets emitted by the asyncio
                # bytecode. ``no branch`` keeps the coverage report honest.
                if self._state.anchor is None:  # re-check  # pragma: no branch
                    self._state.anchor = anchor_obj

        # ---- Step 2. Snapshot mutables OUTSIDE the lock ----------------
        async with self._lock:
            iteration = self._state.iteration
            phase = self._state.phase
            anchor_snapshot = self._state.anchor
            self._state.step_id += 1
            step_id = self._state.step_id
            self._state.tool_call_count += 1

        # ---- Step 2.5. classify with optional cross-turn context --------
        # Adapters (e.g. ClaudeCodeAdapter UserPromptSubmit hook) may
        # supply a ``prior_classification`` in the context dict; the
        # router then inherits one-step-down tier+domain when the current
        # prompt is a "conversational continuation" (bare ack, numbered
        # answer, brief question). This closes the misclass observed in
        # real telemetry where short follow-ups to high-complexity turns
        # were re-classified as minimal/general. See effort_router.
        prior_classification_raw = ctx.get("prior_classification")
        prior_classification = (
            prior_classification_raw
            if isinstance(prior_classification_raw, TaskClassification)
            else None
        )
        if prior_classification is not None:
            classification = classify_with_context(
                query,
                prior_classification=prior_classification,
                config=(
                    config_snapshot
                    if config_snapshot.escalate_low_confidence
                    else None
                ),
            )
        elif config_snapshot.escalate_low_confidence:
            classification = classify_with_config(query, config_snapshot)
        else:
            classification = classify(query)
        if not isinstance(classification, TaskClassification):
            raise KernelContractError(
                what="effort_router.classify returned wrong type",
                why=(
                    f"got {type(classification).__name__!r}, "
                    "expected TaskClassification"
                ),
                fix=(
                    "implementer must update sibling LLD or kernel; "
                    "do NOT silently coerce"
                ),
            )

        # ---- Step 2.6. NOTE-LLD04-A: per-session MAX-tier kill switch --
        # H4: collapse the kill-switch READ + WRITE into a single
        # critical section so concurrent before_step calls cannot all
        # observe ``kill_switch=False`` and skip the cap (DIST-03).
        wants_escalate = (
            "max_tier_escalated" in classification.matched_signals
        )
        async with self._lock:
            kill_switch = self._state.max_tier_kill_switch
            # NOTE: coverage.py's branch tracker cannot record the
            # ``False`` arc through the trailing ``async with`` exit on
            # asyncio bytecode. The False branch (no escalation) IS
            # exercised by every plain-query test. ``pragma: no branch``
            # keeps the report honest — same idiom as the anchor
            # double-check in step 1 above.
            if wants_escalate and not kill_switch:  # pragma: no branch
                # First call to escalate flips the switch atomically; this
                # call still gets MAX but observes kill_switch=False so
                # it doesn't get capped. Subsequent calls observe True
                # and cap to HIGH below — atomicity proved by H4 test.
                self._state.max_tier_kill_switch = True
        if kill_switch and classification.complexity is EffortLevel.MAX:
            classification = replace(
                classification,
                complexity=EffortLevel.HIGH,
                matched_signals=(
                    *classification.matched_signals,
                    "max_tier_kill_switched",
                ),
            )
            LOG.info(
                "amp.max_tier_kill_switch active; capping at HIGH this session"
            )

        # ---- Step 3. Allocate budget for this effort -------------------
        self._budget.allocate(classification.complexity)

        # ---- Step 4. Goal-anchor inject as a function of step_id
        anchor_for_inject = (
            anchor_snapshot
            if anchor_snapshot is not None
            else self._goal.capture(query)
        )
        anchored_query = self._goal.inject(
            query,
            anchor_for_inject,
            step_id,
            interval=config_snapshot.goal_reinjection_interval,
        )

        # ---- Step 5. Modifiers + persona (lock-free) -------------------
        # IP-8 v2: compose from user-selected flavor + iteration strictness.
        # ``self._persona_flavor`` is resolved once at __init__ from config.
        # ``get_strictness_profile`` provides the iteration-escalation axis.
        # get_persona is still imported for code-path backward compat but the
        # role text is now supplied by the flavor; suppress the unused-var warning
        # by not assigning it.
        get_persona(iteration)  # validates clamp behavior; result unused
        strictness_profile = get_strictness_profile(iteration)
        persona_prompt = compose_persona(
            description=self._persona_flavor.description,
            review_focus=tuple(self._persona_flavor.review_focus),
            profile=strictness_profile,
        )
        modifiers = select_modifiers(
            classification.complexity,
            phase,
            persona_role=self._persona_flavor.description,
        )

        # ---- Step 6. Phase prompt (with strict slot validation) --------
        # skip full phase scaffolding for MINIMAL/LOW prompts.
        # The EXPLORE 3-option tree + AWAITING-EVALUATION ceremony is pure
        # overhead on "yes", "fix the typo", or other simple prompts. End
        # users see a verbose scaffolded response when they wanted direct
        # conversation. Gate by complexity:
        #   MINIMAL → no phase wrapper at all
        #   LOW     → no phase wrapper (light nudge only via modifiers)
        #   MEDIUM+ → full phase scaffolding (EXPLORE/EVALUATE/EXECUTE/etc.)
        _skip_phase = classification.complexity in (
            EffortLevel.MINIMAL,
            EffortLevel.LOW,
        )

        if _skip_phase:
            phase_prompt = ""
        else:
            from agent_amplifier.phase_prompts import required_slots as _rs

            needed = _rs(phase)
            phase_slots: dict[str, str] = {"anchor": query}
            for slot_name in ("prev_output", "chosen", "issues"):  # pragma: no cover - multi-iteration
                if slot_name in needed and slot_name in ctx:
                    phase_slots[slot_name] = str(ctx[slot_name])

            try:
                phase_prompt = get_phase_prompt(phase, phase_slots)
            except (KeyError, ValueError, TypeError) as e:
                fallback = os.environ.get("AGENT_AMP_FALLBACK_PHASE")
                if fallback:
                    LOG.warning(
                        "amp.phase_slot_missing key=%s; "
                        "AGENT_AMP_FALLBACK_PHASE=%s in effect",
                        redact(str(e)),
                        fallback,
                    )
                    async with self._lock:
                        self._state.phase = PhaseIndex.EXPLORE
                    phase = PhaseIndex.EXPLORE
                    phase_prompt = get_phase_prompt(phase, {"anchor": query})
                else:
                    raise KernelContractError(
                        what=(
                            f"Adapter did not provide required slot for phase "
                            f"{phase.name}"
                        ),
                        why=(
                            f"phase_prompts.get_phase_prompt raised "
                            f"{type(e).__name__}: {e}"
                        ),
                        fix=(
                            "If you wrote this adapter, see "
                            "https://docs.qualixar.com/agent-amplifier/"
                            "adapter-contract#phase-slots\n"
                            "  - If bundled adapter, open issue at "
                            "https://github.com/qualixar/agent-amplifier/"
                            "issues/new\n"
                            "  - To work around: set "
                            "AGENT_AMP_FALLBACK_PHASE=EXPLORE"
                        ),
                    ) from e

        # ---- Step 7. Render envelope (cache-boundary order) ------------
        envelope = inject_modifiers(
            f"{phase_prompt}\n\n{anchored_query}",
            modifiers,
            persona_role=self._persona_flavor.description,
            session_nonce=self._session_nonce,
        )

        # ---- Step 8. Tool recommendation (lock-free) -------------------
        available = ctx.get("available_tools") or []
        recommended_tools = (
            tuple(recommend_tools(query, list(available)))
            if available
            else ()
        )
        recommended_groups = tuple(classify_tools(query))

        # ---- Step 9. Save classification under lock --------------------
        async with self._lock:
            self._state.last_classification = classification

        env = StepEnvelope(
            classification=classification,
            thinking_trigger=suggest_thinking_trigger(
                classification.complexity
            )
            or None,
            phase=phase.name,
            persona=persona_prompt,
            recommended_tools=recommended_tools,
            recommended_groups=recommended_groups,
            iteration=iteration,
            step_id=step_id,
            envelope=envelope,
            budget=_budget_report_to_dict(self._budget.report()),
            recalled_patterns=seed_texts,
            suggested_model=self._model_router.suggest(
                classification.complexity,
                domain=classification.domain,
            ).tier if self._model_router.enabled else None,
            extras=ctx,
        )

        # ---- Step 10. Emit BEFORE_STEP OUTSIDE the lock ----------------
        self._emit(
            AmplifierEvent.BEFORE_STEP,
            {
                "iteration": iteration,
                "step_id": step_id,
                "phase": phase.name,
                "effort": classification.complexity.value,
                "thinking_trigger": env.thinking_trigger,
                "recommended_groups": list(recommended_groups),
            },
        )
        return env

    def _degenerate_envelope(
        self,
        query: str,
        context: dict[str, Any] | None,
    ) -> StepEnvelope:
        """returned by ``before_step`` when the inner path raised.

        Mirrors input context with no amplification applied — the host
        agent continues normally without any kernel injection.
        """
        ctx = dict(context or {}) if isinstance(context, dict) else {}
        return StepEnvelope(
            classification=TaskClassification(
                complexity=EffortLevel.MINIMAL,
                domain="degraded",
                estimated_tokens=0,
                confidence=0.0,
                matched_signals=("amp_degraded",),
            ),
            thinking_trigger=None,
            phase=self._state.phase.name,
            persona="",
            recommended_tools=(),
            recommended_groups=(),
            iteration=self._state.iteration,
            step_id=self._state.step_id,
            envelope=query,
            budget={"used": 0, "limit": 0, "mode": "degraded"},
            recalled_patterns=(),
            suggested_model=None,
            extras=ctx,
        )

    # ------------------------------------------------------------------
    # after_step
    # ------------------------------------------------------------------

    async def after_step(
        self,
        context: dict[str, Any] | StepEnvelope,
        result: str,
    ) -> dict[str, Any]:
        """Update convergence + drift; decide continue / stop / re_anchor.

        Lock topology: snapshot inputs under lock; pure compute OUTSIDE;
        commit under lock; emit OUTSIDE.

        B5: re-entry guard mirrors ``before_step``.
        """
        if _IN_KERNEL_LOCK.get():
            raise KernelReentrancyError(
                "after_step re-entered from inside a kernel callback. "
                "User callbacks (observability_callback, memory_recall, "
                "memory_remember) MUST NOT call AgentAmplifier methods."
            )
        token = _IN_KERNEL_LOCK.set(True)
        try:
            return await self._after_step_inner(context, result)
        except Exception as e:
            LOG.warning(
                "amp.after_step failed, returning continue: %r",
                redact(repr(e)),
            )
            return {
                "action": "continue",
                "iteration": self._state.iteration,
                "phase": self._state.phase.name,
                "warning": "amp_degraded",
            }
        finally:
            _IN_KERNEL_LOCK.reset(token)

    async def _after_step_inner(
        self,
        context: dict[str, Any] | StepEnvelope,
        result: str,
    ) -> dict[str, Any]:
        if isinstance(context, StepEnvelope):
            ctx: dict[str, Any] = context.to_dict()
        else:
            ctx = dict(context)

        # ---- Step 1. Snapshot under lock (H5) -----------------
        # H5: snapshot iteration + tool_call_count atomically once at the
        # top, use the snapshot for ALL decisions. Without this, subsequent
        # reads (after the lock releases) can observe newer values written
        # by the same task's ``_continue_decision`` or by concurrent tasks
        # under the kernel's ``anyio.Lock``.
        async with self._lock:
            iteration = self._state.iteration
            tool_call_count_snapshot = self._state.tool_call_count
            anchor = self._state.anchor

        # ---- Step 2. Track tokens (budget has its own lock) -------------
        tokens = int(ctx.get("amp_tokens_used") or len(result) // 4)
        self._budget.track(tokens)

        # ---- Step 3. Convergence + drift (each module thread-safe) -----
        conv_state = self._convergence.update(result, iteration)
        drift = (
            self._goal.measure_drift(result, anchor)
            if anchor is not None
            else 0.0
        )
        drift_level = self._goal.classify_drift(drift)

        # ---- Step 4. Commit + decide under lock -------------------------
        async with self._lock:
            self._state.last_convergence = conv_state
            self._state.last_drift = drift

        # ---- Step 5. Decision (pure of snapshots; OUTSIDE lock) --------
        decision: dict[str, Any] = {}
        budget_stop = self._budget.should_stop_for_budget()

        if budget_stop:
            decision = {
                "action": "stop",
                "reason": "budget_exhausted",
                "output": self._convergence.best_output(),
            }
            self._emit(
                AmplifierEvent.ON_BUDGET_HIT,
                {"report": _budget_report_to_dict(self._budget.report())},
            )
        elif conv_state is ConvergenceState.CONVERGED:
            decision = {
                "action": "stop",
                "reason": "converged",
                "output": self._convergence.best_output(),
            }
            self._emit(
                AmplifierEvent.ON_CONVERGE, {"iteration": iteration}
            )
        elif conv_state is ConvergenceState.OSCILLATING:
            decision = {
                "action": "stop",
                "reason": "oscillating",
                "output": self._convergence.best_output(),
                "warning": "iteration outputs are oscillating",
            }
        elif drift_level is DriftLevel.DRIFTED:
            decision = {
                "action": "re_anchor",
                "drift_score": drift,
                "drift_level": drift_level.value,
                "warning": (
                    "drift exceeds alert threshold; goal anchor will be "
                    "re-injected on next iteration."
                ),
            }
            self._emit(AmplifierEvent.ON_DRIFT, {"drift": drift})
        elif drift_level is DriftLevel.DRIFTING:
            decision = await self._continue_decision()
            decision["drift_level"] = drift_level.value
            decision["drift_score"] = drift
            decision["warning"] = (
                "drift approaching alert threshold; consider explicit re-anchor."
            )
        elif self._convergence.should_stop():
            decision = {
                "action": "stop",
                "reason": "max_iterations",
                "output": self._convergence.best_output(),
            }
        else:
            decision = await self._continue_decision()

        # ---- Step 6. Emit AFTER_STEP OUTSIDE the lock -------------------
        # H5: report the SNAPSHOT'd iteration + tool_call_count
        # so observers see consistent values even when ``_continue_decision``
        # has already bumped the live state by the time the emit runs.
        self._emit(
            AmplifierEvent.AFTER_STEP,
            {
                "iteration": iteration,
                "tool_call_count": tool_call_count_snapshot,
                "decision": decision.get("action"),
                "drift": drift,
                "convergence": conv_state.value,
            },
        )
        return decision

    async def _continue_decision(self) -> dict[str, Any]:
        """Helper: advance phase and bump iteration under lock."""
        async with self._lock:
            self._state.phase = advance_phase(self._state.phase)
            self._state.iteration += 1
            iteration = self._state.iteration
            phase = self._state.phase
        self._emit(
            AmplifierEvent.ON_ITERATION,
            {"iteration": iteration, "phase": phase.name},
        )
        return {
            "action": "continue",
            "iteration": iteration,
            "phase": phase.name,
        }

    # ------------------------------------------------------------------
    # finalize
    # ------------------------------------------------------------------

    async def finalize(self) -> dict[str, Any]:
        """End-of-session: persist outcome via memory plane, return budget report.

        idempotent — second call is a no-op returning the same
        report shape. V2.1: write goes through ``_resolve_remember`` which
        prefers the explicit ``memory_remember`` callback, then falls back
        to ``adapter.default_memory_remember``, then no-op.
        """
        async with self._lock:
            if self._state.finalized:
                return self._build_finalize_report()
            self._state.finalized = True
            cls = self._state.last_classification
            iteration = self._state.iteration
            convergence = self._state.last_convergence
            anchor_str = (
                self._state.anchor.text if self._state.anchor else ""
            )

        if cls is not None:
            outcome = Outcome(
                query=anchor_str,
                effort=cls.complexity,
                iterations=iteration + 1,
                quality=(
                    1.0
                    if convergence is ConvergenceState.CONVERGED
                    else 0.5
                ),
                converged=(convergence is ConvergenceState.CONVERGED),
                tokens_used=int(self._budget.report().used),
            )
            self._resolve_remember(outcome)

        report = self._build_finalize_report()
        return report

    def _build_finalize_report(self) -> dict[str, Any]:
        report = _budget_report_to_dict(self._budget.report())
        report["iterations_completed"] = self._state.iteration + 1
        report["final_state"] = self._state.last_convergence.value
        report["drift_at_end"] = self._state.last_drift
        report["max_tier_kill_switched"] = self._state.max_tier_kill_switch
        return report


# ---------------------------------------------------------------------------
# Public facades — AgentAmplifier (sync) and AsyncAgentAmplifier (async)
# ---------------------------------------------------------------------------


class AgentAmplifier:
    """Synchronous facade. Calls into ``_AmplifierCore`` via a
    ``BlockingPortal``.

    Threading: safe to call from any thread. The underlying portal serializes
    calls through anyio's event loop. For high concurrency, prefer
    ``AsyncAgentAmplifier`` from within an async runtime.

    DO NOT instantiate more than one ``AgentAmplifier`` per session — the
    ``step_id`` invariant is per-instance.
    """

    def __init__(
        self,
        config: AmplifierConfig | None = None,
        *,
        adapter: AdapterBase | None = None,
        memory_recall: Callable[[str, int], list[RecalledPattern]] | None = None,
        memory_remember: Callable[[Outcome], None] | None = None,
    ) -> None:
        self._portal_holder = PortalHolder()
        # Construction itself is synchronous — _AmplifierCore.__init__ does
        # no await. The portal is needed only for method calls.
        self._core = _AmplifierCore(
            config=config,
            adapter=adapter,
            memory_recall=memory_recall,
            memory_remember=memory_remember,
        )

    # ------------------------------------------------------------------
    # Public API (each method = one portal.call away from the async core)
    # ------------------------------------------------------------------

    def before_step(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> StepEnvelope:
        return self._portal_holder.run_sync(
            self._core.before_step, query, context
        )

    def after_step(
        self, context: Any, result: str
    ) -> dict[str, Any]:
        return self._portal_holder.run_sync(
            self._core.after_step, context, result
        )

    def finalize(self) -> dict[str, Any]:
        return self._portal_holder.run_sync(self._core.finalize)

    def close(self) -> None:
        self._portal_holder.close()

    # context manager
    def __enter__(self) -> AgentAmplifier:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # config + diagnostic
    @property
    def config(self) -> AmplifierConfig:
        return self._core.config

    @property
    def iteration(self) -> int:
        return self._core._state.iteration

    @property
    def phase(self) -> PhaseIndex:
        return self._core._state.phase

    # V2.1: expose memory plane resolution for tests/observability.
    def _resolve_recall(
        self, query: str, limit: int = 3
    ) -> list[RecalledPattern]:
        return self._core._resolve_recall(query, limit)

    def _resolve_remember(self, outcome: Outcome) -> None:
        self._core._resolve_remember(outcome)


class AsyncAgentAmplifier:
    """Async-native facade. Use directly inside an async runtime.

    DO NOT instantiate more than one ``AsyncAgentAmplifier`` per session —
    the ``step_id`` invariant is per-instance.
    """

    def __init__(
        self,
        config: AmplifierConfig | None = None,
        *,
        adapter: AdapterBase | None = None,
        memory_recall: Callable[[str, int], list[RecalledPattern]] | None = None,
        memory_remember: Callable[[Outcome], None] | None = None,
    ) -> None:
        self._core = _AmplifierCore(
            config=config,
            adapter=adapter,
            memory_recall=memory_recall,
            memory_remember=memory_remember,
        )

    async def before_step(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> StepEnvelope:
        return await self._core.before_step(query, context)

    async def after_step(
        self, context: Any, result: str
    ) -> dict[str, Any]:
        return await self._core.after_step(context, result)

    async def finalize(self) -> dict[str, Any]:
        return await self._core.finalize()

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> AsyncAgentAmplifier:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    @property
    def config(self) -> AmplifierConfig:
        return self._core.config

    # V2.1: expose memory plane resolution for tests/observability.
    def _resolve_recall(
        self, query: str, limit: int = 3
    ) -> list[RecalledPattern]:
        return self._core._resolve_recall(query, limit)

    def _resolve_remember(self, outcome: Outcome) -> None:
        self._core._resolve_remember(outcome)


def amplify(
    query: str, config: AmplifierConfig | None = None
) -> dict[str, Any]:
    """One-shot helper. Construct sync facade, run before_step, return as dict.

    Closes the portal in ``finally``. Use this for trivial scripted flows
    where a single amplifier injection is sufficient.
    """
    with AgentAmplifier(config) as amp:
        env = amp.before_step(query, {})
        return env.to_dict()


__all__ = [
    "AgentAmplifier",
    "AsyncAgentAmplifier",
    "KernelContractError",
    "KernelReentrancyError",
    "StepEnvelope",
    "amplify",
]
