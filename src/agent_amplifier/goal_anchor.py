# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""IP-3 Goal Anchor Protocol.

Per . Stateless functional helper that captures the original
user request, decides re-injection cadence, measures semantic drift against
a current iteration's output, and classifies that drift into a stable
``DriftLevel`` enum.

V2.0 thread-safety contract:
    * ``GoalAnchorService`` holds only frozen config after ``__init__``;
      it has no mutable instance state.
    * ``GoalAnchor`` is a frozen dataclass.
    * Therefore both are safe to share across threads without an external
      lock — see .1.

V2.0 verifications inlined into the module:
    * ``threading.Lock`` is NOT reentrant (Python docs `threading` module).
      Verified during pre-flight; we therefore use NO lock here because there
      is no shared mutable state. The kernel () holds its own lock.
    * PEP 703 free-threading data-race surface (Gemini-grounded, 2026-04-26):
      compound check-then-act on ``deque`` is unsafe even with PEP 703's
      per-object micro-locks; module-internal compound operations live in
      ``convergence.py``/``token_budget.py`` under explicit ``threading.Lock``.
      ``goal_anchor`` performs no compound operation; it is therefore safe.

Cross-LLD spot-fix ticket filed during implementation:
    * STAGE-5C-TICKET-1: .2 imports ``DriftLevel`` from
      ``agent_amplifier.types``, but neither
      ``types.py`` defines it. Per the user-prompt instruction (Cluster C
      ) and the goal-anchor-specific scope, ``DriftLevel`` lives here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Final

from agent_amplifier._internal.keyword_set import (
    keyword_set as _keyword_set,
)

LOG = logging.getLogger("agent_amplifier.goal_anchor")


# ---------------------------------------------------------------------------
# Module constants (immutable;  anti-drift)
# ---------------------------------------------------------------------------

#: Drift score at or above this returns :data:`DriftLevel.DRIFTING`.
DRIFT_WARN_THRESHOLD: Final[float] = 0.5

#: Drift score at or above this returns :data:`DriftLevel.DRIFTED`.
DRIFT_ALERT_THRESHOLD: Final[float] = 0.7

#: Default kernel cadence — re-inject every Nth tool call.
DEFAULT_REINJECTION_INTERVAL: Final[int] = 5

#: DECISIONS-LOCKED C-6 — soft anchor expiration. ``LOG.warning`` past this
#: point; we still inject (the original request does not expire just because
#: time passed).
ANCHOR_MAX_AGE_SECONDS: Final[float] = 3600.0

#: Approximate fixed cost (in tokens) of one re-injection's template
#: overhead. Used by callers for budget pre-allocation.
INJECTION_TOKEN_COST_ESTIMATE: Final[int] = 50

#:  / Sec F-07 — hard cap on the escaped anchor text size.
MAX_ANCHOR_ESCAPED_CHARS: Final[int] = 8192

#: Re-injection template. The ``current_context`` token is appended verbatim;
#: ``text`` is already escaped at ``capture()`` time.
INJECTION_TEMPLATE: Final[str] = (
    'GOAL ANCHOR (original request): "{text}"\n'
    "Verify your current action serves this goal.\n"
    "{current_context}"
)


# ---------------------------------------------------------------------------
# Enum: DriftLevel — STABLE PUBLIC API per .
# ---------------------------------------------------------------------------


class DriftLevel(str, Enum):
    """Classification of semantic drift between an iteration's output and the
    original goal anchor (.10).

    Ordering by severity:
        ON_TRACK   < DRIFTING < DRIFTED

    Cross-LLD contract ():
        * kernel calls ``classify_drift(drift)`` once per ``after_step``
        * kernel sets ``decision["drift_level"] = level.value``
        * if level is ``DRIFTED``: kernel emits ``ON_DRIFT`` + sets warning
        * if level is ``DRIFTING``: kernel sets advisory warning
    """

    ON_TRACK = "on_track"
    DRIFTING = "drifting"
    DRIFTED = "drifted"


# ---------------------------------------------------------------------------
# Dataclass: GoalAnchor (frozen, slots)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoalAnchor:
    """Frozen capture of the original user request.

    Immutable. All fields populated by :meth:`GoalAnchorService.capture`.

    Attributes:
        text: Escaped, length-capped, control-char-stripped form of the
            original request. See :func:`_escape_for_template`.
        keyword_set: ASCII-tokenized lower-cased frozenset (drops stop-words
            and tokens of length < 2). Tokenized from the *raw* request, so
            quote-replacement in ``text`` does not affect the keyword set.
        captured_at_monotonic: Result of ``time.monotonic()`` at capture.
            Used only for age-warning logic; never compared to wall time.
        captured_at_wall_iso: ISO-8601 string for log lines only.
        char_count: ``len(text)`` after escaping.
        token_estimate: ``max(1, char_count // 4)`` — a coarse upper bound
            for budget allocation.
    """

    text: str
    keyword_set: frozenset[str]
    captured_at_monotonic: float
    captured_at_wall_iso: str
    char_count: int
    token_estimate: int


# ---------------------------------------------------------------------------
# Service: GoalAnchorService
# ---------------------------------------------------------------------------


class GoalAnchorService:
    """Stateless helper for goal-anchor lifecycle.

    The kernel owns the anchor object and the tool-call counter; this service
    holds only frozen config (thresholds, intervals) after construction.
    Therefore it is safe to share across threads without an external lock.

    See .1 for the full thread-safety contract for the
    Cluster-C cluster.
    """

    def __init__(
        self,
        *,
        reinjection_interval: int = DEFAULT_REINJECTION_INTERVAL,
        warn_threshold: float = DRIFT_WARN_THRESHOLD,
        alert_threshold: float = DRIFT_ALERT_THRESHOLD,
        max_anchor_age_s: float = ANCHOR_MAX_AGE_SECONDS,
    ) -> None:
        if reinjection_interval < 1:
            raise ValueError(
                "reinjection_interval must be >= 1, got "
                f"{reinjection_interval}"
            )
        if not (0.0 <= warn_threshold < alert_threshold <= 1.0):
            raise ValueError(
                f"thresholds invalid: warn={warn_threshold} "
                f"alert={alert_threshold} "
                "(must satisfy 0 <= warn < alert <= 1)"
            )
        self._interval = reinjection_interval
        self._warn = warn_threshold
        self._alert = alert_threshold
        self._max_age_s = float(max_anchor_age_s)

    # -- lifecycle ----------------------------------------------------------

    def capture(self, original_request: str | None) -> GoalAnchor:
        """Freeze the original user request into a :class:`GoalAnchor`.

        escapes the stored text, caps it at
        :data:`MAX_ANCHOR_ESCAPED_CHARS`, and strips ASCII control chars.

        ``None`` is treated as ``""`` — an empty anchor is still a valid
        carrier; ``inject()`` and ``measure_drift()`` short-circuit when the
        anchor's text is empty.
        """
        text_raw = "" if original_request is None else original_request.strip()
        text_escaped = _escape_for_template(text_raw)
        # Tokenize the SOURCE text — keyword set is invariant of escaping
        # transforms (e.g. straight-quote → curly-quote).
        kw = _keyword_set(text_raw)
        char_count = len(text_escaped)
        return GoalAnchor(
            text=text_escaped,
            keyword_set=kw,
            captured_at_monotonic=time.monotonic(),
            captured_at_wall_iso=time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime()
            ),
            char_count=char_count,
            token_estimate=max(1, char_count // 4),
        )

    # -- injection ---------------------------------------------------------

    def inject(
        self,
        current_context: str | None,
        anchor: GoalAnchor,
        tool_call_count: int,
        *,
        interval: int | None = None,
        force: bool = False,
    ) -> str:
        """Decide whether the current tool-call boundary re-injects the anchor.

        Returns the (possibly modified) context. Empty anchors are passthrough.

        Trigger conditions:
            * ``force=True``
            * ``tool_call_count > 0 AND tool_call_count % interval == 0``
            * anchor age > ``max_anchor_age_s`` (``LOG.warning`` then
              still inject — the request does not expire on a wall clock).

        this method does NOT classify drift; the kernel calls
        :meth:`classify_drift` separately after :meth:`measure_drift`.
        """
        if not anchor.text:
            return current_context if current_context is not None else ""

        iv = interval if interval is not None else self._interval
        if iv < 1:
            iv = 1

        age = time.monotonic() - anchor.captured_at_monotonic
        too_old = age > self._max_age_s
        if too_old:

            LOG.warning(
                "Anchor age %.0fs exceeds max %.0fs; re-injecting anyway",
                age,
                self._max_age_s,
            )

        should_inject = (
            force
            or (tool_call_count > 0 and tool_call_count % iv == 0)
            or too_old
        )
        if not should_inject:
            return current_context if current_context is not None else ""

        return INJECTION_TEMPLATE.format(
            text=anchor.text,
            current_context=current_context if current_context is not None else "",
        )

    # -- drift measurement -------------------------------------------------

    def measure_drift(
        self,
        current_output: str,
        anchor: GoalAnchor,
        *,
        precomputed_kw: frozenset[str] | None = None,
    ) -> float:
        """Compute Jaccard-based drift in ``[0, 1]``.

        ``drift = 1 - Jaccard(anchor.keyword_set, current.keyword_set)``.

        when ``precomputed_kw`` is supplied (kernel hot path), we
        skip internal tokenization entirely. Default ``None`` keeps the
        method usable in isolation.

        Edge cases:
            * empty anchor               → 0.0
            * empty current and no kw    → 1.0
            * empty precomputed_kw       → 1.0
            * both empty                 → 0.0
        """
        if not anchor.text or not anchor.keyword_set:
            return 0.0

        if precomputed_kw is not None:
            cur_kw = precomputed_kw
        elif not current_output:
            return 1.0
        else:
            cur_kw = _keyword_set(current_output)

        if not cur_kw:
            return 1.0

        union = anchor.keyword_set | cur_kw
        if not union:  # pragma: no cover - both empty handled above
            return 0.0
        inter = anchor.keyword_set & cur_kw
        jaccard = len(inter) / len(union)
        return max(0.0, min(1.0, 1.0 - jaccard))

    # -- classification ----------------------------------------------------

    def classify_drift(self, drift_score: float) -> DriftLevel:
        """STABLE PUBLIC API.

        Maps a drift score in ``[0, 1]`` to a :class:`DriftLevel`.

        Cross-LLD contract documented on :class:`DriftLevel`.
        """
        if drift_score >= self._alert:
            return DriftLevel.DRIFTED
        if drift_score >= self._warn:
            return DriftLevel.DRIFTING
        return DriftLevel.ON_TRACK

    # -- cost estimation ---------------------------------------------------

    def estimated_injection_tokens(self, anchor: GoalAnchor) -> int:
        """Approximate per-injection cost = anchor token estimate + template."""
        return anchor.token_estimate + INJECTION_TOKEN_COST_ESTIMATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escape_for_template(text: str) -> str:
    """Make ``text`` safe to embed in :data:`INJECTION_TEMPLATE`.

     (Sec F-07):

    1. Truncate to :data:`MAX_ANCHOR_ESCAPED_CHARS` (8192).
    2. Strip non-printable control chars except ``\\n`` and space (which we
       then collapse to a single space).
    3. Replace ``"`` with the curly variant ``“`` to avoid breaking the
       JSON-ish template.
    4. Replace ``\\r`` and remaining ``\\n`` with a space.

    The result is ASCII-printable plus curly quotes plus spaces. Null bytes,
    bell characters, ESC bytes (which carry CSI / terminal-control
    sequences) are eliminated.

    NOTE: This is for prompt-injection-safety in the goal anchor only,
    NOT a cryptographic sanitizer.  owns ``<system-reminder>``
    tag escaping.
    """
    if not text:
        return ""
    if len(text) > MAX_ANCHOR_ESCAPED_CHARS:
        text = text[:MAX_ANCHOR_ESCAPED_CHARS]
    # Drop control chars (preserve printable + \n + space). Per Python's
    # ``str.isprintable``, this excludes ESC (0x1B), null (0x00), bell (0x07),
    # vertical tab, and friends — exactly the chars we need to drop.
    cleaned = "".join(c for c in text if c.isprintable() or c in (" ", "\n"))
    cleaned = cleaned.replace('"', "“")
    cleaned = cleaned.replace("\n", " ").replace("\r", " ")
    return cleaned


__all__ = [
    "ANCHOR_MAX_AGE_SECONDS",
    "DEFAULT_REINJECTION_INTERVAL",
    "DRIFT_ALERT_THRESHOLD",
    "DRIFT_WARN_THRESHOLD",
    "INJECTION_TEMPLATE",
    "INJECTION_TOKEN_COST_ESTIMATE",
    "MAX_ANCHOR_ESCAPED_CHARS",
    "DriftLevel",
    "GoalAnchor",
    "GoalAnchorService",
]
