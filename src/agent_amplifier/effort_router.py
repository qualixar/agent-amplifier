# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Agent Amplifier — Dynamic Effort Router (IP-2).

Heuristic, ML-free, regex-driven classification of free-text queries into
five :class:`~agent_amplifier.types.EffortLevel` tiers.

V2.0 fixes ():

* **** — Pre-compiled alternation regex per tier (5 calls, not 250).
  Eliminates ``query.lower()`` allocation by relying on
  :data:`re.IGNORECASE`. Longest-first sort prevents prefix-stealing of
  multi-word keywords (``"system design"``).

* **** — Denial-of-Wallet hardening:

  - :data:`MAX_QUERY_CHARS` = 8192 cap with ``"query_truncated"`` signal.
  - :data:`MAX_DISTINCT_FOR_ESCALATE` = 2 — single MAX-keyword hits cap at
    HIGH instead of escalating to MAX (rejects ``"security " * 50``).
  - :func:`classify_with_config` honors
    :attr:`AmplifierConfig.escalate_low_confidence`. The bare
    :func:`classify` always plays safe (back off to MEDIUM when
    ``confidence < LOW_CONFIDENCE_THRESHOLD`` and base tier ≥ HIGH).

* **** — Effort router uses regex ``\\b(...)\\b`` alternation, not
  tokenization. It deliberately does **NOT** import
  ``agent_amplifier._internal.keyword_set`` (which is owned by  for
  convergence/goal_anchor). The module-level keyword sets here are
  semantically distinct from that helper's stop-word list.

* **** — Empty / whitespace-only queries log INFO and return
  ``MINIMAL`` deterministically; truncation logs WARNING. No exceptions
  are raised for any ``str`` input.

Threat model: see .2. Out of scope (V1, deferred): semantic
disambiguation of phrases such as ``"the secret to good code"`` (locked
decision B-5).

Performance budget (.8):
    * Module-load: < 50 ms (regex compilation amortized).
    * Per-call P50: < 0.3 ms for queries < 1000 chars.
    * Per-call P99: < 2 ms for any query ≤ MAX_QUERY_CHARS
      (perf gate test in tests/perf/test_classify_p99.py).

Thread-safety: pure functions, no shared mutable state. Module-level
constants are frozen at import time and asserted unchanged by an autouse
``_assert_globals_unchanged`` fixture.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Final

from agent_amplifier.types import (
    AmplifierConfig,
    EffortLevel,
    TaskClassification,
)

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Adversarial-input contract — public constants
# ---------------------------------------------------------------------------

#: Hard cap on classified query length. Anything longer is truncated and a
#: ``"query_truncated"`` signal is appended to ``matched_signals``.
MAX_QUERY_CHARS: Final[int] = 8192

#: Minimum number of *distinct* MAX-tier keywords required to escalate to
#: :attr:`EffortLevel.MAX`. Single-keyword hits cap at :attr:`HIGH`.
MAX_DISTINCT_FOR_ESCALATE: Final[int] = 2

#: Confidence below this threshold causes :func:`classify` (and
#: :func:`classify_with_config` when ``escalate_low_confidence=False``) to
#: back off any HIGH-or-MAX tier to MEDIUM.
LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.6

# ---------------------------------------------------------------------------
# 2. Keyword catalog (frozen at module load)
# ---------------------------------------------------------------------------

MINIMAL_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "typo", "rename", "format", "comment", "lint", "prettier",
        "indent", "whitespace", "alphabetize", "uppercase", "lowercase",
    }
)

LOW_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "fix", "bug", "patch", "explain", "describe", "what is",
        "what does", "show me", "print", "log", "console.log", "find",
        "search", "lookup", "translate", "convert", "rewrite",
    }
)

MEDIUM_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "refactor", "test", "tests", "unit test", "add", "implement",
        "function", "method", "class", "endpoint", "handler", "validate",
        "parse", "serialize", "deserialize", "migrate",
    }
)

HIGH_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "redesign", "review", "audit", "optimize", "performance",
        "scalability", "architecture", "design", "decompose", "extract",
        "module", "service", "api contract", "schema", "data model",
        "concurrency", "race", "deadlock", "memory leak",
    }
)

MAX_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "security", "vulnerability", "cve", "owasp", "injection",
        "auth", "authn", "authz", "encryption", "secret", "key rotation",
        "system design", "distributed", "consensus", "byzantine",
        "compliance", "soc2", "hipaa", "gdpr", "ip review",
        "production deploy", "rollout", "blast radius",
    }
)


# ---------------------------------------------------------------------------
# 3. Pre-compiled alternation regex per tier
# ---------------------------------------------------------------------------


def _build_tier_regex(kw_set: Iterable[str]) -> re.Pattern[str]:
    """Build a case-insensitive alternation regex with longest-first ordering.

    Word boundaries (``\\b``) prevent ``"security"`` from matching inside
    ``"insecurity"``. Longest-first ordering prevents prefix-stealing on
    multi-word keywords (``"system design"`` must not be eaten by a
    hypothetical shorter ``"system"`` rule).

    An empty keyword set returns a regex that never matches anything —
    chosen over ``None`` to keep call sites uniform (``pat.search(...)``
    is always safe to call).
    """
    materialized = list(kw_set)
    if not materialized:
        return re.compile(r"(?!x)x")  # never-matches sentinel
    escaped = sorted((re.escape(k) for k in materialized), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


# Module-level — built ONCE at import time. 5 alternation patterns
# replace 250 substring searches.
_TIER_REGEX: Final[dict[EffortLevel, re.Pattern[str]]] = {
    EffortLevel.MINIMAL: _build_tier_regex(MINIMAL_KEYWORDS),
    EffortLevel.LOW: _build_tier_regex(LOW_KEYWORDS),
    EffortLevel.MEDIUM: _build_tier_regex(MEDIUM_KEYWORDS),
    EffortLevel.HIGH: _build_tier_regex(HIGH_KEYWORDS),
    EffortLevel.MAX: _build_tier_regex(MAX_KEYWORDS),
}


# ---------------------------------------------------------------------------
# 4. Trace / multi-file / question / domain regexes
# ---------------------------------------------------------------------------

# All quantifiers are bounded (``\S{1,256}``, ``.{1,512}``) to remove ReDoS
# attack surface (CWE-1333). Bounded quantifiers preserve semantics for any
# realistic stack trace shape.
TRACE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(
        r"^\s{1,32}at\s+\S{1,256}\s+\(.{1,512}:\d+:\d+\)$", re.MULTILINE
    ),
    re.compile(r"\bException in thread\b"),
    re.compile(r"^panic:", re.MULTILINE),
    re.compile(r"thread '[^']{1,256}' panicked at"),
    re.compile(r"^Error:[^\n]{1,512}\n\s{1,32}at ", re.MULTILINE),
    re.compile(r"\b(SegmentationFault|SIGSEGV|core dumped)\b"),
)

MULTI_FILE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b\d{1,5}\s+files?\b", re.IGNORECASE),
    re.compile(r"\bacross\b[^\n]{0,128}\b(modules?|files?|services?)\b"),
    re.compile(r"\bmono(repo|lith)\b", re.IGNORECASE),
    re.compile(
        r"\b(refactor|rename|migrate)\b[^\n]{0,128}\bevery\b", re.IGNORECASE
    ),
)

QUESTION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"\b(?:why|how|when should|should i|what.{1,32}best)\b", re.IGNORECASE
    ),
)

DESIGN_NOUNS: Final[frozenset[str]] = frozenset(
    {
        "architecture", "system", "approach", "pattern", "trade-off",
        "tradeoff",
    }
)
_DESIGN_NOUN_RE: Final[re.Pattern[str]] = _build_tier_regex(DESIGN_NOUNS)

DOMAIN_HINTS: Final[dict[str, tuple[str, ...]]] = {
    "security": ("security", "vulnerability", "auth", "encrypt", "owasp"),
    "performance": (
        "performance", "latency", "throughput", "optimize", "memory",
    ),
    "data": (
        "database", "sql", "postgres", "schema", "query", "migration",
    ),
    "api": ("api", "endpoint", "rest", "graphql", "handler", "route"),
    "frontend": ("react", "next", "tailwind", "css", "ui", "component"),
    "infra": ("docker", "kubernetes", "k8s", "terraform", "cloud"),
    "tests": ("test", "pytest", "jest", "vitest", "coverage"),
    "docs": ("readme", "docstring", "docs", "tutorial"),
    "general": (),
}

_DOMAIN_REGEX: Final[dict[str, re.Pattern[str]]] = {
    name: (_build_tier_regex(hints) if hints else re.compile(r"(?!x)x"))
    for name, hints in DOMAIN_HINTS.items()
}

# Word-boundary regex for ``estimate_tokens``. Bounded run length blocks
# pathological inputs.
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\S{1,256}")


# ---------------------------------------------------------------------------
# 5. Tier ranking helpers (private)
# ---------------------------------------------------------------------------

_TIER_RANK: Final[dict[EffortLevel, int]] = {
    EffortLevel.MINIMAL: 0,
    EffortLevel.LOW: 1,
    EffortLevel.MEDIUM: 2,
    EffortLevel.HIGH: 3,
    EffortLevel.MAX: 4,
}
_RANK_TIER: Final[dict[int, EffortLevel]] = {v: k for k, v in _TIER_RANK.items()}


def _bump(level: EffortLevel, by: int = 1) -> EffortLevel:
    """Bump tier up by ``by`` steps, capped at :attr:`EffortLevel.MAX`."""
    return _RANK_TIER[
        min(_TIER_RANK[level] + by, _TIER_RANK[EffortLevel.MAX])
    ]


def _max_tier(a: EffortLevel, b: EffortLevel) -> EffortLevel:
    return a if _TIER_RANK[a] >= _TIER_RANK[b] else b


def _distinct_max_count(query: str) -> set[str]:
    """Return the set of *distinct* MAX-tier keywords matched in ``query``.

    Encapsulated per CRIT #2 mitigation — future refactors must update this
    one helper. Lower-cases all hits because :data:`re.IGNORECASE` returns
    the original substring (which preserves case).
    """
    hits = _TIER_REGEX[EffortLevel.MAX].findall(query)
    return {h.lower() for h in hits}


# ---------------------------------------------------------------------------
# 6. Public helpers (independently exported for tests + kernel use)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate token count from word count.

    Rule of thumb: ``1 token ≈ 0.75 English words``, so
    ``tokens ≈ words * 1.3``.

    The 1.3 multiplier is an empirical approximation that matches BPE
    tokenizers (cl100k / o200k families) on English prose to within
    ~10%.  Token budgets in this project are sized with margin so a
    minor under- or over-estimate does not cause kill-switch trips.
    Callers needing exact counts SHOULD plug in the host's native
    tokenizer (e.g. ``tiktoken``); this helper exists as a zero-dep
    fallback for when no tokenizer is available.
    """
    if not text:
        return 0
    word_count = len(_WORD_RE.findall(text))
    return int(word_count * 1.3)


def is_code_heavy(query: str) -> bool:
    """Detect code-heavy queries.

    A query is code-heavy if any of:
      * it contains a fenced code block ``\\`\\`\\``` ;
      * length > 80 chars AND non-alphanumeric ratio > 0.30
        (CRIT #3 retained from V1 — short punctuation is not code);
      * it contains ``function `` / ``class `` / ``def `` / ``=>`` /
        ``();``.

    The ratio computation is an O(L) Python loop. Bounded by
    :data:`MAX_QUERY_CHARS` upstream of any caller in
    :func:`classify`, so worst-case L ≤ 8192 — well within the perf gate.
    """
    if not query:
        return False
    if "```" in query:
        return True
    if len(query) > 80:
        non_alnum = sum(
            1 for c in query if not c.isalnum() and not c.isspace()
        )
        ratio = non_alnum / len(query)
        if ratio > 0.30:
            return True
    return any(
        marker in query
        for marker in ("function ", "class ", "def ", "=>", "();")
    )


def infer_domain(query: str) -> str:
    """Return the first matching domain name; ``"general"`` if none match.

    Iteration order is :data:`DOMAIN_HINTS` insertion order
    (``security`` first → ``general`` last as fallback).
    """
    if not query:
        return "general"
    for name, pat in _DOMAIN_REGEX.items():
        if name == "general":
            continue
        if pat.search(query):
            return name
    return "general"


def suggest_thinking_trigger(level: EffortLevel) -> str:
    """Map an :class:`EffortLevel` to a Claude-style thinking-trigger keyword.

    Conventional mapping used by Claude-class hosts.  Exact internal token
    budgets are host-specific and non-contractual; the keywords below are
    the documented public surface for opt-in extended thinking.

    * ``""`` (no injection) for MINIMAL
    * ``"think"`` for LOW
    * ``"think hard"`` for MEDIUM
    * ``"think harder"`` for HIGH
    * ``"ultrathink"`` for MAX (== Claude Code 2.1.111+ ``xhigh`` effort
      dial on Opus 4.7; same upper-bound semantics, different mechanism —
      we inject the keyword, the host translates it to the model's effort
      parameter)
    """
    return {
        EffortLevel.MINIMAL: "",
        EffortLevel.LOW: "think",
        EffortLevel.MEDIUM: "think hard",
        EffortLevel.HIGH: "think harder",
        EffortLevel.MAX: "ultrathink",
    }[level]


def should_iterate(level: EffortLevel) -> bool:
    """True if the kernel should run the iterative refinement loop."""
    return level in (
        EffortLevel.MEDIUM,
        EffortLevel.HIGH,
        EffortLevel.MAX,
    )


# ---------------------------------------------------------------------------
# 7. classify() — the hot path
# ---------------------------------------------------------------------------


def classify(query: str) -> TaskClassification:
    """Classify ``query`` into a :class:`TaskClassification`.

    Pure function. Deterministic. Idempotent under leading/trailing
    whitespace. Total over ``str`` input — never raises.

    Always plays *safe* on low-confidence HIGH+ tiers (back off to MEDIUM).
    Power users opt into "trust the classifier even at low confidence" via
    :func:`classify_with_config` and
    :attr:`AmplifierConfig.escalate_low_confidence`.

    Performance: P99 < 2 ms for any input ≤ :data:`MAX_QUERY_CHARS`.
    """
    return _classify(query, escalate_low_confidence=False)


def classify_with_config(
    query: str, config: AmplifierConfig
) -> TaskClassification:
    """Classify ``query`` honoring ``config.escalate_low_confidence``.

    The kernel calls this when it has loaded a user config that opts into
    low-confidence escalation. The bare :func:`classify` always plays safe.

    Raises:
        TypeError: if ``config`` is not an :class:`AmplifierConfig`.
    """
    if not isinstance(config, AmplifierConfig):
        raise TypeError(
            "classify_with_config requires AmplifierConfig; got "
            f"{type(config).__name__}"
        )
    return _classify(
        query, escalate_low_confidence=config.escalate_low_confidence
    )


def _classify(
    query: str | None, *, escalate_low_confidence: bool
) -> TaskClassification:
    """Shared classification engine. Single entry point for both public APIs."""
    # Step 0 — sanitation.
    if query is None or not query.strip():
        LOG.info(
            "effort_router.classify received empty query; returning MINIMAL"
        )
        return TaskClassification(
            complexity=EffortLevel.MINIMAL,
            domain="empty",
            estimated_tokens=0,
            confidence=1.0,
            matched_signals=("empty_query",),
        )

    matched_signals: list[str] = []

    # Step 0.5 — cap query length BEFORE any regex.
    if len(query) > MAX_QUERY_CHARS:
        LOG.warning(
            "effort_router.classify: query truncated from %d to %d chars "
            "(MAX_QUERY_CHARS); classification uses prefix only",
            len(query),
            MAX_QUERY_CHARS,
        )
        query = query[:MAX_QUERY_CHARS]
        matched_signals.append("query_truncated")

    # Step 1 — base tier from PRE-COMPILED alternation regex per tier.
    base = EffortLevel.MINIMAL
    distinct_max_hits: set[str] = _distinct_max_count(query)

    if distinct_max_hits:
        matched_signals.append(
            f"max_keywords_distinct={len(distinct_max_hits)}"
        )

    if len(distinct_max_hits) >= MAX_DISTINCT_FOR_ESCALATE:

        base = EffortLevel.MAX
        matched_signals.append("max_tier_escalated")
    elif len(distinct_max_hits) == 1:
        # Single MAX kw caps at HIGH (DoW mitigation).
        base = EffortLevel.HIGH
        matched_signals.append("max_keyword_single_hit_capped_at_high")
    else:
        # Walk HIGH → MINIMAL until first match.
        for tier in (
            EffortLevel.HIGH,
            EffortLevel.MEDIUM,
            EffortLevel.LOW,
            EffortLevel.MINIMAL,
        ):
            if _TIER_REGEX[tier].search(query):
                base = tier
                matched_signals.append(f"tier_match:{tier.value}")
                break

    # Step 2 — escalators (deterministic order).
    if any(p.search(query) for p in TRACE_PATTERNS):
        base = _max_tier(base, EffortLevel.MEDIUM)
        matched_signals.append("trace_detected")

    if (
        is_code_heavy(query)
        and _TIER_RANK[base] <= _TIER_RANK[EffortLevel.LOW]
    ):
        base = EffortLevel.MEDIUM
        matched_signals.append("code_heavy_bump")

    if any(p.search(query) for p in MULTI_FILE_PATTERNS):
        base = _bump(base, 1)
        matched_signals.append("multi_file")

    tokens = estimate_tokens(query)
    if tokens > 2000:
        base = _bump(base, 2)
        matched_signals.append(f"len>2000:{tokens}")
    elif tokens > 500:
        base = _bump(base, 1)
        matched_signals.append(f"len>500:{tokens}")

    if (
        any(p.search(query) for p in QUESTION_PATTERNS)
        and _DESIGN_NOUN_RE.search(query)
    ):
        base = _max_tier(base, EffortLevel.HIGH)
        matched_signals.append("design_question")

    # Step 3 — confidence (signal density). Bounded in [0.5, 1.0].
    n_signals = len(matched_signals)
    confidence = (
        0.5 if n_signals == 0 else min(1.0, 0.5 + 0.1 * n_signals)
    )

    # Step 3.5 —  + low-confidence HIGH+ back-off.
    if (
        not escalate_low_confidence
        and confidence < LOW_CONFIDENCE_THRESHOLD
        and _TIER_RANK[base] >= _TIER_RANK[EffortLevel.HIGH]
    ):
        base = EffortLevel.MEDIUM
        matched_signals.append("low_confidence_no_escalate")

    # Step 4 — domain inference.
    domain = infer_domain(query)

    return TaskClassification(
        complexity=base,
        domain=domain,
        estimated_tokens=tokens,
        confidence=confidence,
        matched_signals=tuple(matched_signals),
    )


# ---------------------------------------------------------------------------
# 7.5 Cross-turn context awareness (continuation-pattern inheritance)
# ---------------------------------------------------------------------------

#: Maximum length (after strip) for a query to even be considered a
#: conversational continuation. Anything longer is content in its own right
#: and goes through the regular pure-prompt path.
_CONTINUATION_MAX_CHARS: Final[int] = 60

#: Compiled regex that matches the shapes typical of "I'm answering / acking
#: a prior turn" — bare affirmatives, refusals, numbered answers, brief
#: questions, option picks. Case-insensitive. ``re.fullmatch`` is the entry
#: point so partial matches inside longer prompts do not qualify.
#:
#: The keyword set is deliberately conservative. False positives here cause
#: a turn to inherit a tier instead of starting at MINIMAL — the worst case
#: is a slightly higher effort budget than the prompt alone would request.
#: False negatives leave a turn at MINIMAL (existing pre-fix behavior).
_CONTINUATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"""
    ^\s*
    (?:
        # one or more continuation tokens, optionally space-separated
        (?:
            # bare affirmatives / refusals / continuation cues
            ok | okay | yes+ | y | yep | yeah | yup |
            no | nope | n |
            go | sure | approved | approve | fine | cool | good | right |
            continue | next | more | proceed |
            do\s+it | fire\s+it | ship\s+it | do\s+this | do\s+that |
            # short questions
            why | how | what | when | where |
            # numbered references / option picks
            \d+\s*\.? (?:\s*\w+)? |
            option\s+[a-z](?:\s*[\+\&]\s*[a-z])* |
            tagline\s*\#?\s*\d+ |
            # standalone punctuation (bare "?" / "!" / "...")
            [\?\!]+ |
            \.{1,3}
        )
        \s*[\.\!\?,]?\s*
    )+
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _inherit_one_step_down(prior: EffortLevel) -> EffortLevel:
    """Return the one-step-down tier from ``prior`` (clamped at MINIMAL).

    Inheritance rule: a continuation turn gets the prior turn's tier minus
    one — the prior turn already paid for the heavy thinking; the
    continuation is usually a confirmation or small follow-up. We do NOT
    inherit at the same tier (over-allocation) or two steps down
    (under-allocation).
    """
    rank = _TIER_RANK[prior]
    return _RANK_TIER[max(0, rank - 1)]


def classify_with_context(
    query: str,
    *,
    prior_classification: TaskClassification | None,
    config: AmplifierConfig | None = None,
) -> TaskClassification:
    """Classify ``query`` with cross-turn context inheritance.

    When the immediate prompt is short and matches a "conversational
    continuation" shape (a bare affirmative, a numbered answer, a brief
    question, an option pick) AND a non-``MINIMAL`` prior classification
    is supplied, the result inherits the one-step-down tier and domain
    from the prior turn instead of falling back to ``MINIMAL/general``.

    In every other case this function delegates to the existing
    :func:`classify` / :func:`classify_with_config` path — fully backward
    compatible.

    Args:
        query: free-text user prompt.
        prior_classification: the immediately preceding turn's
            :class:`TaskClassification`, or ``None`` if no prior turn is
            known. The kernel populates this from ``state.db`` (envelopes
            row at the prior ``turn_id``).
        config: optional :class:`AmplifierConfig` to honor
            ``escalate_low_confidence`` on the pure-prompt fallback path.

    Raises:
        TypeError: if ``config`` is provided but is not an
            :class:`AmplifierConfig`.
    """
    if config is not None and not isinstance(config, AmplifierConfig):
        raise TypeError(
            "classify_with_context: config must be AmplifierConfig or None; got "
            f"{type(config).__name__}"
        )

    # Continuation detection — short prompt + matches pattern + has prior to inherit
    if (
        query is not None
        and prior_classification is not None
        and prior_classification.complexity != EffortLevel.MINIMAL
    ):
        stripped = query.strip()
        if (
            0 < len(stripped) <= _CONTINUATION_MAX_CHARS
            and _CONTINUATION_PATTERN.fullmatch(stripped) is not None
        ):
            inherited_tier = _inherit_one_step_down(
                prior_classification.complexity
            )
            return TaskClassification(
                complexity=inherited_tier,
                domain=prior_classification.domain,
                estimated_tokens=estimate_tokens(query),
                confidence=0.85,
                matched_signals=(
                    "context_inherited_from_prior",
                    f"prior_complexity:{prior_classification.complexity.value}",
                    f"prior_domain:{prior_classification.domain}",
                ),
            )

    # Fallback: existing pure-prompt path. config wins if supplied.
    if config is not None:
        return classify_with_config(query, config)
    return classify(query)


# ---------------------------------------------------------------------------
# 8. Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "DESIGN_NOUNS",
    "DOMAIN_HINTS",
    "HIGH_KEYWORDS",
    "LOW_CONFIDENCE_THRESHOLD",
    "LOW_KEYWORDS",
    "MAX_DISTINCT_FOR_ESCALATE",
    "MAX_KEYWORDS",
    "MAX_QUERY_CHARS",
    "MEDIUM_KEYWORDS",
    "MINIMAL_KEYWORDS",
    "classify",
    "classify_with_config",
    "classify_with_context",
    "estimate_tokens",
    "infer_domain",
    "is_code_heavy",
    "should_iterate",
    "suggest_thinking_trigger",
]
