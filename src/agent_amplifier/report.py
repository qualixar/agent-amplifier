# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Read-only dashboard over the Claude Code adapter's state.db.

Renders amplification statistics from the per-user state DB
(``~/.claude/agent-amp/state.db`` by default) as a tabulated report.
Used by ``agent-amp report`` and as a programmatic summary surface for
paper / blog data.

Sections rendered (in order):
  * Health         — row counts and envelope-vs-outcome coverage
  * Last N turns   — table of recent turns with classification + duration
  * Classification — complexity-by-domain distribution
  * Convergence    — share of turns marked converged
  * Sweep          — synthetic abandoned outcomes vs real Stop outcomes
"""
from __future__ import annotations

import contextlib
import sqlite3
import sys
from pathlib import Path
from typing import Any

from agent_amplifier.adapters.claude_code import state as _state
from agent_amplifier.model_router import ModelRouter


def render_report(
    *,
    db_path: Path | None = None,
    last: int = 10,
) -> int:
    """Render the dashboard to stdout. Returns 0 on success, 1 if no state.db.

    Parameters:
        db_path: SQLite file. ``None`` resolves to the default location
            without instantiating ``StateStore`` (which would create the
            schema on first call and mask the ``not found`` UX path).
        last: number of most-recent turns to show in the per-turn table.
    """
    if db_path is None:
        db_path = Path(_state._DEFAULT_STATE_DIR) / _state._STATE_DB_FILENAME
    if not db_path.exists():
        print(
            f"agent-amp state.db not found at {db_path}",
            file=sys.stderr,
        )
        print(
            "Hint: install hooks via 'agent-amp install claude-code', then "
            "use Claude Code for at least one turn.",
            file=sys.stderr,
        )
        return 1

    print("Agent Amplifier Report")
    print("=" * 60)
    print()
    _print_health(db_path)
    print()
    _print_last_turns(db_path, n=last)
    print()
    _print_classification(db_path)
    print()
    _print_convergence(db_path)
    print()
    _print_sweep(db_path)
    print()
    _print_model_routing(db_path)
    return 0


# ---------------------------------------------------------------------------
# Section helpers — each opens a short-lived sqlite connection (matches the
# StateStore idiom; never holds a connection across calls).
# ---------------------------------------------------------------------------


def _open(db_path: Path) -> sqlite3.Connection:
    """Read-only connection. Caller wraps in ``contextlib.closing``."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", timeout=5.0, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _print_health(db_path: Path) -> None:
    counts: dict[str, int] = {}
    with contextlib.closing(_open(db_path)) as conn:
        for table in ("sessions", "envelopes", "events", "outcomes"):

            # never user input — so the f-string interpolation cannot
            # introduce SQL injection.
            (n,) = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()
            counts[table] = int(n)
    print("## Health")
    print(f"  Sessions:  {counts['sessions']:>6}")
    print(f"  Envelopes: {counts['envelopes']:>6}")
    print(f"  Events:    {counts['events']:>6}")
    print(f"  Outcomes:  {counts['outcomes']:>6}")
    if counts["envelopes"]:
        pct = round(100.0 * counts["outcomes"] / counts["envelopes"], 1)
        print(
            f"  Coverage:  {pct}% "
            f"({counts['outcomes']}/{counts['envelopes']} turns)"
        )


def _print_last_turns(db_path: Path, *, n: int) -> None:
    print(f"## Last {n} turns")
    rows: list[dict[str, Any]] = []
    with contextlib.closing(_open(db_path)) as conn:
        cur = conn.execute(
            """
            SELECT
                e.session_id                AS session_id,
                e.turn_id                   AS turn_id,
                e.classification_complexity AS complexity,
                e.classification_domain     AS domain,
                COALESCE(e.thinking_trigger, '-') AS trigger,
                e.phase                     AS phase,
                COALESCE(o.duration_ms, '-') AS dur_ms,
                CASE
                    WHEN o.converged IS NULL THEN '-'
                    WHEN o.converged = 1     THEN 'yes'
                    ELSE 'no'
                END                         AS converged,
                COALESCE(
                    json_extract(o.finalize_report_json, '$.stop_reason'),
                    '-'
                )                           AS stop_reason
            FROM envelopes e
            LEFT JOIN outcomes o
              ON o.session_id = e.session_id AND o.turn_id = e.turn_id
            ORDER BY e.created_at DESC
            LIMIT ?
            """,
            (n,),
        )
        for row in cur.fetchall():
            rows.append(dict(row))
    if not rows:
        print("  (no turns recorded yet)")
        return
    header = (
        f"  {'sess':<8}  {'turn':>4}  {'cmplx':<8}  {'domain':<14}  "
        f"{'trigger':<11}  {'phase':<10}  {'dur_ms':>7}  {'conv':<4}  reason"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for r in rows:
        sess = str(r["session_id"])[:8]
        domain = str(r["domain"])[:14]
        trigger = str(r["trigger"])[:11]
        print(
            f"  {sess:<8}  {int(r['turn_id']):>4}  "
            f"{r['complexity']!s:<8}  "
            f"{domain:<14}  "
            f"{trigger:<11}  "
            f"{r['phase']!s:<10}  "
            f"{r['dur_ms']!s:>7}  "
            f"{r['converged']!s:<4}  "
            f"{r['stop_reason']}"
        )


def _print_classification(db_path: Path) -> None:
    print("## Classification distribution")
    with contextlib.closing(_open(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT classification_complexity AS complexity,
                   classification_domain     AS domain,
                   COUNT(*)                  AS n
              FROM envelopes
             GROUP BY classification_complexity, classification_domain
             ORDER BY n DESC
             LIMIT 20
            """,
        ).fetchall()
    if not rows:
        print("  (no envelopes)")
        return
    print(f"  {'complexity':<10}  {'domain':<20}  {'count':>5}")
    for r in rows:
        domain = str(r["domain"])[:20]
        print(
            f"  {r['complexity']!s:<10}  "
            f"{domain:<20}  "
            f"{int(r['n']):>5}"
        )


def _print_convergence(db_path: Path) -> None:
    print("## Convergence")
    with contextlib.closing(_open(db_path)) as conn:
        total, converged = conn.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN converged=1 THEN 1 ELSE 0 END) FROM outcomes"
        ).fetchone()
    total = int(total)
    if not total:
        print("  (no outcomes recorded yet)")
        return
    # SUM over a non-empty table cannot be NULL (CASE returns 0 or 1, never
    # NULL), so a direct int() is safe here.
    converged = int(converged)
    pct = round(100.0 * converged / total, 1)
    print(f"  Converged: {converged}/{total} ({pct}%)")


def _print_sweep(db_path: Path) -> None:
    print("## Sweep efficacy (abandoned-envelope healing)")
    with contextlib.closing(_open(db_path)) as conn:
        (abandoned,) = conn.execute(
            "SELECT COUNT(*) FROM outcomes "
            "WHERE json_extract(finalize_report_json, '$.stop_reason')"
            " = 'abandoned'"
        ).fetchone()
        (total,) = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()
    abandoned = int(abandoned)
    total = int(total)
    if not total:
        print("  (no outcomes)")
        return
    real = total - abandoned
    pct = round(100.0 * abandoned / total, 1)
    print(f"  Real Stop:  {real:>5}")
    print(f"  Abandoned:  {abandoned:>5}  ({pct}%)")


def _print_model_routing(db_path: Path) -> None:
    # Shows what the CURRENT config would suggest, not what was actually
    # used at envelope-creation time. Historical suggested_model is on
    # StepEnvelope but not persisted to state.db in v1.0; v1.1 will add
    # a suggested_model column to the envelopes table.
    print("## Model Routing (suggested model per complexity)")
    router = ModelRouter()
    with contextlib.closing(_open(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT classification_complexity AS complexity, COUNT(*) AS n
            FROM envelopes
            GROUP BY classification_complexity
            ORDER BY n DESC
            """,
        ).fetchall()
    if not rows:
        print("  (no envelopes)")
        return
    print(f"  {'complexity':<10}  {'count':>5}  {'suggested model':<12}  display")
    for r in rows:
        cmplx = str(r["complexity"])
        suggestion = router.suggest(cmplx)
        print(
            f"  {cmplx:<10}  {int(r['n']):>5}  "
            f"{suggestion.tier:<12}  {suggestion.display}"
        )


__all__ = [
    "render_report",
]
