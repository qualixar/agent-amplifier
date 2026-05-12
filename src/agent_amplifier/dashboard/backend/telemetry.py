# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Read-only telemetry queries over Claude Code adapter state.db."""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

from agent_amplifier.dashboard.backend.models import (
    ConvergencePoint,
    ConvergenceResponse,
    CountSummary,
    TelemetrySummaryResponse,
    TurnInfo,
    TurnsResponse,
)

_TABLES: tuple[str, ...] = ("sessions", "envelopes", "events", "outcomes")


def telemetry_summary(db_path: Path) -> TelemetrySummaryResponse:
    if not db_path.exists():
        return _empty_summary(db_exists=False)
    counts = _read_counts(db_path)
    coverage = _rate(counts.outcomes, counts.envelopes)
    converged, total = _read_converged(db_path)
    return TelemetrySummaryResponse(
        db_exists=True,
        counts=counts,
        coverage_rate=coverage,
        convergence_rate=_rate(converged, total),
    )


def recent_turns(db_path: Path, *, limit: int) -> TurnsResponse:
    bounded = max(1, min(limit, 500))
    if not db_path.exists():
        return TurnsResponse(limit=bounded, turns=[])
    rows: list[TurnInfo] = []
    with contextlib.closing(_open(db_path)) as conn:
        cur = conn.execute(
            """
            SELECT
                e.session_id AS session_id,
                e.turn_id AS turn_id,
                e.classification_complexity AS complexity,
                e.classification_domain AS domain,
                e.thinking_trigger AS trigger,
                e.phase AS phase,
                e.created_at AS created_at,
                o.duration_ms AS duration_ms,
                o.converged AS converged,
                o.tokens_used AS tokens_used,
                json_extract(o.finalize_report_json, '$.stop_reason') AS stop_reason
            FROM envelopes e
            LEFT JOIN outcomes o
              ON o.session_id = e.session_id AND o.turn_id = e.turn_id
            ORDER BY e.created_at DESC
            LIMIT ?
            """,
            (bounded,),
        )
        for row in cur.fetchall():
            rows.append(
                TurnInfo(
                    session_id=str(row["session_id"]),
                    turn_id=int(row["turn_id"]),
                    complexity=str(row["complexity"]),
                    domain=str(row["domain"]),
                    trigger=_optional_str(row["trigger"]),
                    phase=str(row["phase"]),
                    created_at=float(row["created_at"]),
                    duration_ms=_optional_int(row["duration_ms"]),
                    converged=_optional_bool(row["converged"]),
                    stop_reason=_optional_str(row["stop_reason"]),
                    tokens_used=_optional_int(row["tokens_used"]),
                )
            )
    return TurnsResponse(limit=bounded, turns=rows)


def convergence_series(db_path: Path, *, days: int) -> ConvergenceResponse:
    bounded = max(1, min(days, 365))
    if not db_path.exists():
        return ConvergenceResponse(days=bounded, points=[])
    points: list[ConvergencePoint] = []
    with contextlib.closing(_open(db_path)) as conn:
        cur = conn.execute(
            """
            SELECT
                date(written_at, 'unixepoch') AS day,
                COUNT(*) AS total,
                SUM(CASE WHEN converged = 1 THEN 1 ELSE 0 END) AS converged
              FROM outcomes
             WHERE written_at >= strftime('%s', 'now', ?)
             GROUP BY day
             ORDER BY day ASC
            """,
            (f"-{bounded} days",),
        )
        for row in cur.fetchall():
            total = int(row["total"])
            converged = int(row["converged"] or 0)
            points.append(
                ConvergencePoint(
                    date=str(row["day"]),
                    total=total,
                    converged=converged,
                    rate=_rate(converged, total),
                )
            )
    return ConvergenceResponse(days=bounded, points=points)


def _open(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", timeout=5.0, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _read_counts(db_path: Path) -> CountSummary:
    values: dict[str, int] = {}
    with contextlib.closing(_open(db_path)) as conn:
        for table in _TABLES:
            (count,) = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()
            values[table] = int(count)
    return CountSummary(
        sessions=values["sessions"],
        envelopes=values["envelopes"],
        events=values["events"],
        outcomes=values["outcomes"],
    )


def _read_converged(db_path: Path) -> tuple[int, int]:
    with contextlib.closing(_open(db_path)) as conn:
        total, converged = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN converged = 1 THEN 1 ELSE 0 END) "
            "FROM outcomes"
        ).fetchone()
    return int(converged or 0), int(total)


def _empty_summary(*, db_exists: bool) -> TelemetrySummaryResponse:
    return TelemetrySummaryResponse(
        db_exists=db_exists,
        counts=CountSummary(sessions=0, envelopes=0, events=0, outcomes=0),
        coverage_rate=0.0,
        convergence_rate=0.0,
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str, bytes, bytearray)):
        return int(value)
    raise TypeError(f"expected integer-compatible value, got {type(value).__name__}")


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    parsed = _optional_int(value)
    return None if parsed is None else bool(parsed)


__all__ = ["convergence_series", "recent_turns", "telemetry_summary"]
