# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Telemetry tab — metrics, charts, recent sessions."""

from __future__ import annotations

import datetime
from collections import Counter
from typing import Any

import altair as alt
import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from agent_amplifier.dashboard.ui.api import DashboardClient, DashboardError


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _fmt_tokens(val: int) -> str:
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if val >= 1_000:
        return f"{val / 1_000:.1f}k"
    return str(val)


def render_telemetry_tab(client: DashboardClient) -> None:
    st.header("Telemetry")

    # Summary metrics
    try:
        summary = client.telemetry_summary()
    except DashboardError as exc:
        st.error(str(exc))
        return

    counts = summary.get("counts", {})
    coverage = summary.get("coverage_rate", 0.0)
    convergence = summary.get("convergence_rate", 0.0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sessions", counts.get("sessions", 0))
    with col2:
        st.metric("Envelopes", counts.get("envelopes", 0))
    with col3:
        st.metric("Coverage %", _fmt_pct(coverage))
    with col4:
        st.metric("Convergence %", _fmt_pct(convergence))

    st.divider()

    # Charts row
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Convergence Rate (7 Days)")
        try:
            conv_data = client.convergence(days=7)
        except DashboardError as exc:
            st.error(str(exc))
            conv_data = {"points": []}

        points = conv_data.get("points", [])
        if points:
            df = pd.DataFrame(points)
            line_chart = (
                alt.Chart(df)
                .mark_line(point=True, color="#00f5ff")
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("rate:Q", title="Rate", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["date", "rate", "total", "converged"],
                )
                .properties(height=250)
            )
            st.altair_chart(line_chart, use_container_width=True)
        else:
            st.info("No convergence data available")

    with chart_right:
        st.subheader("IP Firing Frequency")
        try:
            turns_data = client.turns(limit=50)
        except DashboardError as exc:
            st.error(str(exc))
            turns_data = {"turns": []}

        turns: list[dict[str, Any]] = turns_data.get("turns", [])
        if turns:
            triggers = [t.get("trigger") or "none" for t in turns]
            freq = Counter(triggers)
            df_freq = pd.DataFrame(
                [{"trigger": k, "count": v} for k, v in freq.most_common()]
            )
            bar_chart = (
                alt.Chart(df_freq)
                .mark_bar(color="#00f5ff")
                .encode(
                    x=alt.X("trigger:N", title="Trigger", sort="-y"),
                    y=alt.Y("count:Q", title="Count"),
                    tooltip=["trigger", "count"],
                )
                .properties(height=250)
            )
            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.info("No turn data available")

    st.divider()

    # Token spend gauge
    st.subheader("Token Spend Today")
    total_tokens = sum(
        (t.get("tokens_used") or 0)
        for t in turns
        if datetime.datetime.fromtimestamp(t.get("created_at", 0), tz=datetime.UTC).date()
        == datetime.datetime.now(tz=datetime.UTC).date()
    )
    soft_cap = 2_000_000_000
    pct = min(total_tokens / soft_cap, 1.0)
    st.progress(pct, text=f"{_fmt_tokens(total_tokens)} / {_fmt_tokens(soft_cap)}")

    st.divider()

    # Recent sessions table
    st.subheader("Recent Sessions")
    if turns:
        rows: list[dict[str, Any]] = []
        for t in turns:
            created = datetime.datetime.fromtimestamp(
                t.get("created_at", 0), tz=datetime.UTC
            )
            # IP-4: "Converged" is a per-turn boolean (yes/no), not a
            # percentage. The prior v1.0 column read ``quality_estimate``
            # which is unpopulated in v1.0 — wrong field + wrong shape.
            converged = t.get("converged")
            if converged is True:
                converged_display = "✓"
            elif converged is False:
                converged_display = "✗"
            else:
                converged_display = "—"
            rows.append(
                {
                    "Session ID": str(t.get("session_id", ""))[:8],
                    "Time": created.strftime("%H:%M"),
                    "Converged": converged_display,
                    "Tokens": _fmt_tokens(t.get("tokens_used") or 0),
                    "Stop reason": t.get("stop_reason") or "—",
                }
            )
        df_sessions = pd.DataFrame(rows)
        st.dataframe(df_sessions, width="stretch", hide_index=True)
    else:
        st.info("No recent sessions")
