# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Health tab — diagnostics, coverage, errors."""

from __future__ import annotations

import subprocess

import streamlit as st

from agent_amplifier.dashboard.ui.api import DashboardClient, DashboardError


def _run_doctor() -> str:
    try:
        result = subprocess.run(
            ["agent-amp", "doctor"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout + (result.stderr if result.returncode != 0 else "")
    except FileNotFoundError:
        return "agent-amp CLI not found in PATH."
    except subprocess.TimeoutExpired:
        return "agent-amp doctor timed out after 15 seconds."
    except Exception as exc:
        return f"Failed to run diagnostics: {exc}"


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def render_health_tab(client: DashboardClient) -> None:
    st.header("System Health")

    # Backend health
    try:
        health = client.health()
    except DashboardError as exc:
        st.error(str(exc))
        health = {}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Backend Status")
        status = health.get("status", "unknown")
        if status == "ok":
            st.success(f"Status: {status.upper()}")
        else:
            st.warning(f"Status: {status}")
        st.write(f"Version: {health.get('amp_version', '—')}")
        st.write(f"Database: {health.get('db_path', '—')}")

    with col2:
        st.subheader("Cleanup Status")
        try:
            summary = client.telemetry_summary()
        except DashboardError:
            summary = {}
        coverage = summary.get("coverage_rate", 0.0)
        st.write(f"Coverage: {_fmt_pct(coverage)}")
        st.write("Abandoned envelopes: not implemented")
        st.write("Sweep-recovered: not implemented")

    st.divider()

    # Doctor output
    st.subheader("agent-amp Doctor")
    doctor_key = "amp_doctor_output"
    if doctor_key not in st.session_state:
        st.session_state[doctor_key] = _run_doctor()

    st.code(st.session_state[doctor_key], language="text")

    if st.button("Run Diagnostics", key="btn_run_diagnostics"):
        with st.spinner("Running diagnostics..."):
            st.session_state[doctor_key] = _run_doctor()
        st.rerun()

    st.divider()

    # Recent errors
    st.subheader("Recent Errors / Warnings")
    st.info("No errors in last 24h (placeholder)")
