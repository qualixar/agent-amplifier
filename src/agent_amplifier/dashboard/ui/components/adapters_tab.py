# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Adapters tab — host adapter status and install actions."""

from __future__ import annotations

from typing import Any

import streamlit as st

from agent_amplifier.dashboard.ui.api import DashboardClient, DashboardError


def _status_badge(adapter: dict[str, Any]) -> str:
    detected = adapter.get("detected", False)
    installed = adapter.get("installed", False)
    if detected and installed:
        return "ACTIVE"
    if detected:
        return "READY"
    return "SOON"


def _action_label(status: str) -> str:
    if status == "ACTIVE":
        return "Uninstall"
    if status == "READY":
        return "Install"
    return "Notify Me"


def render_adapters_tab(client: DashboardClient) -> None:
    st.header("Host Adapters")
    st.caption("Manage integration with AI agent frameworks.")

    try:
        data = client.get_adapters()
    except DashboardError as exc:
        st.error(str(exc))
        return

    adapters: list[dict[str, Any]] = data.get("adapters", [])

    if not adapters:
        st.info("No adapters registered")
        return

    # Header row
    hdr = st.columns([3, 2, 2, 3])
    with hdr[0]:
        st.markdown("**Name**")
    with hdr[1]:
        st.markdown("**Status**")
    with hdr[2]:
        st.markdown("**Version**")
    with hdr[3]:
        st.markdown("**Action**")

    for adapter in adapters:
        name = adapter.get("display_name", adapter.get("name", "Unknown"))
        status = _status_badge(adapter)
        version = "1.0.0"  # backend does not expose version yet; wireframe expectation
        action = _action_label(status)

        row = st.columns([3, 2, 2, 3])
        with row[0]:
            st.write(name)
        with row[1]:
            color_map: dict[str, str] = {"ACTIVE": "green", "READY": "orange", "SOON": "gray"}
            st.badge(status, color=color_map.get(status, "gray"))  # type: ignore[arg-type]
        with row[2]:
            st.write(version)
        with row[3]:
            btn_key = f"adapter_action_{adapter['name']}"
            if action == "Notify Me":
                st.button(
                    action,
                    key=btn_key,
                    disabled=True,
                    help="Coming soon",
                )
            elif st.button(action, key=btn_key):
                try:
                    result = client.install_adapter(adapter["name"])
                    st.toast(f"{name}: {result.get('status', 'done')}", icon="🔧")
                    st.rerun()
                except DashboardError as exc:
                    if exc.status_code == 404:
                        st.error(f"{name} adapter not yet available in this build.")
                    else:
                        st.error(str(exc))
