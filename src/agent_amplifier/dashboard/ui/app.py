# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Agent Amplifier Dashboard — Streamlit UI entry point.

Run with:
    streamlit run src/agent_amplifier/dashboard/ui/app.py

The backend must be running (``agent-amp dashboard``) on port 8765
or the port specified by ``AGENT_AMP_DASHBOARD_PORT``.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from agent_amplifier import __version__
from agent_amplifier.dashboard.ui.api import DashboardClient, DashboardError
from agent_amplifier.dashboard.ui.components.adapters_tab import render_adapters_tab
from agent_amplifier.dashboard.ui.components.health_tab import render_health_tab
from agent_amplifier.dashboard.ui.components.telemetry_tab import render_telemetry_tab
from agent_amplifier.dashboard.ui.components.tune_tab import render_tune_tab

_TABS = ["Tune", "Telemetry", "Adapters", "Health"]


# ---------------------------------------------------------------------------
# Qualixar brand CSS — softens Streamlit's harsh red error banners to amber,
# tightens section spacing, and stamps the brand purple on focus rings.
# Kept as a pure-string function so it is unit-testable without booting the
# Streamlit runtime.
# ---------------------------------------------------------------------------


def _inject_brand_css_html() -> str:
    """Return the Qualixar brand `<style>` block as a single HTML string.

    Qualixar purple ``#7C3AED`` for focus/accent, amber ``#F59E0B`` for
    "attention" tones in place of Streamlit's default red errors. The CSS
    overrides Streamlit's alert background tokens via the `data-baseweb` and
    `[data-testid='stAlert*']` selectors that Streamlit exposes for every
    alert variant.
    """
    return (
        "<style>\n"
        "/* Qualixar dashboard polish */\n"
        ":root { --qx-purple: #7C3AED; --qx-amber: #F59E0B; }\n"
        "[data-testid='stAlertContainer'][kind='error'],\n"
        "[data-testid='stAlert'][data-baseweb='notification'][kind='error']\n"
        "  { background-color: rgba(245, 158, 11, 0.10) !important;\n"
        "    border-left: 4px solid var(--qx-amber) !important; }\n"
        "[data-testid='stAlertContainer'][kind='warning']\n"
        "  { background-color: rgba(245, 158, 11, 0.08) !important;\n"
        "    border-left: 4px solid var(--qx-amber) !important; }\n"
        "[data-testid='stAlertContainer'][kind='success']\n"
        "  { border-left: 4px solid var(--qx-purple) !important; }\n"
        "h1, h2, h3 { letter-spacing: -0.01em; }\n"
        "[data-testid='stHeader'] { background: transparent; }\n"
        "div[role='radiogroup'] > label { padding: 0.25rem 0.75rem; }\n"
        "/* tighten the gap between st.divider and the next block */\n"
        "hr { margin: 0.75rem 0 !important; }\n"
        "</style>"
    )


def _render_header(client: DashboardClient) -> None:
    left, right = st.columns([4, 1])
    with left:
        st.title(f"Agent-Amplifier Dashboard v{__version__}")
    with right:
        try:
            health = client.health()
            status = health.get("status", "unknown")
            if status == "ok":
                st.success("Status: OK")
            else:
                st.warning(f"Status: {status}")
        except DashboardError:
            st.error("Backend unreachable")


def main() -> None:
    st.set_page_config(
        page_title="Agent Amplifier",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Qualixar brand CSS — softens default error red to amber + tightens spacing.
    st.markdown(_inject_brand_css_html(), unsafe_allow_html=True)

    # Initialize backend client (lives for the full script run)
    client = DashboardClient()

    _render_header(client)

    st.divider()

    # Tab navigation
    active_tab = st.radio(
        "Navigation",
        _TABS,
        horizontal=True,
        label_visibility="collapsed",
        key="amp_active_tab",
    )

    st.divider()

    _TAB_RENDERERS: dict[str, Callable[[DashboardClient], None]] = {
        "Tune": render_tune_tab,
        "Telemetry": render_telemetry_tab,
        "Adapters": render_adapters_tab,
        "Health": render_health_tab,
    }
    renderer = _TAB_RENDERERS.get(active_tab)
    if renderer is not None:  # pragma: no branch
        renderer(client)

    client.close()


if __name__ == "__main__":
    main()
