# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tune tab — IP drag-drop reordering + runtime config + persona management."""

from __future__ import annotations

from typing import Any

import streamlit as st
from streamlit_sortables import sort_items  # type: ignore[import-untyped]

from agent_amplifier.dashboard.ui.api import DashboardClient, DashboardError

# IDs that are foundational and cannot be toggled off.
_IMMUTABLE_IPS: frozenset[str] = frozenset({"kernel", "adapters"})


def _load_personas(client: DashboardClient) -> list[dict[str, Any]]:
    """Fetch personas from the backend; cache the result in session state.

    Defensive: if the backend response is missing the ``personas`` key, is
    not a dict, or contains non-dict entries, return an empty list rather
    than crashing the tab.
    """
    try:
        # Annotated as ``Any`` so the runtime isinstance() guard below is
        # not flagged as unreachable by mypy. The backend contract is dict,
        # but a misbehaving / older backend could return anything.
        data: Any = client.get_personas()
    except DashboardError as exc:
        st.error(str(exc))
        cached: list[dict[str, Any]] = st.session_state.get(
            "amp_personas_cache", []
        )
        return cached
    if not isinstance(data, dict):
        return []
    raw = data.get("personas", [])
    if not isinstance(raw, list):
        return []
    personas: list[dict[str, Any]] = [p for p in raw if isinstance(p, dict)]
    st.session_state.amp_personas_cache = personas
    return personas


def _render_ip_card(ip: dict[str, Any]) -> str:
    return f"{ip['name']}"


def _sync_from_backend(client: DashboardClient) -> list[dict[str, Any]]:
    """Fetch IPs from backend and update session state."""
    try:
        data = client.get_ips()
    except DashboardError as exc:
        st.error(str(exc))
        return []
    ips: list[dict[str, Any]] = data.get("ips", [])
    st.session_state.amp_ips_cache = ips
    return ips


def _apply_changes(client: DashboardClient) -> None:
    config: dict[str, object] = dict(st.session_state.get("amp_config_cache", {}))
    config["max_iterations"] = st.session_state.get("amp_max_iterations", 4)
    config["token_budget"] = st.session_state.get("amp_token_budget", 250000)
    config["persona"] = str(
        st.session_state.get("amp_persona_select", "senior-engineer")
    )
    try:
        result = client.save_config(config)
        st.session_state.amp_config_cache = result.get("config", {})
        st.toast("Changes applied", icon="✅")
    except DashboardError as exc:
        st.error(str(exc))


def _revert_changes(client: DashboardClient) -> None:
    try:
        data = client.get_config()
        st.session_state.amp_config_cache = data.get("config", {})
        st.toast("Reverted to saved config", icon="↩️")
    except DashboardError as exc:
        st.error(str(exc))


def render_tune_tab(client: DashboardClient) -> None:
    st.header("Amplification Features")
    st.caption(
        "Drag to reorder. Drag between columns to enable or disable. "
        "Runtime Kernel and Cross-Framework Adapter Layer are foundational and cannot be disabled."
    )

    # Fetch config if not cached
    if "amp_config_cache" not in st.session_state:
        try:
            st.session_state.amp_config_cache = client.get_config().get("config", {})
        except DashboardError as exc:
            st.error(str(exc))
            st.session_state.amp_config_cache = {}

    config: dict[str, object] = st.session_state.amp_config_cache

    # Fetch IPs if not cached
    ips: list[dict[str, Any]] = st.session_state.get("amp_ips_cache", [])
    if not ips:
        ips = _sync_from_backend(client)

    active = [ip for ip in ips if ip.get("enabled", True)]
    inactive = [ip for ip in ips if not ip.get("enabled", True)]

    left, right = st.columns(2)

    with left:
        st.subheader(f"Active ({len(active)} enabled)")
        if active:
            active_labels = [_render_ip_card(ip) for ip in active]
            new_active = sort_items(
                active_labels,
                key="sort_active",
                direction="vertical",
            )
            # Detect reorder within active list
            if new_active != active_labels:
                # Rebuild order based on labels (names are unique in catalog)
                name_to_id = {ip["name"]: ip["id"] for ip in ips}
                new_active_ids = [name_to_id[n] for n in new_active if n in name_to_id]
                new_inactive_ids = [ip["id"] for ip in inactive]
                new_order = new_active_ids + new_inactive_ids
                try:
                    client.reorder_ips(new_order)
                    st.session_state.amp_ips_cache = client.get_ips().get("ips", [])
                    st.rerun()
                except DashboardError as exc:
                    st.error(str(exc))
        else:
            st.info("No active features")

    with right:
        st.subheader(f"Inactive ({len(inactive)} hidden)")
        if inactive:
            inactive_labels = [_render_ip_card(ip) for ip in inactive]
            new_inactive = sort_items(
                inactive_labels,
                key="sort_inactive",
                direction="vertical",
            )
            if new_inactive != inactive_labels:
                name_to_id = {ip["name"]: ip["id"] for ip in ips}
                new_inactive_ids = [name_to_id[n] for n in new_inactive if n in name_to_id]
                new_active_ids = [ip["id"] for ip in active]
                new_order = new_active_ids + new_inactive_ids
                try:
                    client.reorder_ips(new_order)
                    st.session_state.amp_ips_cache = client.get_ips().get("ips", [])
                    st.rerun()
                except DashboardError as exc:
                    st.error(str(exc))
        else:
            st.info("No inactive features")

    # Per-feature toggle buttons (belt-and-suspenders alongside drag-drop)
    st.divider()
    st.subheader("Feature Toggles")
    toggle_cols = st.columns(3)
    for idx, ip in enumerate(ips):
        col = toggle_cols[idx % 3]
        with col:
            disabled = ip["id"] in _IMMUTABLE_IPS
            label = f"{'🔒 ' if disabled else ''}{ip['name']}"
            toggled = st.toggle(
                label,
                value=ip.get("enabled", True),
                key=f"toggle_{ip['id']}",
                disabled=disabled,
            )
            if not disabled and toggled != ip.get("enabled", True):
                try:
                    client.toggle_ip(ip["id"])
                    st.session_state.amp_ips_cache = client.get_ips().get("ips", [])
                    st.rerun()
                except DashboardError as exc:
                    st.error(str(exc))

    # Runtime Config
    st.divider()
    st.header("Runtime Config")

    max_iter = int(str(config.get("max_iterations", 4)))
    token_budget = int(str(config.get("token_budget", 250000)))

    st.slider("Max Iterations", 1, 10, value=max_iter, key="amp_max_iterations")
    st.slider(
        "Token Budget",
        min_value=1000,
        max_value=1_000_000,
        value=token_budget,
        step=1000,
        key="amp_token_budget",
    )

    st.divider()
    _render_persona_section(client, config)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Apply Changes", type="primary", key="btn_apply"):
            _apply_changes(client)
    with c2:
        if st.button("Revert to Defaults", key="btn_revert"):
            _revert_changes(client)


# ---------------------------------------------------------------------------
# Persona section — value tagline, when-to-use, custom persona form
# ---------------------------------------------------------------------------


def _render_persona_section(
    client: DashboardClient, config: dict[str, object]
) -> None:
    """Render the persona picker + education panel + custom-persona form."""
    st.subheader("Persona")
    st.caption(
        "Pick the reviewer profile the amplifier should adopt. Built-in "
        "personas come ordered by strictness; custom personas let you wire "
        "in domain-specific reviewers (e.g. ML engineer, accessibility "
        "auditor)."
    )

    personas = _load_personas(client)
    slugs = [p["slug"] for p in personas if "slug" in p]
    labels = [p["label"] for p in personas if "slug" in p]
    if not slugs:
        st.info("No personas available yet — backend returned an empty list.")
        return
    current_slug = str(config.get("persona", slugs[0]))
    if current_slug not in slugs:
        current_slug = slugs[0]
    current_index = slugs.index(current_slug)

    selected_label = st.selectbox(
        "Active persona",
        options=labels,
        index=current_index,
        key="amp_persona_label_select",
    )
    # Persist the selected slug into session state so _apply_changes can read it.
    selected_idx = labels.index(selected_label)
    selected = personas[selected_idx]
    st.session_state.amp_persona_select = selected["slug"]

    # Education panel — what this persona catches + when to use it.
    with st.container(border=True):
        if selected["custom"]:
            st.markdown(f"**{selected['label']}** — custom persona")
        else:
            st.markdown(
                f"**{selected['label']}** — built-in (level {selected['level']})"
            )
        st.markdown(f"**Value:** {selected['value_tagline']}")
        st.markdown(f"**When to use:** {selected['when_to_use']}")
        focus_str = ", ".join(selected["focus"]) if selected["focus"] else "—"
        st.markdown(f"**Focus axes:** {focus_str}")
        if not selected["custom"]:
            st.caption(
                f"Strictness {selected['strictness']}, "
                f"severity threshold: {selected['severity_threshold']}."
            )

    # Add custom persona form (collapsed by default).
    with (
        st.expander("Add a custom persona"),
        st.form("amp_add_persona_form", clear_on_submit=True),
    ):
            name = st.text_input(
                "Slug (lowercase, hyphen-safe — e.g. `ml-engineer`)",
                key="amp_new_persona_name",
            )
            label = st.text_input(
                "Display label (e.g. `ML Engineer`)",
                key="amp_new_persona_label",
            )
            description = st.text_area(
                "Description — describe the reviewer's expertise + what "
                "they should catch. Free text is sanitized against "
                "prompt-injection before it reaches the LLM.",
                key="amp_new_persona_description",
                height=120,
            )
            focus_raw = st.text_input(
                "Review focus (comma-separated tags, e.g. `pytorch,ml`)",
                key="amp_new_persona_focus",
            )
            submitted = st.form_submit_button("Add custom persona")
            if submitted:
                review_focus = [
                    s.strip() for s in (focus_raw or "").split(",") if s.strip()
                ]
                try:
                    client.create_persona(
                        name=name.strip(),
                        label=label.strip(),
                        description=description,
                        review_focus=review_focus,
                    )
                    st.session_state.pop("amp_personas_cache", None)
                    st.toast(f"Added custom persona: {name}", icon="✅")
                    st.rerun()
                except DashboardError as exc:
                    st.error(str(exc))

    # List + delete existing custom personas.
    customs = [p for p in personas if p["custom"]]
    if customs:
        st.markdown("**Custom personas**")
        for entry in customs:
            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(
                    f"`{entry['slug']}` — {entry['label']}<br>"
                    f"<span style='color:#6B7280'>{entry['value_tagline']}</span>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button(
                    "Remove",
                    key=f"amp_remove_persona_{entry['slug']}",
                ):
                    try:
                        client.delete_persona(entry["slug"])
                        st.session_state.pop("amp_personas_cache", None)
                        st.toast(
                            f"Removed: {entry['slug']}", icon="🗑️"
                        )
                        st.rerun()
                    except DashboardError as exc:
                        st.error(str(exc))
