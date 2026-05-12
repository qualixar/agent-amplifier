# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for Qualixar dark+light Streamlit theme."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

# ---------------------------------------------------------------------------
# Brand tokens (asserted in tests + sourced by app.py)
# ---------------------------------------------------------------------------

QUALIXAR_PURPLE = "#7C3AED"
QUALIXAR_PURPLE_DARK_MODE = "#A78BFA"
QUALIXAR_AMBER = "#F59E0B"


# ---------------------------------------------------------------------------
# 1. .streamlit/config.toml structural tests
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    # tests/dashboard/ui/test_theme.py -> repo root
    return Path(__file__).resolve().parents[3]


def test_streamlit_config_file_exists() -> None:
    """The .streamlit/config.toml file is committed at repo root."""
    cfg_path = _project_root() / ".streamlit" / "config.toml"
    assert cfg_path.is_file(), f"Missing {cfg_path}"


def test_streamlit_config_parses_as_toml() -> None:
    cfg_path = _project_root() / ".streamlit" / "config.toml"
    data = tomllib.loads(cfg_path.read_text())
    assert "theme" in data


def test_streamlit_config_light_theme_uses_qualixar_purple() -> None:
    cfg_path = _project_root() / ".streamlit" / "config.toml"
    data = tomllib.loads(cfg_path.read_text())
    theme = data["theme"]
    assert theme.get("primaryColor", "").lower() == QUALIXAR_PURPLE.lower()


def test_streamlit_config_has_dark_override_block() -> None:
    """[theme.dark] overrides bg + primaryColor for dark mode."""
    cfg_path = _project_root() / ".streamlit" / "config.toml"
    data = tomllib.loads(cfg_path.read_text())
    assert "theme" in data
    dark = data["theme"].get("dark") or data.get("theme.dark")
    # Streamlit 1.40+ accepts [theme.dark] inline table or nested key
    if dark is None:
        # nested-block form: parsed as table under "theme" key "dark"
        dark = data["theme"].get("dark")
    assert dark is not None, "Missing [theme.dark] override block"
    assert "primaryColor" in dark
    assert "backgroundColor" in dark


# ---------------------------------------------------------------------------
# 2. CSS injection in app.py
# ---------------------------------------------------------------------------


def test_inject_brand_css_returns_css_with_qualixar_purple() -> None:
    """The helper returns a CSS string mentioning the brand purple."""
    from agent_amplifier.dashboard.ui.app import _inject_brand_css_html

    html = _inject_brand_css_html()
    assert "<style>" in html
    assert "</style>" in html
    assert QUALIXAR_PURPLE.lower() in html.lower()


def test_inject_brand_css_softens_error_red() -> None:
    """Error-banner CSS overrides the default red with an amber tone."""
    from agent_amplifier.dashboard.ui.app import _inject_brand_css_html

    html = _inject_brand_css_html()
    # The amber accent must appear in the css (used to soften error/warning).
    assert QUALIXAR_AMBER.lower() in html.lower()


def test_main_injects_brand_css_markdown(
    canned_responses: dict[str, Any],
) -> None:
    """When the app boots, brand CSS lands in the markdown stream."""
    mock_client = MagicMock()

    def _route(method: str, url: str, **kwargs: Any) -> MagicMock:
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.text = "{}"
        if "health" in url:
            response.json.return_value = canned_responses["health"]
        elif "/api/config" in url:
            response.json.return_value = canned_responses["config"]
        elif "ips" in url:
            response.json.return_value = canned_responses["ips"]
        elif "adapters" in url:
            response.json.return_value = canned_responses["adapters"]
        else:
            response.json.return_value = {}
        return response

    mock_client.request.side_effect = _route

    with patch("agent_amplifier.dashboard.ui.api.httpx.Client") as MockHttpx:
        MockHttpx.return_value = mock_client
        at = AppTest.from_file("src/agent_amplifier/dashboard/ui/app.py")
        at.run()

    markdowns = [m.value for m in at.markdown]
    assert any(
        QUALIXAR_PURPLE.lower() in (m or "").lower() for m in markdowns
    ), "Brand purple not present in rendered markdown stream"
