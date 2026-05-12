# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""FastAPI backend for the Agent Amplifier dashboard."""

from __future__ import annotations

from agent_amplifier.dashboard.backend.app import create_app

__all__ = ["create_app"]
