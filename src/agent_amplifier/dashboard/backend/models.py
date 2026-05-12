# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Pydantic models for the dashboard backend API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    amp_version: str
    db_path: str


class ConfigResponse(BaseModel):
    config: dict[str, object]


class ConfigUpdateRequest(BaseModel):
    config: dict[str, object]


class IpInfo(BaseModel):
    id: str
    name: str
    file: str
    enabled: bool
    order: int


class IpsResponse(BaseModel):
    ips: list[IpInfo]


class IpToggleResponse(BaseModel):
    ip: IpInfo


class IpReorderRequest(BaseModel):
    ip_ids: list[str] = Field(min_length=1)


class CountSummary(BaseModel):
    sessions: int
    envelopes: int
    events: int
    outcomes: int


class TelemetrySummaryResponse(BaseModel):
    db_exists: bool
    counts: CountSummary
    coverage_rate: float
    convergence_rate: float


class TurnInfo(BaseModel):
    session_id: str
    turn_id: int
    complexity: str
    domain: str
    trigger: str | None
    phase: str
    created_at: float
    duration_ms: int | None
    converged: bool | None
    stop_reason: str | None
    # IP-10 v2 / Option C: per-turn tokens read from Claude Code transcript
    # JSONL at Stop time. Nullable because v1.0 rows pre-backfill have 0; the
    # dashboard widget treats None as 0.
    tokens_used: int | None = None


class TurnsResponse(BaseModel):
    limit: int
    turns: list[TurnInfo]


class ConvergencePoint(BaseModel):
    date: str
    total: int
    converged: int
    rate: float


class ConvergenceResponse(BaseModel):
    days: int
    points: list[ConvergencePoint]


class AdapterInfo(BaseModel):
    name: str
    display_name: str
    detected: bool
    installed: bool


class AdaptersResponse(BaseModel):
    adapters: list[AdapterInfo]


class AdapterInstallResponse(BaseModel):
    name: str
    status: str


class PersonaInfo(BaseModel):
    slug: str
    label: str
    value_tagline: str
    when_to_use: str
    focus: list[str]
    custom: bool
    level: int | None = None
    role: str | None = None
    strictness: float | None = None
    severity_threshold: str | None = None


class PersonasResponse(BaseModel):
    personas: list[PersonaInfo]


class PersonaCreateRequest(BaseModel):
    name: str
    label: str
    description: str
    review_focus: list[str] = Field(default_factory=list)
