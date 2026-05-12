# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""FastAPI route surface for the dashboard backend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from agent_amplifier.config import ConfigError
from agent_amplifier.custom_personas import (
    CustomPersona,
    InvalidPersonaError,
    delete_custom_persona,
    save_custom_persona,
)
from agent_amplifier.dashboard.backend.models import (
    AdapterInstallResponse,
    AdaptersResponse,
    ConfigResponse,
    ConfigUpdateRequest,
    ConvergenceResponse,
    HealthResponse,
    IpReorderRequest,
    IpsResponse,
    IpToggleResponse,
    PersonaCreateRequest,
    PersonaInfo,
    PersonasResponse,
    TelemetrySummaryResponse,
    TurnsResponse,
)
from agent_amplifier.dashboard.backend.services import (
    DashboardService,
    DashboardSettings,
)
from agent_amplifier.persona_docs import BUILTIN_PERSONA_DOCS, list_all_personas


def create_app(settings: DashboardSettings | None = None) -> FastAPI:
    service = DashboardService(settings or DashboardSettings())
    app = FastAPI(title="Agent Amplifier Dashboard Backend")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    _BUILTIN_SLUGS = {d.slug for d in BUILTIN_PERSONA_DOCS}

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return service.health()

    @app.get("/api/config", response_model=ConfigResponse)
    def get_config() -> ConfigResponse:
        try:
            return service.get_config()
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/config", response_model=ConfigResponse)
    def post_config(payload: ConfigUpdateRequest) -> ConfigResponse:
        try:
            return service.save_config(payload.config)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/ips", response_model=IpsResponse)
    def get_ips() -> IpsResponse:
        try:
            return service.ips()
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/ips/{ip_id}/toggle", response_model=IpToggleResponse)
    def post_ip_toggle(ip_id: str) -> IpToggleResponse:
        try:
            ip = service.toggle_ip(ip_id)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if ip is None:
            raise HTTPException(status_code=404, detail=f"unknown IP id: {ip_id}")
        return IpToggleResponse(ip=ip)

    @app.post("/api/ips/reorder", response_model=IpsResponse)
    def post_ip_reorder(payload: IpReorderRequest) -> IpsResponse:
        try:
            return service.reorder_ips(payload.ip_ids)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/telemetry/summary", response_model=TelemetrySummaryResponse)
    def get_telemetry_summary() -> TelemetrySummaryResponse:
        return service.telemetry_summary()

    @app.get("/api/telemetry/turns", response_model=TurnsResponse)
    def get_telemetry_turns(limit: int = Query(default=50, ge=1, le=500)) -> TurnsResponse:
        return service.turns(limit=limit)

    @app.get("/api/telemetry/convergence", response_model=ConvergenceResponse)
    def get_telemetry_convergence(days: int = Query(default=7, ge=1, le=365)) -> ConvergenceResponse:
        return service.convergence(days=days)

    @app.get("/api/adapters", response_model=AdaptersResponse)
    def get_adapters() -> AdaptersResponse:
        return service.adapters()

    @app.post("/api/adapters/{name}/install", response_model=AdapterInstallResponse)
    def post_adapter_install(name: str) -> AdapterInstallResponse:
        try:
            installed = service.install_adapter(name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if installed is None:
            raise HTTPException(status_code=404, detail=f"unknown adapter: {name}")
        return installed

    @app.get("/api/personas", response_model=PersonasResponse)
    def get_personas() -> PersonasResponse:
        return PersonasResponse(
            personas=[PersonaInfo(**entry) for entry in list_all_personas()]
        )

    @app.post(
        "/api/personas",
        response_model=PersonaInfo,
        status_code=201,
    )
    def post_personas(payload: PersonaCreateRequest) -> PersonaInfo:
        if payload.name in _BUILTIN_SLUGS:
            raise HTTPException(
                status_code=409,
                detail=f"slug '{payload.name}' is reserved by a built-in persona",
            )
        try:
            save_custom_persona(
                CustomPersona(
                    name=payload.name,
                    label=payload.label,
                    description=payload.description,
                    review_focus=tuple(payload.review_focus),
                )
            )
        except InvalidPersonaError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Re-read to surface the post-sanitization values to the client.
        for entry in list_all_personas():
            if entry["slug"] == payload.name:
                return PersonaInfo(**entry)
        # Defensive — save succeeded but listing did not return the entry.
        raise HTTPException(  # pragma: no cover
            status_code=500, detail="persona saved but not visible in listing"
        )

    @app.delete("/api/personas/{name}", status_code=204)
    def delete_personas(name: str) -> Response:
        if name in _BUILTIN_SLUGS:
            raise HTTPException(
                status_code=403,
                detail=f"cannot remove built-in persona: {name}",
            )
        removed = delete_custom_persona(name)
        if not removed:
            raise HTTPException(
                status_code=404, detail=f"persona not found: {name}"
            )
        return Response(status_code=204)

    return app


__all__ = ["create_app"]
