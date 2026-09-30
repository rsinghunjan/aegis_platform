"""Canonical, lightweight FastAPI application for the Aegis control plane."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
import inspect
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from agentic.runtime import AgentRuntime, AgentRuntimeError, AgentStore

logger = logging.getLogger(__name__)


class CreateAgentRunRequest(BaseModel):
    tenant_id: str
    goal: str
    budget: float = 0.0
    idempotency_key: str | None = None
    role: str = "agent"
    environment: str = "local"
    scopes: list[str] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    tenant_id: str
    actor: str
    reason: str = ""


class ResumeAgentRunRequest(BaseModel):
    tenant_id: str
    role: str = "agent"
    environment: str = "local"
    scopes: list[str] = Field(default_factory=list)


def create_app(
    runtime: AgentRuntime | None = None,
    tenant_authorizer: Optional[
        Callable[[Request, str, str, Optional[str]], Any]
    ] = None,
) -> FastAPI:
    selected_runtime = runtime or AgentRuntime()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            selected_runtime.store.initialize()
        except Exception as exc:
            logger.exception("Aegis runtime database initialization failed")
            raise RuntimeError(
                "Aegis database is unavailable; configure AEGIS_AGENT_DATABASE_URL "
                "or DATABASE_URL and install the matching SQLAlchemy database driver"
            ) from exc
        yield

    app = FastAPI(
        title="Aegis Platform",
        version="1.0.0",
        description="Governed ML control plane and durable agent runtime",
        lifespan=lifespan,
    )
    app.state.agent_runtime = selected_runtime

    async def authorize(
        request: Request, tenant_id: str, action: str, actor: Optional[str] = None
    ) -> None:
        if tenant_authorizer is None:
            raise HTTPException(
                status_code=503,
                detail="agent API authorization is not configured",
            )
        allowed = tenant_authorizer(request, tenant_id, action, actor)
        if inspect.isawaitable(allowed):
            allowed = await allowed
        if not allowed:
            raise HTTPException(status_code=403, detail="tenant access denied")

    @app.get("/healthz", tags=["health"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["health"])
    def readyz() -> dict[str, str]:
        try:
            with app.state.agent_runtime.store.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
        except Exception as exc:
            logger.warning("Aegis readiness check failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return {"status": "ready"}

    @app.post("/agent/runs", tags=["agentic"])
    async def create_agent_run(
        request: Request, body: CreateAgentRunRequest
    ) -> dict[str, Any]:
        await authorize(request, body.tenant_id, "execute")
        try:
            agent_runtime = app.state.agent_runtime
            run = agent_runtime.create_run(
                tenant_id=body.tenant_id,
                goal=body.goal,
                budget=body.budget,
                idempotency_key=body.idempotency_key,
            )
            result = await agent_runtime.execute(
                run.run_id,
                body.tenant_id,
                role=body.role,
                environment=body.environment,
                scopes=set(body.scopes),
            )
            return {
                "run": agent_runtime.get_run(run.run_id, body.tenant_id).dict(),
                **result,
            }
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/agent/runs/{run_id}", tags=["agentic"])
    async def get_agent_run(
        request: Request, run_id: str, tenant_id: str
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "read")
        try:
            return app.state.agent_runtime.get_run(run_id, tenant_id).dict()
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/agent/runs/{run_id}/resume", tags=["agentic"])
    async def resume_agent_run(
        request: Request, run_id: str, body: ResumeAgentRunRequest
    ) -> dict[str, Any]:
        await authorize(request, body.tenant_id, "execute")
        try:
            return await app.state.agent_runtime.resume(
                run_id,
                body.tenant_id,
                role=body.role,
                environment=body.environment,
                scopes=set(body.scopes),
            )
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/agent/runs/{run_id}/approve", tags=["agentic"])
    async def approve_agent_run(
        request: Request, run_id: str, body: ApprovalRequest
    ) -> dict[str, Any]:
        await authorize(request, body.tenant_id, "approve", body.actor)
        try:
            agent_runtime = app.state.agent_runtime
            approval = agent_runtime.approve(
                run_id, body.tenant_id, body.actor, body.reason
            )
            result = await agent_runtime.resume(
                run_id, body.tenant_id, autonomous=False
            )
            return {"approval": approval.dict(), **result}
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/agent/runs/{run_id}/evidence", tags=["agentic"])
    async def get_agent_evidence(
        request: Request, run_id: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        await authorize(request, tenant_id, "read")
        try:
            return [
                item.dict()
                for item in app.state.agent_runtime.list_evidence(run_id, tenant_id)
            ]
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if os.getenv("AEGIS_MOUNT_LEGACY_API", "").lower() in {"1", "true", "yes"}:
        try:
            from api.api_server import app as legacy_app

            app.mount("/legacy", legacy_app)
        except ImportError as exc:
            logger.warning(
                "Legacy API routes were requested but optional modules are missing: %s",
                exc.name,
            )
        except Exception:
            logger.exception("Legacy API routes could not be mounted")

    return app


app = create_app()


def create_runtime(store: AgentStore | None = None) -> AgentRuntime:
    """Build a runtime for an embedding service or test."""
    return AgentRuntime(store=store)
