"""Canonical, lightweight FastAPI application for the Aegis control plane."""
from __future__ import annotations

import logging
import os
import hashlib
import json
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


class ApprovalDecisionRequest(BaseModel):
    tenant_id: str
    actor: str
    decision: str
    reason: str = ""


class ResumeAgentRunRequest(BaseModel):
    tenant_id: str
    role: str = "agent"
    environment: str = "local"
    scopes: list[str] = Field(default_factory=list)


def _run_summary(run: Any) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "status": run.status,
        "risk": run.risk.value,
        "goal_hash": hashlib.sha256(run.goal.encode()).hexdigest(),
        "budget": run.budget,
        "spent_cost": run.spent_cost,
        "error_hash": hashlib.sha256(run.error.encode()).hexdigest()
        if run.error
        else None,
        "result_hash": run.final_result_hash,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


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
                "run": _run_summary(
                    agent_runtime.get_run(run.run_id, body.tenant_id)
                ),
                **result,
            }
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/agent/runs/create", tags=["agentic"])
    async def create_agent_run_only(
        request: Request, body: CreateAgentRunRequest
    ) -> dict[str, Any]:
        await authorize(request, body.tenant_id, "execute")
        try:
            run = app.state.agent_runtime.create_run(
                tenant_id=body.tenant_id,
                goal=body.goal,
                budget=body.budget,
                idempotency_key=body.idempotency_key,
            )
            return _run_summary(run)
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/agent/runs/{run_id}/execute", tags=["agentic"])
    async def execute_agent_run(
        request: Request, run_id: str, body: ResumeAgentRunRequest
    ) -> dict[str, Any]:
        await authorize(request, body.tenant_id, "execute")
        try:
            result = await app.state.agent_runtime.execute(
                run_id,
                body.tenant_id,
                role=body.role,
                environment=body.environment,
                scopes=set(body.scopes),
            )
            return {
                "run": _run_summary(
                    app.state.agent_runtime.get_run(run_id, body.tenant_id)
                ),
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
            return _run_summary(app.state.agent_runtime.get_run(run_id, tenant_id))
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

    @app.get("/agent/approvals", tags=["agentic"])
    async def list_agent_approvals(
        request: Request,
        tenant_id: str,
        status: str = "pending",
        offset: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        await authorize(request, tenant_id, "read")
        return [
            item.dict()
            for item in app.state.agent_runtime.list_approvals(
                tenant_id, status, offset, limit
            )
        ]

    @app.post("/agent/approvals/{approval_id}/decision", tags=["agentic"])
    async def decide_agent_approval(
        request: Request, approval_id: str, body: ApprovalDecisionRequest
    ) -> dict[str, Any]:
        await authorize(request, body.tenant_id, "approve", body.actor)
        runtime = app.state.agent_runtime
        approval = next(
            (
                item
                for item in runtime.list_approvals(body.tenant_id)
                if item.approval_id == approval_id
            ),
            None,
        )
        if approval is None or approval.status != "pending":
            raise HTTPException(status_code=404, detail="pending approval not found")
        try:
            if body.decision == "approve":
                decision = runtime.approve(
                    approval.run_id, body.tenant_id, body.actor, body.reason
                )
                result = await runtime.resume(
                    approval.run_id, body.tenant_id, autonomous=False
                )
            elif body.decision == "deny":
                decision = runtime.deny(
                    approval.run_id, body.tenant_id, body.actor, body.reason
                )
                result = {"status": "BLOCKED", "run_id": approval.run_id}
            else:
                raise HTTPException(
                    status_code=422, detail="decision must be approve or deny"
                )
            return {"approval": decision.dict(), **result}
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/operator/agent/runs", tags=["operator"])
    async def operator_runs(
        request: Request, tenant_id: str, offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        runs = app.state.agent_runtime.list_runs(tenant_id, offset, limit)
        return {
            "items": [
                {
                    "run_id": run.run_id,
                    "status": run.status,
                    "risk": run.risk.value,
                    "goal_hash": hashlib.sha256(run.goal.encode()).hexdigest(),
                    "created_at": run.created_at,
                    "updated_at": run.updated_at,
                    "result_hash": run.final_result_hash,
                }
                for run in runs
            ],
            "offset": max(0, offset),
            "limit": max(1, min(limit, 100)),
        }

    @app.get("/operator/agent/runs/{run_id}/timeline", tags=["operator"])
    async def operator_timeline(
        request: Request,
        run_id: str,
        tenant_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        try:
            return app.state.agent_runtime.list_timeline(
                run_id, tenant_id, offset, limit
            )
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/operator/agent/blocked-decisions", tags=["operator"])
    async def operator_blocked_decisions(
        request: Request, tenant_id: str
    ) -> list[dict[str, Any]]:
        await authorize(request, tenant_id, "operator_read")
        return app.state.agent_runtime.list_blocked_decisions(tenant_id)

    @app.get("/operator/agent/evidence-summary", tags=["operator"])
    async def operator_evidence_summary(
        request: Request, tenant_id: str, run_id: str
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        try:
            entries = app.state.agent_runtime.list_evidence(run_id, tenant_id)
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "run_id": run_id,
            "count": len(entries),
            "sha256": hashlib.sha256(
                json.dumps([entry.sha256 for entry in entries]).encode()
            ).hexdigest(),
            "kinds": sorted({entry.kind for entry in entries}),
        }

    @app.get("/operator/agent/capabilities", tags=["operator"])
    async def operator_capabilities(
        request: Request, tenant_id: str, offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        runtime = app.state.agent_runtime
        items = runtime.catalog.list()
        page_offset = max(0, offset)
        page_limit = max(1, min(limit, 100))
        return {
            **runtime.catalog.summary(),
            "items": items[page_offset:page_offset + page_limit],
            "offset": page_offset,
            "limit": page_limit,
            "next_offset": (
                page_offset + page_limit if page_offset + page_limit < len(items) else None
            ),
        }

    @app.get("/operator/agent/capabilities/version", tags=["operator"])
    async def operator_capability_version(
        request: Request, tenant_id: str
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        return app.state.agent_runtime.catalog.summary()

    @app.get("/operator/agent/capabilities/{name}", tags=["operator"])
    async def operator_capability(
        request: Request, name: str, tenant_id: str
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        try:
            return app.state.agent_runtime.catalog.inspect(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capability not found") from exc

    @app.get("/operator/agent/remediation-events", tags=["operator"])
    async def operator_remediation_events(
        request: Request, tenant_id: str, offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        return app.state.agent_runtime.list_remediation_events(
            tenant_id, offset, limit
        )

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
