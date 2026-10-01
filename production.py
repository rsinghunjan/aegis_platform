"""Canonical, lightweight FastAPI application for the Aegis control plane."""
from __future__ import annotations

import logging
import os
import hashlib
import json
import asyncio
import math
from contextlib import asynccontextmanager
import inspect
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from agentic.runtime import AgentRuntime, AgentRuntimeError, AgentStore
from agentic.worker import AgentExecutionMessage, AgentPrincipal
from services.ai_workflow import AIWorkflow, AIWorkflowError

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
    actor: str | None = None
    approval_id: str | None = None
    reason: str = ""


class ApprovalDecisionRequest(BaseModel):
    tenant_id: str
    actor: str | None = None
    decision: str
    reason: str = ""


class ResumeAgentRunRequest(BaseModel):
    tenant_id: str
    role: str = "agent"
    environment: str = "local"
    scopes: list[str] = Field(default_factory=list)


class IngestKnowledgeRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    document: str = Field(min_length=1, max_length=64_000)


class AskAIRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=8_000)
    max_tokens: int = Field(default=512, ge=1, le=4096)


class AgentFeedbackRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    rating: int = Field(ge=1, le=5)
    note: str = Field(default="", max_length=2000)


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
    approval_notifier: Optional[Callable[[Any], Any]] = None,
    principal_resolver: Optional[Callable[[Request], Any]] = None,
    execution_dispatcher: Any = None,
    ai_workflow: AIWorkflow | None = None,
) -> FastAPI:
    selected_runtime = runtime or AgentRuntime()
    selected_ai_workflow = ai_workflow or AIWorkflow()
    if approval_notifier is not None:
        selected_runtime.approval_notifier = approval_notifier

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
        anchor_task = None
        if selected_runtime.evidence_anchor_backend is not None:
            try:
                anchor_interval = float(
                    os.getenv("AEGIS_EVIDENCE_ANCHOR_INTERVAL_SECONDS", "60")
                )
                if not math.isfinite(anchor_interval) or anchor_interval <= 0:
                    raise ValueError
            except ValueError as exc:
                raise RuntimeError(
                    "AEGIS_EVIDENCE_ANCHOR_INTERVAL_SECONDS must be positive"
                ) from exc

            async def anchor_periodically() -> None:
                while True:
                    await asyncio.sleep(anchor_interval)
                    try:
                        await asyncio.to_thread(
                            selected_runtime.anchor_pending_evidence
                        )
                    except Exception as exc:
                        logger.warning(
                            "Periodic evidence anchoring failed: %s",
                            type(exc).__name__,
                        )

            anchor_task = asyncio.create_task(anchor_periodically())
        try:
            yield
        finally:
            if anchor_task is not None:
                anchor_task.cancel()
                try:
                    await anchor_task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(
        title="Aegis Platform",
        version="1.0.0",
        description="Governed ML control plane and durable agent runtime",
        lifespan=lifespan,
    )
    app.state.agent_runtime = selected_runtime
    app.state.ai_workflow = selected_ai_workflow

    async def resolve_principal(request: Request) -> AgentPrincipal:
        if principal_resolver is None:
            return AgentPrincipal()
        principal = principal_resolver(request)
        if inspect.isawaitable(principal):
            principal = await principal
        if isinstance(principal, AgentPrincipal):
            return principal
        if isinstance(principal, dict):
            return AgentPrincipal.parse_obj(principal)
        raise HTTPException(status_code=503, detail="trusted principal is unavailable")

    async def authorize(
        request: Request, tenant_id: str, action: str, actor: Optional[str] = None
    ) -> None:
        if tenant_authorizer is None:
            raise HTTPException(
                status_code=503,
                detail="agent API authorization is not configured",
            )
        principal = await resolve_principal(request)
        allowed = tenant_authorizer(
            request, tenant_id, action, actor or principal.principal_id
        )
        if inspect.isawaitable(allowed):
            allowed = await allowed
        if not allowed:
            raise HTTPException(status_code=403, detail="tenant access denied")

    async def dispatch_run(
        request: Request, run_id: str, tenant_id: str, *, resume: bool = False
    ) -> dict[str, Any]:
        principal = await resolve_principal(request)
        await authorize(request, tenant_id, "execute", principal.principal_id)
        if execution_dispatcher is None:
            raise HTTPException(
                status_code=503, detail="agent execution dispatcher is not configured"
            )
        message = AgentExecutionMessage(
            run_id=run_id, tenant_id=tenant_id, principal=principal, resume=resume
        )
        dispatch = getattr(execution_dispatcher, "dispatch", execution_dispatcher)
        result = dispatch(message)
        if inspect.isawaitable(result):
            result = await result
        return {"status": "QUEUED", "run_id": run_id, "dispatch_id": result}

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

    @app.post("/ai/knowledge", tags=["ai-workflows"])
    async def ingest_knowledge(
        request: Request, body: IngestKnowledgeRequest
    ) -> dict[str, Any]:
        await authorize(request, body.tenant_id, "ai_knowledge_write")
        try:
            return app.state.ai_workflow.ingest(body.tenant_id, body.document)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/ai/answer", tags=["ai-workflows"])
    async def answer_question(request: Request, body: AskAIRequest) -> dict[str, Any]:
        await authorize(request, body.tenant_id, "ai_generate")
        try:
            return app.state.ai_workflow.answer(
                body.tenant_id, body.query, body.max_tokens
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AIWorkflowError as exc:
            logger.warning("AI workflow inference unavailable")
            raise HTTPException(
                status_code=503, detail="configured AI provider is unavailable"
            ) from exc

    @app.post("/agent/runs/{run_id}/feedback", tags=["agentic"])
    async def record_agent_feedback(
        request: Request, run_id: str, body: AgentFeedbackRequest
    ) -> dict[str, Any]:
        await authorize(request, body.tenant_id, "ai_feedback")
        try:
            evidence_id = app.state.agent_runtime.record_evidence(
                run_id,
                body.tenant_id,
                "user_feedback",
                {
                    "rating": body.rating,
                    "note_sha256": hashlib.sha256(body.note.encode()).hexdigest()
                    if body.note
                    else None,
                },
            )
            return {"run_id": run_id, "evidence_id": evidence_id}
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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
            return _run_summary(run)
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
        try:
            return await dispatch_run(request, run_id, body.tenant_id)
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
            return await dispatch_run(request, run_id, body.tenant_id, resume=True)
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/agent/runs/{run_id}/approve", tags=["agentic"])
    async def approve_agent_run(
        request: Request, run_id: str, body: ApprovalRequest
    ) -> dict[str, Any]:
        principal = await resolve_principal(request)
        if not principal.principal_id:
            raise HTTPException(status_code=503, detail="trusted approver identity is unavailable")
        await authorize(request, body.tenant_id, "approve", principal.principal_id)
        try:
            agent_runtime = app.state.agent_runtime
            approval = agent_runtime.approve(
                run_id,
                body.tenant_id,
                principal.principal_id,
                body.reason,
                approval_id=body.approval_id,
            )
            return {"approval": approval.dict(), "status": "APPROVED"}
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
        principal = await resolve_principal(request)
        if not principal.principal_id:
            raise HTTPException(status_code=503, detail="trusted approver identity is unavailable")
        await authorize(request, body.tenant_id, "approve", principal.principal_id)
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
                    approval.run_id,
                    body.tenant_id,
                    principal.principal_id,
                    body.reason,
                    approval_id=approval_id,
                )
                result = {"status": "APPROVED", "run_id": approval.run_id}
            elif body.decision == "deny":
                decision = runtime.deny(
                    approval.run_id,
                    body.tenant_id,
                    principal.principal_id,
                    body.reason,
                    approval_id=approval_id,
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
        chain = app.state.agent_runtime.verify_evidence_chain(run_id, tenant_id)
        return {
            "run_id": run_id,
            "count": len(entries),
            "sha256": hashlib.sha256(
                json.dumps([entry.sha256 for entry in entries]).encode()
            ).hexdigest(),
            "kinds": sorted({entry.kind for entry in entries}),
            "anchors": app.state.agent_runtime.list_evidence_anchors(
                run_id, tenant_id
            ),
            "integrity": chain,
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
