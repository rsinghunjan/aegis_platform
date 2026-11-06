"""Canonical, lightweight FastAPI application for the Aegis control plane."""
from __future__ import annotations

import logging
import hashlib
import json
import asyncio
import math
import os
from contextlib import asynccontextmanager
import inspect
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from datetime import datetime, timezone

from ai_core import AICore, AIQuotaExceeded, anomaly
from agentic.deployment import create_configured_agent_runtime
from agentic.identity import OIDCTenantIdentity
from agentic.runtime import AgentRuntime, AgentRuntimeError, AgentStore
from agentic.worker import (
    AgentExecutionMessage,
    AgentPrincipal,
    CeleryExecutionDispatcher,
)
from services import feedback_loop, regression_sets
from services.ai_evaluation import tuning_recommendations
from services.prompt_eval import build_prompt_evaluator
from services.ai_workflow import AIWorkflow, AIWorkflowError
from services.observability.ai_metrics import CONTENT_TYPE_LATEST, render_metrics
from services.observability.tracing import (
    get_trace_id,
    inject_trace_context,
    start_span,
    trace_id_context,
    trace_id_from_traceparent,
)
from api.otel import init_otel

logger = logging.getLogger(__name__)


def _validate_production_integrations(
    tenant_authorizer: Any,
    principal_resolver: Any,
    execution_dispatcher: Any,
) -> None:
    environment = (
        os.getenv("AEGIS_ENV")
        or os.getenv("ENVIRONMENT")
        or os.getenv("ENV")
        or ""
    ).strip().lower()
    if environment not in {"prod", "production"} and os.getenv(
        "AEGIS_PRODUCTION", ""
    ).strip().lower() not in {"1", "true", "yes"}:
        return

    if os.getenv("AEGIS_ALLOW_DEV_PROVIDERS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        raise RuntimeError("development AI providers are not allowed in production")
    if tenant_authorizer is None or principal_resolver is None:
        raise RuntimeError(
            "production requires trusted tenant authorization and principal resolution"
        )
    if execution_dispatcher is None:
        raise RuntimeError("production requires a configured agent execution dispatcher")

    database_urls = {
        "AEGIS_AGENT_DATABASE_URL": os.getenv("AEGIS_AGENT_DATABASE_URL")
        or os.getenv("DATABASE_URL", ""),
        "AEGIS_AI_DATABASE_URL": os.getenv("AEGIS_AI_DATABASE_URL", ""),
    }
    for setting, database_url in database_urls.items():
        if not database_url.strip().lower().startswith(
            ("postgresql://", "postgresql+", "postgres://")
        ):
            raise RuntimeError(
                f"production requires {setting} to use a shared PostgreSQL database"
            )


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


class PolicyProposalRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    suggestion: dict[str, Any]
    replay_traffic: list[dict[str, Any]] = Field(min_items=1, max_items=500)


class PolicyApprovalRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="", max_length=500)


class PolicyRollbackRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    reason: str = Field(default="", max_length=500)


class AgentFeedbackRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    rating: int = Field(ge=1, le=5)
    note: str = Field(default="", max_length=2000)


class AutoProposeRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    replay_traffic: list[dict[str, Any]] = Field(min_items=1, max_items=500)


class AnomalyScanRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    model_name: str = Field(default="default", min_length=1, max_length=128)
    remediate: bool = False


class PromptProposalRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    template: str = Field(min_length=1, max_length=4000)
    reason: str = Field(default="", max_length=500)


class PromptPromoteRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)


class RoutingApplyRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    order: list[str] = Field(min_items=1, max_items=16)


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
    policy_replay_evaluator: Callable[..., Any] | None = None,
    policy_opa_client: Any = None,
    drift_remediation: Any = None,
) -> FastAPI:
    oidc_identity = OIDCTenantIdentity.from_environment()
    selected_tenant_authorizer = tenant_authorizer or (
        oidc_identity.authorize_tenant if oidc_identity is not None else None
    )
    selected_principal_resolver = principal_resolver or (
        oidc_identity.resolve_principal if oidc_identity is not None else None
    )
    selected_execution_dispatcher = execution_dispatcher
    broker_url = os.getenv("CELERY_BROKER_URL") or os.getenv("AEGIS_BROKER_URL")
    if selected_execution_dispatcher is None and broker_url:
        from celery import Celery

        result_backend = os.getenv("CELERY_RESULT_BACKEND")
        celery_app = Celery(
            "aegis_agent_control_plane",
            broker=broker_url,
            backend=result_backend or broker_url,
        )
        selected_execution_dispatcher = CeleryExecutionDispatcher(
            celery_app, queue=os.getenv("AEGIS_AGENT_QUEUE")
        )
    _validate_production_integrations(
        selected_tenant_authorizer,
        selected_principal_resolver,
        selected_execution_dispatcher,
    )
    selected_runtime = runtime or create_configured_agent_runtime()
    selected_ai_workflow = ai_workflow or AIWorkflow()
    if approval_notifier is not None:
        selected_runtime.approval_notifier = approval_notifier

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            selected_runtime.store.initialize()
            ai_data_store = getattr(selected_ai_workflow, "data_store", None)
            if ai_data_store is not None:
                ai_data_store.initialize()
        except Exception as exc:
            logger.exception("Aegis runtime database initialization failed")
            raise RuntimeError(
                "Aegis database initialization failed; configure the agent and AI "
                "database URLs, install their drivers, and enable pgvector for PostgreSQL"
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
    app.state.ai_core = AICore.durable(selected_ai_workflow)
    selected_runtime.decision_core = app.state.ai_core
    app.state.ai_core.policy.replay_evaluator = policy_replay_evaluator
    if policy_opa_client is not None:
        app.state.ai_core.policy.opa_client = policy_opa_client
    app.state.execution_dispatcher = selected_execution_dispatcher

    @app.middleware("http")
    async def trace_request(request: Request, call_next):
        incoming_trace_id = trace_id_from_traceparent(
            request.headers.get("traceparent")
        )
        with trace_id_context(incoming_trace_id):
            with start_span(
                "aegis.http.request",
                {"http.method": request.method},
            ):
                response = await call_next(request)
            response.headers["X-Trace-ID"] = get_trace_id() or incoming_trace_id or ""
            return response

    init_otel(
        app=app,
        sqlalchemy_engine=getattr(
            getattr(selected_ai_workflow, "data_store", None), "engine", None
        ),
    )

    async def resolve_principal(request: Request) -> AgentPrincipal:
        if selected_principal_resolver is None:
            return AgentPrincipal()
        principal = selected_principal_resolver(request)
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
        if selected_tenant_authorizer is None:
            raise HTTPException(
                status_code=503,
                detail="agent API authorization is not configured",
            )
        principal = await resolve_principal(request)
        allowed = selected_tenant_authorizer(
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
        if selected_execution_dispatcher is None:
            raise HTTPException(
                status_code=503, detail="agent execution dispatcher is not configured"
            )
        message = AgentExecutionMessage(
            run_id=run_id,
            tenant_id=tenant_id,
            principal=principal,
            resume=resume,
            traceparent=inject_trace_context(),
        )
        dispatch = getattr(
            selected_execution_dispatcher,
            "dispatch",
            selected_execution_dispatcher,
        )
        result = dispatch(message)
        if inspect.isawaitable(result):
            result = await result
        app.state.ai_core.record_agent_event(
            tenant_id, run_id, "resume" if resume else "execute", "queued"
        )
        return {"status": "QUEUED", "run_id": run_id, "dispatch_id": result}

    @app.get("/healthz", tags=["health"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        payload = render_metrics()
        if payload is None:
            raise HTTPException(
                status_code=503, detail="Prometheus metrics dependency is unavailable"
            )
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    @app.get("/readyz", tags=["health"])
    def readyz() -> dict[str, str]:
        try:
            stores = [
                app.state.agent_runtime.store,
                getattr(app.state.ai_workflow, "data_store", None),
            ]
            for store in stores:
                engine = getattr(store, "engine", None)
                if engine is not None:
                    with engine.connect() as connection:
                        connection.exec_driver_sql("SELECT 1")
        except Exception as exc:
            logger.warning("Aegis database readiness check failed: %s", type(exc).__name__)
            raise HTTPException(
                status_code=503, detail="application database unavailable"
            ) from exc
        return {"status": "ready"}

    @app.post("/ai/knowledge", tags=["ai-workflows"])
    async def ingest_knowledge(
        request: Request, body: IngestKnowledgeRequest
    ) -> dict[str, Any]:
        await authorize(request, body.tenant_id, "ai_knowledge_write")
        try:
            return app.state.ai_core.ingest(body.tenant_id, body.document)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AIWorkflowError as exc:
            logger.warning(
                "AI workflow ingestion unavailable error_type=%s",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=503, detail="AI workflow is unavailable"
            ) from exc

    @app.delete("/ai/knowledge/{document_id}", tags=["ai-workflows"])
    async def delete_knowledge(
        request: Request, document_id: str, tenant_id: str
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "ai_knowledge_write")
        if not app.state.ai_workflow.delete_document(tenant_id, document_id):
            raise HTTPException(status_code=404, detail="knowledge document not found")
        return {"tenant_id": tenant_id, "document_id": document_id, "deleted": True}

    @app.post("/ai/answer", tags=["ai-workflows"])
    async def answer_question(request: Request, body: AskAIRequest) -> dict[str, Any]:
        await authorize(request, body.tenant_id, "ai_generate")
        try:
            return app.state.ai_core.answer(
                body.tenant_id, body.query, body.max_tokens
            )
        except AIQuotaExceeded as exc:
            raise HTTPException(
                status_code=429,
                detail="AI usage limit exceeded",
                headers={"Retry-After": str(exc.retry_after_seconds)},
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AIWorkflowError as exc:
            logger.warning(
                "AI workflow operation unavailable error_type=%s",
                type(exc).__name__,
            )
            raise HTTPException(status_code=503, detail="AI workflow is unavailable") from exc

    @app.get("/operator/ai/usage", tags=["ai-workflows"])
    async def list_ai_usage(
        request: Request,
        tenant_id: str,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        return {
            "tenant_id": tenant_id,
            "records": app.state.ai_workflow.list_usage(tenant_id, limit),
        }

    @app.get("/operator/ai/insights", tags=["ai-workflows"])
    async def ai_insights(request: Request, tenant_id: str) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        return app.state.ai_core.insights(tenant_id)

    @app.get("/operator/ai/policy/proposals", tags=["ai-workflows"])
    async def list_ai_policy_proposals(
        request: Request, tenant_id: str
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        return {
            "tenant_id": tenant_id,
            "proposals": app.state.ai_core.policy.list_proposals(tenant_id),
        }

    @app.get("/operator/ai/policy/versions", tags=["ai-workflows"])
    async def list_ai_policy_versions(
        request: Request, tenant_id: str
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        return {
            "tenant_id": tenant_id,
            "versions": app.state.ai_core.policy.history(tenant_id),
        }

    @app.post("/operator/ai/policy/proposals", tags=["ai-workflows"])
    async def propose_ai_policy(
        request: Request, body: PolicyProposalRequest
    ) -> dict[str, Any]:
        principal = await resolve_principal(request)
        await authorize(
            request, body.tenant_id, "ai_policy_propose", principal.principal_id
        )
        try:
            return app.state.ai_core.policy.propose(
                body.tenant_id, body.suggestion, body.replay_traffic
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/operator/ai/policy/proposals/{proposal_id}/approve",
        tags=["ai-workflows"],
    )
    async def approve_ai_policy(
        request: Request, proposal_id: str, body: PolicyApprovalRequest
    ) -> dict[str, Any]:
        principal = await resolve_principal(request)
        await authorize(request, body.tenant_id, "approve", principal.principal_id)
        try:
            return app.state.ai_core.policy.approve(
                body.tenant_id, proposal_id, principal.principal_id, body.reason
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/operator/ai/policy/rollback", tags=["ai-workflows"])
    async def rollback_ai_policy(
        request: Request, body: PolicyRollbackRequest
    ) -> dict[str, Any]:
        principal = await resolve_principal(request)
        await authorize(request, body.tenant_id, "approve", principal.principal_id)
        try:
            return app.state.ai_core.policy.rollback(
                body.tenant_id, body.version, principal.principal_id, body.reason
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/operator/ai/improvements", tags=["ai-workflows"])
    async def ai_improvements(request: Request, tenant_id: str) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        workflow = app.state.ai_workflow
        signals = await asyncio.to_thread(
            feedback_loop.collect_run_signals, app.state.agent_runtime, tenant_id
        )
        usage = await asyncio.to_thread(workflow.list_usage, tenant_id, 500)
        candidates = feedback_loop.build_candidates(
            signals, usage, workflow.inference_router.provider_order()
        )
        manifest = feedback_loop.export_retraining_manifest(signals)
        return {
            "tenant_id": tenant_id,
            "candidates": candidates,
            "retraining_manifest_sha256": manifest["manifest_sha256"],
            "retraining_rows": manifest["count"],
        }

    @app.get("/operator/ai/improvements/retraining-manifest", tags=["ai-workflows"])
    async def retraining_manifest(request: Request, tenant_id: str) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        signals = await asyncio.to_thread(
            feedback_loop.collect_run_signals, app.state.agent_runtime, tenant_id
        )
        return feedback_loop.export_retraining_manifest(signals)

    @app.post("/operator/ai/policy/auto-propose", tags=["ai-workflows"])
    async def auto_propose_policies(
        request: Request, body: AutoProposeRequest
    ) -> dict[str, Any]:
        principal = await resolve_principal(request)
        await authorize(
            request, body.tenant_id, "ai_policy_propose", principal.principal_id
        )
        try:
            return app.state.ai_core.propose_from_suggestions(
                body.tenant_id, body.replay_traffic
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/operator/ai/anomalies/scan", tags=["ai-workflows"])
    async def scan_anomalies(
        request: Request, body: AnomalyScanRequest
    ) -> dict[str, Any]:
        principal = await resolve_principal(request)
        await authorize(
            request, body.tenant_id, "ai_policy_propose", principal.principal_id
        )
        workflow = app.state.ai_workflow
        usage = await asyncio.to_thread(workflow.list_usage, body.tenant_id, 500)
        anomalies = anomaly.detect_anomalies(usage)

        def generate(prompt: str) -> str:
            return workflow.generate(body.tenant_id, prompt, 64)["text"]

        triaged = await asyncio.to_thread(anomaly.triage, anomalies, generate)
        remediation = []
        if body.remediate and triaged:
            if drift_remediation is None:
                raise HTTPException(
                    status_code=503, detail="drift remediation is not configured"
                )
            for finding in anomaly.to_drift_findings(
                body.tenant_id,
                body.model_name,
                triaged,
                datetime.now(timezone.utc).isoformat(),
            ):
                # High-risk actions still stop at the runtime approval gate.
                remediation.append(await drift_remediation.consume(finding))
        return {
            "tenant_id": body.tenant_id,
            "anomalies": triaged,
            "remediation": remediation,
        }

    @app.get("/operator/ai/regression-candidates", tags=["ai-workflows"])
    async def regression_candidates(request: Request, tenant_id: str) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        usage = await asyncio.to_thread(
            app.state.ai_workflow.list_usage, tenant_id, 500
        )
        return {
            "tenant_id": tenant_id,
            "failures": regression_sets.failure_candidates(usage),
        }

    @app.get("/operator/ai/prompts", tags=["ai-workflows"])
    async def list_prompts(request: Request, tenant_id: str) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        return {"prompts": app.state.ai_workflow.prompts.list()}

    @app.post("/operator/ai/prompts", tags=["ai-workflows"])
    async def propose_prompt(
        request: Request, body: PromptProposalRequest
    ) -> dict[str, Any]:
        principal = await resolve_principal(request)
        await authorize(
            request, body.tenant_id, "ai_policy_propose", principal.principal_id
        )
        try:
            return app.state.ai_workflow.prompts.propose(body.template, body.reason)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/operator/ai/prompts/{version}/promote", tags=["ai-workflows"])
    async def promote_prompt(
        request: Request, version: int, body: PromptPromoteRequest
    ) -> dict[str, Any]:
        principal = await resolve_principal(request)
        await authorize(request, body.tenant_id, "approve", principal.principal_id)
        workflow = app.state.ai_workflow
        try:
            return await asyncio.to_thread(
                workflow.prompts.promote,
                version,
                principal.principal_id,
                build_prompt_evaluator(workflow),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.warning("Prompt evaluation failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="prompt evaluation failed") from exc

    @app.post("/operator/ai/routing/apply", tags=["ai-workflows"])
    async def apply_routing(
        request: Request, body: RoutingApplyRequest
    ) -> dict[str, Any]:
        principal = await resolve_principal(request)
        await authorize(request, body.tenant_id, "approve", principal.principal_id)
        router = app.state.ai_workflow.inference_router
        try:
            router.reorder(body.order)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        app.state.ai_core.record_agent_event(
            body.tenant_id,
            "routing",
            "routing_applied",
            "applied",
            metadata={"actor_sha256": hashlib.sha256(principal.principal_id.encode()).hexdigest()},
        )
        return {"order": router.provider_order(), "applied_by": principal.principal_id}

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
            app.state.ai_core.record_agent_event(
                body.tenant_id, run_id, "feedback", "recorded"
            )
            return {"run_id": run_id, "evidence_id": evidence_id}
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/operator/agent/runs/{run_id}/feedback-summary", tags=["operator"])
    async def agent_feedback_summary(
        request: Request, run_id: str, tenant_id: str
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        try:
            feedback = [
                item.metadata["rating"]
                for item in app.state.agent_runtime.list_evidence(run_id, tenant_id)
                if item.kind == "user_feedback"
                and isinstance(item.metadata.get("rating"), int)
                and not isinstance(item.metadata.get("rating"), bool)
                and 1 <= item.metadata["rating"] <= 5
            ]
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        average_rating = sum(feedback) / len(feedback) if feedback else None
        return {
            "run_id": run_id,
            "tenant_id": tenant_id,
            "rating_count": len(feedback),
            "average_rating": average_rating,
            "tuning_recommendations": tuning_recommendations({}, feedback),
        }

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
            app.state.ai_core.record_agent_event(
                body.tenant_id, run.run_id, "create", "created"
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
            app.state.ai_core.record_agent_event(
                body.tenant_id, run.run_id, "create", "created"
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
            app.state.ai_core.record_agent_event(
                body.tenant_id, run_id, "approval", "approved"
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
            app.state.ai_core.record_agent_event(
                body.tenant_id,
                approval.run_id,
                "approval_decision",
                "approved" if body.decision == "approve" else "denied",
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

    @app.get("/operator/governance/status", tags=["operator"])
    async def operator_governance_status(
        request: Request, tenant_id: str
    ) -> dict[str, Any]:
        await authorize(request, tenant_id, "operator_read")
        agent_runtime = app.state.agent_runtime
        policy = agent_runtime.policy
        anchor_backend = agent_runtime.evidence_anchor_backend
        # Only non-sensitive configuration flags, hashes, and backend names may
        # be added below. Never include secrets, credentials, or connection
        # strings in this response.
        return {
            "tenant_id": tenant_id,
            "identity": {
                "tenant_authorizer_configured": selected_tenant_authorizer is not None,
                "principal_resolver_configured": selected_principal_resolver is not None,
            },
            "execution": {
                "execution_dispatcher_configured": selected_execution_dispatcher is not None,
                "capability_enforcement": agent_runtime.capability_enforcement,
                "approval_sla_seconds": agent_runtime.approval_sla_seconds,
            },
            "policy": {
                "autonomy_enabled": policy.autonomy_enabled,
                "autonomy_mode": policy.autonomy_mode,
                "medium_requires_approval": policy.medium_requires_approval,
                "policy_version": policy.version,
                "external_engine_count": len(policy.engines),
            },
            "evidence": {
                "anchoring_configured": anchor_backend is not None,
                "anchor_backend": getattr(anchor_backend, "name", None),
            },
        }

    return app


app = create_app()


def create_runtime(store: AgentStore | None = None) -> AgentRuntime:
    """Build a runtime for an embedding service or test."""
    return AgentRuntime(store=store)
