"""Execution messages and worker entry points for the governed agent runtime."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

from agentic.runtime import AgentRuntime


class AgentPrincipal(BaseModel):
    """Trusted identity and execution attributes supplied by the embedding service."""

    principal_id: str | None = None
    role: str = "agent"
    environment: str = "local"
    scopes: list[str] = Field(default_factory=list)


class AgentExecutionMessage(BaseModel):
    """Queue-safe execution request; handlers must deduplicate on run_id."""

    run_id: str
    tenant_id: str
    principal: AgentPrincipal
    resume: bool = False


class ExecutionDispatcher(Protocol):
    def dispatch(self, message: AgentExecutionMessage) -> Any: ...


class AgentWorker:
    """Worker-side executor, intended to run separately from the control-plane API."""

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    async def execute(self, message: AgentExecutionMessage) -> dict[str, Any]:
        method: Callable[..., Any] = self.runtime.resume if message.resume else self.runtime.execute
        result = method(
            message.run_id,
            message.tenant_id,
            role=message.principal.role,
            environment=message.principal.environment,
            scopes=set(message.principal.scopes),
            principal_id=message.principal.principal_id,
        )
        return await result if inspect.isawaitable(result) else result


class CeleryExecutionDispatcher:
    """Queue worker execution through an application-configured Celery instance."""

    def __init__(self, celery_app: Any, task_name: str = "aegis.execute_agent_run"):
        self.celery_app = celery_app
        self.task_name = task_name

    def dispatch(self, message: AgentExecutionMessage) -> str:
        result = self.celery_app.send_task(
            self.task_name, args=[message.dict()]
        )
        return str(result.id)


def register_celery_agent_worker(
    celery_app: Any,
    worker: AgentWorker,
    task_name: str = "aegis.execute_agent_run",
) -> Any:
    """Register the task in a worker process after its runtime/tools are configured."""

    @celery_app.task(name=task_name)
    def execute_agent_run(payload: dict[str, Any]) -> dict[str, Any]:
        message = AgentExecutionMessage.parse_obj(payload)
        return asyncio.run(worker.execute(message))

    return execute_agent_run
