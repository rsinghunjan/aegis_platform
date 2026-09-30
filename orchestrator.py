"""Reusable facade for creating, executing, approving, and resuming agent runs."""
from __future__ import annotations

from typing import Any, Optional

from agentic.runtime import AgentRuntime, AgentStore, ToolSpec


class Orchestrator:
    def __init__(self, runtime: Optional[AgentRuntime] = None):
        self.runtime = runtime or AgentRuntime()

    def register_tool(self, spec: ToolSpec, handler: Any) -> None:
        self.runtime.register_tool(spec, handler)

    def create_run(
        self,
        tenant_id: str,
        goal: str,
        budget: float = 0.0,
        idempotency_key: Optional[str] = None,
    ):
        return self.runtime.create_run(tenant_id, goal, budget, idempotency_key)

    async def execute(self, run_id: str, tenant_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self.runtime.execute(run_id, tenant_id, **kwargs)

    async def resume(self, run_id: str, tenant_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self.runtime.resume(run_id, tenant_id, **kwargs)

    def approve(self, run_id: str, tenant_id: str, actor: str, reason: str = ""):
        return self.runtime.approve(run_id, tenant_id, actor, reason)


def create_orchestrator(database_url: Optional[str] = None) -> Orchestrator:
    return Orchestrator(AgentRuntime(store=AgentStore(database_url)))
