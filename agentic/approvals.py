"""Approval SLA operations and optional notification/escalation adapters."""
from __future__ import annotations

from typing import Any, Callable, Optional

from agentic.runtime import AgentRuntime, Approval


class ApprovalService:
    def __init__(
        self,
        runtime: AgentRuntime,
        notifier: Optional[Callable[[Approval], Any]] = None,
        escalator: Optional[Callable[[Approval], Any]] = None,
    ):
        self.runtime = runtime
        self.notifier = notifier
        self.escalator = escalator

    def pending(self, tenant_id: Optional[str] = None) -> list[Approval]:
        return self.runtime.list_approvals(tenant_id, status="pending")

    def approve(
        self, run_id: str, tenant_id: str, actor: str, reason: str = ""
    ) -> Approval:
        approval = self.runtime.approve(run_id, tenant_id, actor, reason)
        self.notify(approval)
        return approval

    def deny(
        self, run_id: str, tenant_id: str, actor: str, reason: str
    ) -> Approval:
        approval = self.runtime.deny(run_id, tenant_id, actor, reason)
        self.notify(approval)
        return approval

    def expire(self) -> int:
        return self.runtime.expire_approvals()

    def escalate(self) -> int:
        already_escalated = {
            approval.approval_id
            for approval in self.runtime.list_approvals(status="pending")
            if approval.escalation_status == "escalated"
        }
        escalated = self.runtime.escalate_approvals()
        if escalated and self.escalator:
            for approval in self.runtime.list_approvals(status="pending"):
                if (
                    approval.escalation_status == "escalated"
                    and approval.approval_id not in already_escalated
                ):
                    self.escalator(approval)
        return escalated

    def notify(self, approval: Approval) -> None:
        if self.notifier:
            self.notifier(approval)
