"""Tenant-scoped, bounded memory shared by every AI operation."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque
from typing import Any

from ai_core.store import AICoreStore

logger = logging.getLogger(__name__)


class AIMemory:
    """Stores hashed interaction metadata (never raw prompts) per tenant."""

    def __init__(
        self,
        max_tenants: int = 100,
        max_events_per_tenant: int = 200,
        store: AICoreStore | None = None,
    ):
        self.max_tenants = max_tenants
        self.max_events_per_tenant = max_events_per_tenant
        self.store = store
        self._events: OrderedDict[str, deque[dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def remember(self, tenant_id: str, kind: str, payload: dict[str, Any]) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        event = {"kind": kind, "at": time.time(), **payload}
        with self._lock:
            events = self._events.get(tenant_id)
            if events is None:
                if len(self._events) >= self.max_tenants:
                    self._events.popitem(last=False)
                events = self._events[tenant_id] = deque(
                    maxlen=self.max_events_per_tenant
                )
            self._events.move_to_end(tenant_id)
            events.append(event)
        if self.store is not None:
            try:
                self.store.append(tenant_id, f"mem:{kind}", payload)
            except Exception as exc:
                # Memory is advisory; never fail the user request on a store outage.
                logger.warning("AI memory persistence failed: %s", type(exc).__name__)

    def recall(
        self, tenant_id: str, kind: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        if self.store is not None:
            try:
                stored = self.store.recent(
                    tenant_id,
                    [f"mem:{kind}"] if kind else None,
                    max(1, limit),
                )
                return [
                    {**event, "kind": event["kind"].removeprefix("mem:")}
                    for event in stored
                    if event["kind"].startswith("mem:")
                ]
            except Exception as exc:
                logger.warning("AI memory read failed: %s", type(exc).__name__)
        with self._lock:
            events = list(self._events.get(tenant_id, ()))
        if kind is not None:
            events = [event for event in events if event["kind"] == kind]
        return events[-limit:]

    def seen_query(self, tenant_id: str, query: str) -> bool:
        digest = query_digest(query)
        return any(
            event.get("query_sha256") == digest
            for event in self.recall(tenant_id, "answer", self.max_events_per_tenant)
        )

    def forget(self, tenant_id: str) -> None:
        with self._lock:
            self._events.pop(tenant_id, None)
        if self.store is not None:
            self.store.forget(tenant_id)


def query_digest(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()
