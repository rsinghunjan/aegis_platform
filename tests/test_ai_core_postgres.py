import os
import uuid

import pytest

from ai_core import AICoreStore


POSTGRES_URL = os.getenv("AEGIS_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="AEGIS_TEST_POSTGRES_URL is not configured"
)


def test_postgres_store_retention_and_tenant_isolation():
    store = AICoreStore(POSTGRES_URL, max_events_per_tenant_kind=3)
    tenant_a = f"postgres-test-a-{uuid.uuid4()}"
    tenant_b = f"postgres-test-b-{uuid.uuid4()}"

    try:
        for value in range(5):
            store.append(tenant_a, "mem:answer", {"value": value})
        store.append(tenant_b, "mem:answer", {"value": 99})

        assert [event["value"] for event in store.recent(tenant_a)] == [2, 3, 4]
        assert [event["value"] for event in store.recent(tenant_b)] == [99]
        store.forget(tenant_a)
        assert store.recent(tenant_a) == []
        assert [event["value"] for event in store.recent(tenant_b)] == [99]
    finally:
        store.forget(tenant_a)
        store.forget(tenant_b)
