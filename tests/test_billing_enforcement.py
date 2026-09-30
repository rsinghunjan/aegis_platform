"""
Unit test for BillingEnforcementMiddleware.

This test:
- Creates an in-memory SQLite DB and a BillingAccount row with billing_suspended=True.
- Monkeypatches api.db.SessionLocal to use the test session factory.
- Spins up a small FastAPI app including the BillingEnforcementMiddleware and a sample route.
- Calls the route with X-Tenant-ID header and asserts 402 Payment Required is returned.

Run with: pytest tests/test_billing_enforcement.py -q
"""
import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.db as db_mod
from api.models import Base, BillingAccount
from api.middleware.billing_enforcement import BillingEnforcementMiddleware

# Create an in-memory SQLite engine and session factory for tests
TEST_SQLITE_URL = "sqlite:///:memory:"


@pytest.fixture
def test_session_factory():
    engine = create_engine(
        TEST_SQLITE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine)
    # create tables
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal


@pytest.fixture(autouse=True)
def patch_session_local(monkeypatch, test_session_factory):
    # monkeypatch the real SessionLocal to use test session
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_factory)
    yield


def test_billing_suspended_returns_402(test_session_factory):
    # seed a billing account with billing_suspended=True
    session = test_session_factory()
    ba = BillingAccount(tenant_id="tenant-test", billing_suspended=True, dunning_level=2)
    session.add(ba)
    session.commit()
    session.close()

    app = FastAPI()

    # include middleware
    app.add_middleware(BillingEnforcementMiddleware, billing_portal_url="/fake/portal")

    @app.get("/ping")
    def ping():
        return {"ok": True}

    client = TestClient(app)
    # call endpoint with X-Tenant-ID header to simulate tenant identity
    resp = client.get("/ping", headers={"X-Tenant-ID": "tenant-test"})
    assert resp.status_code == 402
    body = resp.json()
    assert body.get("billing_portal") == "/fake/portal"
    assert body.get("dunning_level") == 2


def test_non_suspended_allows_request(test_session_factory):
    session = test_session_factory()
    ba = BillingAccount(tenant_id="tenant-ok", billing_suspended=False, dunning_level=0)
    session.add(ba)
    session.commit()
    session.close()

    app = FastAPI()
    app.add_middleware(BillingEnforcementMiddleware, billing_portal_url="/fake/portal")

    @app.get("/ping")
    def ping(request=None):
        # middleware should have set request.state.billing_dunning_level (0)
        return {"ok": True}

    client = TestClient(app)
    resp = client.get("/ping", headers={"X-Tenant-ID": "tenant-ok"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
