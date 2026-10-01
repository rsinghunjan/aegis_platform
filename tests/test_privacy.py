# tests/test_privacy.py
import os
import json
import pytest
from datetime import datetime, timedelta

from api.db import SessionLocal
from api import privacy, audit
from api.models import DataRetentionPolicy, AuditLog, SafetyEvent, User

@pytest.fixture(autouse=True)
def setup_db():
    # ensure DB is fresh for tests (sqlite test DB configured by tests/conftest)
    yield
    # cleanup: simple truncation for determinism
    session = SessionLocal()
    for t in ("safety_events", "audit_logs", "data_retention_policies", "users"):
        try:
            session.execute(f"DELETE FROM {t};")
        except Exception:
            pass
    session.commit()
    session.close()


def test_register_and_dryrun_enforce(tmp_path):
    # create a table entry to be cleaned up: safety_event with old timestamp
    session = SessionLocal()
    old_ts = datetime.utcnow() - timedelta(days=365)
    ev = SafetyEvent(request_id="r1", user_id=None, model_version_id=None, decision="FLAG", reasons=["test"], input_snapshot="x", created_at=old_ts)
    session.add(ev)
    session.commit()

    # register retention policy for safety_events (30d -> delete)
    p = privacy.register_retention_policy("safety_30", table="safety_events", timestamp_column="created_at", retention_days=30, action="delete")
    assert isinstance(p, DataRetentionPolicy)

    summary = privacy.enforce_retention_policies(dry_run=True)
    # find our policy summary
    found = [s for s in summary if s["policy"] == "safety_30"]
    assert found and found[0]["rows"] >= 1


def test_anonymize_user_and_audit():
    session = SessionLocal()
    # create a user
    u = User(username="to_delete", password_hash="hash", scopes=["predict"])
    session.add(u)
    session.commit()
    uid = u.id
    session.close()

    ok = privacy.anonymize_user(uid, actor="tester")
    assert ok

    session = SessionLocal()
    updated = session.query(User).filter_by(id=uid).one_or_none()
    assert updated.username.startswith("anon-")
    assert updated.password_hash != "hash"
    assert updated.disabled is True

    # audit log should contain entry
    logs = session.query(AuditLog).filter_by(action="user.anonymize").all()
    assert len(logs) >= 1
    session.close()


def test_federated_protect_blocks_when_insufficient():
    with pytest.raises(PermissionError):
        privacy.federated_aggregation_protect({"metric": 10}, participant_count=2, min_participants=5)


def test_federated_dp_noise_and_audit():
    out = privacy.federated_aggregation_protect({"metric": 100}, participant_count=10, min_participants=5, dp_epsilon=0.5)
    # returned value should be numeric and different than input sometimes
    assert "metric" in out
    # audit event exists
    session = SessionLocal()
    ev = session.query(AuditLog).filter_by(action="federated.dp").one_or_none()
    assert ev is not None
    session.close()
