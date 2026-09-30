import os
import tempfile
import json
import pytest

pytest.importorskip("aegis_multimodal_ai_system.policy.policy_engine")
pytest.importorskip("aegis_multimodal_ai_system.review.db")

from aegis_multimodal_ai_system.policy.policy_engine import PolicyEngine
from aegis_multimodal_ai_system.review import db as review_db

def test_policy_engine_keywords():
    pe = PolicyEngine(model_callable=None, blocked_keywords=["kill", "bomb"])
    d = pe.decide("I will kill you")
    assert d.action == "block" and d.reason == "keyword"

def test_policy_engine_allow():
    pe = PolicyEngine(model_callable=lambda t: 0.0, blocked_keywords=["bad"])
    d = pe.decide("hello world")
    assert d.action == "allow"

def test_review_db_add_get_set(tmp_path):
    # redirect DB path to temp for isolation
    original = os.environ.get("REVIEW_DB_PATH")
    try:
        # monkeypatch DB_PATH by changing module variable
        from pathlib import Path
        review_db.DB_PATH = Path(str(tmp_path / "review_queue.db"))
        review_db.init_db()
        rid = review_db.add_item(request_id="r1", flagged=True, reason="test", text_snippet="hello", metadata={"a":1})
        items = review_db.list_pending()
        assert any(i["id"] == rid for i in items)
        it = review_db.get_item(rid)
        assert it["request_id"] == "r1"
        ok = review_db.set_review(rid, reviewer="alice", verdict="allow", notes="ok")
        assert ok is True
    finally:
        if original is not None:
            os.environ["REVIEW_DB_PATH"] = original
