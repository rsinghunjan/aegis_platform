  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
#!/usr/bin/env python3
"""
Helper functions to record model audit events (verification/promote/approve) using the DB engine.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime
import uuid

from aegis_multimodal_ai_system.db.engine import session_scope
from aegis_multimodal_ai_system.db.models.audit import ModelAudit

def record_audit_event(model_name: str, action: str, actor: str, verification_passed: bool, verification_details: dict = None, signature_issuer: str = None, notes: str = None):
    with session_scope() as session:
        a = ModelAudit(
            model_name=model_name,
            model_version=verification_details.get("version") if verification_details else None,
            actor=actor,
            action=action,
            timestamp=datetime.utcnow(),
            verification_passed=verification_passed,
            verification_details=verification_details,
            signature_issuer=signature_issuer,
            notes=notes,
        )
        session.add(a)
        session.flush()
