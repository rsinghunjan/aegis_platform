  1
  2
  3
  4
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
#!/usr/bin/env python3
"""
SQLAlchemy model for audit logs related to model promotions / verifications / approvals.
This table is used to record who attempted a promotion, verification results, and cosign signer identity.
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ModelAudit(Base):
    __tablename__ = "model_audit"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(256), nullable=False, index=True)
    model_version = Column(String(64), nullable=True)
    actor = Column(String(256), nullable=False)  # identity from OIDC token
    action = Column(String(64), nullable=False)  # e.g., "promote", "verify", "approve"
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    verification_passed = Column(Boolean, default=False)
    verification_details = Column(JSON, nullable=True)
    signature_issuer = Column(String(256), nullable=True)
    notes = Column(Text, nullable=True)
aegis_multimodal_ai_system/db/models/audit.py
