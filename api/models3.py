# (patch) Add AuditLog and DataRetentionPolicy models to existing api/models.py
"""
Canonical datastore models for Aegis (SQLAlchemy ORM)
- Existing models retained; additions below: AuditLog, DataRetentionPolicy
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from .db import Base


# ... existing User, RefreshToken, Model, ModelVersion, Job, SafetyEvent definitions unchanged ...
# (omitted here for brevity in the patch; ensure existing models are still present in the file)

# New models (append to the same file)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(200), nullable=False, index=True)
    actor = Column(String(200), nullable=False)
    target_type = Column(String(200), nullable=False)
    target_id = Column(Integer, nullable=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DataRetentionPolicy(Base):
    __tablename__ = "data_retention_policies"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, unique=True)
    table_name = Column(String(200), nullable=False)
    timestamp_column = Column(String(200), nullable=False, default="created_at")
    retention_days = Column(Integer, nullable=False, default=90)
    action = Column(String(20), nullable=False, default="delete")  # delete | anonymize
    tenant_column = Column(String(200), nullable=True)
    filter_sql = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
