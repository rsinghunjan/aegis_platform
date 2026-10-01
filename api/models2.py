# (updated) add SafetyEvent model to the existing api/models.py
"""
Canonical datastore models for Aegis (SQLAlchemy ORM)
- User, RefreshToken, Model, ModelVersion, Job, SafetyEvent
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from .db import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    scopes = Column(JSON, nullable=False, default=[])  # list of scope strings
    disabled = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    refresh_tokens = relationship("RefreshToken", back_populates="user")
    jobs = relationship("Job", back_populates="user")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token = Column(Text, nullable=False, unique=True, index=True)
    revoked = Column(Boolean, default=False, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="refresh_tokens")


class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    versions = relationship("ModelVersion", back_populates="model")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(String(100), nullable=False)
    metadata = Column(JSON, nullable=True)
    artifact_path = Column(String(1000), nullable=True)  # object store or filesystem path
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    model = relationship("Model", back_populates="versions")
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_version"),)


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True, index=True)
    status = Column(String(50), nullable=False, default="PENDING", index=True)
    input_payload = Column(JSON, nullable=True)
    output_payload = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="jobs")


class SafetyEvent(Base):
    __tablename__ = "safety_events"
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True, index=True)
    decision = Column(String(20), nullable=False, index=True)  # ALLOW|FLAG|BLOCK
    reasons = Column(JSON, nullable=True)  # list of reasons
    input_snapshot = Column(Text, nullable=True)  # truncated JSON payload for audit
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # relationship backrefs (optional)
    # user = relationship("User", backref="safety_events")  # if you want backref
