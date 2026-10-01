from __future__ import annotations

from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, JSON, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from pgvector.sqlalchemy import Vector

from aegis_db.settings import get_db_settings


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), ForeignKey("tenants.id"), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    tenant = relationship("Tenant")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), ForeignKey("tenants.id"), index=True)
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.id"), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    artifacts_uri: Mapped[str | None] = mapped_column(Text, nullable=True)

    tenant = relationship("Tenant")
    job = relationship("Job")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(255))
    action: Mapped[str] = mapped_column(String(255))
    resource: Mapped[str] = mapped_column(String(255))
    decision: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RagDocument(Base):
    __tablename__ = "rag_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    namespace: Mapped[str] = mapped_column(String(128), index=True)
    doc_id: Mapped[str] = mapped_column(String(255), index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RagChunk(Base):
    __tablename__ = "rag_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    namespace: Mapped[str] = mapped_column(String(128), index=True)
    doc_id: Mapped[str] = mapped_column(String(255), index=True)
    chunk_id: Mapped[str] = mapped_column(String(255), index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # IMPORTANT: dim is fixed per DB schema; env is read at import time here,
    # but real fixation happens in Alembic migration.
    embedding: Mapped[list[float]] = mapped_column(Vector(get_db_settings().embedding_dim))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
