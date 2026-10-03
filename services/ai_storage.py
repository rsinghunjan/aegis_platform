"""Durable tenant-scoped knowledge and hash-only inference audit storage."""
from __future__ import annotations

import math
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    Index,
    JSON,
    String,
    Text,
    create_engine,
    delete,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from services.embeddings.vector_store import VectorRecord, VectorStore, _cosine_similarity


class AIStorageBase(DeclarativeBase):
    pass


class KnowledgeDocumentRow(AIStorageBase):
    __tablename__ = "ai_knowledge_documents"

    tenant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class KnowledgeChunkRow(AIStorageBase):
    __tablename__ = "ai_knowledge_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class InferenceAuditRow(AIStorageBase):
    __tablename__ = "ai_inference_audit"
    __table_args__ = (Index("ix_ai_audit_tenant_created", "tenant_id", "created_at"),)

    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    response_sha256: Mapped[str | None] = mapped_column(String(64))
    error_type: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AIDataStore:
    """SQL-backed stores; vector queries always constrain by the bound tenant."""

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.getenv(
            "AEGIS_AI_DATABASE_URL", "sqlite:///./aegis_ai.db"
        )
        self.engine = create_engine(self.database_url, pool_pre_ping=True)
        self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._initialized = False

    def initialize(self) -> None:
        if not self._initialized:
            AIStorageBase.metadata.create_all(self.engine)
            self._initialized = True

    def vector_store(self, tenant_id: str) -> TenantVectorStore:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        return TenantVectorStore(self, tenant_id)

    def ingest_document(
        self,
        tenant_id: str,
        document_id: str,
        char_count: int,
        max_documents: int,
        max_indexed_chars: int,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors) or not chunks:
            raise ValueError("document chunks and embeddings must be non-empty and aligned")
        if any(not math.isfinite(float(value)) for vector in vectors for value in vector):
            raise ValueError("embeddings must contain only finite numbers")
        self.initialize()
        with self._sessions.begin() as session:
            doc_count, char_total = session.execute(
                select(
                    func.count(KnowledgeDocumentRow.document_id),
                    func.coalesce(func.sum(KnowledgeDocumentRow.char_count), 0),
                ).where(KnowledgeDocumentRow.tenant_id == tenant_id)
            ).one()
            if doc_count >= max_documents:
                raise ValueError("tenant document limit reached")
            if char_total + char_count > max_indexed_chars:
                raise ValueError("tenant indexed text limit reached")
            session.add(
                KnowledgeDocumentRow(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    char_count=char_count,
                    created_at=datetime.now(timezone.utc),
                )
            )
            for chunk_index, (text, embedding) in enumerate(zip(chunks, vectors)):
                session.add(
                    KnowledgeChunkRow(
                        id=str(uuid.uuid4()),
                        tenant_id=tenant_id,
                        document_id=document_id,
                        text=text,
                        embedding=embedding,
                        metadata_json={
                            "tenant_id": tenant_id,
                            "document_id": document_id,
                            "chunk_index": chunk_index,
                        },
                    )
                )

    def remove_document(self, tenant_id: str, document_id: str) -> bool:
        self.initialize()
        with self._sessions.begin() as session:
            session.execute(
                delete(KnowledgeChunkRow).where(
                    KnowledgeChunkRow.tenant_id == tenant_id,
                    KnowledgeChunkRow.document_id == document_id,
                )
            )
            result = session.execute(
                delete(KnowledgeDocumentRow).where(
                    KnowledgeDocumentRow.tenant_id == tenant_id,
                    KnowledgeDocumentRow.document_id == document_id,
                )
            )
            deleted = result.rowcount == 1
        return deleted

    def record_inference(
        self,
        *,
        request_id: str,
        tenant_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float,
        status: str,
        request_sha256: str,
        response_sha256: str | None = None,
        error_type: str | None = None,
    ) -> None:
        self.initialize()
        with self._sessions.begin() as session:
            session.add(
                InferenceAuditRow(
                    request_id=request_id,
                    tenant_id=tenant_id,
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cost_usd=cost_usd,
                    status=status,
                    request_sha256=request_sha256,
                    response_sha256=response_sha256,
                    error_type=error_type,
                    created_at=datetime.now(timezone.utc),
                )
            )

    def list_usage(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        self.initialize()
        with self._sessions() as session:
            rows = session.scalars(
                select(InferenceAuditRow)
                .where(InferenceAuditRow.tenant_id == tenant_id)
                .order_by(
                    InferenceAuditRow.created_at.desc(),
                    InferenceAuditRow.request_id.desc(),
                )
                .limit(limit)
            ).all()
        return [
            {
                "request_id": row.request_id,
                "tenant_id": row.tenant_id,
                "provider": row.provider,
                "model": row.model,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "latency_ms": row.latency_ms,
                "cost_usd": row.cost_usd,
                "status": row.status,
                "request_sha256": row.request_sha256,
                "response_sha256": row.response_sha256,
                "error_type": row.error_type,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]


class TenantVectorStore(VectorStore):
    """VectorStore view whose reads and mutations are bound to one tenant."""

    name = "tenant-sql"

    def __init__(self, store: AIDataStore, tenant_id: str):
        self.store = store
        self.tenant_id = tenant_id

    def upsert(
        self,
        vectors: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        ttl_seconds: float | None = None,
    ) -> list[str]:
        del ttl_seconds
        if len(vectors) != len(texts):
            raise ValueError("vectors and texts must be the same length")
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        if len(metadatas) != len(texts) or len(ids) != len(texts):
            raise ValueError(
                "vectors, texts, metadata, and ids must have equal lengths"
            )
        self.store.initialize()
        with self.store._sessions.begin() as session:
            for vector, text, metadata, record_id in zip(vectors, texts, metadatas, ids):
                document_id = metadata.get("document_id")
                if metadata.get("tenant_id") != self.tenant_id or not document_id:
                    raise ValueError("tenant-scoped document metadata is required")
                document = session.get(
                    KnowledgeDocumentRow, (self.tenant_id, document_id)
                )
                if document is None:
                    raise ValueError("document is not registered for this tenant")
                row = session.get(KnowledgeChunkRow, record_id)
                if row is not None and row.tenant_id != self.tenant_id:
                    raise ValueError("chunk id belongs to a different tenant")
                if row is None:
                    row = KnowledgeChunkRow(
                        id=record_id,
                        tenant_id=self.tenant_id,
                        document_id=document_id,
                        text=text,
                        embedding=vector,
                        metadata_json=metadata,
                    )
                    session.add(row)
                else:
                    row.document_id = document_id
                    row.text = text
                    row.embedding = vector
                    row.metadata_json = metadata
        return ids

    def query(
        self, vector: list[float], top_k: int = 5
    ) -> list[tuple[VectorRecord, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.store.initialize()
        with self.store._sessions() as session:
            rows = session.scalars(
                select(KnowledgeChunkRow).where(
                    KnowledgeChunkRow.tenant_id == self.tenant_id
                )
            ).all()
        scored = [
            (
                VectorRecord(
                    id=row.id,
                    vector=row.embedding,
                    text=row.text,
                    metadata=row.metadata_json,
                ),
                _cosine_similarity(vector, row.embedding),
            )
            for row in rows
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def delete(self, ids: list[str]) -> None:
        self.store.initialize()
        with self.store._sessions.begin() as session:
            session.execute(
                delete(KnowledgeChunkRow).where(
                    KnowledgeChunkRow.tenant_id == self.tenant_id,
                    KnowledgeChunkRow.id.in_(ids),
                )
            )
