"""Durable, tenant-scoped event storage shared by AI memory and observability."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Index,
    Integer,
    String,
    create_engine,
    delete,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class AICoreBase(DeclarativeBase):
    pass


class AICoreEventRow(AICoreBase):
    __tablename__ = "ai_core_events"
    __table_args__ = (Index("ix_ai_core_tenant_kind_id", "tenant_id", "kind", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AICoreStore:
    """Append-only event log with per-tenant, per-kind retention.

    Payloads must be metadata and hashes only; callers never pass raw prompts.
    Works with any SQLAlchemy URL (use PostgreSQL for multi-replica services).
    """

    def __init__(
        self, database_url: str | None = None, max_events_per_tenant_kind: int = 500
    ):
        self.database_url = database_url or os.getenv(
            "AEGIS_AI_CORE_DATABASE_URL",
            os.getenv("AEGIS_AI_DATABASE_URL", "sqlite:///./aegis_ai.db"),
        )
        self.max_events = max_events_per_tenant_kind
        self.engine = create_engine(self.database_url, pool_pre_ping=True)
        self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._initialized = False

    def initialize(self) -> None:
        if not self._initialized:
            AICoreBase.metadata.create_all(self.engine)
            self._initialized = True

    def append(self, tenant_id: str, kind: str, payload: dict[str, Any]) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        self.initialize()
        with self._sessions.begin() as session:
            row = AICoreEventRow(
                tenant_id=tenant_id,
                kind=kind,
                payload=payload,
                created_at=datetime.now(timezone.utc),
            )
            session.add(row)
            session.flush()
            cutoff = session.scalars(
                select(AICoreEventRow.id)
                .where(
                    AICoreEventRow.tenant_id == tenant_id,
                    AICoreEventRow.kind == kind,
                )
                .order_by(AICoreEventRow.id.desc())
                .offset(self.max_events)
                .limit(1)
            ).first()
            if cutoff is not None:
                session.execute(
                    delete(AICoreEventRow).where(
                        AICoreEventRow.tenant_id == tenant_id,
                        AICoreEventRow.kind == kind,
                        AICoreEventRow.id <= cutoff,
                    )
                )

    def recent(
        self, tenant_id: str, kinds: list[str] | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        """Return the newest events for one tenant in chronological order."""
        self.initialize()
        query = select(AICoreEventRow).where(AICoreEventRow.tenant_id == tenant_id)
        if kinds:
            query = query.where(AICoreEventRow.kind.in_(kinds))
        query = query.order_by(AICoreEventRow.id.desc()).limit(max(1, limit))
        with self._sessions() as session:
            rows = session.scalars(query).all()
        return [
            {"kind": row.kind, "at": row.created_at.timestamp(), **row.payload}
            for row in reversed(rows)
        ]

    def forget(self, tenant_id: str) -> None:
        self.initialize()
        with self._sessions.begin() as session:
            session.execute(
                delete(AICoreEventRow).where(AICoreEventRow.tenant_id == tenant_id)
            )
