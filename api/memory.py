"""Bounded conversation memory with in-process and tenant-scoped SQL backends."""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import DateTime, Integer, String, Text, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from aegis_db.session import create_sessionmaker
from aegis_db.migrations import upgrade_database

logger = logging.getLogger("aegis.memory")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ConversationMemory:
    """Backward-compatible bounded in-process history."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._msgs: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, message: Dict[str, Any]):
        with self._lock:
            self._msgs.append({"ts": time.time(), **message})
            if len(self._msgs) > self.capacity:
                self._msgs = self._msgs[-self.capacity :]

    def get_conversation(self, max_chars: int = 2000) -> str:
        with self._lock:
            out = [
                f"[{message.get('role', 'user')}] {message.get('content', '')}"
                for message in self._msgs
            ]
            conversation = "\n".join(out)
            return conversation[-max_chars:] if max_chars > 0 else ""

    def clear(self):
        with self._lock:
            self._msgs = []


class _MemoryBase(DeclarativeBase):
    pass


class ConversationMessage(_MemoryBase):
    __tablename__ = "conversation_memory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)


class PersistentConversationMemory:
    """Persist bounded memory by tenant/session and reject cross-tenant reads."""

    def __init__(
        self,
        session_id: str,
        tenant_id: str,
        capacity: int = 100,
        retention_days: int = 30,
        database_url: Optional[str] = None,
        summary_fn: Optional[Callable[[str], str]] = None,
        retrieval_hook: Optional[Callable[[str, str, str, int], list[Dict[str, Any]]]] = None,
    ):
        if not session_id or not tenant_id:
            raise ValueError("session_id and tenant_id are required")
        if capacity < 1 or retention_days < 1:
            raise ValueError("capacity and retention_days must be positive")
        self.session_id = session_id
        self.tenant_id = tenant_id
        self.capacity = capacity
        self.retention_days = retention_days
        self.summary_fn = summary_fn
        self.retrieval_hook = retrieval_hook
        self.engine, self.session_factory = create_sessionmaker(
            database_url
            or os.getenv("AEGIS_MEMORY_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or "sqlite:///./aegis_memory.db"
        )
        upgrade_database(
            database_url
            or os.getenv("AEGIS_MEMORY_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or "sqlite:///./aegis_memory.db"
        )

    def add(self, message: Dict[str, Any]) -> None:
        role = str(message.get("role", "user"))[:32]
        content = str(message.get("content", ""))
        with self.session_factory() as session:
            now = _now()
            session.execute(
                delete(ConversationMessage).where(
                    ConversationMessage.tenant_id == self.tenant_id,
                    ConversationMessage.session_id == self.session_id,
                    ConversationMessage.created_at
                    < now - timedelta(days=self.retention_days),
                )
            )
            session.add(
                ConversationMessage(
                    tenant_id=self.tenant_id,
                    session_id=self.session_id,
                    role=role,
                    content=content,
                    created_at=now,
                )
            )
            session.commit()
            self._compact(session)
            session.commit()

    def get_messages(
        self,
        tenant_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Dict[str, Any]]:
        if tenant_id is not None and tenant_id != self.tenant_id:
            raise PermissionError("Conversation memory is tenant-scoped")
        maximum = max(1, min(limit or self.capacity, self.capacity))
        with self.session_factory() as session:
            rows = session.scalars(
                select(ConversationMessage)
                .where(
                    ConversationMessage.tenant_id == self.tenant_id,
                    ConversationMessage.session_id == self.session_id,
                    ConversationMessage.created_at
                    >= _now() - timedelta(days=self.retention_days),
                )
                .order_by(ConversationMessage.created_at.desc())
                .limit(maximum)
            ).all()
            return [
                {"role": row.role, "content": row.content, "ts": row.created_at.timestamp()}
                for row in reversed(rows)
            ]

    def get_conversation(
        self, max_chars: int = 2000, tenant_id: Optional[str] = None
    ) -> str:
        history = "\n".join(
            f"[{message['role']}] {message['content']}"
            for message in self.get_messages(tenant_id=tenant_id)
        )
        return history[-max_chars:] if max_chars > 0 else ""

    def retrieve(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Dict[str, Any]]:
        if tenant_id is not None and tenant_id != self.tenant_id:
            raise PermissionError("Conversation memory is tenant-scoped")
        maximum = max(1, min(limit or self.capacity, self.capacity))
        if self.retrieval_hook is None:
            return self.get_messages(tenant_id=tenant_id, limit=maximum)
        results = self.retrieval_hook(
            query, self.tenant_id, self.session_id, maximum
        )
        authorized = []
        for item in results[:maximum]:
            if (
                item.get("tenant_id") != self.tenant_id
                or item.get("session_id") != self.session_id
            ):
                raise PermissionError("Retrieval hook returned unauthorized memory")
            authorized.append(item)
        return authorized

    def clear(self, tenant_id: Optional[str] = None) -> None:
        if tenant_id is not None and tenant_id != self.tenant_id:
            raise PermissionError("Conversation memory is tenant-scoped")
        with self.session_factory() as session:
            session.execute(
                delete(ConversationMessage).where(
                    ConversationMessage.tenant_id == self.tenant_id,
                    ConversationMessage.session_id == self.session_id,
                )
            )
            session.commit()

    def _compact(self, session: Any) -> None:
        rows = session.scalars(
            select(ConversationMessage)
            .where(
                ConversationMessage.tenant_id == self.tenant_id,
                ConversationMessage.session_id == self.session_id,
            )
            .order_by(ConversationMessage.created_at.desc())
        ).all()
        if len(rows) <= self.capacity and not any(
            row.role == "summary" for row in rows
        ):
            return

        recent = [
            row for row in rows if row.role != "summary"
        ][: max(0, self.capacity - 1)]
        recent_ids = {row.id for row in recent}
        excess = [row for row in rows if row.id not in recent_ids]
        compacted_text = "\n".join(
            f"[{row.role}] {row.content}"
            for row in reversed(excess)
            if row.role != "summary"
        )
        prior_summaries = [row.content for row in excess if row.role == "summary"]
        if prior_summaries:
            compacted_text = "\n".join(prior_summaries + [compacted_text])
        summary = (
            self.summary_fn(compacted_text)
            if self.summary_fn
            else compacted_text[-1000:]
        )
        session.execute(
            delete(ConversationMessage).where(
                ConversationMessage.id.in_([row.id for row in excess])
            )
        )
        summary_time = (
            min((row.created_at for row in recent), default=_now())
            - timedelta(microseconds=1)
        )
        session.add(
            ConversationMessage(
                tenant_id=self.tenant_id,
                session_id=self.session_id,
                role="summary",
                content=summary[:2000],
                created_at=summary_time,
            )
        )
