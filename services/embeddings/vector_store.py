"""Vector database abstraction with pluggable backends.

The default :class:`InMemoryVectorStore` has no external dependencies and
is suitable for tests and small deployments. :class:`PGVectorStore` shows
the shape of a production backend (lazy psycopg2/pgvector import); other
backends (Milvus, Pinecone, Weaviate, Qdrant) can be added by implementing
the same :class:`VectorStore` interface.
"""
from __future__ import annotations

import abc
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VectorRecord:
    id: str
    vector: List[float]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.ttl_seconds is None:
            return False
        return (now or time.time()) - self.created_at > self.ttl_seconds


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (norm_a * norm_b)


class VectorStore(abc.ABC):
    name: str = "vector-store"

    @abc.abstractmethod
    def upsert(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        ttl_seconds: Optional[float] = None,
    ) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def query(self, vector: List[float], top_k: int = 5) -> List[Tuple[VectorRecord, float]]:
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, ids: List[str]) -> None:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """Simple in-process vector store using brute-force cosine similarity.

    Suitable for tests, small corpora, and as a fallback when no external
    vector database (Milvus/Pinecone/Weaviate/Qdrant/pgvector) is
    configured.
    """

    name = "in-memory"

    def __init__(self) -> None:
        self._records: Dict[str, VectorRecord] = {}

    def upsert(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        ttl_seconds: Optional[float] = None,
    ) -> List[str]:
        if len(vectors) != len(texts):
            raise ValueError("vectors and texts must be the same length")
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        for vector, text, metadata, record_id in zip(vectors, texts, metadatas, ids):
            self._records[record_id] = VectorRecord(
                id=record_id,
                vector=vector,
                text=text,
                metadata=metadata,
                ttl_seconds=ttl_seconds,
            )
        return ids

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [rid for rid, rec in self._records.items() if rec.is_expired(now)]
        for rid in expired:
            del self._records[rid]

    def query(self, vector: List[float], top_k: int = 5) -> List[Tuple[VectorRecord, float]]:
        self._purge_expired()
        scored = [
            (record, _cosine_similarity(vector, record.vector))
            for record in self._records.values()
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def delete(self, ids: List[str]) -> None:
        for record_id in ids:
            self._records.pop(record_id, None)

    def __len__(self) -> int:
        return len(self._records)


class PGVectorStore(VectorStore):
    """Vector store backed by PostgreSQL + the pgvector extension.

    Connection/SDK usage is deferred to first use so importing this module
    doesn't require ``psycopg2`` to be installed.
    """

    name = "pgvector"

    def __init__(self, dsn: str, table: str = "aegis_embeddings"):
        self.dsn = dsn
        self.table = table

    def _connect(self):
        import psycopg2

        return psycopg2.connect(self.dsn)

    def upsert(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        ttl_seconds: Optional[float] = None,
    ) -> List[str]:
        import json

        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                for record_id, vector, text, metadata in zip(ids, vectors, texts, metadatas):
                    cur.execute(
                        f"""
                        INSERT INTO {self.table} (id, embedding, text, metadata)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata
                        """,
                        (record_id, vector, text, json.dumps(metadata)),
                    )
            conn.commit()
        finally:
            conn.close()
        return ids

    def query(self, vector: List[float], top_k: int = 5) -> List[Tuple[VectorRecord, float]]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, embedding, text, metadata, 1 - (embedding <=> %s) AS score
                    FROM {self.table}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (vector, vector, top_k),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        results = []
        for row in rows:
            record = VectorRecord(id=row[0], vector=row[1], text=row[2], metadata=row[3] or {})
            results.append((record, float(row[4])))
        return results

    def delete(self, ids: List[str]) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table} WHERE id = ANY(%s)", (ids,))
            conn.commit()
        finally:
            conn.close()
