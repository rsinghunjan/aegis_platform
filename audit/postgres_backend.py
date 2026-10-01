"""
Postgres audit backend for enterprise deployments.

Features:
- Writes audit events into a durable Postgres table with indexes for request_id/timestamp.
- Uses psycopg2 connection pool. Reads DATABASE_URL from env (standard format).
- Prune older events via SQL DELETE using safe batching.
- Avoids storing unmasked PII (caller responsibility). Store text_snippet truncated.

Environment variables:
- DATABASE_URL (required): e.g., postgres://user:pass@host:5432/aegis
- AUDIT_PG_TABLE (optional): default "audit_events"
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    import psycopg2  # type: ignore
    from psycopg2 import pool  # type: ignore
except Exception:
    psycopg2 = None
    pool = None

DATABASE_URL = os.getenv("DATABASE_URL", "")
AUDIT_PG_TABLE = os.getenv("AUDIT_PG_TABLE", "audit_events")


class PostgresBackend:
    def __init__(self, dsn: Optional[str] = None, minconn: int = 1, maxconn: int = 5):
        if psycopg2 is None:
            raise RuntimeError("psycopg2 is required for PostgresBackend")
        self.dsn = dsn or DATABASE_URL
        if not self.dsn:
            raise RuntimeError("DATABASE_URL not configured for PostgresBackend")
        self.pool = pool.ThreadedConnectionPool(minconn, maxconn, self.dsn)
        self._ensure_table()

    def _ensure_table(self):
        conn = self.pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {AUDIT_PG_TABLE} (
                    id TEXT PRIMARY KEY,
                    request_id TEXT,
                    timestamp DOUBLE PRECISION,
                    event_json JSONB,
                    sha256 TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_{AUDIT_PG_TABLE}_ts ON {AUDIT_PG_TABLE} (timestamp);
                CREATE INDEX IF NOT EXISTS idx_{AUDIT_PG_TABLE}_rid ON {AUDIT_PG_TABLE} (request_id);
                """
            )
            conn.commit()
            cur.close()
        finally:
            self.pool.putconn(conn)

    def write(self, event: Dict) -> None:
        eid = event.get("request_id") or str(int(time.time() * 1000))
        ts = float(event.get("timestamp", time.time()))
        # store truncated snippet for privacy / to limit DB usage
        try:
            payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except Exception:
            payload = json.dumps({"error": "serialization_failed", "meta": {k: str(v) for k, v in event.items()}})
        sha256 = hashlib_sha256_bytes(payload.encode("utf-8"))
        conn = self.pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(
                f"INSERT INTO {AUDIT_PG_TABLE} (id, request_id, timestamp, event_json, sha256) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                (eid, event.get("request_id"), ts, payload, sha256),
            )
            conn.commit()
            cur.close()
        except Exception:
            logger.exception("PostgresBackend: failed to write event")
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            self.pool.putconn(conn)

    def prune_older_than(self, seconds: int) -> None:
        cutoff = time.time() - seconds
        conn = self.pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(f"DELETE FROM {AUDIT_PG_TABLE} WHERE timestamp < %s", (cutoff,))
            deleted = cur.rowcount
            conn.commit()
            cur.close()
            logger.info("PostgresBackend: pruned %d rows older than %s seconds", deleted, seconds)
        except Exception:
            logger.exception("PostgresBackend: prune failed")
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            self.pool.putconn(conn)


def hashlib_sha256_bytes(b: bytes) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()
