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
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
#!/usr/bin/env python3
"""
SQLAlchemy engine factory and Session maker.

- Reads DATABASE_URL from env or config.cfg (12-factor)
- Provides get_engine(), get_session(), and contextmanager helper
- Use this everywhere for DB access so switching DB providers is a config change only.
"""
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

DB_URL = os.environ.get("DATABASE_URL")  # e.g. postgresql://user:pass@host:5432/dbname

if not DB_URL:
    # fallback to config module if available
    try:
        from aegis_multimodal_ai_system import config
        DB_URL = getattr(config.cfg, "DATABASE_URL", None)
    except Exception:
        DB_URL = None

if DB_URL is None:
    # default to local sqlite for developer convenience
    DB_URL = "sqlite:///./aegis_dev.db"

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None

def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(DB_URL, future=True, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    return _engine

def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        get_engine()
    return _SessionLocal()

@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
aegis_multimodal_ai_system/db/engine.py
