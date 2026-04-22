from __future__ import annotations

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aegis_db.settings import get_db_settings

log = logging.getLogger(__name__)

_ENGINE = None
_SessionLocal = None


def _try_import_psycopg3() -> bool:
    try:
        import psycopg  # noqa: F401
        return True
    except Exception:
        return False


def _try_import_psycopg2() -> bool:
    try:
        import psycopg2  # noqa: F401
        return True
    except Exception:
        return False


def get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    settings = get_db_settings()
    url = settings.database_url

    # If URL requests psycopg3 but psycopg isn't installed, fallback to psycopg2.
    if "postgresql+psycopg://" in url and not _try_import_psycopg3():
        if _try_import_psycopg2():
            log.warning("psycopg3 not available; falling back to psycopg2")
            url = url.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)

    _ENGINE = create_engine(url, pool_pre_ping=True)
    return _ENGINE


def get_sessionmaker():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal
