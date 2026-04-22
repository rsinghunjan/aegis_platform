from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote_plus


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class DbSettings:
    database_url: str
    embedding_dim: int


def _build_db_url_from_split() -> str:
    host = _env("DB_HOST", "postgres")
    port = _env("DB_PORT", "5432")
    name = _env("DB_NAME", "aegis")
    user = _env("DB_USER", "aegis")
    password = _env("DB_PASSWORD", "aegis")
    sslmode = _env("DB_SSLMODE", None)
    driver = _env("DB_DRIVER", "psycopg")  # psycopg (3) default, psycopg2 fallback

    user_q = quote_plus(user or "")
    pw_q = quote_plus(password or "")

    scheme = "postgresql"
    if driver == "psycopg2":
        scheme = "postgresql+psycopg2"
    else:
        scheme = "postgresql+psycopg"

    url = f"{scheme}://{user_q}:{pw_q}@{host}:{port}/{name}"
    if sslmode:
        url += f"?sslmode={quote_plus(sslmode)}"
    return url


def get_db_settings() -> DbSettings:
    raw_url = _env("DATABASE_URL", None)
    if raw_url:
        # Normalize plain postgresql:// to psycopg3 by default
        if raw_url.startswith("postgresql://"):
            raw_url = raw_url.replace("postgresql://", "postgresql+psycopg://", 1)
        database_url = raw_url
    else:
        database_url = _build_db_url_from_split()

    embedding_dim = int(_env("AEGIS_EMBEDDING_DIM", "1536") or "1536")
    return DbSettings(database_url=database_url, embedding_dim=embedding_dim)
