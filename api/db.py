"""
SQLAlchemy DB setup for Aegis API.

- Exposes `engine`, `SessionLocal`, and `Base` for models and Alembic.
- Reads DATABASE_URL env var (default for local dev).
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://postgres:password@localhost:5432/aegis")

# echo=True for debugging; set False in production
engine = create_engine(DATABASE_URL, echo=False, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()
