from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config


def upgrade_database(database_url: str) -> None:
    """Upgrade a database through the repository's Alembic revision chain."""
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    config.attributes["database_url"] = database_url
    command.upgrade(config, "head")
