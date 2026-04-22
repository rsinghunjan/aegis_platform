from __future__ import annotations

import sys
from alembic import command
from alembic.config import Config


def main() -> int:
    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
