#!/usr/bin/env python
"""
Repo hygiene filename checks.

- Ensures mlflow compose filename consistency: accepts either docker/docker-compose-mlflow.yml or docker/docker-compose.mlflow.yml but flags if none or both are present.
- Optional additional checks can be added to enforce naming conventions.

Usage:
  python scripts/check_filenames.py
Exit non-zero on failure.
"""
from pathlib import Path
import sys

def main():
    repo_root = Path(".")
    f1 = repo_root / "docker" / "docker-compose-mlflow.yml"
    f2 = repo_root / "docker" / "docker-compose.mlflow.yml"
    has1 = f1.exists()
    has2 = f2.exists()
    if not has1 and not has2:
        print("ERROR: No MLflow docker compose file found. Expected one of:")
        print("  docker/docker-compose-mlflow.yml")
        print("  docker/docker-compose.mlflow.yml")
        return 2
    if has1 and has2:
        print("WARNING: Both compose filenames present. Please keep a single canonical name to avoid confusion.")
    else:
        print("OK: Found compose file:", f1 if has1 else f2)
    return 0

if __name__ == "__main__":
    sys.exit(main())
