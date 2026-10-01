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
#!/usr/bin/env python3
"""
Postgres backup script that:
- reads DATABASE_URL from env
- dumps DB to a local file using pg_dump
- uploads the dump to object-store via StorageClient
- provides a timestamped filename and prints the upload URI

Usage:
  DATABASE_URL=... OBJECT_STORE_TYPE=s3 OBJECT_STORE_BUCKET=... python3 scripts/backup/backup_db.py
"""
from __future__ import annotations
import os
import subprocess
import time
from pathlib import Path
from aegis_multimodal_ai_system.storage.factory import create_storage_client

def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")
    bucket = os.environ.get("OBJECT_STORE_BUCKET")
    if not bucket:
        raise RuntimeError("OBJECT_STORE_BUCKET not set")
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_name = f"db-backups/aegis-db-{ts}.sql.gz"
    local = Path("/tmp") / f"aegis-db-{ts}.sql.gz"
