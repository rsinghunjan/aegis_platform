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
#!/usr/bin/env python3
"""
Restore a Postgres dump from object store:

Usage:
  DATABASE_URL=postgresql://user:pass@host:5432/db OBJECT_STORE_TYPE=s3 OBJECT_STORE_BUCKET=... python3 scripts/backup/restore_db.py s3://bucket/path/to/backup.sql.gz

Notes:
- This script will download the file to /tmp and run psql to restore. Ensure you have appropriate permissions and psql installed.
"""
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from aegis_multimodal_ai_system.storage.factory import create_storage_client
from urllib.parse import urlparse

def main():
    if len(sys.argv) < 2:
        print("Usage: restore_db.py <artifact_uri>", file=sys.stderr)
        sys.exit(2)
    artifact_uri = sys.argv[1]
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")
    bucket = os.environ.get("OBJECT_STORE_BUCKET")
    if not bucket:
        raise RuntimeError("OBJECT_STORE_BUCKET not set")

    # Expect s3://bucket/key or just key; use registry.loader.resolve_artifact_uri for robust parsing if desired
    if artifact_uri.startswith("s3://"):
        parsed = artifact_uri[len("s3://"):]
        parts = parsed.split("/", 1)
        b = parts[0]; key = parts[1]
    else:
        # fallback to key under bucket
        b = bucket; key = artifact_uri.lstrip("/")
    client = create_storage_client(bucket=b)
    local = Path("/tmp") / Path(key).name
    client.download(key, local)
    # decompress & restore
    import gzip
    with gzip.open(str(local), "rb") as gz:
        # psql restore
        p = subprocess.Popen(["psql", db_url], stdin=subprocess.PIPE)
        for chunk in iter(lambda: gz.read(8192), b""):
            p.stdin.write(chunk)
        p.stdin.close()
        p.wait()
    print("Restore completed from", artifact_uri)

if __name__ == "__main__":
    main()
scripts/backup/restore_db.py
