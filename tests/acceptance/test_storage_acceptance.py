#!/usr/bin/env python3
"""
Acceptance tests that validate the storage adapters against the MinIO dev stack.

Expect environment variables (set by scripts/local_dev_setup.sh or CI step):
  OBJECT_STORE_TYPE=minio
  OBJECT_STORE_ENDPOINT=http://localhost:9000
  OBJECT_STORE_ACCESS_KEY=minioadmin
  OBJECT_STORE_SECRET_KEY=minioadmin
  OBJECT_STORE_BUCKET=aegis-models
"""
import os
import tempfile
from pathlib import Path
import uuid

from aegis_multimodal_ai_system.storage.factory import create_storage_client

def test_minio_upload_download_roundtrip():
    # Create a small temp file
    tmp = tempfile.mkdtemp()
    data_file = Path(tmp) / "hello.txt"
    data_file.write_text("hello aegis", encoding="utf-8")

    client = create_storage_client()
    key = f"acceptance/{uuid.uuid4().hex}/hello.txt"
    # upload
    client.upload(data_file, key)
    # exists
    assert client.exists(key)
    # download to new location
    dl = Path(tmp) / "hello.download.txt"
    client.download(key, dl)
    assert dl.exists()
    assert dl.read_text(encoding="utf-8") == "hello aegis"
