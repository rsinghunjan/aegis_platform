#!/usr/bin/env python3
"""
Presigned / PAR semantics test.

This test tries to create a storage client via factory and call get_presigned_url() for a small object.
It skips if no provider is configured or if the provider does not support presigned/PAR in the adapter.
Run in CI only if OBJECT_STORE_* secrets are provided.
"""
import os
import tempfile
from pathlib import Path
import pytest

from storage.factory import create_storage_client

@pytest.mark.skipif(not os.environ.get("OBJECT_STORE_TYPE"), reason="No OBJECT_STORE_TYPE configured")
def test_get_presigned_url_roundtrip():
    client = create_storage_client()
    # create a tiny file and upload
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "tiny.txt"
    p.write_text("presign test", encoding="utf-8")
    key = "tests/presign/tiny.txt"
    client.upload(p, key)
    # try get presigned url (adapter may raise if unsupported)
    try:
        url = client.get_presigned_url(key, expires_in=60)
    except NotImplementedError:
        pytest.skip("Presigned URL not implemented for this provider in adapter")
    assert url and isinstance(url, str)
