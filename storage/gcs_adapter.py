#!/usr/bin/env python3
"""
GCS adapter implementing StorageClient.

Requires google-cloud-storage installed in the environment to use.
Uses GOOGLE_APPLICATION_CREDENTIALS for auth (standard GCP ADC).
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
from .abstract import StorageClient

try:
    from google.cloud import storage as gcs_storage  # type: ignore
except Exception:  # pragma: no cover - optional dep
    gcs_storage = None

class GCSStorageAdapter(StorageClient):
    def __init__(self, bucket: str):
        if gcs_storage is None:
            raise RuntimeError("google-cloud-storage is required for GCSStorageAdapter")
        self.client = gcs_storage.Client()
        self.bucket_name = bucket
        self.bucket = self.client.bucket(bucket)
        # bucket existence is infra-managed; best-effort
        if not self.bucket.exists():
            try:
                self.client.create_bucket(self.bucket_name)
            except Exception:
                pass

    def upload(self, local_path: Path, remote_path: str, content_type: Optional[str] = None) -> None:
        blob = self.bucket.blob(remote_path)
        if content_type:
            blob.content_type = content_type
        blob.upload_from_filename(str(local_path))

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob = self.bucket.blob(remote_path)
        blob.download_to_filename(str(local_path))

    def exists(self, remote_path: str) -> bool:
        blob = self.bucket.blob(remote_path)
        return blob.exists()

    def list(self, prefix: str) -> Iterable[str]:
        for blob in self.client.list_blobs(self.bucket_name, prefix=prefix):
            yield blob.name

    def get_presigned_url(self, remote_path: str, expires_in: int = 3600) -> str:
        # GCS signed URLs require credentials; attempt to create one
        blob = self.bucket.blob(remote_path)
        return blob.generate_signed_url(expiration=expires_in)
