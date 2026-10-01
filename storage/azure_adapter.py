#!/usr/bin/env python3
"""
Azure Blob Storage adapter implementing StorageClient.

Requires azure-storage-blob package. Authentication can be via
AZURE_STORAGE_CONNECTION_STRING or environment credentials supported by azure SDK.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
from .abstract import StorageClient

try:
    from azure.storage.blob import BlobServiceClient  # type: ignore
except Exception:  # pragma: no cover - optional dep
    BlobServiceClient = None

class AzureBlobStorageAdapter(StorageClient):
    def __init__(self, container: str, connection_string: Optional[str] = None):
        if BlobServiceClient is None:
            raise RuntimeError("azure-storage-blob is required for AzureBlobStorageAdapter")
        if connection_string:
            self.client = BlobServiceClient.from_connection_string(connection_string)
        else:
            # will use default Azure auth chain (env or MSI)
            self.client = BlobServiceClient(account_url=None)  # let SDK resolve via env
        self.container_name = container
        self.container = self.client.get_container_client(container)
        try:
            self.container.get_container_properties()
        except Exception:
            try:
                self.client.create_container(container)
            except Exception:
                pass

    def upload(self, local_path: Path, remote_path: str, content_type: Optional[str] = None) -> None:
        blob_client = self.container.get_blob_client(remote_path)
        with open(local_path, "rb") as fh:
            blob_client.upload_blob(fh, overwrite=True, content_settings=None)

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob_client = self.container.get_blob_client(remote_path)
        with open(local_path, "wb") as fh:
            stream = blob_client.download_blob()
            fh.write(stream.readall())

    def exists(self, remote_path: str) -> bool:
        blob_client = self.container.get_blob_client(remote_path)
        try:
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False

    def list(self, prefix: str) -> Iterable[str]:
        for blob in self.container.list_blobs(name_starts_with=prefix):
            yield blob.name

    def get_presigned_url(self, remote_path: str, expires_in: int = 3600) -> str:
        # SAS generation is more involved and requires account key or azure.identity; for now raise
        raise RuntimeError("Presigned URL generation for Azure not implemented in adapter")
