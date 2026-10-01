#!/usr/bin/env python3
"""
Storage abstraction for Aegis.

Define a StorageClient interface that the rest of the codebase uses.
Implementations (S3, MinIO, GCS, Azure) should implement this interface.
"""
from __future__ import annotations
from typing import Protocol, Iterable, Optional
from pathlib import Path

class StorageClient(Protocol):
    """
    Minimal storage client interface.

    - upload(local_path, remote_path, content_type=None)
    - download(remote_path, local_path)
    - exists(remote_path) -> bool
    - list(prefix) -> Iterable[str]
    - get_presigned_url(remote_path, expires_in=3600) -> str
    """
    def upload(self, local_path: Path, remote_path: str, content_type: Optional[str] = None) -> None:
        ...

    def download(self, remote_path: str, local_path: Path) -> None:
        ...

    def exists(self, remote_path: str) -> bool:
        ...

    def list(self, prefix: str) -> Iterable[str]:
        ...

    def get_presigned_url(self, remote_path: str, expires_in: int = 3600) -> str:
        ...
