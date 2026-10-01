"""Data lake abstraction with local filesystem and S3 backends."""
from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import List, Optional


class DataLake(abc.ABC):
    name: str = "data-lake"

    @abc.abstractmethod
    def put(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, key: str) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    def list(self, prefix: str = "") -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        try:
            self.get(key)
            return True
        except (FileNotFoundError, KeyError):
            return False


class LocalDataLake(DataLake):
    """Filesystem-backed data lake rooted at ``base_path``.

    Serves as the default/fallback implementation and is useful for tests
    and single-node deployments. Cloud-backed lakes (S3/GCS/ADLS) share the
    same interface so callers can switch backends via configuration.
    """

    name = "local"

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        resolved = (self.base_path / key).resolve()
        if self.base_path.resolve() not in resolved.parents and resolved != self.base_path.resolve():
            raise ValueError(f"key '{key}' escapes data lake root")
        return resolved

    def put(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def list(self, prefix: str = "") -> List[str]:
        results = []
        for path in self.base_path.rglob("*"):
            if path.is_file():
                rel = str(path.relative_to(self.base_path))
                if rel.startswith(prefix):
                    results.append(rel)
        return sorted(results)

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()


class S3DataLake(DataLake):
    """S3-backed data lake. Requires ``boto3`` at call time."""

    name = "s3"

    def __init__(self, bucket: str, prefix: str = "", region_name: Optional[str] = None):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region_name = region_name or os.environ.get("AWS_REGION")

    def _client(self):
        import boto3

        return boto3.client("s3", region_name=self.region_name)

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes) -> None:
        self._client().put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)

    def get(self, key: str) -> bytes:
        response = self._client().get_object(Bucket=self.bucket, Key=self._full_key(key))
        return response["Body"].read()

    def list(self, prefix: str = "") -> List[str]:
        client = self._client()
        full_prefix = self._full_key(prefix)
        paginator = client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self.prefix:
                    key = key[len(self.prefix) + 1 :]
                keys.append(key)
        return sorted(keys)

    def delete(self, key: str) -> None:
        self._client().delete_object(Bucket=self.bucket, Key=self._full_key(key))
