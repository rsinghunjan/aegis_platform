#!/usr/bin/env python3
"""
MinIO / generic S3-compatible adapter.

MinIO speaks the S3 API so we reuse boto3 but allow specifying endpoint_url explicitly.
This adapter is intended for local dev (docker-compose MinIO) or any S3-compatible store.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
from .abstract import StorageClient
import boto3
from botocore.exceptions import ClientError

class MinIOStorageAdapter(StorageClient):
    def __init__(self, bucket: str, endpoint_url: str, access_key: str, secret_key: str, region: Optional[str] = "us-east-1"):
        self.bucket = bucket
        session_kwargs = {"aws_access_key_id": access_key, "aws_secret_access_key": secret_key}
        self.s3 = boto3.resource("s3", region_name=region, endpoint_url=endpoint_url, **session_kwargs)
        self.client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url, **session_kwargs)
        # create bucket if missing (best-effort for local dev)
        try:
            self.client.head_bucket(Bucket=bucket)
        except ClientError:
            try:
                # create bucket
                self.client.create_bucket(Bucket=bucket)
            except Exception:
                pass

    def upload(self, local_path: Path, remote_path: str, content_type: Optional[str] = None) -> None:
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        # MinIO may not support ACLs; use simple upload
        self.client.upload_file(str(local_path), self.bucket, remote_path, ExtraArgs=extra_args)

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, remote_path, str(local_path))

    def exists(self, remote_path: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except ClientError:
            return False

    def list(self, prefix: str) -> Iterable[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def get_presigned_url(self, remote_path: str, expires_in: int = 3600) -> str:
        return self.client.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": remote_path}, ExpiresIn=expires_in)
