#!/usr/bin/env python3
"""
AWS S3 adapter using boto3.

This adapter uses standard boto3 configuration (env vars, shared credentials).
Pass endpoint_url via config for S3-compatible providers if needed.
"""
from __future__ import annotations
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Iterable, Optional
from .abstract import StorageClient

class S3StorageAdapter(StorageClient):
    def __init__(self, bucket: str, region: Optional[str] = None, endpoint_url: Optional[str] = None, access_key: Optional[str] = None, secret_key: Optional[str] = None):
        session_kwargs = {}
        if access_key and secret_key:
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key
        self.s3 = boto3.resource("s3", region_name=region, endpoint_url=endpoint_url, **session_kwargs)
        self.client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url, **session_kwargs)
        self.bucket = bucket
        # ensure bucket exists (best-effort)
        try:
            self.s3.meta.client.head_bucket(Bucket=bucket)
        except ClientError:
            # bucket may not exist or we don't have permissions; creation should be infra-managed
            pass

    def upload(self, local_path: Path, remote_path: str, content_type: Optional[str] = None) -> None:
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
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
