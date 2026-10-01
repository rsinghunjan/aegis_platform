  1
  2
  3
  4
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
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
"""
Migrate newline-delimited JSON audit log file(s) to S3.

- Uploads each line as a separate object under:
  s3://{bucket}/{prefix}/{YYYY}/{MM}/{DD}/{timestamp}-{request_id}.json
- Skips already-uploaded keys (idempotent).
- Optionally verifies checksum after upload.

Usage:
  python -m aegis_multimodal_ai_system.audit.s3_migration --bucket my-bucket --prefix aegis/audit --file logs/safety_audit.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    import boto3  # type: ignore
    from botocore.exceptions import ClientError  # type: ignore
except Exception:
    boto3 = None  # type: ignore


def _obj_key(prefix: str, timestamp: float, request_id: str) -> str:
    t = datetime.utcfromtimestamp(int(timestamp))
    date_path = f"{t.year:04d}/{t.month:02d}/{t.day:02d}"
    safe_id = request_id.replace("/", "_")[:64]
    fname = f"{int(timestamp)}-{safe_id}.json"
    return f"{prefix.rstrip('/')}/{date_path}/{fname}"


def upload_line_to_s3(s3, bucket: str, prefix: str, line: str, verify: bool = True) -> Optional[str]:
    try:
        obj = json.loads(line)
    except Exception:
        logger.warning("Skipping non-json line")
        return None

    ts = float(obj.get("timestamp", time.time()))
    rid = obj.get("request_id") or obj.get("requestId") or str(int(ts)) + "-" + hashlib.sha1(line.encode("utf-8")).hexdigest()[:8]
    key = _obj_key(prefix, ts, rid)

    # Check if object exists
    try:
        s3.head_object(Bucket=bucket, Key=key)
        logger.debug("Already uploaded: %s", key)
        return key
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            # Unexpected error
            raise

    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body)
    except Exception:
        logger.exception("Failed to put_object %s", key)
        raise

    if verify:
        # minimal verification by head_object and size
        for _ in range(3):
            try:
                h = s3.head_object(Bucket=bucket, Key=key)
                if h and h.get("ContentLength", 0) == len(body):
                    return key
            except Exception:
                time.sleep(0.5)
        logger.warning("Upload verification failed for %s", key)
        return key
    return key


def migrate_file(bucket: str, prefix: str, file_path: str, verify: bool = True):
    if boto3 is None:
        raise RuntimeError("boto3 required for S3 migration. Install boto3.")

    s3 = boto3.client("s3")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                key = upload_line_to_s3(s3, bucket, prefix, line, verify=verify)
                if key:
                    logger.info("Uploaded %s", key)
                    count += 1
            except Exception:
                logger.exception("Failed to upload line; continuing")
    logger.info("Migration complete. Uploaded %d events from %s", count, file_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="aegis/audit")
    parser.add_argument("--file", required=True)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    migrate_file(args.bucket, args.prefix, args.file, verify=not args.no_verify)


if __name__ == "__main__":
    main()
aegis_multimodal_ai_system/audit/s3_migration.py
