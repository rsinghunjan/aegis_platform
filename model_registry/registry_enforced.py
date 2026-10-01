"""
ModelRegistry with enforced artifact signature verification.

Drop-in replacement for the existing registry module if you want to require signatures
in CI and at runtime.

Behavior:
- If MODEL_SIGNING_REQUIRED is "true", the registry will refuse to register or load any
  model artifact that does not have a valid signature verified against MODEL_SIGNING_PUBKEY.
- If MODEL_SIGNING_REQUIRED is "false" (default), registry will verify when signature is present
  but will allow unsigned models (useful for dev).
- Signature verification uses model_registry.signing.verify_rsa_signature (RSA PKCS1v15+SHA256).
- Also verifies sha256 checksum when provided.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from .signing import compute_sha256, load_public_key_bytes, verify_rsa_signature

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODEL_CACHE_DIR", "models/cache"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIGNING_PUBKEY = os.getenv("MODEL_SIGNING_PUBKEY", "")  # PEM text or path
MODEL_SIGNING_REQUIRED = os.getenv("MODEL_SIGNING_REQUIRED", "false").lower() in ("1", "true", "yes")

# Optional metric integration: increment a counter on signature failures if you have metrics
try:
    from ..metrics.metrics import AEGIS_MODEL_SIGNATURE_ERRORS
except Exception:
    AEGIS_MODEL_SIGNATURE_ERRORS = None


class ModelMetadata:
    def __init__(self, name: str, version: str, path: str, checksum: Optional[str] = None, loaded_at: Optional[float] = None):
        self.name = name
        self.version = version
        self.path = path
        self.checksum = checksum
        self.loaded_at = loaded_at or time.time()

    def as_dict(self):
        return {"name": self.name, "version": self.version, "path": str(self.path), "checksum": self.checksum, "loaded_at": self.loaded_at}


class ModelRegistry:
    def __init__(self, s3_bucket: Optional[str] = None, s3_prefix: Optional[str] = None):
        self._lock = threading.RLock()
        self._models: Dict[str, ModelMetadata] = {}
        self.s3_bucket = s3_bucket or os.getenv("MODEL_REGISTRY_S3_BUCKET", "")
        self.s3_prefix = s3_prefix or os.getenv("MODEL_REGISTRY_S3_PREFIX", "models")

    def _key(self, name: str, version: str) -> str:
        return f"{name}:{version}"

    def _require_and_verify(self, artifact_path: Path, checksum: Optional[str] = None, sig_path: Optional[Path] = None):
        # verify checksum if provided
        if checksum:
            computed = compute_sha256(artifact_path)
            if computed.lower() != checksum.lower():
                raise RuntimeError(f"Checksum mismatch for {artifact_path}: expected {checksum}, got {computed}")

        # if signature file provided, verify it
        if sig_path:
            pub = load_public_key_bytes(MODEL_SIGNING_PUBKEY)
            if not pub:
                raise RuntimeError("MODEL_SIGNING_PUBKEY not configured but signature present")
            ok = verify_rsa_signature(pub, str(artifact_path), str(sig_path))
            if not ok:
                if AEGIS_MODEL_SIGNATURE_ERRORS is not None:
                    try:
                        AEGIS_MODEL_SIGNATURE_ERRORS.labels(model_name=os.path.basename(str(artifact_path)), model_version="unknown").inc()
                    except Exception:
                        pass
                raise RuntimeError(f"Signature verification failed for {artifact_path}")
            logger.info("Signature verified for artifact %s", artifact_path)
            return True

        # if no signature and signatures are required, fail
        if MODEL_SIGNING_REQUIRED:
            raise RuntimeError(f"Model signing is required (MODEL_SIGNING_REQUIRED=true) but {artifact_path} has no signature")
        return False

    def register_local(self, name: str, version: str, local_path: str, checksum: Optional[str] = None, sig_path: Optional[str] = None) -> ModelMetadata:
        p = Path(local_path)
        if not p.exists():
            raise FileNotFoundError(f"Model path not found: {local_path}")

        # perform verification / enforcement
        self._require_and_verify(p, checksum=checksum, sig_path=Path(sig_path) if sig_path else None)

        meta = ModelMetadata(name, version, str(p.resolve()), checksum, loaded_at=time.time())
        with self._lock:
            self._models[self._key(name, version)] = meta
        logger.info("Registered local model %s:%s at %s", name, version, meta.path)
        return meta

    def fetch_from_s3(self, name: str, version: str, s3_key: Optional[str] = None, checksum: Optional[str] = None, sig_s3_key: Optional[str] = None) -> ModelMetadata:
        """
        Download artifact from S3 and require verification if configured.
        sig_s3_key: optional s3 key for detached signature (same directory).
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
        except Exception:
            raise RuntimeError("boto3 required for S3 fetch")

        key = s3_key or f"{self.s3_prefix}/{name}/{version}/model.tar"
        local_dir = MODELS_DIR / name / version
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / Path(key).name

        s3 = boto3.client("s3")
        try:
            s3.download_file(self.s3_bucket, key, str(local_path))
        except ClientError as e:
            logger.exception("S3 download failed: %s", e)
            raise

        # optionally download signature
        local_sig_path = None
        if sig_s3_key:
            local_sig_path = local_dir / Path(sig_s3_key).name
            try:
                s3.download_file(self.s3_bucket, sig_s3_key, str(local_sig_path))
            except ClientError:
                logger.exception("Failed to download signature from s3: %s", sig_s3_key)
                if MODEL_SIGNING_REQUIRED:
                    raise

        # verify as enforced
        self._require_and_verify(local_path, checksum=checksum, sig_path=local_sig_path)

        meta = ModelMetadata(name=name, version=version, path=str(local_path.resolve()), checksum=checksum, loaded_at=time.time())
        with self._lock:
            self._models[self._key(name, version)] = meta
        logger.info("Fetched and registered model %s:%s from s3 key %s", name, version, key)
        return meta

    def get(self, name: str, version: str) -> Optional[ModelMetadata]:
        with self._lock:
            return self._models.get(self._key(name, version))

    def list_models(self) -> Dict[str, Dict]:
        with self._lock:
            return {k: v.as_dict() for k, v in self._models.items()}

    def health(self) -> Dict:
        with self._lock:
            models = {k: v.as_dict() for k, v in self._models.items()}
        stat = MODELS_DIR.stat()
        return {"model_count": len(models), "models": models, "models_dir": str(MODELS_DIR), "models_dir_mtime": stat.st_mtime, "checked_at": time.time()}
