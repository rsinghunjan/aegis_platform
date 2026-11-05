"""RSA (PKCS#1 v1.5 / SHA-256) signing for model artifacts.

Keys come from AEGIS_MODEL_SIGN_PRIVATE_KEY / AEGIS_MODEL_SIGN_PUBLIC_KEY (PEM text or file path).
Verification is compatible with model_registry.signing.verify_rsa_signature. Everything fails closed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class SigningError(RuntimeError):
    pass


def _load_pem(setting: str) -> bytes:
    value = os.getenv(setting, "").strip()
    if not value:
        raise SigningError(f"{setting} is not set")
    path = Path(value)
    try:
        return path.read_bytes() if path.is_file() else value.encode()
    except OSError as exc:
        raise SigningError(f"cannot read {setting}") from exc


def signature_path(artifact_path: str) -> str:
    return f"{artifact_path}.sig"


def sign_model_artifact(artifact_path: str, key_id: Optional[str] = None) -> str:
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise SigningError(f"artifact not found: {artifact_path}")
    try:
        key = serialization.load_pem_private_key(_load_pem("AEGIS_MODEL_SIGN_PRIVATE_KEY"), password=None)
        signature = key.sign(artifact.read_bytes(), padding.PKCS1v15(), hashes.SHA256())
    except SigningError:
        raise
    except Exception as exc:
        raise SigningError("signing failed") from exc
    out = signature_path(artifact_path)
    Path(out).write_bytes(signature)
    return out


def verify_model_artifact(artifact_path: str, sig_path: Optional[str] = None) -> bool:
    from model_registry.signing import verify_rsa_signature

    try:
        public_key = _load_pem("AEGIS_MODEL_SIGN_PUBLIC_KEY")
    except SigningError:
        return False
    return verify_rsa_signature(public_key, artifact_path, sig_path or signature_path(artifact_path))
