"""
Model artifact signing / verification helpers.

This supports:
- RSA PKCS#1 v1.5 + SHA256 signature verification (useful for CI-produced signatures),
- A thin wrapper so CI can sign artifacts with a private key (e.g. openssl),
  and runtime can verify with a public key.

Notes:
- For production prefer cosign / sigstore. This file provides a portable fallback.
- Do NOT log secret keys or signature bytes.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_rsa_signature(public_key_pem: bytes, artifact_path: str, sig_path: str) -> bool:
    """
    Verify RSA PKCS1v15 SHA256 signature on artifact_path using public_key_pem.
    Returns True on success, False otherwise.
    """
    try:
        pub = serialization.load_pem_public_key(public_key_pem)
    except Exception:
        logger.exception("Failed to load public key for signature verification")
        return False

    art = Path(artifact_path)
    sig = Path(sig_path)
    if not art.exists() or not sig.exists():
        logger.warning("Artifact or signature file missing (%s, %s)", art, sig)
        return False

    data = _read_bytes(art)
    signature = _read_bytes(sig)

    try:
        pub.verify(
            signature,
            data,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except InvalidSignature:
        logger.warning("Invalid signature for %s", artifact_path)
        return False
    except Exception:
        logger.exception("Signature verification error for %s", artifact_path)
        return False


def load_public_key_bytes(path_or_pem: Optional[str]) -> Optional[bytes]:
    """
    Accept either a PEM string or a path to a PEM file. Return bytes or None.
    """
    if not path_or_pem:
        return None
    p = Path(path_or_pem)
    if p.exists():
        return p.read_bytes()
    # treat as raw pem string
    return path_or_pem.encode("utf-8")
