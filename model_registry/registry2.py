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
# NOTE: This is an additive change to call signature verification when registering/fetching models.
# Only the relevant portions are shown; drop into your existing registry module or replace the file.
import logging
import os
from pathlib import Path
from typing import Optional

from .signing import compute_sha256, load_public_key_bytes, verify_rsa_signature

logger = logging.getLogger(__name__)

# existing globals
MODELS_DIR = Path(os.getenv("MODEL_CACHE_DIR", "models/cache"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Environment: path to PEM public key used to verify model artifacts
MODEL_SIGNING_PUBKEY = os.getenv("MODEL_SIGNING_PUBKEY", "")  # can be PEM text or path to PEM file


def _verify_signature_if_present(artifact_path: Path, sig_path: Optional[Path]) -> None:
    """
    If a signature file is provided, attempt verification using MODEL_SIGNING_PUBKEY.
    Raises RuntimeError on verification failure.
    """
    if sig_path is None:
        logger.debug("No signature file provided for %s", artifact_path)
        return
    pubkey_bytes = load_public_key_bytes(MODEL_SIGNING_PUBKEY)
    if not pubkey_bytes:
        raise RuntimeError("Model signing public key not configured (MODEL_SIGNING_PUBKEY)")
    ok = verify_rsa_signature(pubkey_bytes, str(artifact_path), str(sig_path))
    if not ok:
        raise RuntimeError("Model signature verification failed for %s" % artifact_path)
    logger.info("Model signature verified for %s", artifact_path)


# Example: modify fetch_from_s3 and register_local to accept optional signature file
def register_local(self, name: str, version: str, local_path: str, checksum: Optional[str] = None, sig_path: Optional[str] = None):
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(f"Model path not found: {local_path}")
    if checksum:
        if not compute_sha256(p) == checksum.lower():
            raise ValueError("Checksum mismatch for model artifact")
    if sig_path:
        _verify_signature_if_present(p, Path(sig_path))
    meta = ModelMetadata(name, version, str(p.resolve()), checksum, loaded_at=time.time())
    with self._lock:
        self._models[self._key(name, version)] = meta
    logger.info("Registered local model %s version %s at %s", name, version, meta.path)
    return meta


def fetch_from_s3(self, name: str, version: str, s3_key: Optional[str] = None, checksum: Optional[str] = None, sig_s3_key: Optional[str] = None):
    # download artifact as before to local_path
    # if sig_s3_key provided, download into same dir as local_path with .sig extension
    # After download, verify checksum and signature:
    if checksum and not self._verify_checksum(local_path, checksum):
        try:
            local_path.unlink()
        except Exception:
            pass
        raise ValueError("Checksum mismatch after S3 download")
    if sig_s3_key:
        # assume we downloaded signature to local_sig_path
        _verify_signature_if_present(local_path, local_sig_path)
    # register meta...
    ```
