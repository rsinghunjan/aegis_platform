from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from aegis_signing.contracts import Signature, Signer
from aegis_signing.hmac_signer import HmacSigner


@dataclass(frozen=True)
class SigningConfig:
    mode: str  # "hmac" now; later: "aws-kms", "gcp-kms", "azure-kv", "hmac+kms"


def get_signer(cfg: SigningConfig) -> Signer:
    # Best-effort: for this patch, only HMAC is implemented.
    if cfg.mode in ("hmac", "hmac+kms"):
        return HmacSigner()
    raise NotImplementedError(f"Signing mode not implemented in best-effort patch: {cfg.mode}")


def sign_json(signer: Signer, payload: Mapping[str, Any]) -> tuple[str, Signature]:
    b = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sha = __import__("hashlib").sha256(b).hexdigest()
    sig = signer.sign(b)
    return sha, sig
