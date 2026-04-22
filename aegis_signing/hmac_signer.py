from __future__ import annotations

import base64
import hmac
import hashlib
import os
from dataclasses import dataclass

from aegis_signing.contracts import Signature


@dataclass(frozen=True)
class HmacSigner:
    """
    Bootstrap signer. Store key in a secret manager in production.
    Env:
      - AEGIS_HMAC_KEY_BASE64
      - AEGIS_SIGNING_KEY_ID (optional)
    """

    key_id: str = "hmac-v1"

    @staticmethod
    def _get_key() -> bytes:
        b64 = os.getenv("AEGIS_HMAC_KEY_BASE64", "")
        if not b64:
            raise RuntimeError("AEGIS_HMAC_KEY_BASE64 is required for HMAC signing")
        return base64.b64decode(b64)

    def sign(self, payload: bytes) -> Signature:
        key = self._get_key()
        mac = hmac.new(key, payload, hashlib.sha256).digest()
        return Signature(
            signature=base64.b64encode(mac).decode("ascii"),
            key_ref=os.getenv("AEGIS_SIGNING_KEY_ID", self.key_id),
            algorithm="hmac-sha256",
        )

    def verify(self, payload: bytes, sig: Signature) -> bool:
        if sig.algorithm != "hmac-sha256":
            return False
        key = self._get_key()
        expected = hmac.new(key, payload, hashlib.sha256).digest()
        got = base64.b64decode(sig.signature)
        return hmac.compare_digest(expected, got)
