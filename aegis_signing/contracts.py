from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Signature:
    signature: str  # base64 or provider-specific encoding
    key_ref: str     # key id / uri
    algorithm: str   # e.g. "hmac-sha256", "rsa-pss", "ecdsa-p256"


class Signer(Protocol):
    def sign(self, payload: bytes) -> Signature: ...
    def verify(self, payload: bytes, sig: Signature) -> bool: ...
