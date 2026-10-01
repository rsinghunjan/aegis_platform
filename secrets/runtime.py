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
"""
Runtime secrets loader and fail-fast checks.

Usage:
- Import RuntimeSecrets at service startup and call RuntimeSecrets.ensure_required(required_list)
- Use RuntimeSecrets.get(...) to fetch secrets (delegates to SecretsManager / Vault / AWS / env)
- This module intentionally avoids printing secret values.

Behavior:
- Autodetects Vault / AWS / ENV backends via SecretsManager (existing abstraction).
- If required secrets are missing, logs a safe error and exits (fail-fast).
- Supports fetching an enrollment list (allowed client ids / cert CNs) stored under a secrets path.
"""
from __future__ import annotations

import os
import logging
import sys
from typing import Optional, List, Dict

from .backends import SecretsManager  # existing abstraction; prefer Vault/AWS

logger = logging.getLogger(__name__)


class RuntimeSecrets:
    def __init__(self, prefer: Optional[List[str]] = None):
        """
        prefer: optional backend order e.g. ["vault", "aws", "env"]
        """
        self.sm = SecretsManager(prefer=prefer)
        # cache for repeated calls
        self._cache: Dict[str, Optional[str]] = {}

    def get(self, name: str, field: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret by name. Does not log secret values.
        If the secret was previously fetched it is returned from cache.
        """
        key = f"{name}::{field}" if field else name
        if key in self._cache:
            return self._cache[key]
        val = self.sm.get_secret(name, key=field, default=default)
        # do not log val
        self._cache[key] = val
        return val

    def ensure_required(self, required: List[str]) -> None:
        """
        Ensure that all secrets in `required` exist (non-empty). If any are missing, fail-fast.
        Each name may be "secret_path" or "secret_path:field" to read a field out of a structured secret.
        Example: ["inference/jwt:JWKS_URI", "inference/model_pubkey"]
        """
        missing = []
        for entry in required:
            if ":" in entry:
                name, field = entry.split(":", 1)
            else:
                name, field = entry, None
            val = self.get(name, field=field)
            if not val:
                missing.append(entry)
        if missing:
            # Fail-fast: do not continue running with missing critical secrets
            logger.error("Missing required runtime secrets: %s. Exiting.", ", ".join(missing))
            # Consider raising an exception instead; use exit for immediate failure during container startup
            sys.exit(2)

    def get_enrollment_allowed_list(self, path: str = "clients/allowed") -> List[str]:
        """
        Fetch a newline/comma-separated list of allowed client identifiers (or cert CNs),
        stored as a secret at `path`. Return as list[str]. Empty list if not configured.
        """
        val = self.get(path)
        if not val:
            return []
        # try JSON list first
        try:
            import json
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        # fallback: split on newline/commas
        parts = [p.strip() for p in val.replace(",", "\n").splitlines() if p.strip()]
        return parts
