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
#!/usr/bin/env python3
"""
OIDC token verification middleware helpers for FastAPI/Flask style apps.

- validate_oidc_token: validates JWT using JWKS endpoint, checks audience and optional roles claim.
- get_actor_from_token: extracts a stable identifier (email or sub) for audit logging.

This is minimal; adapt to your OIDC provider (Keycloak, Okta, Google) and caching strategies.
"""
from __future__ import annotations
import os
import time
import json
import requests
import jwt  # PyJWT
from jwt import PyJWKClient
from typing import Optional

OIDC_ISSUER = os.environ.get("OIDC_ISSUER")  # e.g. https://auth.example.com/
OIDC_AUDIENCE = os.environ.get("OIDC_AUDIENCE")  # client_id for your API

_jwks_client = None
def _get_jwks_client():
    global _jwks_client
    if _jwks_client is None:
        if not OIDC_ISSUER:
            raise RuntimeError("OIDC_ISSUER not set")
        jwks_url = OIDC_ISSUER.rstrip("/") + "/.well-known/jwks.json"
