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
"""
FastAPI dependency that enforces one of:
- OIDC (JWT verified via JWKS) OR
- mTLS (identity asserted by verified client certificate forwarded by ingress/mesh header)

Configuration via environment:
- AUTH_MODE: "none" | "oidc" | "mtls"   (default: "none")
- For OIDC:
  - OIDC_JWKS_URI or store it in Vault and fetch at startup
  - OIDC_AUDIENCE optionally
- For mTLS:
  - MTLS_ID_HEADER: header name that contains client identity (default: "x-ssl-client-cn")
  - ENROLLED_CLIENTS_SECRET_PATH: secret path (in Vault) containing allowed client CNs/IDs (optional)
- Use RuntimeSecrets to fetch required secrets at startup.

Behavior:
- On oidc: validates Authorization: Bearer <token> via OIDC JWKS helper
- On mtls: checks configured header for client identity and validates against enrolled list (from Vault or file)
"""
from __future__ import annotations

import os
import logging
from typing import Dict, Optional

from fastapi import Depends, HTTPException, Request, status

from ..secrets.runtime import RuntimeSecrets
from ..auth.oidc_jwks import verify_jwt, JWKS_URI, OIDC_AUDIENCE, OIDC_ISSUER  # earlier helper

logger = logging.getLogger(__name__)

AUTH_MODE = os.getenv("AUTH_MODE", "none").lower()
MTLS_ID_HEADER = os.getenv("MTLS_ID_HEADER", "x-ssl-client-cn")
ENROLLED_CLIENTS_SECRET_PATH = os.getenv("ENROLLED_CLIENTS_SECRET_PATH", "clients/allowed")

_runtime_secrets = RuntimeSecrets()


def _verify_oidc(request: Request) -> Dict:
    # Validate bearer token via OIDC JWKS
    auth = request.headers.get("authorization", "")
    if not auth:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    try:
        scheme, token = auth.split(" ", 1)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header")
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization must be Bearer token")

    # If JWKS URI not configured in env, try reading from secrets
    jwks_uri = os.getenv("OIDC_JWKS_URI", "") or _runtime_secrets.get("oidc/jwks_uri")
    if not jwks_uri:
        # fallback to earlier module-level JWKS_URI if present
        jwks_uri = JWKS_URI
    if not jwks_uri:
        logger.error("OIDC configured but no JWKS URI found")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OIDC not configured")

    # call the OIDC helper (verify_jwt reads JWKS_URI global in its module if not provided; adjust as needed)
    try:
        # we delegate to verify_jwt that fetches JWKS from JWKS_URI configured earlier
        claims = verify_jwt(token, audience=os.getenv("OIDC_AUDIENCE", OIDC_AUDIENCE), issuer=os.getenv("OIDC_ISSUER", OIDC_ISSUER))
        return claims
    except Exception as e:
        logger.debug("OIDC token verification failed: %s", e)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def _verify_mtls(request: Request) -> Dict:
    """
    Expect the ingress/service-mesh to verify client certs and forward the client's identity
    (usually CN or a verified subject) in a header such as x-ssl-client-cn.
    This function checks that header and validates against the enrolled clients list stored in secrets.
    """
    header_val = request.headers.get(MTLS_ID_HEADER)
    if not header_val:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Client certificate identity header missing")

    # normalize
    client_id = header_val.strip()
    # check against enrolled list from Vault (or other secret backend)
    allowed = _runtime_secrets.get_enrollment_allowed_list(ENROLLED_CLIENTS_SECRET_PATH)
    if allowed:
        # exact match check, constant-time compare not necessary for IDs but do safe check
        if client_id not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Client not enrolled / authorized")
    # else: if no enrolled list configured, we accept any verified client cert - still warn
    else:
        logger.warning("MTLS mode enabled but enrolled clients list is empty; consider configuring ENROLLED_CLIENTS_SECRET_PATH")

    # Return identity info for downstream use
    return {"client_id": client_id}


def require_auth(request: Request) -> Optional[Dict]:
    """
    FastAPI dependency to enforce configured auth mode. Returns parsed claims or identity dict.
    Usage:
        @app.post("/predict")
        def predict(..., auth=Depends(require_auth)):
            ...
    """
    mode = AUTH_MODE
    if mode == "none":
        return None
    if mode == "oidc":
        return _verify_oidc(request)
    if mode == "mtls":
        return _verify_mtls(request)
    # invalid mode
    logger.error("Invalid AUTH_MODE configured: %s", AUTH_MODE)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Auth misconfigured")
aegis_multimodal_ai_system/auth/require_auth.py
