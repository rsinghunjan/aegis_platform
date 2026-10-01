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
"""

# Simple in-memory cache
_jwks_cache: Dict[str, Dict] = {"keys": [], "fetched_at": 0}
JWKS_TTL = 60 * 60  # 1 hour cache


def _fetch_jwks() -> Dict:
    global _jwks_cache
    now = time.time()
    if _jwks_cache["keys"] and (now - _jwks_cache["fetched_at"] < JWKS_TTL):
        return _jwks_cache
    if not JWKS_URI:
        raise RuntimeError("OIDC_JWKS_URI not configured")
    resp = requests.get(JWKS_URI, timeout=5)
    resp.raise_for_status()
    jwks = resp.json()
    _jwks_cache = {"keys": jwks.get("keys", []), "fetched_at": now}
    return _jwks_cache


def _get_key_for_kid(kid: str) -> Optional[Dict]:
    jwks = _fetch_jwks()
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key
    return None


def verify_jwt(token: str, audience: Optional[str] = None, issuer: Optional[str] = None) -> Dict:
    """
    Verify a JWT using JWKS fetch and python-jose.
    Returns decoded claims (dict) on success, raises ValueError / JWTError on failure.
    """
    if jwt is None:
        raise RuntimeError("python-jose not installed; install 'python-jose[cryptography]' to use OIDC verification")

    # Decode header to find kid
    try:
        headers = jwt.get_unverified_header(token)
    except JWTError as e:
        logger.debug("JWT header parse error: %s", e)
        raise

    kid = headers.get("kid")
    if not kid:
        logger.debug("JWT missing kid header")
        raise JWTError("missing kid in token header")

    key = _get_key_for_kid(kid)
    if not key:
        logger.debug("No JWKS key for kid=%s; refreshing JWKS", kid)
        # force refresh and retry once
        _jwks_cache["fetched_at"] = 0
        key = _get_key_for_kid(kid)
        if not key:
            raise JWTError("unable to find key for kid")

    # Let python-jose verify token (it will check alg etc)
    verify_kwargs = {}
    if audience or OIDC_AUDIENCE:
        verify_kwargs["audience"] = audience or OIDC_AUDIENCE
    if issuer or OIDC_ISSUER:
        verify_kwargs["issuer"] = issuer or OIDC_ISSUER

    try:
        claims = jwt.decode(token, key, algorithms=[key.get("alg", "RS256")], **verify_kwargs)
        return claims
    except JWTError as e:
        logger.debug("JWT verification failed: %s", e)
        raise
