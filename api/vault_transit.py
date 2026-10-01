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
    pass


def _ensure_configured():
    if not VAULT_ADDR:
        raise VaultTransitError("VAULT_ADDR not set in environment")
    if not VAULT_TOKEN:
        raise VaultTransitError("VAULT_TOKEN not set (use OIDC/AppRole/Agent to provision token)")


def _sha256_b64(data: bytes) -> str:
    h = hashlib.sha256(data).digest()
    return base64.b64encode(h).decode("utf-8")


def _headers():
    return {"X-Vault-Token": VAULT_TOKEN, "Content-Type": "application/json"}


def sign_bytes(key_name: str, data: bytes, key_version: Optional[int] = None) -> str:
    """
    Sign `data` using Vault Transit key `key_name`.
    Returns Vault signature string (e.g., "vault:v1:BASE64...")
    """
    _ensure_configured()
    digest_b64 = _sha256_b64(data)
    url = f"{VAULT_ADDR}/v1/{TRANSIT_MOUNT}/sign/{key_name}"
    payload = {"hash_algorithm": "sha2-256", "input": digest_b64}
    if key_version:
        payload["key_version"] = key_version
    resp = requests.post(url, json=payload, headers=_headers(), timeout=10)
    if resp.status_code != 200:
        logger.exception("Vault transit sign failed: %s %s", resp.status_code, resp.text)
        raise VaultTransitError(f"sign failed: {resp.status_code} {resp.text}")
    body = resp.json()
    signature = body.get("data", {}).get("signature")
    if not signature:
        raise VaultTransitError("no signature returned from Vault transit")
    return signature


def verify_bytes(key_name: str, data: bytes, signature: str) -> bool:
    """
    Verify the signature for `data` using Vault Transit verify endpoint.
    Returns True if verified, False otherwise.
    """
    _ensure_configured()
    digest_b64 = _sha256_b64(data)
    url = f"{VAULT_ADDR}/v1/{TRANSIT_MOUNT}/verify/{key_name}"
    payload = {"hash_algorithm": "sha2-256", "input": digest_b64, "signature": signature}
    resp = requests.post(url, json=payload, headers=_headers(), timeout=10)
    if resp.status_code != 200:
        logger.exception("Vault transit verify failed: %s %s", resp.status_code, resp.text)
        # treat non-200 as failure
        return False
    body = resp.json()
    return bool(body.get("data", {}).get("valid", False))


# Convenience file helpers

def sign_file(key_name: str, path: str) -> str:
    """
    Read file and sign its SHA256 digest via Transit. Returns signature.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    return sign_bytes(key_name, data)


def verify_file(key_name: str, path: str, signature: str) -> bool:
    with open(path, "rb") as fh:
        data = fh.read()
    return verify_bytes(key_name, data, signature)
api/vault_transit.py
