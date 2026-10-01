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
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
"""
    Read a KV v2 secret; path is the logical path under the mount.
    Returns the data dict or None.
    """
    client = _get_client()
    if client is None:
        logger.debug("vault client not available; returning None for %s", path)
        return None
    try:
        # hvac KV v2 read_secret_version expects path relative to mount root
        resp = client.secrets.kv.v2.read_secret_version(path=path, mount_point=mount)
        return resp.get("data", {}).get("data", {})
    except Exception as exc:
        logger.debug("vault read_kv failed for %s: %s", path, exc)
        return None


def get_secret_value(path: str, key: str, mount: str = VAULT_MOUNT) -> Optional[str]:
    """
    Read a single secret value from KV v2. If not present in Vault,
    fallback to environment variable matching key.
    """
    data = get_kv(path, mount=mount)
    if data and key in data:
        return data.get(key)
    # fallback: environment variable
    env_key = key
    value = os.environ.get(env_key)
    if value:
        logger.debug("Using fallback env var for %s", key)
    return value


def get_jwt_secret() -> str:
    """
    Return the SECRET_KEY used to sign JWTs. Fallback to AEGIS_SECRET_KEY env var.
    """
    v = get_secret_value("aegis/secrets/jwt", "SECRET_KEY")
    if v:
        return v
    return os.environ.get("AEGIS_SECRET_KEY", "dev-secret-key-change-me")


def get_model_sign_key() -> str:
    v = get_secret_value("aegis/keys/model_sign/active", "active_key_id")
    if v:
        # read the actual key id value
        key_data = get_kv(f"aegis/keys/model_sign/{v}")
        if key_data and "key" in key_data:
            return key_data["key"]
    # fallback to simple single key location
    v2 = get_secret_value("aegis/keys/model_sign", "MODEL_SIGN_KEY")
    if v2:
        return v2
    return os.environ.get("AEGIS_MODEL_SIGN_KEY", "dev-model-sign-key-change-me")
api/vault.py
