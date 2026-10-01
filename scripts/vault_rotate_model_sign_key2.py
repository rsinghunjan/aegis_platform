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
#!/usr/bin/env python3
"""
Rotate the AEGIS model signing key in Vault (KV v2).

This script supports Vault authentication via VAULT_TOKEN (recommended from ephemeral OIDC/AppRole login in CI).
It creates a new key entry and updates the active pointer.

Usage:
  # preview (no writes)
  python scripts/vault_rotate_model_sign_key.py --preview

  # actually write (expects VAULT_ADDR & VAULT_TOKEN in env)
  python scripts/vault_rotate_model_sign_key.py --preview=false

Outputs:
  On success (non-preview), prints a JSON blob including the new key_id (but not the raw key).
"""
import os
import argparse
import uuid
import secrets
import json
from datetime import datetime
import hvac

VAULT_MOUNT = os.environ.get("VAULT_MOUNT", "secret")  # KV v2 mount
ACTIVE_PATH = f"{VAULT_MOUNT}/data/aegis/keys/model_sign/active"
KEY_BASE_SUBPATH = "aegis/keys/model_sign"  # used with hvac kv v2 API


def hvac_client():
    addr = os.environ.get("VAULT_ADDR")
    token = os.environ.get("VAULT_TOKEN")
    if not addr:
        raise RuntimeError("Set VAULT_ADDR in env")
    client = hvac.Client(url=addr, token=token)
    if token and not client.is_authenticated():
        raise RuntimeError("Vault auth failed with provided token")
    return client


def gen_key_hex(length_bytes=32):
    return secrets.token_hex(length_bytes)


def write_kv_v2(client, logical_path, data):
    # logical_path e.g. "model_sign/<key_id>" or "model_sign/active"
    client.secrets.kv.v2.create_or_update_secret(path=logical_path, secret={"value": data}, mount_point=VAULT_MOUNT)


def read_active(client):
    try:
        res = client.secrets.kv.v2.read_secret_version(path="aegis/keys/model_sign/active", mount_point=VAULT_MOUNT)
        return res["data"]["data"]["value"]
    except Exception:
        return None


def main(dry_run=True):
    client = hvac_client() if os.environ.get("VAULT_ADDR") else None
    key_id = str(uuid.uuid4())
    key_value = gen_key_hex(32)
    ts = datetime.utcnow().isoformat() + "Z"

    payload = {
        "key_id": key_id,
        "created_at": ts,
        "created_by": os.environ.get("ROTATED_BY", "automation"),
        "status": "active"
    }

    if dry_run:
        # show a preview including checksum-style info (but not the raw key)
        print(json.dumps({"preview": True, "key_id": key_id, "payload": payload}, indent=2))
        return payload

    if client is None:
        raise RuntimeError("VAULT_ADDR not configured; cannot write")

    # store new key object (store key under a per-key subpath)
    write_kv_v2(client, f"aegis/keys/model_sign/{key_id}", {"key": key_value, "created_at": ts, "created_by": payload["created_by"], "status": "active"})

    # update active pointer
    active_payload = {"active_key_id": key_id, "activated_at": ts}
    write_kv_v2(client, "aegis/keys/model_sign/active", active_payload)

    # Return a JSON with non-sensitive fields for CI verification (do NOT echo raw key)
    out = {"key_id": key_id, "activated_at": ts}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preview", dest="preview", action="store_true", help="preview only (no writes)")
    p.add_argument("--preview-false", dest="preview", action="store_false", help="write to Vault (aliases --no-preview)")
    args = p.parse_args()
    main(dry_run=args.preview)
