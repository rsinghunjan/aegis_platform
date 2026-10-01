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
#!/usr/bin/env python3
"""
Simple verification utility: confirm that active_key_id exists at the expected place in Vault.

Usage:
  VAULT_ADDR=... VAULT_TOKEN=... python scripts/vault_verify_rotation.py --key-id <uuid>
"""
import os
import argparse
import hvac
import sys

VAULT_MOUNT = os.environ.get("VAULT_MOUNT", "secret")


def hvac_client():
    addr = os.environ.get("VAULT_ADDR")
    token = os.environ.get("VAULT_TOKEN")
    if not addr:
        raise RuntimeError("Set VAULT_ADDR in env")
    client = hvac.Client(url=addr, token=token)
    if token and not client.is_authenticated():
        raise RuntimeError("Vault auth failed with provided token")
    return client


def verify(key_id: str):
    client = hvac_client()
    # read active pointer
    try:
        active = client.secrets.kv.v2.read_secret_version(path="aegis/keys/model_sign/active", mount_point=VAULT_MOUNT)
        active_data = active["data"]["data"]["value"]
        if active_data.get("active_key_id") != key_id:
            print(f"Active key id mismatch: expected={key_id} got={active_data.get('active_key_id')}", file=sys.stderr)
            return 2
        # check that the key entry exists
        key_entry = client.secrets.kv.v2.read_secret_version(path=f"aegis/keys/model_sign/{key_id}", mount_point=VAULT_MOUNT)
        if "data" not in key_entry or "data" not in key_entry["data"]:
            print("Key entry missing", file=sys.stderr)
            return 3
        print(f"Verification OK: active_key_id={key_id} exists in Vault")
        return 0
    except Exception as exc:
        print("Verification failed:", exc, file=sys.stderr)
        return 4


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--key-id", required=True)
    args = p.parse_args()
    rc = verify(args.key_id)
    sys.exit(rc)
