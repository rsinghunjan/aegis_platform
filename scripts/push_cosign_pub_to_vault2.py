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
#!/usr/bin/env python3
"""
Push cosign public key into HashiCorp Vault (KV v2) via HTTP API.

Requires:
  - VAULT_ADDR env (e.g. https://vault.example.com)
  - VAULT_TOKEN env with write privileges to the target path

Usage:
  python3 scripts/push_cosign_pub_to_vault.py /path/to/cosign.pub secret/data/aegis/cosign
"""
from __future__ import annotations
import sys
import os
import json
import requests
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: push_cosign_pub_to_vault.py <cosign_pub.pem> <vault_kv_v2_path>", file=sys.stderr)
        sys.exit(2)
    pub_path = Path(sys.argv[1])
    vault_path = sys.argv[2].lstrip("/")
    vault_addr = os.environ.get("VAULT_ADDR")
    vault_token = os.environ.get("VAULT_TOKEN")
    if not vault_addr or not vault_token:
        print("VAULT_ADDR and VAULT_TOKEN must be set in env", file=sys.stderr)
        sys.exit(2)
    if not pub_path.exists():
        print("Pubkey not found:", pub_path, file=sys.stderr)
        sys.exit(2)
    pub = pub_path.read_text(encoding="utf-8")
    url = f"{vault_addr}/v1/{vault_path}"
    payload = {"data": {"public_key": pub}}
    headers = {"X-Vault-Token": vault_token}
    r = requests.post(url, json=payload, headers=headers, timeout=10)
    r.raise_for_status()
    print("Wrote cosign public key to", vault_path)

if __name__ == "__main__":
    main()
scripts/push_cosign_pub_to_vault.py
