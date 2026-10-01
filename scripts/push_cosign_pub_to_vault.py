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
#!/usr/bin/env python3
"""
Push a cosign public key file to Vault KV v2.

Usage:
  VAULT_ADDR=https://vault.example:8200 VAULT_TOKEN=s.xxxxxx python3 scripts/push_cosign_pub_to_vault.py /path/to/cosign.pub secret/data/aegis/cosign

This writes JSON payload { "data": { "public_key": "<content>" } } to the V2 KV endpoint.
"""
from __future__ import annotations
import sys
import os
import requests
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/push_cosign_pub_to_vault.py <cosign_pub.pem> <vault_kv_v2_path>", file=sys.stderr)
        print("Example: python3 ... cosign.pub secret/data/aegis/cosign", file=sys.stderr)
        sys.exit(2)

    pub_path = Path(sys.argv[1])
    vault_path = sys.argv[2].lstrip("/")

    vault_addr = os.environ.get("VAULT_ADDR")
    token = os.environ.get("VAULT_TOKEN")
    if not vault_addr or not token:
        print("Please set VAULT_ADDR and VAULT_TOKEN environment variables", file=sys.stderr)
        sys.exit(2)

    if not pub_path.exists():
        print(f"Public key file not found: {pub_path}", file=sys.stderr)
        sys.exit(2)

    pub_content = pub_path.read_text(encoding="utf-8")

    url = f"{vault_addr.rstrip('/')}/v1/{vault_path}"
    # KV v2 expects body { "data": { ... } }
    payload = {"data": {"public_key": pub_content}}
    headers = {"X-Vault-Token": token}
    r = requests.post(url, json=payload, headers=headers, timeout=10)
    try:
        r.raise_for_status()
    except Exception as e:
        print("Vault write failed:", r.status_code, r.text, file=sys.stderr)
        raise
    print("Wrote cosign public key to Vault path:", vault_path)

if __name__ == "__main__":
    main()
scripts/push_cosign_pub_to_vault.pysc
