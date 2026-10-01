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
"""
Rotate per-client API keys and store them in Vault or AWS Secrets Manager.

Usage:
  python scripts/rotate_api_keys.py --client-id client123 --backend vault
  python scripts/rotate_api_keys.py --client-id client123 --backend aws --secret-name aegis/clients/client123

Notes:
- This script returns the new API key on stdout. Distribute it securely to the client (out-of-band).
- Do NOT commit the new key to any repository.
"""
import argparse
import os
import sys
import json
import logging
import secrets
import string

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def gen_key(length: int = 40) -> str:
    alphabet = string.ascii_letters + string.digits + "-_"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def store_in_vault(path: str, data: dict) -> bool:
    try:
        import hvac
    except Exception:
        logger.error("hvac not installed; cannot use vault backend")
        return False
    addr = os.getenv("VAULT_ADDR")
    token = os.getenv("VAULT_TOKEN")
    if not addr or not token:
        logger.error("VAULT_ADDR or VAULT_TOKEN not set")
        return False
    client = hvac.Client(url=addr, token=token)
    if not client.is_authenticated():
        logger.error("Vault client not authenticated")
        return False
    # expect path to be a kv v2 path relative to mount; use create_or_update_secret
    try:
        # Try to write under kv v2 helper
        # If path contains "secret/data/" remove that prefix
        rel = path
        if path.startswith("secret/data/"):
            rel = path.split("secret/data/", 1)[1]
        client.secrets.kv.v2.create_or_update_secret(path=rel, secret=data)
        return True
    except Exception as e:
        logger.exception("Vault write failed: %s", e)
        return False


def store_in_aws(secret_name: str, data: dict) -> bool:
    try:
        import boto3
    except Exception:
        logger.error("boto3 not installed; cannot use aws backend")
        return False
    sess = boto3.session.Session()
    client = sess.client("secretsmanager")
    try:
        # Use put_secret_value if secret exists else create_secret
        try:
            client.put_secret_value(SecretId=secret_name, SecretString=json.dumps(data))
        except client.exceptions.ResourceNotFoundException:
            client.create_secret(Name=secret_name, SecretString=json.dumps(data))
        return True
    except Exception:
        logger.exception("AWS Secrets Manager write failed")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--backend", choices=["vault", "aws"], default="vault")
    parser.add_argument("--path", default=None, help="Vault path or AWS secret name")
    args = parser.parse_args()

    new_key = gen_key()
    payload = {"api_key": new_key, "rotated_at": int(__import__("time").time())}

    if args.backend == "vault":
        path = args.path or f"clients/{args.client_id}"
        ok = store_in_vault(path, payload)
    else:
        secret_name = args.path or f"aegis/clients/{args.client_id}"
        ok = store_in_aws(secret_name, payload)

    if not ok:
        logger.error("Failed to store rotated key")
        sys.exit(2)

    # Print the key to stdout (caller should capture it securely)
    print(new_key)


if __name__ == "__main__":
    main()
scripts/rotate_api_keys.py
