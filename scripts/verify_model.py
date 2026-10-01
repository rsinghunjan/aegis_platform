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
"""
Runtime/CI helper to verify a model artifact's RSA signature using the public key.
Usage:
  python scripts/verify_model.py --artifact model.tar --sig model.tar.sig --pubkey /path/to/pub.pem
Exits with 0 on success, non-zero on failure.
"""
import argparse
import sys
from pathlib import Path

from aegis_multimodal_ai_system.model_registry.signing import load_public_key_bytes, verify_rsa_signature

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", required=True)
    p.add_argument("--sig", required=True)
    p.add_argument("--pubkey", required=True)
    args = p.parse_args()

    pub = load_public_key_bytes(args.pubkey)
    if not pub:
        print("Failed to load public key", file=sys.stderr)
        sys.exit(2)
    ok = verify_rsa_signature(pub, args.artifact, args.sig)
    if ok:
        print("Signature verified")
        sys.exit(0)
    else:
        print("Signature verification FAILED", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
