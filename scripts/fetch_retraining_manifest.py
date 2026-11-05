#!/usr/bin/env python
"""Download the hash-only retraining manifest from a running Aegis control plane."""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--tenant-id", required=True)
    p.add_argument("--out", default="retraining_manifest.json")
    args = p.parse_args()
    token = os.getenv("AEGIS_OPERATOR_TOKEN", "")
    if not token:
        print("AEGIS_OPERATOR_TOKEN is required", file=sys.stderr)
        return 2
    query = urllib.parse.urlencode({"tenant_id": args.tenant_id})
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/operator/ai/improvements/retraining-manifest?{query}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        manifest = json.load(response)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True)
    print(json.dumps({"count": manifest["count"], "manifest_sha256": manifest["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
