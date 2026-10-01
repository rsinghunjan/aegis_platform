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
#!/usr/bin/env bash
# quick helper to validate CI-provided storage credentials locally
set -euo pipefail
echo "Checking S3 (boto3) credentials..."
python3 - <<'PY'
import os, boto3
ep = os.environ.get("OBJECT_STORE_ENDPOINT")
ak = os.environ.get("OBJECT_STORE_ACCESS_KEY")
sk = os.environ.get("OBJECT_STORE_SECRET_KEY")
if not ak or not sk:
    print("S3 creds not set; skipping")
else:
    try:
        s3 = boto3.client("s3", aws_access_key_id=ak, aws_secret_access_key=sk, endpoint_url=ep)
        buckets = s3.list_buckets()
        print("S3 list_buckets ok:", [b["Name"] for b in buckets.get("Buckets",[])][:5])
    except Exception as e:
        print("S3 check failed:", e)
PY
echo "Check GCS credentials (GOOGLE_APPLICATION_CREDENTIALS)..."
python3 - <<'PY'
import os
if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    try:
        from google.cloud import storage
        client = storage.Client()
        print("GCS buckets sample:", [b.name for b in client.list_buckets(max_results=5)])
    except Exception as e:
        print("GCS check failed:", e)
else:
    print("GCS credentials not set; skipping")
PY
echo "Check Azure..."
python3 - <<'PY'
import os
conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
if conn:
    try:
        from azure.storage.blob import BlobServiceClient
        c = BlobServiceClient.from_connection_string(conn)
        print("Azure containers sample:", [c.name for c in c.list_containers()][:5])
    except Exception as e:
        print("Azure check failed:", e)
else:
    print("Azure credentials not set; skipping")
PY
echo "Check OCI (basic) - requires oci sdk and env/private key)"
python3 - <<'PY'
import os
if os.environ.get("OCI_PRIVATE_KEY") or os.environ.get("OCI_PRIVATE_KEY_FILE"):
    print("OCI private key appears set - please validate with oci SDK in your environment as required.")
else:
    print("OCI credentials not set; skipping")
PY
scripts/verify_ci_creds.sh
