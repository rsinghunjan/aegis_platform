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
#!/usr/bin/env bash
# Quick local dev bootstrap: brings up docker-compose and creates the MinIO bucket via boto3 Python helper.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$ROOT/docker-compose.dev.yml"

echo "Starting local dev stack (MinIO + Postgres)..."
docker compose -f "$COMPOSE_FILE" up -d

echo "Waiting for MinIO to be healthy..."
# simple wait loop
for i in $(seq 1 30); do
  if docker inspect --format='{{json .State.Health.Status}}' $(docker ps -q --filter ancestor=minio/minio) 2>/dev/null | grep -q healthy; then
    echo "MinIO healthy"
    break
  fi
  sleep 2
done

echo "Creating bucket (aegis-models) in MinIO using Python helper..."
python3 - <<'PY'
import os, boto3, time
endpoint = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
access = os.environ.get("MINIO_ACCESS", "minioadmin")
secret = os.environ.get("MINIO_SECRET", "minioadmin")
bucket = os.environ.get("OBJECT_STORE_BUCKET", "aegis-models")
# boto3 expects endpoint without scheme for resource.client? use client with endpoint_url
s3 = boto3.client("s3", aws_access_key_id=access, aws_secret_access_key=secret, endpoint_url=endpoint)
try:
    s3.head_bucket(Bucket=bucket)
    print("Bucket exists:", bucket)
except Exception:
    print("Creating bucket:", bucket)
    s3.create_bucket(Bucket=bucket)
    print("Bucket created.")
PY

echo "Local dev stack ready."
echo "Set env vars for Aegis local dev:"
echo "  export OBJECT_STORE_TYPE=minio"
echo "  export OBJECT_STORE_ENDPOINT=http://localhost:9000"
echo "  export OBJECT_STORE_ACCESS_KEY=minioadmin"
echo "  export OBJECT_STORE_SECRET_KEY=minioadmin"
echo "  export OBJECT_STORE_BUCKET=aegis-models"
echo "  export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/aegis_dev"
