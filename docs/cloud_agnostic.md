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
```markdown
# Aegis: Cloud-agnostic dev adapter (MinIO + S3 adapter)

What this provides
- Centralized config (aegis_multimodal_ai_system.config.cfg)
- Storage abstraction (StorageClient) and two concrete adapters:
  - MinIO (aegis_multimodal_ai_system.storage.minio_adapter.MinIOStorageAdapter)
  - S3 (aegis_multimodal_ai_system.storage.s3_adapter.S3StorageAdapter)
- Local dev docker-compose (MinIO + Postgres) and a bootstrap script (scripts/local_dev_setup.sh)

How to use (developer quickstart)
1. Start local dev stack:
   ./scripts/local_dev_setup.sh

2. In a Python REPL or your orchestrator module:
```python
from aegis_multimodal_ai_system import config
from aegis_multimodal_ai_system.storage.minio_adapter import MinIOStorageAdapter
from aegis_multimodal_ai_system.storage.s3_adapter import S3StorageAdapter

cfg = config.cfg
if cfg.OBJECT_STORE_TYPE == "minio":
    client = MinIOStorageAdapter(bucket=cfg.OBJECT_STORE_BUCKET, endpoint_url=cfg.OBJECT_STORE_ENDPOINT, access_key=cfg.OBJECT_STORE_ACCESS_KEY, secret_key=cfg.OBJECT_STORE_SECRET_KEY)
else:
    client = S3StorageAdapter(bucket=cfg.OBJECT_STORE_BUCKET, region=cfg.OBJECT_STORE_REGION, endpoint_url=cfg.OBJECT_STORE_ENDPOINT, access_key=cfg.OBJECT_STORE_ACCESS_KEY, secret_key=cfg.OBJECT_STORE_SECRET_KEY)

# Use client.upload / client.download / client.get_presigned_url
```

Extending to other clouds
- To add GCS or Azure: implement the StorageClient interface in a new adapter module (gcs_adapter.py or azure_blob_adapter.py) and wire it behind config. Keep the rest of the application working with StorageClient only.

Best practices
- Keep provider resource lifecycle (creating buckets, DB instances) in infra (Terraform) — runtime code should not create provider resources except for local dev convenience.
- Use signed artifacts and verify signatures before loading in orchestrator (cosign).
- Keep credentials out of git; use Vault/Kubernetes Secrets or GitHub OIDC for CI.

Next steps
- Replace direct uses of provider SDKs in model loaders / mlflow clients with the StorageClient interface.
- Add a CI job that runs acceptance tests against the docker-compose.dev stack to validate portability.
```
