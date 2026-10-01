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
```markdown
Feast S3 + Redis minimal production wiring (Aegis)

Goal
- Use S3 (MinIO for local dev) as the offline store and Redis as the online store.
- Provide dataset versioning by uploading parquet datasets to S3 under a versioned path and maintaining a 'latest' pointer.
- Materialize features from the offline store into Redis for serving to close the train/serve feature gap.

Local quickstart (dev)
1. Start MinIO + Redis:
   docker compose -f docker/docker-compose-feast.yml up -d

   MinIO console: http://localhost:9001 (user/minioadmin, pass/minioadmin)
   Redis: localhost:6379

2. Ensure environment variables so boto3 can reach MinIO:
   export AWS_ACCESS_KEY_ID=minioadmin
   export AWS_SECRET_ACCESS_KEY=minioadmin
   export AWS_REGION=us-east-1
   export MLFLOW_S3_ENDPOINT_URL=http://localhost:9001
   export MINIO_ACCESS_KEY=minioadmin
   export MINIO_SECRET_KEY=minioadmin

3. Install feast deps:
   python -m pip install -r feast/requirements-feast.txt

4. Ingest a versioned dataset and materialize:
   python feast/feature_repo/ingest_to_s3.py --bucket feast-offline --prefix aegis

   This will:
   - create a parquet at s3://feast-offline/aegis/datasets/user_events/{version}/user_events.parquet
   - write a manifest.json alongside it
   - update 'latest' pointer under s3://feast-offline/aegis/datasets/user_events/latest/
   - fs.apply(...) and fs.materialize(...) to populate Redis online store

Production notes
- In production, replace MinIO with a real S3 bucket and ensure Feast has IAM credentials (or use IRSA on EKS).
- Keep dataset versions immutable; write manifests describing provenance (source job id, git sha, data checksum).
- Use persistent storage and lifecycle policies for offline datasets; maintain retention/cleanup policy.
- Materialize incrementally via scheduled jobs (k8s CronJob) to keep online store fresh.
- Namespace datasets (by tenant or project) to avoid collisions.

Integration ideas
- On training pipelines, pass the dataset version (s3 prefix) as a training input; store dataset_version metadata in MLflow run tags and ModelRegistry entries so training is fully reproducible.
- Add a lightweight registry API or metadata store that maps model_version -> dataset_version -> feature_view_version for provenance.
- Add a CI smoke test that runs ingest_to_s3.py, runs the training script using features (examples/train_with_features.py), and verifies end-to-end.

Security & credentials
- For CI, use ephemeral credentials or OIDC to grant producers short-lived access to S3.
- MinIO for dev is convenient, but use proper S3 buckets and IAM policies in staging/production.

Next steps I can help with
- A) Add a k8s CronJob manifest to run periodic materialize-incremental in cluster.
- B) Wire the dataset version into MLflow run tagging and the promote -> registry flow.
- C) Create a small metadata registry (YAML or simple DB) to record dataset versions and link them to model registry entries.

Which would you like to do next?
```
docs/feast_s3_setup.md
