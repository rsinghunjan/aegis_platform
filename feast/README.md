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
```markdown
Feast minimal feature-store demo for Aegis

Goal
- Provide a minimal, reproducible Feast example that demonstrates:
  - feature definitions and a local feature repo
  - ingestion to an offline store (Parquet) and materialization to an online store (Redis)
  - example usage in a training script (historical feature retrieval)
  - an online retrieval snippet for runtime serving

What is included
- feast/feature_repo/feature_store.yaml   : Feast repo config (offline=parquet local, online=redis)
- feast/feature_repo/features.py         : Entity + FeatureView + FileSource definitions
- feast/feature_repo/ingest.py           : Create synthetic sample data, write parquet and materialize to online store
- examples/train_with_features.py        : Example training script that retrieves historical features and trains a model
- api/feature_service_online.py          : Minimal snippet to fetch online features in a serving request
- requirements-feast.txt                 : Minimal Python deps for running the demo

Quickstart (local)
1. Install deps:
   python -m pip install -r feast/requirements-feast.txt

2. Start a Redis instance (for the online store). Quick local (dev) option:
   docker run -d --name feast-redis -p 6379:6379 redis:7

3. Run ingestion (creates local parquet and materializes into Redis):
   python feast/feature_repo/ingest.py

4. Train with features:
   python examples/train_with_features.py

5. Online retrieval (example usage in a serving app):
   Use the snippet in api/feature_service_online.py to fetch features from Redis for an entity.

Notes & pointers
- This demo uses local filesystem for the offline store (parquet files) to keep it simple.
  For production, set offline store to S3 (e.g., aws s3://bucket/path) and configure Feast accordingly.
- For production online store you may use Redis (cluster), Amazon DynamoDB, or Feast's supported stores.
- For multi-tenant Aegis you should namespace feature tables by tenant (prefix keys or use per-tenant feature stores).
- Materialize step writes feature values to the online store; r/o TTLs are controlled in the FeatureView config.
- Extend ingestion to write to object storage and schedule materialize-incremental in k8s CronJob for production.

```
feast/README.md
