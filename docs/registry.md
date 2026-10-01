# Model Registry & Deployment Lifecycle (Aegis)

Overview
--------
This document describes the automated flow for onboarding, validating, and promoting model artifacts.

Key concepts:
- Artifact: model binary or archive (Torch .pt, ONNX, TorchScript, etc.).
- Registration: publishing an artifact to the registry; computes checksum + signature.
- Stages: staging (dev), canary (limited rollout), prod (full rollout).
- Promotion: moving a version between stages. Promotions are recorded in metadata._registry.promotion_history.
- Validation: signature + checksum verification.

Local developer flow
--------------------
1. Add artifact under `model_registry/<model_name>/<version>/` or use the API to upload.
2. On push, CI can generate a metadata.json containing checksum + signature for audit (example workflow below).
3. Operator/admin calls the registry API:
   - POST /v1/registry/register (admin) to register an artifact in DB
   - POST /v1/registry/{model}/{version}/promote to push to canary/prod
   - GET /v1/registry/{model}/{version}/validate to verify signature

CI integration (example)
------------------------
We include a GitHub Action (model-publish.yml) that runs on pushes to `model_registry/**` and:
- computes SHA256 checksum
- signs the stamp with a repository secret `MODEL_SIGN_KEY`
- writes `metadata.json` adjacent to the artifact with checksum/signature
- (optional) creates a PR or pushes commit with metadata

Rollout strategy (recommended)
------------------------------
1. Register: admin registers model version in staging.
2. Smoke tests: run a battery of model QA tests against staging.
3. Canary: promote to canary and route a small % of production traffic to canary instances
   - Use weighted routing at the API gateway or a traffic router (Envoy).
   - Monitor inference latency, error rate, and quality metrics (example: top_k recall).
4. Promote to prod if canary metrics are healthy.
5. Support rollback: promotion history stored in metadata allows reverting stage.

Security & attestation
----------------------
- Registry computes HMAC signature over "model_name|version|checksum" using `AEGIS_MODEL_SIGN_KEY`.
- CI should compute/check similar signature; operator can verify with `/validate` endpoint.
- For stronger guarantees, add artifact signing via GPG or an external signing service and store signatures in a separate attestation store.

Observability
-------------
- Prometheus metrics (aegis_registry_registrations_total, aegis_registry_promotions_total) provided by the registry code.
- Instrument model-serving metrics (latency, error rate) and tie them to model_name/version labels for rollout decisions.

Next steps
----------
- Add an operator that watches GitHub metadata files and auto-registers models (optional).
- Add an automated canary traffic shaper (Envoy + A/B routing + Prometheus alerts).
- Add immutable artifact storage (S3 with versioning) and signed download URLs for workers / model servers.
