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
# Aegis Model Serving API (reference)

Files added:
- openapi.yaml           (OpenAPI 3.0 spec)
- api_server.py          (FastAPI reference server)
- requirements-api.txt   (dependencies)

## Quickstart (local, dev)

1. Create a venv and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r api/requirements-api.txt
```

2. Run locally:
```bash
python -m api.api_server
```

3. Browse docs:
- Swagger UI: http://localhost:8080/docs
- OpenAPI JSON: http://localhost:8080/openapi.json
- Health: http://localhost:8080/health
- Readiness: http://localhost:8080/ready

## Notes for productionization

- Replace the ModelRegistry stub in `api/api_server.py` with a model loader that:
  - loads model artifacts from `model_registry/` or an artifact store
  - supports GPU device placement and warmup
  - exposes model metadata/signatures for the OpenAPI spec

- Add authentication/authorization middleware (JWT/OAuth2) and require tokens on predict endpoints.

- Add streaming/file handling for large inputs (don't load large files fully in memory in the API process).

- Enable OpenTelemetry tracing and Prometheus metrics (instrument predict latencies, error rates, concurrency).

- Add Alembic migrations + Postgres for usage logging, job state, and model metadata.

- Add K8s manifests/Helm chart with proper liveness/readiness, resource requests/limits, and HPA/VPA.

## Next step options I can implement for you
- Wire ModelRegistry to a real model artifact (tell me which runtime: torch/onnx/tf).
- Add JWT auth and example token issuance endpoint.
- Add a simple Celery/RQ job queue and a /jobs endpoint for async workloads.
If you want one of these now, tell me which and I’ll implement it.
