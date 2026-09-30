# Aegis model promotion golden path

```text
Client
  | POST /promote {run_id, model_name, version, tenant_id, user}
  v
Governance API -- MLflow: require an existing FINISHED run + artifact URI
  | SQLAlchemy transaction
  +--> Tenant + Job(payload contains model/version/artifact reference)
  +--> Run(metrics and artifact URI)
  +--> AuditLog(decision, actor, run ID, notes)
  |
  +--> GET /models/{model_name}/promoted?tenant_id=...
       resolves the latest matching approved Job/Run reference

Multimodal API -- POST /generate {query}
  | existing generate_safe_response() + SafetyChecker hook
  v
{response}
```

The promotion API keeps `GET /runs/<experiment_name>` and `POST /promote`.
Promotion JSON accepts `run_id` (required), `model_name` (defaults to the run ID),
`version` (defaults to `1`), `tenant_id` (defaults to `default`), `tenant_name`,
`user`, and `notes`. A completed promotion retains the legacy
`{"ok": true, "run_id": "..."}` response. Validation failures return HTTP 400
and persist a rejected job/audit decision. Missing MLflow configuration can be
provided using `MLFLOW_TRACKING_URI`.

## Local setup

```bash
python -m pip install -r governance/requirements.txt -r requirements-db.txt
# The existing Aegis DB models also require the pgvector Python package.
python -m pip install 'pgvector>=0.3.6'
# Install MLflow to serve real run-listing and promotion requests.
python -m pip install mlflow
export DATABASE_URL=sqlite:///./governance.db
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m governance.api
```

Promote a run and resolve its persisted reference:

```bash
curl -X POST http://localhost:8080/promote \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"<RUN_ID>","model_name":"example","version":"1","tenant_id":"local"}'
curl 'http://localhost:8080/models/example/promoted?tenant_id=local'
```

Run the multimodal inference API separately (install its component requirements
first):

```bash
uvicorn aegis_multimodal_ai_system.app:app --reload
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' -d '{"query":"Hello"}'
```

Targeted tests use fake MLflow runs, SQLite, and a mocked multimodal system; they
do not download models or call external services.

## Scope and production caveats

- The promotion adapter records a model name/version and MLflow artifact URI in
  the existing SQLAlchemy tenant/job/run/audit tables. There is no usable
  persistent registry implementation in `model_registry/` to receive the
  handoff; this persisted reference adapter is deliberately local and minimal.
- With no `DATABASE_URL`, promotion uses `sqlite:///./governance.db` for local
  development. SQLite is not a production durability or concurrency guarantee.
  Production deployments should configure a supported database and apply
  schema changes through their migration process; this path creates only its
  existing required tables for convenience.
- Promotion does not copy, download, sign, checksum-verify, or deploy MLflow
  artifacts. The resolve endpoint returns a reference, not a loaded model.
  Deployment, artifact trust, rollback, caller authentication, and durable
  audit retention remain operator responsibilities.
- `/generate` invokes the existing multimodal system, which loads models lazily.
  The current `SafetyChecker.is_unsafe()` implementation is a no-op placeholder;
  it is wired as an API hook, not a production safety control. The endpoint
  does not automatically load the promoted MLflow artifact.
