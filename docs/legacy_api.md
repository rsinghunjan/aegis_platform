# Legacy API status and migration

## Decision

The conditional `/legacy` mount has been removed. Its only declared route was
not reachable: `api/api_server.py` is an incomplete fragment with an unterminated
triple-quoted string, so it cannot be imported as a Python module. The mount
therefore could not load the route, and no deprecation response header or
removal period applies. This retirement is effective immediately with this
change (2026-10-01). `AEGIS_MOUNT_LEGACY_API` is no longer read; setting it has
no effect.

## Route inventory

| Status | Method and path | Original purpose | Current usage and replacement |
| --- | --- | --- | --- |
| Retired; never mounted by `production:app` | `POST /legacy/v1/models/{model_name}/versions/{version}/predict-multipart` | Reference multipart text, image, or audio prediction, declared in the `api/api_server.py` fragment and described by `api/openapi.yaml`. | No in-repository deployment enables the mount flag, and the route has no usage telemetry. External deployment usage cannot be determined from this repository. The canonical app does not expose general model prediction; `/ai/answer` is a tenant-authorized RAG workflow, not a drop-in replacement. Deploy a separately maintained model-serving service if multipart model inference is required. |

The supported canonical app is `production:app` (also the command in the root
`Dockerfile`). Its routes are health/readiness (`/healthz`, `/readyz`), AI
workflows (`/ai/knowledge`, `/ai/answer`), agent runs and approvals
(`/agent/runs...`, `/agent/approvals...`), and operator read models
(`/operator/agent/...`). It does not serve the legacy model-prediction route.

The source audit found no deployment configuration that sets the mount flag;
the root Dockerfile starts `production:app`. Older examples invoke other
servers directly: the Celery Compose file and hardened Dockerfile invoke the
unimportable `api.api_server`, while the integration Compose file starts
`api.api_server3` (which reuses `api.api_server2`). These are separate
historical model-serving implementations, not routes mounted by this flag or
supported production deployment examples. The integration Compose example is
the only in-repository deployment use found for that separate variant. No
external deployment inventory or usage metrics are available here. Operators
should check their own deployment manifests and traffic before upgrading if
they have copied or modified these legacy files.

## Dependencies and migration

`api/requirements-api.txt` is a historical dependency list for the abandoned
reference API, not an installable extra for `production:app`. Its older FastAPI
and Pydantic pins are separate from the canonical
[`requirements-control-plane.txt`](../requirements-control-plane.txt), which
is the only dependency file used by the root production `Dockerfile`. Do not
combine the legacy pins with the control-plane requirements.

There is no compatible in-repository model-serving replacement to migrate to.
Deploy and maintain a model-serving application separately if this functionality
is needed; do not redirect prediction requests to `/ai/answer`. The canonical
routes and their authorization requirements are documented in the
[repository README](../README.md).
