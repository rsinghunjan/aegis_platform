# Legacy model-serving API (retired)

The reference multipart prediction endpoint is not part of the supported API.
The historical `api/api_server.py` is an incomplete, syntactically invalid
fragment; `python -m api.api_server` and the former
`AEGIS_MOUNT_LEGACY_API=true` option are not supported. The route was never
available through the canonical `production:app`.

See [`docs/legacy_api.md`](../docs/legacy_api.md) for the route inventory,
retirement decision, dependency status, and migration guidance. Do not install
`api/requirements-api.txt` for the canonical control plane; use
`requirements-control-plane.txt`.
