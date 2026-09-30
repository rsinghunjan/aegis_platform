# Repository hygiene

The repository contains numerous `*.patch` and `*.diff` files. Treat these as
historical or proposed artifacts; they are not applied automatically and are
not the authoritative runtime implementation.

Canonical paths for the runnable control plane are:

- `production.py` — FastAPI app and health/readiness endpoints.
- `orchestrator.py` — reusable orchestration facade.
- `agentic/runtime.py` — durable planning, policy, execution, verification,
  approvals, and evidence.
- `agentic/remediation.py` — monitoring finding adapter.
- `api/memory.py` — in-process and persistent memory.
- `api/tasks.py` — registered job handler lifecycle.
- `requirements.txt`, `pyproject.toml`, and `Dockerfile` — server dependencies
  and container startup.

Legacy service variants and patch artifacts are retained for history and
compatibility; they should not be imported as the canonical application.
