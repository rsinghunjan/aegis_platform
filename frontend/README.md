# Aegis AI Operations Dashboard

An operator dashboard for production AI/ML/LLM workflows, talking to the
read-only `/operator/agent/*` API exposed by `production.py`. It surfaces the
governance, evidence, and operational status surrounding agent runs.

## Features
- **AI Run Monitor** — polls `/operator/agent/runs` and shows run status.
- **Governance Queue** — lists pending approvals for operator decisions.
- **AI Evidence Explorer** — renders a run's evidence hash chain and flags breaks.
- **Run Analytics** — aggregates run counts by status.

## Development

```bash
cd frontend
npm install
npm run dev
```

The dev server proxies `/operator` and `/agent` requests to
`http://localhost:8000` (the FastAPI control plane) — see `vite.config.ts`.
It also proxies `/ai` for the knowledge workflow. Set `VITE_AEGIS_TENANT_ID`
to prefill the tenant selector, or enter a tenant ID in the dashboard. A tenant
ID selects a resource only; the server must still be configured with trusted
authentication and tenant authorization, and protected requests fail closed
without those integrations.

The client follows the API contract in `production.py`: operator run reads are
paginated, approval and evidence routes require `tenant_id`, and approval
decisions are posted to `/agent/approvals/{approval_id}/decision`.

## Build

```bash
npm run build
```

Outputs a static bundle to `dist/`, which can be served by any static file
server or embedded behind the control plane's reverse proxy.
