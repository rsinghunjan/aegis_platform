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

## Build

```bash
npm run build
```

Outputs a static bundle to `dist/`, which can be served by any static file
server or embedded behind the control plane's reverse proxy.
