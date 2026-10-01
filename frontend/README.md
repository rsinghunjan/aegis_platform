# Aegis Platform Dashboard

A minimal React + TypeScript operator dashboard for the Aegis control plane,
talking to the read-only `/operator/agent/*` API exposed by `production.py`.

## Features
- **Run Monitor** — polls `/operator/agent/runs` and shows live run status.
- **Approval Queue** — lists pending approvals and lets an operator approve/deny them.
- **Evidence Explorer** — renders a run's evidence hash chain and flags breaks.
- **Analytics** — aggregate run counts by status.

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
