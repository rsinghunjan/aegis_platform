# Graceful Shutdown & Drain Runbook

Purpose
-------
Ensure in-flight inference requests, model uploads, and background jobs finish cleanly during deployments, scaling events, or node terminations. Minimize dropped requests and avoid corrupting state.

Goals / SLOs
- RTO (service unavailable during deploy): < 2 minutes for rolling updates
- RPO (data loss): zero for committed DB writes; best-effort for in-flight requests (log + retry)
- No dropped high-cost model inferences — prefer draining to completion or enqueue as a job

Key concepts
- Readiness vs liveness: set readiness to false to stop receiving new traffic; keep liveness true until graceful period ends.
- PreStop hook: run preStop to mark instance NotReady and signal in-flight drain.
- TerminationGracePeriodSeconds: set to allow sufficient time for draining (recommended: 120–600s depending on model latency).
- Draining procedure: stop accepting new requests, wait for in-flight requests, flush logs/metrics, checkpoint models or in-flight state if required, then exit.

Kubernetes best practices
- readinessProbe -> /ready (returns false when preparing to shutdown)
- livenessProbe -> /health (keeps container alive until graceful shutdown completes)
- lifecycle.preStop -> curl -X POST http://127.0.0.1:8080/-/shutdown/drain (or run `sleep` + local drain script)
- terminationGracePeriodSeconds: set to 300 (5 minutes) for steady workloads; increase if large batch jobs run

Application-side recommendations
1. Add a /-/shutdown/drain endpoint that:
   - flips an in-memory readiness flag (so /ready returns not ready)
   - stops accepting new predictions (returns 503 quickly)
   - returns current in-flight count and estimated finish time
2. Worker awareness:
   - Celery: use `--statedb` and `--max-tasks-per-child` to avoid long-lived leaked resources; on SIGTERM let worker finish current task (Celery worker handles SIGTERM as graceful shutdown by default).
   - Ensure tasks check for revoke flags for cancellable work.
3. Model server:
   - On drain, stop accepting new gRPC/HTTP requests (toggle readiness)
   - Optionally checkpoint model state or save cache/warmup artifacts
4. Database connections:
   - Use SQLAlchemy connection pool `pool_pre_ping=True` and ensure sessions are closed on shutdown.
5. Signal handling:
   - Run Uvicorn/Hypercorn with `--graceful-timeout` or handle SIGTERM to stop accepting new connections while finishing existing ones.

Example sequence for rolling update
1. K8s sends SIGTERM to Pod; preStop runs and calls /-/shutdown/drain
2. Service proxy stops routing new requests (readiness false)
3. Pod finishes in-flight requests (observed via /ready or metrics)
4. Pod calls shutdown hooks to flush logs and checkpoints
5. Pod exits; Deployment proceeds to next pod

Operational checks & alerts
- Add Prometheus alert: pod readiness flip -> investigate if pods stay NotReady for > X minutes
- Track metrics: in_flight_requests, drain_duration, graceful_shutdown_count, forced_kill_count

Emergency forced kill
- If Pod does not exit after terminationGracePeriodSeconds, kubelet issues SIGKILL. Monitor forced_kill_count and investigate slow drains.

See also
- `k8s/prestop-readiness-lifecycle.yaml` (example manifest)
- `docs/ops/dr_plan.md` (DR playbook)
