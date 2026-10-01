```markdown
Production testing runbook — integration + load + chaos

Goals
- Validate that autoscaling (HPA/KEDA) responds to load, canary rollouts behave under load, and billing throttles/enforcements trigger correctly.
- Surface OOMs, noisy-neighbor effects, and tail-latency regressions before production.

Test scenarios
1) Baseline functional test
   - Start integration stack, run migrations, seed DB.
   - Run small load (k6) to verify endpoints, auth and job enqueue paths.

2) Sustained load + autoscale validation
   - Run sustained load that pressures queue + prediction throughput.
   - Observe HPA / KEDA metrics (replica count increases) and ensure throughput increases while p95 remains under SLO.

3) Canary under load
   - Deploy a canary version and inject traffic split (Flagger or manual). Validate Flagger analysis metrics: success-rate and p95 remain within thresholds.
   - If analysis fails, Flagger should rollback.

4) Billing throttle enforcement
   - Simulate tenant overuse by generating usage that exceeds daily quota.
   - Verify rate limiter rejects requests and billing endpoints reflect invoice creation and throttling.

5) Chaos: pod or node failure
   - During sustained load, kill an API/worker pod or simulate node preemption.
   - Verify requests failover, queue processing continues, eviction and restart behavior is healthy.

Metrics & thresholds (examples)
- HTTP error rate < 1% during load
- Prediction p95 < 2s (tune per model & device)
- HPA replicas scale up by >1 within 2 minutes of sustained load
- Billing throttles produce 429 responses for offending tenant when daily_quota exceeded

How to run locally
1) Start stack:
   docker compose -f docker/docker-compose.integration.yml up -d --build
2) Wait for health:
   ./scripts/wait_for_services.sh http://localhost:8081/health 120 2
3) Run baseline load:
   API_BASE=http://localhost:8081 ADMIN_USER=admin ADMIN_PASS=adminpass ./scripts/run_k6_load.sh
4) Run chaos:
   # on k8s
   ./scripts/chaos_kill_random_pod.sh default "app=aegis-api"
   # on docker-compose (simulate pause)
   docker compose -f docker/docker-compose.integration.yml pause api && sleep 10 && docker compose -f docker/docker-compose.integration.yml unpause api

CI notes
- CI runners are resource-limited. Use these CI tests for functional/traffic-pattern verification and lightweight load. Run heavier production-like load on a dedicated self-hosted runner or load lab with GPU resources.

Next steps & improvement ideas
- Add Prometheus queries to validate autoscaler reaction (verify metrics before/after).
- Build a "load lab" with a self-hosted GitHub Actions runner (or dedicated runner) sized to run more realistic workloads.
- Add test assertions that fail CI if autoscaling/canary/billing behaviors diverge from expected thresholds (be cautious — flaky infra can cause false positives).
