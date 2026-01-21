# Speculative Decoding Pilot Plan

Goal: Safely evaluate speculative decoding effectiveness (latency, cost, energy, safety) and validate metrics before gradual rollout.

Phases:
1. Dev & Unit
   - Deploy student and teacher test endpoints in staging.
   - Deploy `services/speculative/orchestrator.py` and `services/serving/speculative_proxy.py`.
   - Configure env vars (SPEC_ACCEPT_THRESHOLD, BATCH_INTERVAL_MS, etc).
   - Run simple unit tests to exercise acceptance logic.

2. Shadow Mode (2 weeks)
   - Enable Shadow: orchestrator runs in "shadow" where it logs student proposals and teacher verification results but returns teacher-only outputs to users.
   - Collect metrics: acceptance_rate, tokens_saved (simulated), teacher_steps_saved if accepted, teacher_score distributions and safety rejections.
   - Review metrics daily and inspect sample artifacts (stored in /var/log/aegis/speculative_events.jsonl).

3. Canary (1-2 weeks)
   - Enable actual speculative decoding for 1% of traffic for low-risk endpoints.
   - Compare p95 latency, throughput, and safety signals (verifier scores, DLP hits).
   - Maintain circuit breakers if acceptance_rate < X or verifier fail rate rises.

4. Incremental Rollout
   - Increase traffic share slowly (1% → 5% → 10% → 25%) while monitoring.
   - Keep conservative thresholds (L=8, ACCEPT_THRESHOLD high).
   - Stop rollout if safety or quality regressions exceed defined SLO bounds.

Metrics to track
- aegis_spec_acceptance_rate
- aegis_spec_tokens_saved_total
- aegis_spec_tokens_proposed_total
- aegis_spec_fallback_count_total
- p95 latency spec vs baseline
- verifier pass-rate for speculative outputs
- CO2e and kWh estimated saved

Safety & Guardrails
- Speculative disabled for risk_level == "high_risk".
- Circuit breaker that disables speculative mode if acceptance_rate < 0.25 or verifier fail-rate increases > 2x baseline.
- Manual approval via Manual Approval service for any flagged promotions or experiments.

Artifacts & Reporting
- Collect logs: /var/log/aegis/speculative_events.jsonl
- Daily dashboard: Grafana panels for acceptance_rate, tokens_saved, p95 latency, energy saved.
- Run locust harness (ops/tests/locust_speculative_locustfile.py) to gather load/latency numbers.

Rollback criteria
- Safety regression: verifier precision drop > 5% absolute on canary traffic.
- Latency regression: p95 of non-spec path degraded or user-facing errors increase.
- Acceptance collapse: acceptance_rate < 0.2 for 1 hour.

Runbook
1. Start orchestrator and proxy in staging.
2. Configure shadow mode flag (env or route) and run for 7–14d.
3. Run locust pilot and collect baseline & spec metrics.
4. Review with SME/safety team and decide canary enablement.
