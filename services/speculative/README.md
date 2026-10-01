Speculative Decoding Orchestrator

Files:
- services/speculative/orchestrator.py : FastAPI orchestrator with batched teacher scoring and metrics.
- services/serving/speculative_proxy.py : Gateway proxy endpoint that forwards to orchestrator with budget/DLP checks.
- ops/speculative/pilot_plan.md : Pilot plan and rollout steps.
- ops/tests/locust_speculative_locustfile.py : Locust test harness for load testing.

Deployment notes:
- Deploy orchestrator and proxy services as Kubernetes deployments in your inference namespace.
- Ensure STUDENT_ENDPOINT, TEACHER_SCORE_URL, TEACHER_GENERATE_URL env vars are set.
- Hook Prometheus to METRICS_PORT (default 9460).
- Start in shadow mode by not routing real traffic; use ops/speculative/pilot_plan.md to run the staged rollout.

Integration tips:
- Teacher scoring endpoint must accept a batch of {job_id, context, draft} and return corresponding token lists and scores.
- Tokenization must be consistent between student and teacher for acceptance checks.
- For production, ensure all network calls have proper auth and timeouts. Use TLS for endpoints.

Safety:
- This implementation disallows speculative decoding for risk_level == "high_risk".
- Metrics and logging are provided for monitoring acceptance and safety signals.
