  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
# SLO and Autoscale guidance for Aegis Inference

This document explains recommended SLOs, autoscaling signals, and batch tuning for a production pilot.

1) Define SLOs (example)
- Availability: 99.5% for inference HTTP endpoint (5xx/total < 0.5% in 30d)
- Latency:
  - P99 < 1.5s for single-request latency (for the pilot; adjust to needs)
  - P50 < 200ms for low-latency interactive use
- Safety: < 0.1% false-negative rate on safety-critical inputs (requires tests & manual review)

2) Autoscaling signals (prioritized)
- CPU utilization (k8s resource metric) — good baseline for compute-bound workloads.
- Custom metrics (recommended):
  - aegis_inference_queue_depth (exposed via a gauge): scale up if queue_size per pod > X
  - aegis_inference_latency_seconds (P95) via Prometheus adapter to HPA (requires prometheus-adapter)
  - aegis_inference_requests_per_second per pod (rate) — scale on QPS
- For GPU workloads:
  - scale by custom metrics (e.g., GPU utilization or pending queue length) rather than CPU.

3) Batching tradeoffs & tuning
- Batch size increases throughput but increases tail latency for small/interactive requests.
- Configure:
  - max_batch_size: start with 4–8 for CPU models, 16–32 for small GPU models.
  - max_latency_ms: 20–100ms (shorter for interactive).
- Tuning approach:
  - Run loadtest for expected QPS and measure throughput vs P95/P99 latency.
  - Increase batch size until P99 approaches SLO, then back off.

4) Readiness & health
- /health must include model_version and a boolean "loaded". Treat not-loaded as NotReady.
- Use readinessProbe for deployment to avoid routing to pods that are still warming up.

5) Resource sizing
- Start with conservative values:
  - CPU requests: 250m, limits: 2000m
  - Memory depends on model size (2–8 GiB typical for medium models)
  - GPU: 1 GPU for heavy models; ensure node pool autoscaling enabled
- Use vertical profiling to adjust.

6) Fast rollback & circuit-breaker
- Implement circuit-breaker: if model latency or 5xx rate exceeds thresholds, mark instance unhealthy and route to fallback.
- Provide a canary deployment path and traffic-split by model_version to test new models.

7) Observability & dashboards
- Key metrics:
  - aegis_inference_requests_total{model_version}
  - aegis_inference_latency_seconds (histogram)
  - aegis_safety_flags_total (by reason)
  - aegis_inference_queue_depth (gauge)
- Alerts:
  - P95 latency > SLO for > 5m
  - Queue depth per pod > threshold for > 5m
  - Safety flags spike (rate) > baseline

8) CI / staging checklist before scaling to production pilot
- Model artifact checksum and signature verification in registry
- Load testing for expected QPS (with safety checks turned on)
- End-to-end test verifying health/readiness and graceful shutdown
- Auth and secrets configured (mTLS/OIDC) and tested

9) Next steps for pilot
- Expose queue depth metric (aegis_inference_queue_depth) and integrate with prometheus-adapter to drive HPA
- Add canary/traffic-splitting layer (Linkerd/NGINX/Traefik)
- Add autoscaler policies for GPU node pools if using GPUs

Notes
- Using batching means autoscaling should consider the effective throughput per pod (not raw QPS).
- For low-latency interactive use prefer small batch sizes and horizontal autoscaling. For high-throughput offline inference use larger batches and fewer, larger nodes.
