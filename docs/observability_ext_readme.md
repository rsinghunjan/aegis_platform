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
```markdown
Observability completeness — dashboards, Alertmanager & Prometheus Adapter

What I added
- prometheus/alertmanager.yml : Alertmanager routes and receivers (Slack + PagerDuty placeholders).
- prometheus/prometheus-adapter-config.yaml : ConfigMap rules to enable Prometheus Adapter to expose P95 and throughput metrics as Kubernetes external metrics.
- grafana/dashboards/aegis-slo-dashboard.json : Grafana dashboard JSON you can import for model-level SLOs.
- k8s/hpa-external-p95.yaml : Example HPA that uses an external metric (P95 latency) to scale the deployment.

How to deploy
1) Alertmanager:
   - Mount `prometheus/alertmanager.yml` into your Alertmanager deployment (or use Helm values.alertmanager.config).
   - Replace the Slack webhook and PagerDuty keys with secrets (do NOT store keys in repo). Use K8s Secret and set `api_url` via templating or secret injection.

2) Prometheus Adapter:
   - Apply the ConfigMap `prometheus/prometheus-adapter-config.yaml` to the `monitoring` namespace (or namespace where adapter runs).
   - Install prometheus-adapter with helm and point it to this ConfigMap; ensure it can reach Prometheus.
   - Validate by running `kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1"` and checking the metric names.

3) Grafana:
   - Import `grafana/dashboards/aegis-slo-dashboard.json` via the Grafana UI (Dashboards -> Import) or provision it via ConfigMap/Helm provisioning.
   - Ensure datasource "Prometheus" is configured.

4) HPA example:
   - After configuring prometheus-adapter so the external metric `aegis_prediction_latency_p95` is available, apply `k8s/hpa-external-p95.yaml`.
   - Tune min/max replicas and target value to match SLO (example target = 500 ms P95).

Secrets & security
- Use Kubernetes Secrets for Slack webhook, PagerDuty integration key, and Prometheus Adapter credentials (if Prometheus is secured).
- Do not commit API keys in the repo. Use sealed-secrets/ExternalSecrets/Vault Agent Injector to provision into cluster.

Verification steps
- Fire a synthetic test load (k6/locust) to increase P95 and observe HPA scaling decisions.
- Trigger a SLO breach (e.g., inject a latency error) and confirm Alertmanager routes to PagerDuty for severity=page.
- Validate Prometheus Adapter metrics appear under `kubectl get --raw /apis/external.metrics.k8s.io/v1beta1 | jq .`.

Notes & next steps I can implement
- I can generate a Grafana provisioning ConfigMap so dashboards are auto-imported on Grafana startup.
- I can create Alertmanager Kubernetes Secret templates and a Helm values patch to wire secrets into Alertmanager securely.
- I can add a small set of Grafana alert rules (in dashboard) and Alertmanager webhook hooks for runbook automation (auto-create Jira/PagerDuty incidents).
- I can help tune the Prometheus Adapter rules to produce metrics with the exact labels your HPA needs (model/version/tenant).

If you want, I will:
- Add the Grafana provisioning ConfigMap and a Helm patch so the SLO dashboard is auto-deployed,
- And create a Helm values example to mount Alertmanager config from a k8s Secret (encrypted via SealedSecrets).
```

If you want, I’ll:
- apply these into your Helm charts (add adapter configmap & provisioning, mount Alertmanager config as secret) and open a branch with the changes, then run a smoke test of Prometheus Adapter + HPA in a small kind cluster and report back with verification steps. Which would you like me to do next?
