```markdown
Kubernetes Autoscaling & Resource Controls — Aegis

What I added
- Helm chart (helm/aegis-api) with Deployment, Service, HPA, VPA and PodDisruptionBudget templates.
- Example production manifests (k8s/deployment-prod.yaml, k8s/hpa-prod.yaml).
- Values file with CPU/memory requests & limits, GPU toggles (nodeSelector + tolerations), and guidance for custom metric scaling.

How to use (quick)
1. Customize values.yaml: set image.repository, image.tag, resource requests/limits and GPU flags.
2. Install via helm:
   helm install aegis-api helm/aegis-api -f helm/aegis-api/values.yaml --namespace aegis --create-namespace

3. Enable HPA autoscaling on CPU by default. For SLO-based scaling:
   - Deploy Prometheus + prometheus-adapter that maps Prometheus metrics (e.g., aegis_prediction_latency_ms or aegis_predictions_total)
     to Kubernetes External metrics API.
   - Set `.Values.hpa.useCustomMetric=true` and configure metric name/target in values.yaml, or edit the HPA template.

Scheduling GPUs
- To schedule model servers on GPU nodes:
  - Set `gpu.enabled: true` and adjust `gpu.count` in values.yaml.
  - The chart will request `nvidia.com/gpu: <count>` in container limits (cluster must have NVIDIA device plugin).
  - Use node labels (example: accelerator=nvidia) in nodeSelector and matching tolerations.

Pod disruption and rolling upgrades
- PDB prevents too many pods being evicted at once (minAvailable in values.yaml).
- Ensure readinessProbe is accurate to avoid traffic sent to unready pods during updates.

VPA & vertical scaling
- VPA can be enabled (values.vpa.enabled=true) if a VPA controller is installed in the cluster.
- VPA works well in stable workloads; use VPA in Recommend or Auto mode with caution for stateful/pinned deployments.

Prometheus Adapter for custom metrics
- To scale on application-level SLOs:
  - Expose metrics (already instrumented by code: aegis_predictions_total, aegis_prediction_latency_ms).
  - Deploy prometheus-adapter and map a Prometheus query to a k8s external metric (see adapter config).
  - Update HPA to reference the external metric (the Helm template includes an example block).

Best practices
- Right-size requests for predictable bin-packing; set requests to what the application normally needs.
- Set conservative limits to allow node autoscaler (cluster autoscaler) to scale nodes appropriately.
- Use separate queues/namespaces for GPU workloads and CPU-only workloads and separate HPA profiles.
- Use PodPriority to ensure critical control-plane pods are not evicted.
- Test scaling behavior in a staging environment before enabling in prod.

Next steps I can implement
- Add a sample prometheus-adapter config that maps aegis metrics to k8s external metrics and update Helm HPA template accordingly.
- Create a Helm values.example optimized for GPU-based model servers.
- Add Kubernetes PodPreset / admission annotations to inject secrets (if you use Vault Agent Injector).
```
