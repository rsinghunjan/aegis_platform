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
- Sensor subscribes to EventSource events and submits an Argo Workflow when a drift alert arrives.
- Workflow runs training (MLflow logging) and then packages & signs the resulting model artifact using your existing packaging/sign helpers.

Files
- argo/events/eventsource-alertmanager.yaml
- argo/events/sensor-retrain.yaml
- argo/workflows/retrain_on_drift_workflow.yaml
- scripts/retrain_trigger_helper.sh

Apply (staging / audit)
1. Install Argo Events and Argo Workflows in your cluster (if not present).
2. Apply EventSource and Sensor:
   kubectl apply -f argo/events/eventsource-alertmanager.yaml
   kubectl apply -f argo/events/sensor-retrain.yaml

3. Apply Workflow RBAC / ServiceAccount used by workflows (ensure it can create pods and use required secrets):
   # ensure aegis-workflow-sa exists and has permissions; use your existing workflow SA if you have one.

4. Test by sending a sample Alertmanager payload:
   curl -XPOST -H "Content-Type: application/json" --data @test-alert.json http://<EVENTSOURCE_HOST>:12000/alertmanager

Example test payload (test-alert.json)
{
  "version": "4",
  "groupKey": "{}:{alertname=\"ModelDrift\"}",
  "status": "firing",
  "receiver": "aegis",
  "groupLabels": {"alertname":"ModelDrift"},
  "commonLabels": {"alertname":"ModelDrift","severity":"critical"},
  "alerts": [
    {
      "status":"firing",
      "labels": {"alertname":"ModelDrift","severity":"critical"},
      "annotations": {"summary":"model drift detected","dataset_uri":"s3://my-staging-bucket/datasets/cifar/v2/"},
      "startsAt":"2025-01-01T00:00:00Z"
    }
  ]
}

Verification
- After posting the alert, the Sensor should create an Argo Workflow (kubectl get wf -n aegis).
- Workflow logs show the training step; MLflow run appears in your MLflow tracking server.
- Package-and-sign step produces an artifact and calls your package_and_sign_vault.sh helper.

Notes & next steps
- Tune Sensor filters to only trigger for desired alerts/labels or severity.
- Secure the EventSource endpoint (ingress, auth) so only Alertmanager or authorized systems can post alerts.
- Consider adding retries, dedup keys, and rate-limits to avoid accidental retrain storms.
- Ensure the workflow's serviceAccount has permission to read secrets (cosign public key or Vault k8s auth) needed by package/sign steps.
```
