# OPA Policies (Pattern 2)

This repo uses OPA "Pattern 2" decisions:

- Query: `data.aegis.decision`
- Response: an object with keys:
  - `allow` (bool)
  - `reason` (string)
  - `policy_id`, `policy_version` (strings)
  - `obligations` (array of objects, each with at least `"type"`)
  - `labels` (object)

## Local test (example)

Run OPA:

```bash
opa run --server --addr :8181 .
```

Query:

```bash
curl -s http://localhost:8181/v1/data/aegis/decision \
  -H 'content-type: application/json' \
  -d '{"input":{"request":{"request_id":"r1","ts":"2026-04-22T00:00:00Z","action":"deploy.request","actor":{"principal_id":"p1","principal_type":"user","org_id":"org1","roles":[],"claims":{}}},"resource":{"type":"deployment","id":"dep1","attributes":{"org_id":"org1","project_id":"proj1","environment_id":"env1","data_classification":"pci"}},"environment":{"id":"env1","risk_tier":"pci","cloud_targets":["kubernetes"],"region_constraints":["us-east-1"],"budget":{}},"policy":{"bundle_sha256":"x","pin_scope_used":"environment","mode":"central+sidecar","engine":"opa-central-http"}}}'
```
