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
#!/usr/bin/env bash
set -euo pipefail
echo "1) run scanner (strict)"
python3 scripts/scan_cloud_sdk_usage.py --strict

echo "2) run unit tests"
pytest -q

echo "3) terraform plan (overlay must be configured)"
cd infra/terraform/overlays/aws
terraform init
terraform plan -var="db_password=CHANGEME"

echo "4) helm template (validate)"
cd "$(git rev-parse --show-toplevel)"
helm template aegis ./helm -f helm/values.aws.yaml > /tmp/aegis.yaml
kubectl apply --dry-run=client -f /tmp/aegis.yaml

echo "5) verifier smoke test (needs AWS creds via OIDC or env)"
python3 scripts/verify_model_signatures.py

echo "6) orchestrator verification unit test"
pytest -q tests/test_orchestrator_verify_enforcement.py

echo "If steps above pass and infra was applied, proceed to actual helm deploy and canary test per runbook."
scripts/verify_end_to_end.sh
