
#!/usr/bin/env bash
# Apply the terraform staging overlay and export outputs for downstream Helm/CI usage.
# Usage: ./scripts/apply_staging.sh infra/terraform/overlays/aws
set -euo pipefail

TF_DIR=${1:-"infra/terraform/overlays/aws"}
WORKDIR=$(realpath "$TF_DIR")
echo "Applying terraform in $WORKDIR"

pushd "$WORKDIR" > /dev/null
terraform init -input=false
terraform plan -out=tfplan -input=false
terraform apply -input=false -auto-approve tfplan

terraform output -json > tf_outputs.json
popd > /dev/null

# convert outputs to env exports for consumption
./scripts/terraform/output_to_env.sh "$WORKDIR/tf_outputs.json" > staging.env
echo "Wrote staging.env; review and source it before running Helm or CI steps."
