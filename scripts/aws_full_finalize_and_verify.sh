#!/usr/bin/env bash
set -euo pipefail
#
# aws_full_finalize_and_verify.sh
#
# Top-level orchestrator to:
#  1) Attach strict S3 + KMS policies to IRSA role (creates managed policies if missing)
#  2) Trigger staging backup -> restore drill
#  3) Verify restore artifacts (size, SSE-KMS metadata) and download samples
#  4) Tag backup objects for chargeback
#  5) Produce instructions/artifacts for running Milvus restore in staging
#
# Relies on the helper scripts present in the repo:
#  - scripts/aws_attach_strict_policies.sh
#  - scripts/aws_backup_restore_drill.sh
#  - scripts/aws_verify_restore_integrity.sh
#  - scripts/aws_tag_backup_objects.sh
#
# Usage:
#   ./scripts/aws_full_finalize_and_verify.sh \
#     --bucket my-bucket \
#     --kms-arn arn:aws:kms:us-west-2:123456789012:key/abcd... \
#     --irsa-role aegis-backup-irsa-role \
#     --namespace aegis \
#     --backup-cronjob aegis-milvus-backup \
#     --tag cost-center=test \
#     [--run-workflow local|gh] [--dry-run]
#

print_usage() {
  cat <<EOF
Usage: $0 --bucket BUCKET --kms-arn KMS_ARN --irsa-role ROLE_NAME [options]

Options:
  --namespace NS               (default: aegis)
  --backup-cronjob NAME        (default: aegis-milvus-backup)
  --tag key=value              tag to apply to backup objects for chargeback
  --run-workflow local|gh      (default: local) gh will dispatch workflow instead of running local drill
  --dry-run                    print actions but don't execute cloud changes
  -h, --help
EOF
}

# defaults
NAMESPACE="aegis"
BACKUP_CRONJOB="aegis-milvus-backup"
RUN_WORKFLOW="local"
DRY_RUN="false"
TAG_ARG=""

# args
while [ $# -gt 0 ]; do
  case "$1" in
    --bucket) BUCKET="$2"; shift 2;;
    --kms-arn) KMS_ARN="$2";](#)

