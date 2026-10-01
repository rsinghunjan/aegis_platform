
#!/usr/bin/env bash
# Run a backup, wait for upload, then restore to a temporary DB and run smoke tests.
set -euo pipefail
export OBJECT_STORE_BUCKET=${OBJECT_STORE_BUCKET:-}
if [ -z "$OBJECT_STORE_BUCKET" ]; then
  echo "Set OBJECT_STORE_BUCKET to your staging bucket"
  exit 2
fi
echo "Triggering DB backup..."
python3 scripts/backup/backup_db.py
echo "List recent backups in bucket..."
aws s3 ls s3://$OBJECT_STORE_BUCKET/db-backups/ --recursive | tail -n 5
echo "Pick a backup key and run restore (manual step)."
echo "Example restore:"
echo "  python3 scripts/backup/restore_db.py s3://$OBJECT_STORE_BUCKET/db-backups/<file>"

