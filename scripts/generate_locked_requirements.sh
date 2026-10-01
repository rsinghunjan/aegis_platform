
#!/usr/bin/env bash
# Generate requirements.txt using pip-compile from requirements.in
set -euo pipefail
REQ_IN="requirements.in"
REQ_OUT="requirements.txt"
if [ ! -f "${REQ_IN}" ]; then
  echo "ERROR: ${REQ_IN} not found. Create it first."
  exit 1
fi
if ! command -v pip-compile >/dev/null 2>&1; then
  echo "pip-compile not found. Installing pip-tools..."
  python -m pip install --user pip-tools
fi
pip-compile --output-file "${REQ_OUT}" --generate-hashes "${REQ_IN}"
echo "Generated ${REQ_OUT} (commit this file for reproducible installs)."
