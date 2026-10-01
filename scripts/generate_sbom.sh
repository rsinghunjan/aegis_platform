
#!/usr/bin/env bash
# Generate SBOM for a file or image using syft
set -euo pipefail

TARGET=${1:-}
if [ -z "$TARGET" ]; then
  echo "Usage: $0 <image-or-file>"
  exit 2
fi

if ! command -v syft >/dev/null 2>&1; then
  echo "syft not installed. Install from https://github.com/anchore/syft"
  exit 3
fi

syft "$TARGET" -o spdx-json > sbom.json
echo "Wrote sbom.json"
