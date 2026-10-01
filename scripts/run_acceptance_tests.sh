
 
#!/usr/bin/env bash
set -euo pipefail
# Run pytest for acceptance tests (expects local dev stack running)
pytest -q tests/acceptance/test_storage_acceptance.py
