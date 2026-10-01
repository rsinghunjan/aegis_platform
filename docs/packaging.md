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
 26
 27
# Packaging & dependency policy (canonical)

We use pip-compile (pip-tools) with the following conventions:

- requirements.in — the canonical top-level dependency list (source-of-truth). You should add top-level direct dependencies here (no transitive pins).
- requirements.txt — generated lock file produced by `pip-compile requirements.in` (committed). CI installs from this file.
- Archive legacy requirements files under `archive/requirements/` (consolidation script moves them).
- No editable/development-only artifacts should be placed into requirements.in; use an optional dev-requirements.in for development-only deps and compile to dev-requirements.txt.

Local workflow
1. Consolidate legacy files:
   - python3 scripts/consolidate_requirements.py --dry-run
   - python3 scripts/consolidate_requirements.py
2. Generate locked file:
   - bash scripts/generate_locked_requirements.sh
3. Review and commit:
   - git add requirements.in requirements.txt
   - git commit -m "chore(deps): add canonical requirements.in and generated requirements.txt"
4. Push and open PR.

CI expectations
- CI verifies that requirements.txt is up-to-date with requirements.in by running pip-compile and diffing the result. If it differs the job fails and requests regenerate + commit.
- CI installs from requirements.txt (locked) to ensure reproducible builds.

Notes
- For security, run periodic dependency scans (Dependabot is enabled) and do not auto-merge major updates without integration testing.
- For local development, you may create a virtualenv and `pip install -r requirements.txt`.
docs/packaging.md
