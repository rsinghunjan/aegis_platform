```markdown
# Quickstart (consolidated repository)

This quickstart shows the minimal steps to run the orchestrator locally, run the smoke test, and perform repository consolidation safely.

Prereqs
- Python 3.10+
- pip and virtualenv
- Optional: pip-tools (pip-compile) for lock generation

Local dev steps
1. Create a branch:
   git checkout -b chore/repo-consolidation

2. Preview consolidation moves (dry-run):
   python3 scripts/organize_repo.py --dry-run
   Review printed moves carefully.

3. Execute consolidation (if OK):
   python3 scripts/organize_repo.py
