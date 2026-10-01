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
 28
# Branch protection guidance (require CI + manual review for major bumps)

To enforce the policy "require CI green + manual review for major dependency bumps", apply the following combination of controls:

1) Branch protection settings (Recommended)
   - Require status checks to pass before merging:
     - build-and-test  (CI job that runs unit + smoke tests)
     - staged-integration (heavy integration tests triggered when dependency files change)
   - Require at least 1 approving review before merging.
   - Optionally enable "Require review from Code Owners" if you use CODEOWNERS.
   - Enforce for administrators to make the policy strict.

   You can set protection via the GitHub UI (Settings → Branches → Add rule → main) or via the provided `scripts/setup_branch_protection.sh` (requires gh CLI).

2) Dependabot & automerge policy
   - Disable automerge for major version bumps; if you use Dependabot automerge, configure it to only automerge patch/minor updates.
   - For major bump PRs, rely on the dependency-guard workflow to add the 'major-bump' label and request maintainer attention; maintainers should run staged integration and approve only after validation.

3) Workflows we added
   - `.github/workflows/dependency-guard.yml` — detects major bumps and labels/annotates PRs.
   - `.github/workflows/staged-integration.yml` — runs integration tests on PRs modifying dependency manifests.
   - `.github/workflows/nightly-regression.yml` — nightly regression runs to catch flakes/regressions early.

4) Owner process
   - When a PR is labeled `major-bump`:
     - Ensure `build-and-test` CI passes.
     - Re-run staged integration (the staged integration job should run automatically on dependency file changes).
     - Approve after verifying integration results and any manual testing.
