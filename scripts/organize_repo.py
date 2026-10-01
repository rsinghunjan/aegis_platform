102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
                match = True
                break
        if match:
            dst = SCRIPTS_TARGET / name
            move_path(p, dst, dry_run)

def archive_readme(dry_run: bool):
    r2 = ROOT / "README2.md"
    canonical = ROOT / "README.md"
    if r2.exists():
        dst = ARCHIVE_READMES / "README2.md"
        move_path(r2, dst, dry_run)
        note = "\n\n> NOTE: README2.md was archived to docs/archived_readmes/README2.md during consolidation.\n"
        if canonical.exists():
            if dry_run:
                print(f"[DRY] would append archival note to {canonical}")
            else:
                with canonical.open("a", encoding="utf-8") as f:
                    f.write(note)
                print(f"Appended archival note to {canonical}")

def plan_git_commands():
    print("\n--- Suggested git commands (review before running) ---")
    print("git checkout -b chore/repo-consolidation")
    print("git add -A")
    print('git commit -m "chore: consolidate scripts, archive binaries and duplicates"')
    print("git push --set-upstream origin chore/repo-consolidation")
    print("Open PR and review files before merging")
    print("-----------------------------------------------------\n")

def main(dry_run: bool):
    print(f"Repo organizer running (dry_run={dry_run})\nROOT={ROOT}\n")
    ensure_dirs()
    files = find_files(ROOT)
    # 1) archive duplicates
    archive_duplicates(files, dry_run)
    files = find_files(ROOT)
    # 2) move binaries
    move_binaries(files, dry_run)
    files = find_files(ROOT)
    # 3) move scripts into package scripts/
    move_scripts(files, dry_run)
    # 4) archive README2.md
    archive_readme(dry_run)
    plan_git_commands()
    print("Done. Review moved files and run the git commands above when ready.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Show planned moves without executing")
    args = ap.parse_args()
    main(args.dry_run)
scripts/organize_repo.py
