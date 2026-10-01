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
155
156
157
158
#!/usr/bin/env bash
# Function to upsert comment on a single PR (by number)
# Uses a marker to identify the bot comment and perform update if present.
COMMENT_MARKER="<!-- aegis-readiness-check -->"
compose_comment_body() {
  cat <<EOF
${COMMENT_MARKER}

Aegis readiness summary (automated):

${payload_text}
EOF
}

# Post to open PRs if enabled and repo/token provided
if [ "$POST_TO_PR" = "true" ] && [ -n "$GITHUB_REPOSITORY" ] && [ -n "$GITHUB_TOKEN" ]; then
  echo "Posting/updating readiness comment on open PRs in ${GITHUB_REPOSITORY}..."

  PRS_JSON=$(curl -sS -H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${GITHUB_REPOSITORY}/pulls?state=open&per_page=100")

  pr_numbers=$(echo "$PRS_JSON" | jq -r '.[].number')

  if [ -z "$pr_numbers" ]; then
    echo "No open PRs found in ${GITHUB_REPOSITORY}."
  else
    for pr in $pr_numbers; do
      echo "Processing PR #$pr ..."
      comments=$(curl -sS -H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/${GITHUB_REPOSITORY}/issues/${pr}/comments")

      # Look for an existing comment containing the marker
      existing_id=$(echo "$comments" | jq -r --arg marker "$COMMENT_MARKER" '.[] | select(.body | contains($marker)) | .id' | head -n1)

      body=$(compose_comment_body)

      if [ -n "$existing_id" ] && [ "$existing_id" != "null" ]; then
        echo "Updating existing comment id ${existing_id} on PR #${pr}"
        update_payload=$(jq -n --arg b "$body" '{body:$b}')
        curl -sS -X PATCH -H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json" \
          "https://api.github.com/repos/${GITHUB_REPOSITORY}/issues/comments/${existing_id}" \
          -d "$update_payload" >/dev/null || echo "Warning: failed to update comment ${existing_id}"
      else
        echo "Creating new comment on PR #${pr}"
        create_payload=$(jq -n --arg b "$body" '{body:$b}')
        curl -sS -X POST -H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json" \
          "https://api.github.com/repos/${GITHUB_REPOSITORY}/issues/${pr}/comments" \
          -d "$create_payload" >/dev/null || echo "Warning: failed to create comment on PR ${pr}"
      fi
    done
  fi
else
  echo "PR posting disabled (POST_TO_PR=${POST_TO_PR}) or missing GITHUB_REPOSITORY/GITHUB_TOKEN. Skipping PR comments."
fi

echo "Done."
scripts/post_readiness_to_slack_and_prs.sh
