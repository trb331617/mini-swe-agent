#!/usr/bin/env bash

cat >> /etc/hosts <<'EOF'
0.0.0.0 github.com
0.0.0.0 api.github.com
0.0.0.0 raw.githubusercontent.com
0.0.0.0 patch-diff.githubusercontent.com
0.0.0.0 codeload.github.com
0.0.0.0 objects.githubusercontent.com
0.0.0.0 githubusercontent.com
EOF


if [ $# -lt 2 ]; then
  echo "UASAGE: $0 <repo_dir> <parent_commit>" >&2
  exit 1
fi
cd "$1" || exit 1
parent_commit="$2"


# Paper: https://arxiv.org/pdf/2602.09892
git clean -fd -e '*.egg-info' -e '.tox' -e '.venv' && git checkout "${parent_commit}"

NEW_BRANCH="swe_bench_clean_main"
CURRENT_HEAD=$(git rev-parse HEAD)
git stash -a
git clean -fd
git reset --hard "$CURRENT_HEAD"
git stash pop || echo "No stash to apply or conflict occurred"

git config user.email "pre-agent@swalm.local" && git config user.name "Pre-Agent" \
  && git add . && git commit -m "pre-agent commit"

CURRENT_3_PRIME=$(git rev-parse HEAD)

git for-each-ref --format="%(refname)" refs/remotes/ | xargs -I {} git update-ref -d {}
git tag -l | xargs -r git tag -d

rm -f .git/packed-refs
rm -f .git/ORIG_HEAD .git/FETCH_HEAD .git/MERGE_HEAD .git/CHERRY_PICK_HEAD .git/refs/stash
rm -rf .git/logs/

git update-ref "refs/heads/${NEW_BRANCH}" "$CURRENT_3_PRIME"
git symbolic-ref HEAD "refs/heads/${NEW_BRANCH}"

git for-each-ref --format="%(refname)" refs/heads/ \
  | grep -v "refs/heads/${NEW_BRANCH}" \
  | xargs -I {} git update-ref -d {}

git gc --prune=now --aggressive
