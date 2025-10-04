#!/usr/bin/env bash
set -e

REMOTE_URL="$1"
BRANCH="${2:-main}"

if [ -z "$REMOTE_URL" ]; then
  echo "Usage: $0 <git-remote-url> [branch]"
  echo "Example: $0 git@github.com:username/repo.git main"
  exit 1
fi

# init git if needed
if [ ! -d .git ]; then
  git init
  git checkout -b "$BRANCH"
fi

git add .
git commit -m "Initial commit: face project" || echo "No changes to commit"

# add or update remote
if git remote | grep -q origin; then
  git remote remove origin
fi
git remote add origin "$REMOTE_URL"

# push
git push -u origin "$BRANCH"
echo "Pushed to $REMOTE_URL (branch $BRANCH)"
