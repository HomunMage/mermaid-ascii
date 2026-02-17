#!/bin/bash
# rollback.sh — Reset to last checkpoint
# Usage: ./scripts/rollback.sh [commit-hash]

cd "$(dirname "$0")/.." || exit 1

if [ -n "$1" ]; then
  echo "Rolling back to: $1"
  git reset --hard "$1"
else
  echo "Rolling back to last commit"
  git reset --hard HEAD
fi

echo "Rollback complete. Current state:"
git log --oneline -3
