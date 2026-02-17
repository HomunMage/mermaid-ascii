#!/bin/bash
# checkpoint.sh — Git commit current state as a checkpoint
# Usage: ./scripts/checkpoint.sh "message"

cd "$(dirname "$0")/.." || exit 1

MSG="${1:-auto checkpoint}"
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')

git add -A
git commit -m "checkpoint: ${MSG} [${TIMESTAMP}]" --no-verify 2>/dev/null

echo "checkpoint saved: ${MSG}"
