#!/usr/bin/env bash
# Generate output for all examples.
# Usage: bash examples/gen.sh
#
# Renders each .mm.md example and writes the output alongside
# the input as .out.txt.

set -euo pipefail
cd "$(dirname "$0")/.."

# Remove old generated files
find examples -name '*.out.txt' -delete 2>/dev/null || true

# Process all .mm.md files recursively
find examples -name '*.mm.md' | sort | while read -r src; do
    out="${src%.mm.md}.out.txt"
    echo "  $src -> $out"
    uv run python -m mermaid_ascii "$src" -o "$out"
done

echo ""
echo "Done. Generated output:"
find examples -name '*.out.txt' | sort | while read -r f; do
    echo "  $f"
done
