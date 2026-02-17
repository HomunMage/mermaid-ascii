#!/usr/bin/env bash
# Generate output for all examples.
# Usage: ./scripts/gen_examples.sh
#
# Builds the project, then renders each .txt example
# and writes the output alongside the input as .out.txt.
# Open the .out.txt files to visually inspect the graph output.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "Building..."
cargo build --release --quiet

BIN=target/release/text-graph

# Remove old generated files
find examples -name '*.out.txt' -delete 2>/dev/null || true

# Process all .txt files recursively
find examples -name '*.txt' ! -name '*.out.txt' | sort | while read -r src; do
    out="${src%.txt}.out.txt"
    echo "  $src -> $out"
    "$BIN" "$src" -o "$out"
done

echo ""
echo "Done. Generated output:"
find examples -name '*.out.txt' | sort | while read -r f; do
    echo "  $f"
done
