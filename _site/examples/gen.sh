#!/usr/bin/env bash
# Generate output for all examples and verify against .expect.txt/.expect.svg files.
# Usage: bash _site/examples/gen.sh          # generate only
#        bash _site/examples/gen.sh --check  # generate + verify against .expect.txt and .expect.svg

set -euo pipefail
cd "$(dirname "$0")/../.."

CHECK=false
if [[ "${1:-}" == "--check" ]]; then
    CHECK=true
fi

# Build Rust binary (release)
echo "Building Rust binary..."
cargo build --release
BINARY="./target/release/mermaid-ascii"

# Remove old generated files
find _site/examples -name '*.out.txt' -delete 2>/dev/null || true
find _site/examples -name '*.out.svg' -delete 2>/dev/null || true

# Process all .mm.md files recursively
find _site/examples -name '*.mm.md' | sort | while read -r src; do
    out_txt="${src%.mm.md}.out.txt"
    out_svg="${src%.mm.md}.out.svg"
    echo "  $src -> $out_txt"
    "$BINARY" "$src" -o "$out_txt"
    echo "  $src -> $out_svg"
    "$BINARY" --svg "$src" -o "$out_svg"
done

echo ""
echo "Done. Generated output:"
find _site/examples -name '*.out.txt' -o -name '*.out.svg' | sort | while read -r f; do
    echo "  $f"
done

# Verify against expected files
if $CHECK; then
    OVERALL_FAIL=0

    echo ""
    echo "Checking against .expect.txt files..."
    FAIL=0
    for expect in _site/examples/*.expect.txt; do
        base="${expect%.expect.txt}"
        out="${base}.out.txt"
        if [[ ! -f "$out" ]]; then
            echo "  MISSING: $out (no output generated for $expect)"
            FAIL=1
            continue
        fi
        if diff -q "$expect" "$out" > /dev/null 2>&1; then
            echo "  OK: $(basename "$base")"
        else
            echo "  FAIL: $(basename "$base")"
            diff --color=auto "$expect" "$out" || true
            FAIL=1
        fi
    done
    if [[ $FAIL -ne 0 ]]; then
        echo ""
        echo "FAILED: some outputs differ from .expect.txt files"
        OVERALL_FAIL=1
    else
        echo ""
        echo "All outputs match .expect.txt files."
    fi

    echo ""
    echo "Checking against .expect.svg files..."
    FAIL=0
    for expect in _site/examples/*.expect.svg; do
        base="${expect%.expect.svg}"
        out="${base}.out.svg"
        if [[ ! -f "$out" ]]; then
            echo "  MISSING: $out (no output generated for $expect)"
            FAIL=1
            continue
        fi
        if diff -q "$expect" "$out" > /dev/null 2>&1; then
            echo "  OK: $(basename "$base")"
        else
            echo "  FAIL: $(basename "$base")"
            diff --color=auto "$expect" "$out" || true
            FAIL=1
        fi
    done
    if [[ $FAIL -ne 0 ]]; then
        echo ""
        echo "FAILED: some outputs differ from .expect.svg files"
        OVERALL_FAIL=1
    else
        echo ""
        echo "All outputs match .expect.svg files."
    fi

    if [[ $OVERALL_FAIL -ne 0 ]]; then
        exit 1
    fi
fi
