#!/bin/bash
# Batch decompile cross-project binaries with Ghidra for SymGen comparison
# Runs Ghidra headless to decompile stripped binaries → JSON

set -euo pipefail
cd "$(dirname "$0")/.."

GHIDRA_HOME="<home>/ghidra"
GHIDRA_HEADLESS="$GHIDRA_HOME/support/analyzeHeadless"
SYMGEN_DIR="<home>/SymGen"
PROJECT_DIR="/tmp/ghidra_project"
OUTPUT_DIR="data/symgen_decomp"

mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR"

# Fix the output_dir in SymGen's decompilation script
DECOMP_SCRIPT="$SYMGEN_DIR/scripts/decompilation/decomp_for_stripped.py"
# Create a patched version with our output dir
PATCHED_SCRIPT="/tmp/decomp_stripped_patched.py"
sed "s|output_dir = \"\"|output_dir = \"$(pwd)/$OUTPUT_DIR\"|" "$DECOMP_SCRIPT" > "$PATCHED_SCRIPT"

echo "═══════════════════════════════════════════════"
echo " Ghidra Decompilation for SymGen Comparison"
echo " Output: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════"

# Cross-project stripped binaries (handle both naming conventions)
BINARIES=$(ls data/stripped/{rcs,tree,bzip2,nginx,dos2unix,curl,hello,cppi,datamash,csplit2}_* demo/stripped/diffutils_* 2>/dev/null || true)

total=$(echo "$BINARIES" | wc -w)
count=0

for binary in $BINARIES; do
    count=$((count + 1))
    name=$(basename "$binary")
    output_file="$OUTPUT_DIR/${name}.json"

    if [ -f "$output_file" ]; then
        echo "[$count/$total] $name (exists, skipping)"
        continue
    fi

    echo "[$count/$total] Decompiling: $name"

    # Clean project dir for each binary
    rm -rf "$PROJECT_DIR/proj_${name}"*

    # Run Ghidra headless analysis + decompilation
    timeout 600 "$GHIDRA_HEADLESS" \
        "$PROJECT_DIR" "proj_${name}" \
        -import "$binary" \
        -postScript "$PATCHED_SCRIPT" \
        -deleteProject \
        2>&1 | grep -E "write result|REPORT|Error|Exception" || true

    # Check if output was created
    if [ -f "$output_file" ]; then
        echo "  ✓ Done: $(du -h "$output_file" | cut -f1)"
    else
        echo "  ✗ Failed: $name"
    fi
done

echo ""
echo "═══════════════════════════════════════════════"
echo " Done! Decompiled files:"
ls "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l
echo " files in $OUTPUT_DIR"
echo "═══════════════════════════════════════════════"
