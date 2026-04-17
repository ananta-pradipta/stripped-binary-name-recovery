#!/bin/bash
# ============================================================
# Setup symlinks from webapp/ to the parent project directory.
#
# This avoids duplicating src/, data/, checkpoints/, etc.
# Run this once after cloning or unzipping.
#
# Usage: cd bfnr-project/webapp && bash setup.sh
# ============================================================
set -e

WEBAPP_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$WEBAPP_DIR")"

echo "=========================================="
echo "  Webapp Setup"
echo "=========================================="
echo "  Webapp dir:    $WEBAPP_DIR"
echo "  Project root:  $PROJECT_ROOT"
echo ""

# Directories to symlink from the project root
SHARED_DIRS="src data checkpoints configs demo results"

for dir in $SHARED_DIRS; do
    target="$PROJECT_ROOT/$dir"
    link="$WEBAPP_DIR/$dir"

    if [ -L "$link" ]; then
        echo "  [ok] $dir -> already symlinked"
    elif [ -d "$link" ]; then
        echo "  [skip] $dir -> exists as real directory (remove it first if you want a symlink)"
    elif [ -d "$target" ]; then
        ln -s "../$dir" "$link"
        echo "  [created] $dir -> ../$dir"
    else
        echo "  [missing] $dir -> $target does not exist"
    fi
done

echo ""

# Verify critical files
CRITICAL_FILES=(
    "data/bpe_model/bpe.model"
    "data/external_calls/external_vocab.json"
    "src/models/function_namer.py"
    "src/preprocessing/parse_bap.py"
    "src/evaluation/metrics.py"
)

echo "Checking critical files..."
all_ok=true
for f in "${CRITICAL_FILES[@]}"; do
    if [ -f "$WEBAPP_DIR/$f" ]; then
        echo "  [ok] $f"
    else
        echo "  [MISSING] $f"
        all_ok=false
    fi
done

echo ""

# Check checkpoints
ckpt_count=$(find "$WEBAPP_DIR/checkpoints" -name "*.pt" 2>/dev/null | wc -l)
if [ "$ckpt_count" -gt 0 ]; then
    echo "Checkpoints: $ckpt_count .pt files found"
else
    echo "Checkpoints: NONE found. Run 'python download_checkpoints.py' or check symlink."
fi

echo ""
if $all_ok; then
    echo "Setup complete. Run: streamlit run app.py"
else
    echo "Some files are missing. Check that the project root has been fully set up."
fi
echo "=========================================="
