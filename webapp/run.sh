#!/bin/bash
# Quick start script for the Function Name Recovery web app.
# Usage: cd cs785-project/webapp && bash run.sh
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "  Function Name Recovery Web App"
echo "=========================================="

# Step 1: Create symlinks if any are missing
needs_setup=false
for dir in src data checkpoints configs demo results; do
    if [ ! -e "$dir" ]; then
        needs_setup=true
        break
    fi
done

if $needs_setup; then
    echo "Running setup (creating symlinks)..."
    bash setup.sh
    echo ""
fi

# Step 2: Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found."
    exit 1
fi

# Step 3: Check/install dependencies
echo "Checking dependencies..."
pip3 install -q -r requirements.txt 2>/dev/null || pip3 install -r requirements.txt

# Step 4: Check critical symlinks
for dir in src data checkpoints; do
    if [ ! -e "$dir" ]; then
        echo "ERROR: $dir not found. Run: bash setup.sh"
        exit 1
    fi
done

# Step 5: Check checkpoints
if ! find -L checkpoints -name "*.pt" 2>/dev/null | grep -q .; then
    echo ""
    echo "Model checkpoints not found. Downloading..."
    python3 download_checkpoints.py
fi

# Step 6: Check BAP
if command -v bap &> /dev/null; then
    echo "BAP: $(bap --version 2>&1 | head -1)"
else
    echo "BAP: not installed (live binary analysis will be disabled)"
fi

# Step 7: Start
echo ""
echo "Starting Streamlit..."
echo "Open http://localhost:8501 in your browser"
echo ""
streamlit run app.py