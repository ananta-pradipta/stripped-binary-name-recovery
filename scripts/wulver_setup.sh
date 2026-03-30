#!/bin/bash
# First-time setup on Wulver HPC
# Run this AFTER uploading data with wulver_upload.sh
# Run this ON Wulver (after SSH-ing in)
#
# Usage: bash cs785/scripts/wulver_setup.sh

COURSE_DIR="/course/2026/spring/cs/785/hz79/adp232"
cd ${COURSE_DIR}

echo "=== Setting up Python environment on Wulver ==="

# Load Python module
module load bright 2>/dev/null
module load Python/3.10 2>/dev/null || module load python3 2>/dev/null || module load python/3.10 2>/dev/null || {
    echo "Could not load Python 3.10 module. Checking available versions..."
    module avail Python 2>&1 | head -20
    echo "Load the appropriate Python module manually, then re-run this script."
    exit 1
}

echo "Python: $(python3 --version)"

# Create virtual environment
if [ ! -d "cs785-env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv cs785-env
else
    echo "Virtual environment already exists."
fi

source cs785-env/bin/activate

# Install dependencies
echo "Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing other dependencies..."
pip install pyyaml sentencepiece tqdm

# DGL may need special install for the CUDA version available
echo "Installing DGL..."
pip install dgl -f https://data.dgl.ai/wheels/torch-2.5/cu121/repo.html 2>/dev/null || \
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html 2>/dev/null || \
pip install dgl

# Verify
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import dgl
print(f'DGL: {dgl.__version__}')
import yaml, sentencepiece, tqdm
print('All dependencies OK')
"

echo ""
echo "=== Setup complete! ==="
echo "To train: cd cs785 && sbatch scripts/wulver_train.sbatch"
