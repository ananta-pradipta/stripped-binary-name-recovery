#!/bin/bash
# ============================================================
# STEP 1: Setup Environment (run inside WSL2 Ubuntu)
# ============================================================
# This installs ALL system packages, Python, PyTorch, BAP, etc.
# Run: bash scripts/01_setup_environment.sh
# Time: ~15-30 minutes depending on internet speed
# ============================================================

set -e  # Stop on any error

echo "═══════════════════════════════════════════════"
echo " Step 1: Setting Up Environment"
echo "═══════════════════════════════════════════════"

# ── 1.1 System packages ──
echo ""
echo "── 1.1 Installing system packages ──"
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    build-essential git curl wget \
    python3 python3-pip python3-venv \
    cmake clang pkg-config \
    m4 unzip zlib1g-dev libffi-dev \
    libgmp-dev libssl-dev \
    binutils elfutils \
    autoconf automake libtool

echo "✓ System packages installed"

# ── 1.2 Create project directory ──
echo ""
echo "── 1.2 Creating project directory ──"
PROJECT_DIR="$HOME/bfnr-project"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Creating $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
else
    echo "$PROJECT_DIR already exists"
fi
cd "$PROJECT_DIR"

# ── 1.3 Python virtual environment ──
echo ""
echo "── 1.3 Setting up Python virtual environment ──"
python3 -m venv bfnr-env
source bfnr-env/bin/activate
pip install --upgrade pip setuptools wheel

echo "✓ Virtual environment created: bfnr-env"

# ── 1.4 PyTorch ──
echo ""
echo "── 1.4 Installing PyTorch ──"
# Check if NVIDIA GPU is available via WSL
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected! Installing PyTorch with CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No GPU detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Verify
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed. CUDA: {torch.cuda.is_available()}')"
echo "✓ PyTorch installed"

# ── 1.5 PyTorch Geometric ──
echo ""
echo "── 1.5 Installing PyTorch Geometric ──"
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python3 -c "import torch; print(torch.__version__.split('+')[0])")+$(python3 -c "import torch; print('cu121' if torch.cuda.is_available() else 'cpu')").html 2>/dev/null || echo "Note: Some PyG optional deps may have failed, but core PyG should work"

# Verify
python3 -c "from torch_geometric.nn import GATConv; print('✓ PyTorch Geometric installed')"

# ── 1.6 Other Python packages ──
echo ""
echo "── 1.6 Installing other Python packages ──"
pip install \
    sentencepiece \
    scikit-learn \
    networkx \
    pandas \
    numpy \
    tqdm \
    pyyaml \
    matplotlib \
    seaborn \
    pytest \
    jsonlines

echo "✓ Python packages installed"

# ── 1.7 Install BAP (Binary Analysis Platform) ──
echo ""
echo "── 1.7 Installing BAP ──"
echo "This takes 10-20 minutes..."

# Install OPAM (OCaml package manager)
if ! command -v opam &> /dev/null; then
    echo "Installing OPAM..."
    sudo apt install -y opam
    opam init --auto-setup --disable-sandboxing -y
    eval $(opam env)
else
    echo "OPAM already installed"
    eval $(opam env)
fi

# Install BAP
echo "Installing BAP via OPAM (this is slow, be patient)..."
opam install bap -y 2>&1 | tail -5
eval $(opam env)

# Verify
if command -v bap &> /dev/null; then
    echo "✓ BAP installed: $(bap --version)"
else
    echo "⚠ BAP installation may have failed. Trying alternative..."
    echo "  You can also use Docker: docker pull binaryanalysisplatform/bap"
    echo "  Or install from source: https://github.com/BinaryAnalysisPlatform/bap"
fi

# ── 1.8 Freeze requirements ──
echo ""
echo "── 1.8 Freezing requirements ──"
pip freeze > requirements_frozen.txt
echo "✓ Requirements frozen to requirements_frozen.txt"

# ── 1.9 Add activation helper ──
echo ""
echo "── 1.9 Creating activation script ──"
cat > "$PROJECT_DIR/activate.sh" << 'ACTIVATE'
#!/bin/bash
# Source this file to activate the environment:
#   source activate.sh
cd ~/bfnr-project
source bfnr-env/bin/activate
eval $(opam env 2>/dev/null) || true
export PROJECT_ROOT=$(pwd)
echo "✓ Environment activated. Project root: $PROJECT_ROOT"
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  BAP: $(bap --version 2>/dev/null || echo 'not found')"
ACTIVATE
chmod +x "$PROJECT_DIR/activate.sh"

echo ""
echo "═══════════════════════════════════════════════"
echo " ✓ ENVIRONMENT SETUP COMPLETE"
echo "═══════════════════════════════════════════════"
echo ""
echo " To activate in the future, run:"
echo "   source ~/bfnr-project/activate.sh"
echo ""
echo " Next step: bash scripts/02_compile_dataset.sh"
echo "═══════════════════════════════════════════════"
