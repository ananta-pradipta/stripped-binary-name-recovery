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
