# Task: BLens Baseline Evaluation

**Project:** CS785 Binary Function Name Recovery
**Date:** 2026-04-09
**Estimated time:** 1-2 days
**Where:** Wulver HPC (GPU required)

---

## What is BLens?

BLens (USENIX Security 2025) is a function name prediction system that uses an ensemble of 4 pretrained embedding models (PalmTree + CLAP + DEXTER + VarCLR) fused with a "LORD" decoder. It reports **cross-project F1=0.46** and **cross-binary F1=0.77** in their paper.

We want to run BLens on our cross-project data to compare with our model (F1=0.650).

---

## Prerequisites

### 1. Access the project on Wulver

```bash
ssh wulver    # Duo 2FA required
ls /project/hz79/_shared/cs785/
# Should see: configs/ data/ src/ scripts/ baselines/ checkpoints/ ...
```

### 2. Wulver environment

```bash
module load bright
module load python3
```

### 3. SLURM job template

All GPU jobs use:
```bash
#!/bin/bash
#SBATCH --job-name=cs785-eval-blens
#SBATCH --account=hz79
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/hz79/_shared/cs785/slurm_logs/cs785-eval-blens.%j.out
```

We have **full NVIDIA A100-80GB** GPUs via the `hz79` account.

---

## Step-by-Step

### Step 1: Clone BLens repo

```bash
ssh wulver
cd /project/hz79/_shared/cs785/baselines/
git clone https://github.com/lmu-plai/blens.git BLens
cd BLens
```

### Step 2: Download data and pretrained models

Their data is on Zenodo: https://doi.org/10.5281/zenodo.14713022

```bash
cd /project/hz79/_shared/cs785/baselines/BLens

# Download data.tar.gz from Zenodo (~20-30GB)
wget 'https://zenodo.org/records/14713022/files/data.tar.gz?download=1' -O data.tar.gz

# Extract
tar xzf data.tar.gz
```

### Step 3: Install dependencies

Read their `INSTALL.md` first:
```bash
cat INSTALL.md
```

They use virtualenvwrapper. On Wulver, a regular venv is easier:
```bash
python3 -m venv /project/hz79/_shared/cs785/baselines/blens_env
source /project/hz79/_shared/cs785/baselines/blens_env/bin/activate
pip install -r requirements.txt
```

Also check `PRETRAINED.md` for pretrained model setup:
```bash
cat PRETRAINED.md
```

### Step 4: Verify installation on their test set

Submit a SLURM job to verify BLens works:
```bash
cat > /tmp/blens_test.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=cs785-eval-blens
#SBATCH --account=hz79
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/project/hz79/_shared/cs785/slurm_logs/cs785-eval-blens.%j.out

module load bright
module load python3
source /project/hz79/_shared/cs785/baselines/blens_env/bin/activate
cd /project/hz79/_shared/cs785/baselines/BLens

echo "=== BLens Evaluation ==="
nvidia-smi -L

python3 evaluation/evaluator_all.py -data-dir=data/
EOF

sbatch /tmp/blens_test.sbatch
```

**Expected:** Their reported numbers (~F1=0.46 cross-project). If this works, proceed to Step 5.

**Troubleshooting:**
- If `evaluator_all.py` errors, check `evaluation/` for other scripts
- If OOM, check if they support batch size reduction
- Check their GitHub issues for known setup problems

### Step 5: Understand BLens input format

Before running on our data, understand what BLens expects:
```bash
# Check data structure
ls data/
find data/ -name "*.json" | head -5
find data/ -name "*.pkl" | head -5

# Check what the eval script loads
grep -r "load\|read\|open" evaluation/evaluator_all.py | head -20

# Check if they have a --symlm-subdataset flag (compatible with SymLM format)
grep -r "symlm" evaluation/ --include="*.py"
```

Key questions to answer:
- What format are function embeddings in? (pickle, numpy, json?)
- Do they expect pre-computed embeddings or raw binaries?
- Can they process stripped ELF binaries directly?

### Step 6: Run BLens on our data

Two options — **do Option A first** (faster, inference only). Option B is bonus if time permits.

---

#### Option A: Run their pretrained model on our cross-project data (PRIORITY)

Tests BLens' generalization to our unseen packages. No retraining.

**Our cross-project packages:** tengine, angie, nginx118, recutils
**Our stripped binaries:** `/project/hz79/_shared/cs785/data/stripped/`

##### A.1: Understand BLens' input format

```bash
cd /project/hz79/_shared/cs785/baselines/BLens

# Examine their test data structure
find data/ -name "*test*" -type f | head -10
find data/ -name "*test*" -type d | head -10

# Check file formats
python3 -c "
import os
for root, dirs, fnames in os.walk('data'):
    for f in fnames[:3]:
        path = os.path.join(root, f)
        print(f'{path} ({os.path.getsize(path)} bytes)')
    if fnames: break
"
```

##### A.2: Prepare our binaries

```bash
# Copy our cross-project stripped binaries into BLens workspace
mkdir -p data/our_xproj/stripped/ data/our_xproj/labels/

for pkg in angie nginx118 tengine recutils; do
    cp /project/hz79/_shared/cs785/data/stripped/${pkg}_*_stripped data/our_xproj/stripped/
    cp /project/hz79/_shared/cs785/data/labels/${pkg}_*_labels.json data/our_xproj/labels/
done

echo "Copied $(ls data/our_xproj/stripped/ | wc -l) binaries"
```

##### A.3: Run BLens preprocessing on our binaries

BLens needs embeddings from 4 upstream models. Find their preprocessing scripts:
```bash
# Find embedding extraction scripts
find . -name "*.py" | xargs grep -l "embed\|preprocess\|extract" | head -10
ls scripts/ preprocessing/ 2>/dev/null
cat PRETRAINED.md
```

Run their embedding extraction (adapt based on what you find):
```bash
cat > /tmp/blens_preprocess_xproj.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=cs785-blens-prep
#SBATCH --account=hz79
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/project/hz79/_shared/cs785/slurm_logs/cs785-blens-prep.%j.out

module load bright
module load python3
source /project/hz79/_shared/cs785/baselines/blens_env/bin/activate
cd /project/hz79/_shared/cs785/baselines/BLens

echo "=== BLens Preprocessing on our xproj data ==="
# ADAPT THIS: Run their embedding extraction on our binaries
# Example (find the actual script name):
# python3 preprocessing/extract_embeddings.py --input data/our_xproj/stripped/ --output data/our_xproj/embeddings/
EOF

sbatch /tmp/blens_preprocess_xproj.sbatch
```

##### A.4: Run inference with pretrained model

```bash
cat > /tmp/blens_infer_xproj.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=cs785-eval-blens-xproj
#SBATCH --account=hz79
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/project/hz79/_shared/cs785/slurm_logs/cs785-eval-blens-xproj.%j.out

module load bright
module load python3
source /project/hz79/_shared/cs785/baselines/blens_env/bin/activate
cd /project/hz79/_shared/cs785/baselines/BLens

echo "=== BLens Inference on our cross-project ==="
nvidia-smi -L

# ADAPT: Use their inference command with pretrained weights
# Key flags to look for: -data-dir, -d=test, --cross-binary, -inferBest, -inferT0
CUDA_VISIBLE_DEVICES=0 python3 RunExp.py \
    -data-dir=data/our_xproj/ \
    -d=test \
    --cross-binary \
    -inferBest
EOF

sbatch /tmp/blens_infer_xproj.sbatch
```

##### A.5: Compute metrics

```bash
# After BLens produces predictions, compute F1
# Adapt based on their output format
python3 -c "
import json, os

# Load predictions (find the output file from BLens)
# pred_file = 'results/predictions.json'  # adapt path

# Load our ground truth
gt = {}
for lf in os.listdir('data/our_xproj/labels/'):
    with open(f'data/our_xproj/labels/{lf}') as f:
        d = json.load(f)
    binary = d['binary']
    for name, addr in d.get('functions', {}).items():
        gt[f'{binary}_{addr}'] = name

# Compare and compute sub-token F1
# ... (adapt to BLens output format)
"
```

---

#### Option B: Train BLens end-to-end on our dataset (BONUS)

Retrains BLens on our 300K training data, then evaluates on our cross-project set. Fairest comparison but takes longer (~1-2 days).

##### B.1: Prepare our full training data

```bash
cd /project/hz79/_shared/cs785/baselines/BLens

# Our training set: ~242K functions (300K minus val/test/cross-project)
# Need: stripped binaries + ground truth for all training binaries

mkdir -p data/our_train/stripped/ data/our_train/labels/

# Copy training binaries (exclude cross-project)
python3 -c "
import json, shutil, os
with open('/project/hz79/_shared/cs785/data/match_index.json') as f:
    mi = json.load(f)
xproj = {'tengine', 'angie', 'nginx118', 'recutils'}
bins = set()
for v in mi.values():
    pkg = v['binary'].split('_')[0]
    if pkg not in xproj:
        bins.add(v['binary'])
print(f'Training binaries: {len(bins)}')

src = '/project/hz79/_shared/cs785/data/stripped/'
dst = 'data/our_train/stripped/'
os.makedirs(dst, exist_ok=True)
copied = 0
for b in bins:
    s = os.path.join(src, b + '_stripped')
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(dst, b + '_stripped'))
        copied += 1
print(f'Copied: {copied}')
"

# Copy labels
cp /project/hz79/_shared/cs785/data/labels/*_labels.json data/our_train/labels/
```

##### B.2: Run BLens preprocessing on training data

```bash
cat > /tmp/blens_preprocess_train.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=cs785-blens-prep-train
#SBATCH --account=hz79
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/hz79/_shared/cs785/slurm_logs/cs785-blens-prep-train.%j.out

module load bright
module load python3
source /project/hz79/_shared/cs785/baselines/blens_env/bin/activate
cd /project/hz79/_shared/cs785/baselines/BLens

echo "=== BLens Preprocessing (full training set) ==="
# ADAPT: Extract embeddings for all training binaries
# This will take many hours (700+ binaries × 4 embedding models)
EOF

sbatch /tmp/blens_preprocess_train.sbatch
```

##### B.3: Train BLens

```bash
cat > /tmp/blens_train.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=cs785-train-blens
#SBATCH --account=hz79
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/hz79/_shared/cs785/slurm_logs/cs785-train-blens.%j.out

module load bright
module load python3
source /project/hz79/_shared/cs785/baselines/blens_env/bin/activate
cd /project/hz79/_shared/cs785/baselines/BLens

echo "=== BLens Training on our data ==="
nvidia-smi -L

# ADAPT: Their training command
# From their README: -pretrain -train -inferBest
CUDA_VISIBLE_DEVICES=0 python3 RunExp.py \
    -data-dir=data/our_train/ \
    -d=test \
    --cross-binary \
    -pretrain -train -inferBest
EOF

sbatch /tmp/blens_train.sbatch
```

##### B.4: Evaluate on cross-project

After training, evaluate on our held-out cross-project set:
```bash
# Same as Option A Step A.4, but using newly trained model
CUDA_VISIBLE_DEVICES=0 python3 RunExp.py \
    -data-dir=data/our_xproj/ \
    -d=test \
    --cross-binary \
    -inferBest
```

---

### What to report

For **each option** completed, fill in this table:

| Metric | Option A (pretrained) | Option B (retrained) |
|--------|----------------------|---------------------|
| Overall F1 | | |
| Overall Precision | | |
| Overall Recall | | |
| angie F1 | | |
| nginx118 F1 | | |
| tengine F1 | | |
| recutils F1 | | |
| Inference time | | |
| GPU memory used | | |

Save all results to: `/project/hz79/_shared/cs785/baselines/BLens/results/`

**Option A results** = "BLens (pretrained) on our data"
**Option B results** = "BLens (retrained on our data)"

---

## Monitoring Jobs

```bash
squeue -u $USER                           # List running jobs
sacct -j JOBID --format=State,Elapsed     # Check completed job
tail -f /project/hz79/_shared/cs785/slurm_logs/cs785-eval-blens.JOBID.out
```

---

## Context for Comparison

| System | Cross-Project F1 | Params | Notes |
|--------|-----------------|--------|-------|
| **Our model** | 0.650 | 32M | 4 packages, 10,467 functions |
| **BLens (paper)** | 0.46 | Ensemble | Their test set |
| **BLens (on our data)** | ??? | Ensemble | **This is what we need** |
| **SYMGEN** | ~0.16* | 34B | Running on our data |
| **SymLM** | 0.021 | ~100M | Vocab mismatch |

---
