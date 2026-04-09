# Task: BLens Baseline Evaluation

**For:** Robert Blacha
**Project:** CS785 Binary Function Name Recovery
**Date:** 2026-04-09
**Deadline:** April 22, 2026
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

### Step 6: Run BLens on our cross-project data

**Our cross-project packages:** tengine, angie, nginx118, recutils
**Our stripped binaries:** `/project/hz79/_shared/cs785/data/stripped/`

**Option A: Use their preprocessing pipeline**
- Feed our stripped binaries through their embedding extraction
- Then run their inference/evaluation
- This is the most accurate approach

**Option B: Use SymLM data compatibility**
- BLens has `--symlm-subdataset` flag
- Our SymLM preprocessed data: `/project/hz79/_shared/cs785/baselines/SymLM/`
- May work without additional preprocessing

**Option C: Convert our data to their format**
- Study their data format from Step 5
- Write a conversion script
- Run their evaluation on converted data

### What to report

After running BLens, record:
- **F1 score, precision, recall** on our cross-project set
- **Per-package breakdown** (angie, nginx118, tengine, recutils)
- **Runtime** (how long inference took)
- **Any issues or limitations** encountered

Save results to: `/project/hz79/_shared/cs785/baselines/BLens/results/`

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

## Questions?

Ask Ananta on Discord.
