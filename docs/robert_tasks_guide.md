# Robert's Task Guide — BLens Evaluation & Dataset Expansion

**For:** Robert Blacha
**Project:** CS785 Binary Function Name Recovery
**Date:** 2026-04-09
**Deadline:** April 22, 2026

---

## Overview

You have two tasks:
1. **Set up and run BLens baseline evaluation** on Wulver (~1-2 days)
2. **Add more packages to our dataset** to improve training diversity (~1 day)

Both tasks are independent — you can work on them in parallel or sequentially.

---

## Prerequisites

### 1. Get access to the project on Wulver

Our shared project directory is:
```
/project/hz79/_shared/cs785/
```

SSH into Wulver:
```bash
ssh wulver    # You'll need Duo 2FA
```

Check you can access the directory:
```bash
ls /project/hz79/_shared/cs785/
# Should see: configs/ data/ src/ scripts/ baselines/ checkpoints/ ...
```

### 2. Local machine setup (for BAP preprocessing — Task 2 only)

BAP is not on Wulver. If you need to preprocess new packages:
```bash
cd ~/cs785-project
bash scripts/01_setup_environment.sh   # Installs BAP + Python env (~30 min)
source ~/cs785-project/activate.sh
bap --version   # Should be 2.5.0
```

If you already have the repo cloned, just pull latest:
```bash
cd ~/cs785-project
git pull origin v3
```

### 3. Wulver environment

For running jobs on Wulver:
```bash
module load bright
module load python3
# For our model's env:
source /project/hz79/_shared/cs785-env/bin/activate
# For baselines env (BLens, SymLM, etc.):
source /project/hz79/_shared/cs785/baselines/symlm_env/bin/activate
```

### 4. SLURM job submission

All GPU jobs use this template:
```bash
#!/bin/bash
#SBATCH --job-name=cs785-YOUR-JOB-NAME
#SBATCH --account=hz79
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/hz79/_shared/cs785/slurm_logs/YOUR-JOB.%j.out
```

We have **full NVIDIA A100-80GB** GPUs via the `hz79` account.

---

## Task 1: BLens Baseline Evaluation

### What is BLens?
BLens (USENIX Security 2025) is a function name prediction system that uses an ensemble of 4 pretrained embedding models (PalmTree + CLAP + DEXTER + VarCLR) fused with a "LORD" decoder. It reports **cross-project F1=0.46** and **cross-binary F1=0.77** in their paper.

We want to run BLens on our cross-project data to compare with our model (F1=0.650).

### Step 1: Clone BLens repo

```bash
ssh wulver
cd /project/hz79/_shared/cs785/baselines/
git clone https://github.com/lmu-plai/blens.git BLens
cd BLens
```

### Step 2: Download BLens data and pretrained models

Their data is on Zenodo: https://doi.org/10.5281/zenodo.14713022

```bash
cd /project/hz79/_shared/cs785/baselines/BLens

# Download data.tar.gz from Zenodo (~20-30GB)
wget 'https://zenodo.org/records/14713022/files/data.tar.gz?download=1' -O data.tar.gz

# Extract
tar xzf data.tar.gz
```

### Step 3: Follow their INSTALL.md

```bash
cat INSTALL.md
# Follow their installation steps
# They use virtualenvwrapper — you may need:
pip install virtualenvwrapper
export WORKON_HOME=~/.virtualenvs
source $(which virtualenvwrapper.sh)
mkvirtualenv blens
# Then install their requirements
```

If virtualenvwrapper doesn't work on Wulver, create a regular venv:
```bash
python3 -m venv /project/hz79/_shared/cs785/baselines/blens_env
source /project/hz79/_shared/cs785/baselines/blens_env/bin/activate
cd /project/hz79/_shared/cs785/baselines/BLens
pip install -r requirements.txt   # or follow INSTALL.md
```

### Step 4: Check their pretrained models

After extracting `data.tar.gz`, check:
```bash
ls data/
# Should contain: pretrained models, tokenizers, test data
ls pretrained_data/   # or wherever their models are
```

Read `PRETRAINED.md` for details on model files.

### Step 5: Run their evaluation on their own test set first

This verifies the installation works:
```bash
# Submit as SLURM job (needs A100-80GB)
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

source /project/hz79/_shared/cs785/baselines/blens_env/bin/activate
cd /project/hz79/_shared/cs785/baselines/BLens

python3 evaluation/evaluator_all.py -data-dir=data/
EOF

sbatch /tmp/blens_test.sbatch
```

If this works and matches their reported numbers (~F1=0.46 cross-project), proceed to Step 6.

### Step 6: Run BLens on our cross-project data

This is the tricky part — we need to convert our data into BLens' expected format.

**Our cross-project packages:** tengine, angie, nginx118, recutils
**Our data location:** `/project/hz79/_shared/cs785/data/`

BLens expects pre-computed embeddings from their pipeline. You'll need to either:

**Option A:** Use their preprocessing to generate embeddings from our stripped binaries
- Our stripped binaries: `/project/hz79/_shared/cs785/data/stripped/`
- Run their embedding extraction on angie, nginx118, tengine, recutils binaries
- Then run their inference

**Option B:** Check if BLens can work with SymLM's data format
- BLens has a `--symlm-subdataset` flag
- Our SymLM data is at: `/project/hz79/_shared/cs785/baselines/SymLM/`

Read their code to understand the input format:
```bash
# Check what data format they expect
grep -r "load.*data\|read.*binary\|input.*format" evaluation/ --include="*.py" | head -20
```

### What to report

After running BLens, we need:
- **F1 score, precision, recall** on our cross-project set
- **Per-package breakdown** (angie, nginx118, tengine, recutils)
- **Comparison:** Our model F1=0.650 vs BLens on same data

---

## Task 2: Add More Packages to Dataset

### Goal
Add diverse packages to our 300K training dataset. More packages = better generalization.

### What we already have (77 packages)
```bash
# Check current packages
cd /project/hz79/_shared/cs785
python3 -c "
import json
from collections import Counter
with open('data/match_index.json') as f:
    mi = json.load(f)
pkgs = Counter(v['binary'].split('_')[0] for v in mi.values())
print(f'Total: {len(mi)} functions, {len(pkgs)} packages')
for p, c in pkgs.most_common():
    print(f'  {p}: {c}')
"
```

### What to add (priority order)

**Priority 1: Packages from SYMGEN's dataset that we don't have**
These are already downloaded on Wulver:
```
/project/hz79/_shared/cs785/baselines/SymGen/zenodo/extracted_bins/x86_64/
```

Packages we DON'T have (18 new packages):
- adns, cflow, dico, freeipmi, gettext, gmp, gss
- libiconv, libidn2, libmicrohttpd, libredwg, libtool
- libunistring, ncurses, openssl, poke, readline, wget2

**Priority 2: Other common packages**
- redis, p7zip, nmap, vim — different domains for diversity

### Step-by-step for each new package

**All preprocessing must be done on your LOCAL machine (BAP required).**

#### 1. Get the stripped binaries

If from SYMGEN's dataset, copy from Wulver to local:
```bash
# On local machine:
scp -r wulver:/project/hz79/_shared/cs785/baselines/SymGen/zenodo/extracted_bins/x86_64/O2/gettext-0.21/ ~/cs785-project/data/symgen_bins/
```

If compiling from source:
```bash
cd ~/cs785-project
# Edit configs/packages.conf to add the package, then:
bash scripts/02_compile_dataset.sh
```

**IMPORTANT:** Always compile with `-no-pie`:
```bash
./configure CFLAGS="-g -O2 -no-pie" LDFLAGS="-no-pie"
make -j$(nproc)
```

#### 2. Strip the binary (if using SYMGEN's unstripped binaries)

```bash
# SYMGEN provides unstripped binaries — strip them for BAP
for bin in data/symgen_bins/gettext-0.21/*; do
    name=$(basename "$bin")
    strip -s "$bin" -o "data/stripped/gettext_${name}_O2_stripped"
    cp "$bin" "data/raw/gettext_${name}_O2_sym"
done
```

#### 3. Extract ground truth labels

```bash
for f in data/raw/gettext_*_sym; do
    name=$(basename "$f" _sym)
    nm --defined-only "$f" | awk '$2 ~ /[tT]/ {printf "{\"0x%s\": \"%s\"}\n", $1, $3}' | python3 -c "
import sys, json
binary = '$name'
labels = {}
for line in sys.stdin:
    try: d = json.loads(line.strip()); labels.update(d)
    except: pass
name_to_addr = {v: k for k, v in labels.items()}
output = {
    'binary': binary,
    'num_functions': len(labels),
    'functions': name_to_addr,
    'addr_to_name': labels,
    'name_to_addr': name_to_addr
}
with open('data/labels/${name}_labels.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'{len(labels)} labels')
"
done
```

**CRITICAL:** The `functions` field MUST be `{name: addr}` format. NOT `{addr: name}`.

#### 4. BAP lift

```bash
source ~/cs785-project/activate.sh

for f in data/stripped/gettext_*_stripped; do
    name=$(basename "$f" _stripped)
    bir="data/bir/${name}.bir"
    [ -f "$bir" ] && continue
    echo -n "$name... "
    timeout 600 bap "$f" --dump=bir:"$bir" 2>/dev/null && echo "OK" || \
    timeout 600 bap "$f" --no-byteweight --dump=bir:"$bir" 2>/dev/null && echo "OK (fb)" || \
    echo "FAILED"
done
```

BAP takes ~1-5 min per binary. If it takes >10 min or OOMs, skip that binary.

#### 5. Parse graphs + extract external calls

```bash
for bir in data/bir/gettext_*.bir; do
    name=$(basename "$bir" .bir)
    python3 -m src.preprocessing.parse_bap \
        --bir "$bir" --binary-name "$name" --output-dir data/graphs
    python3 -m src.preprocessing.extract_external \
        --bir "$bir" --binary-name "$name" --output-dir data/external_calls
done
```

#### 6. Update match_index

```bash
bash scripts/expand_bap_pipeline.sh
# This incrementally adds new functions to data/match_index.json
```

#### 7. Verify

```bash
python3 -c "
import json
with open('data/match_index.json') as f:
    mi = json.load(f)
gettext = sum(1 for v in mi.values() if v['binary'].startswith('gettext'))
print(f'gettext: {gettext} functions matched')
print(f'Total dataset: {len(mi)} functions')
"
```

#### 8. Sync to Wulver

```bash
REMOTE="wulver:/project/hz79/_shared/cs785"
rsync -avz data/match_index.json "$REMOTE/data/"
rsync -avz data/votes_vocab.json "$REMOTE/data/"
rsync -az data/graphs/gettext_*.json "$REMOTE/data/graphs/"
rsync -az data/labels/gettext_*_labels.json "$REMOTE/data/labels/"
rsync -az data/external_calls/gettext_*.json "$REMOTE/data/external_calls/"
```

### After adding packages: rebuild vocabularies

```bash
# On local machine
python3 -m src.preprocessing.build_votes \
    --match-index data/match_index.json \
    --output data/votes_vocab.json \
    --min-count 2

# Sync updated vocab to Wulver
rsync -avz data/votes_vocab.json wulver:/project/hz79/_shared/cs785/data/
```

---

## Important Rules

1. **BAP is LOCAL ONLY** — not installed on Wulver
2. **Always use `-no-pie`** when compiling — prevents address mismatch
3. **Label format:** `functions` must be `{name: addr}`, NOT `{addr: name}`
4. **Don't add cross-project packages to training:** tengine, angie, nginx118, recutils
5. **Don't modify** `split_assignments.json` without coordinating with Ananta
6. **SLURM jobs:** Use `--account=hz79 --partition=gpu --qos=standard` for GPU access
7. **Job names:** Use pattern `cs785-{task}-{detail}` (e.g., `cs785-eval-blens`)

---

## File Locations on Wulver

```
/project/hz79/_shared/cs785/              # Main project directory
├── data/                                  # All training data
│   ├── match_index.json                  # Central: 300K function mappings
│   ├── graphs/                           # ~470K per-function CFG graphs
│   ├── labels/                           # Ground truth per binary
│   └── external_calls/                   # Library call data
├── baselines/
│   ├── BLens/                            # BLens repo (you'll set this up)
│   ├── SymGen/                           # SYMGEN baseline
│   │   └── zenodo/extracted_bins/x86_64/ # SYMGEN's 33-project binaries
│   ├── SymLM/                            # SymLM baseline
│   ├── hf_cache/                         # HuggingFace models (139GB, symlinked)
│   └── symlm_env/                        # Shared Python env for baselines
├── checkpoints/
│   └── best_model.pt                     # Our trained model (300K dataset)
├── configs/
│   └── optimized_large.yaml             # Model config
├── scripts/
│   ├── eval_cross_project.py            # Cross-project evaluation
│   └── expand_bap_pipeline.sh           # Incremental preprocessing
├── slurm_logs/                           # Job output logs
└── docs/
    └── preprocessing_contributor_guide.md # Full preprocessing guide
```

---

## Questions?

Ask Ananta on Discord. Key context:
- Our model: 32.2M params, F1=0.650 on cross-project (10,467 functions, 4 packages)
- Paper deadline: April 22
- We need BLens comparison numbers and more training data diversity
