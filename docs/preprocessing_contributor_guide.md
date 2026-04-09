# Data Collection & Preprocessing Contributor Guide

**For:** Robert Blacha (dataset compilation, BAP extraction, data preprocessing)
**Project:** CS785 Binary Function Name Recovery
**Last updated:** 2026-04-09

---

## Current Dataset State (2026-04-09)

```
300,013 matched functions | 77 packages | 851 binaries
Opt: O0=47% | O1=16% | O2=16% | O3=15% | default=6%
Votes vocab: 7,004 tokens | Ext vocab: 2,237 tokens
```

### Cross-project packages (DO NOT add to training):
- `tengine`, `angie`, `nginx118`, `recutils`

---

## Pipeline Overview

### Local Machine (preprocessing contributor)

```
Step 1: Compile packages
    configs/packages.conf → scripts/02_compile_dataset.sh
    Output: data/raw/*_sym (debug) + data/stripped/*_stripped

Step 2: Extract ground truth labels
    data/raw/*_sym → nm → data/labels/*_labels.json

Step 3: BAP lift stripped binaries
    data/stripped/*_stripped → bap → data/bir/*.bir

Step 4: Parse BAP-IR into graphs
    data/bir/*.bir → parse_bap.py → data/graphs/*.json

Step 5: Extract external calls
    data/bir/*.bir → extract_external.py → data/external_calls/*_external.json

Step 6: Address matching
    data/labels/ + data/graphs/ → data/match_index.json

Step 7: Build vocabularies
    data/match_index.json → data/votes_vocab.json
    data/external_calls/ → data/external_calls/external_vocab.json
```

### Wulver HPC: /project/hz79/_shared/cs785/

```
Step 8: Sync data to Wulver
    bash scripts/wulver_sync.sh  (syncs code)
    rsync data files manually    (match_index, vocabs, graphs, labels, ext_calls)

Step 9: Train model
    python3 -m src.training.train --config configs/optimized_large.yaml \
        --seed 42 --batch-size 256 --amp --num-workers 4

Step 10: Evaluate
    python3 scripts/eval_cross_project.py checkpoints/best_model.pt \
        --config configs/optimized_large.yaml --amp
```

---

## Step-by-Step: Adding New Packages

### 1. Edit `configs/packages.conf`

```
name | download_url | tarball_name | source_directory | binary1 binary2 ...
```

Example:
```
diffutils | https://ftp.gnu.org/gnu/diffutils/diffutils-3.10.tar.xz | diffutils-3.10.tar.xz | diffutils-3.10 | src/diff src/diff3 src/sdiff src/cmp
```

**Important:** For autotools packages, binary paths may need `.libs/` prefix (e.g., `src/.libs/diff` instead of `src/diff`).

### 2. Compile

```bash
source ~/cs785-project/activate.sh
bash scripts/02_compile_dataset.sh
```

Or compile manually for non-autotools packages (Makefile-based like bzip2, tree):
```bash
cd build_tmp/package-dir
make clean && make -j$(nproc) CFLAGS="-g -O2 -no-pie" LDFLAGS="-no-pie"
cp binary ~/cs785-project/data/raw/pkg_binary_O2_sym
strip -s binary -o ~/cs785-project/data/stripped/pkg_binary_O2_stripped
```

### 3. Verify binaries

```bash
# Must be ELF executable (NOT "pie executable" for reliable address matching)
file data/stripped/pkg_binary_O2_stripped
# Expected: ELF 64-bit LSB executable, x86-64, ...

# Must have debug symbols in raw version
nm --defined-only data/raw/pkg_binary_O2_sym | grep ' [tT] ' | wc -l
# Expected: >0 functions
```

### 4. Extract labels

```bash
for f in data/raw/pkg_*_sym; do
    name=$(basename "$f" _sym)
    nm --defined-only "$f" | awk '$2 ~ /[tT]/ {printf "{\"0x%s\": \"%s\"}\n", $1, $3}' | python3 -c "
import sys, json
binary = '$name'
labels = {}
for line in sys.stdin:
    try: d = json.loads(line.strip()); labels.update(d)
    except: pass
# IMPORTANT: functions must be name->addr format
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

**Critical:** The `functions` field must be `{name: addr}` format, NOT `{addr: name}`. The eval script reads `functions` as `name → addr`.

### 5. BAP lift

```bash
for f in data/stripped/pkg_*_stripped; do
    name=$(basename "$f" _stripped)
    bir="data/bir/${name}.bir"
    [ -f "$bir" ] && continue  # Skip existing
    echo -n "$name... "
    timeout 600 bap "$f" --dump=bir:"$bir" 2>/dev/null && echo "OK" || \
    timeout 600 bap "$f" --no-byteweight --dump=bir:"$bir" 2>/dev/null && echo "OK (fallback)" || \
    echo "FAILED"
done
```

### 6. Parse graphs + extract external calls

```bash
for bir in data/bir/pkg_*.bir; do
    name=$(basename "$bir" .bir)
    # Skip if graphs already exist
    [ -f "data/graphs/${name}_sub_0.json" ] && continue
    python3 -m src.preprocessing.parse_bap --bir "$bir" --binary-name "$name" --output-dir data/graphs
    python3 -m src.preprocessing.extract_external --bir "$bir" --binary-name "$name" --output-dir data/external_calls
done
```

### 7. Update match_index (incremental)

```bash
bash scripts/expand_bap_pipeline.sh
# This runs: labels → BAP → parse → ext calls → incremental match
# Only processes NEW binaries (skips existing)
```

Or run matching standalone:
```bash
python3 -c "
import json, glob, os

with open('data/match_index.json') as f:
    mi = json.load(f)
existing = set(mi.keys())

# Load labels
labels_by_binary = {}
for lf in glob.glob('data/labels/*_labels.json'):
    with open(lf) as f:
        d = json.load(f)
    binary = d.get('binary', '')
    atn = d.get('addr_to_name', {})
    int_to_name = {}
    for addr_str, name in atn.items():
        try: int_to_name[int(addr_str, 16)] = name
        except: pass
    if int_to_name:
        labels_by_binary[binary] = int_to_name

# Match new graphs
new = 0
for gf in glob.glob('data/graphs/*.json'):
    if gf in existing: continue
    with open(gf) as f:
        g = json.load(f)
    binary = g.get('binary', '')
    try: addr = int(g.get('address', ''), 16)
    except: continue
    name = labels_by_binary.get(binary, {}).get(addr)
    if name:
        mi[gf] = {'binary': binary, 'address': g['address'],
                   'address_int': addr, 'bap_name': g.get('function_name',''),
                   'real_name': name}
        new += 1

with open('data/match_index.json', 'w') as f:
    json.dump(mi, f, indent=2)
print(f'New: {new}, Total: {len(mi)}')
"
```

### 8. Rebuild vocabularies

```bash
# Votes (name tokenizer)
python3 -m src.preprocessing.build_votes --match-index data/match_index.json \
    --output data/votes_vocab.json --min-count 2

# External call vocabulary
python3 -c "
import json, glob
all_ext = set()
for f in glob.glob('data/external_calls/*_external.json'):
    with open(f) as fp:
        d = json.load(fp)
    for func in d.get('functions', []):
        for call in func.get('external_calls', []):
            name = call.get('name', '')
            if name: all_ext.add(name)
vocab = {'<NO_EXT>': 0}
for name in sorted(all_ext):
    vocab[name] = len(vocab)
with open('data/external_calls/external_vocab.json', 'w') as f:
    json.dump({'vocab_size': len(vocab), 'vocabulary': vocab}, f, indent=2)
print(f'External vocab: {len(vocab)} tokens')
"
```

---

## Syncing to Wulver

After preprocessing locally, sync data to Wulver for training/eval:

### Quick code sync (excludes data/)
```bash
bash scripts/wulver_sync.sh
```

### Manual data sync
```bash
REMOTE="wulver:/project/hz79/_shared/cs785"

# Essential files (always sync these)
rsync -avz data/match_index.json "$REMOTE/data/"
rsync -avz data/votes_vocab.json "$REMOTE/data/"
rsync -avz data/external_calls/external_vocab.json "$REMOTE/data/external_calls/"
rsync -avz data/split_assignments.json "$REMOTE/data/"

# New graphs (for new packages only)
rsync -az data/graphs/newpkg_*.json "$REMOTE/data/graphs/"

# New labels
rsync -az data/labels/newpkg_*_labels.json "$REMOTE/data/labels/"

# New external calls
rsync -az data/external_calls/newpkg_*.json "$REMOTE/data/external_calls/"
```

### SSH requirement
You must have an active SSH connection to Wulver first:
```bash
ssh wulver  
```

---

## Wulver HPC Guide

### Access
- **Host:** wulver.njit.edu (or use SSH config alias `wulver`)
- **Account:** `hz79` (research account — full A100-80GB GPU access)
- **Project dir:** `/project/hz79/_shared/cs785`
- **Python env:** `/project/hz79/_shared/cs785-env`

### Submitting a training job

```bash
#!/bin/bash
#SBATCH --job-name=cs785-train
#SBATCH --account=hz79
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/cs785-train.%j.out

module load bright
module load python3
source /project/hz79/_shared/cs785-env/bin/activate
cd /project/hz79/_shared/cs785

python3 -m src.training.train \
    --config configs/optimized_large.yaml \
    --seed 42 \
    --batch-size 256 \
    --amp \
    --num-workers 4
```

### Monitoring jobs
```bash
squeue -u adp232                    # List running jobs
sacct -j JOBID --format=State,Elapsed  # Check completed job
tail -f slurm_logs/cs785-train.JOBID.out  # Watch output
seff JOBID                           # Resource usage after completion
```

### Job naming convention
```
cs785-train          — model training
cs785-train-cl       — contrastive learning fine-tuning
cs785-eval-test      — test evaluation
cs785-eval-xproj     — cross-project evaluation
cs785-eval-symgen    — SYMGEN baseline
cs785-prep-symgen    — Ghidra decompilation for SYMGEN
```

---

## Data Directory Structure

```
data/
├── raw/                         # Debug binaries with symbols (LOCAL ONLY)
│   └── {pkg}_{bin}_O{level}_sym
├── stripped/                    # Stripped binaries (LOCAL ONLY)
│   └── {pkg}_{bin}_O{level}_stripped
├── debug/                       # Copy of debug binaries (LOCAL ONLY)
│   └── {pkg}_{bin}_O{level}
├── bir/                         # BAP-IR files (LOCAL ONLY, not on Wulver)
│   └── {pkg}_{bin}_O{level}.bir
├── graphs/                      # Per-function CFG graphs (SYNCED TO WULVER)
│   └── {pkg}_{bin}_O{level}_sub_{addr}.json
├── labels/                      # Ground truth labels (SYNCED TO WULVER)
│   └── {pkg}_{bin}_O{level}_labels.json
├── external_calls/              # External calls (SYNCED TO WULVER)
│   ├── {pkg}_{bin}_O{level}_external.json
│   └── external_vocab.json
├── match_index.json             # Central mapping (SYNCED TO WULVER)
├── split_assignments.json       # Train/val/test splits (SYNCED TO WULVER)
└── votes_vocab.json             # Name tokenizer vocab (SYNCED TO WULVER)
```

---

## Label File Format (CRITICAL)

The eval script expects this exact format:

```json
{
  "binary": "pkg_bin_O2",
  "num_functions": 347,
  "functions": {
    "main": "0x5c40",
    "usage": "0x4a30"
  },
  "addr_to_name": {
    "0x0000000000005c40": "main",
    "0x0000000000004a30": "usage"
  },
  "name_to_addr": {
    "main": "0x5c40",
    "usage": "0x4a30"
  }
}
```

**The `functions` field MUST be `name → addr` (NOT `addr → name`).** The eval's `load_ground_truth()` iterates `functions.items()` as `(name, addr)` pairs.

---

## Critical Rules

1. **BAP is LOCAL ONLY.** BAP is not installed on Wulver. All BAP lifting must be done locally.
2. **Compile with `-no-pie`** to avoid address mismatch between nm and BAP.
3. **Label `functions` must be `name → addr`.** The eval script breaks otherwise.
4. **Use deterministic sort `(-count, name)` for vocab building.** Non-deterministic sort caused 1,232 token mismatches.
5. **Cross-project packages must NOT be in training.**
6. **`match_index.json` is regenerated, not hand-edited.**
7. **NEVER overwrite `external_vocab.json` during inference.** The vocab is saved in checkpoints.
8. **Coordinate with Ananta before changing `match_index.json` or `split_assignments.json`** — training depends on these.

---

## Quick Reference

```bash
# Full local pipeline
source ~/cs785-project/activate.sh
bash scripts/02_compile_dataset.sh       # Compile
bash scripts/03_preprocess.sh             # Full preprocess
# OR incrementally:
bash scripts/expand_bap_pipeline.sh       # Process only new binaries

# Check dataset
python3 -c "
import json
with open('data/match_index.json') as f:
    idx = json.load(f)
print(f'Total: {len(idx)} functions')
from collections import Counter
pkgs = Counter(v['binary'].split('_')[0] for v in idx.values())
for p, c in pkgs.most_common(10):
    print(f'  {p}: {c}')
"

# Sync to Wulver
bash scripts/wulver_sync.sh
rsync -avz data/match_index.json wulver:/project/hz79/_shared/cs785/data/
rsync -avz data/votes_vocab.json wulver:/project/hz79/_shared/cs785/data/
rsync -az data/graphs/newpkg_*.json wulver:/project/hz79/_shared/cs785/data/graphs/
rsync -az data/labels/newpkg_*_labels.json wulver:/project/hz79/_shared/cs785/data/labels/
```

---

## Key Files

| File | Owner | Purpose |
|------|-------|---------|
| `configs/packages.conf` | Robert | Package list for compilation |
| `scripts/02_compile_dataset.sh` | Robert | Compilation pipeline |
| `scripts/03_preprocess.sh` | Robert | Full preprocessing |
| `scripts/expand_bap_pipeline.sh` | Robert | Incremental preprocessing for new binaries |
| `src/preprocessing/parse_bap.py` | Shared | BAP-IR parser + V3 tokenization |
| `src/preprocessing/extract_external.py` | Shared | External call extractor |
| `src/preprocessing/build_votes.py` | Shared | Votes name tokenizer |
| `src/preprocessing/build_dataset.py` | Ananta | Dataset loader (PyTorch) |
| `data/match_index.json` | Shared | Central graph→label mapping |
| `data/split_assignments.json` | Shared | Train/val/test splits |
| `configs/optimized_large.yaml` | Ananta | Model config (25M+ params) |
| `src/training/train.py` | Ananta | Training loop |
| `scripts/eval_cross_project.py` | Ananta | Cross-project evaluation |
| `scripts/wulver_sync.sh` | Shared | Code sync to Wulver |
