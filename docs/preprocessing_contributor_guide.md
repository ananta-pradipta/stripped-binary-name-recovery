# Data Collection & Preprocessing Contributor Guide

**For:** Robert Blacha (dataset compilation, BAP extraction, data preprocessing)
**Project:** CS785 Binary Function Name Recovery
**Last updated:** 2026-04-02

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Adding New Packages](#3-adding-new-packages)
4. [Running the Preprocessing Pipeline](#4-running-the-preprocessing-pipeline)
5. [Data Directory Structure](#5-data-directory-structure)
6. [Key File Formats](#6-key-file-formats)
7. [How Each Stage Works](#7-how-each-stage-works)
8. [Common Tasks](#8-common-tasks)
9. [Debugging & Troubleshooting](#9-debugging--troubleshooting)
10. [Critical Rules (Do Not Break)](#10-critical-rules)

---

## 1. Environment Setup

### First-time setup

```bash
# Clone the repo (if not already)
cd ~
git clone <repo-url> cs785-project
cd cs785-project

# Run the full environment setup (~30 min)
bash scripts/01_setup_environment.sh
```

This installs:
- Python 3.10+ venv with PyTorch, PyTorch Geometric, sentencepiece, networkx
- BAP 2.5.0 (Binary Analysis Platform) via OPAM/OCaml
- System tools: gcc, binutils, elfutils, autotools, build-essential

### Before each session

```bash
source ~/cs785-project/activate.sh
```

This activates the Python venv, sets up OCaml/BAP, and exports `PROJECT_ROOT`. Verify with:
```bash
python3 --version      # Should be 3.10+
bap --version          # Should be 2.5.0
```

### BAP is required locally

BAP cannot run on the Wulver HPC cluster. All BAP-related preprocessing (lifting binaries to IR) must be done on a local machine with BAP installed. Training and evaluation can run on Wulver without BAP.

---

## 2. Pipeline Overview

```
Source packages (GNU FTP)
    |
    v  [02_compile_dataset.sh]
Debug binaries (.sym) + Stripped binaries (.stripped)
    |
    v  [03_preprocess.sh]
    |--- Step 3.1: Extract ground truth labels (nm) --> data/labels/
    |--- Step 3.2: Lift stripped binaries with BAP  --> data/bir/
    |--- Step 3.3: Parse BAP-IR into CFG graphs     --> data/graphs/
    |--- Step 3.4: Extract external calls            --> data/external_calls/
    |--- Step 3.5: Match addresses (labels <-> graphs) --> data/match_index.json
    |--- Step 3.6: Build BPE vocabulary              --> data/bpe_model/
    |--- Step 3.7: Build external call vocabulary    --> data/external_calls/external_vocab.json
    v
Training-ready dataset (loaded by src/preprocessing/build_dataset.py)
```

### What each stage produces

| Stage | Input | Output | ~Time |
|-------|-------|--------|-------|
| 02_compile | Source tarballs | `data/raw/`, `data/stripped/` | 30-60 min |
| 3.1 Labels | Debug binaries | `data/labels/*_labels.json` | 2-5 min |
| 3.2 BAP lift | Stripped binaries | `data/bir/*.bir` | 1-4 hours (slowest) |
| 3.3 Parse | `.bir` files | `data/graphs/*.json` | 10-30 min |
| 3.4 Ext calls | `.bir` files | `data/external_calls/*_external.json` | 5-10 min |
| 3.5 Matching | Labels + graphs | `data/match_index.json` | 2-5 min |
| 3.6 BPE | Function names | `data/bpe_model/` | 1 min |
| 3.7 Ext vocab | Ext call files | `data/external_calls/external_vocab.json` | 1 min |

---

## 3. Adding New Packages

### Step 1: Edit `configs/packages.conf`

Each line follows this format:
```
name | download_url | tarball_name | source_directory | binary1 binary2 ...
```

Example:
```
diffutils | https://ftp.gnu.org/gnu/diffutils/diffutils-3.10.tar.xz | diffutils-3.10.tar.xz | diffutils-3.10 | diff diff3 sdiff cmp
```

Fields:
- **name**: Package identifier (used in all file naming)
- **download_url**: Direct URL to the source tarball
- **tarball_name**: Filename of the downloaded archive
- **source_directory**: Top-level directory inside the tarball after extraction
- **binary names**: Space-separated list of binaries to extract after `make`

### Step 2: Test compilation

```bash
# Compile just the new package (the script is idempotent, skips already-built)
bash scripts/02_compile_dataset.sh
```

The script compiles at optimization levels O0, O1, O2, O3 by default. It produces:
- `data/raw/{name}_{binary}_O{level}_sym` (debug binary)
- `data/stripped/{name}_{binary}_O{level}_stripped` (stripped binary)

### Step 3: Verify the binaries

```bash
# Check that debug symbols exist
nm --defined-only data/raw/diffutils_diff_O2_sym | head

# Check that stripped binary has no symbols
nm data/stripped/diffutils_diff_O2_stripped 2>&1 | head
# Should say "no symbols"

# Check file type
file data/stripped/diffutils_diff_O2_stripped
# Should say "ELF 64-bit LSB executable, x86-64, ..."
```

### Step 4: Run preprocessing on the new package

```bash
bash scripts/03_preprocess.sh
```

This is incremental -- it will process new binaries and skip existing ones.

### Step 5: Update split assignments

New binaries default to the training set. If you want to assign them to val/test/demo, edit `data/split_assignments.json`:

```json
{
  "train": ["existing_binary_1", "new_package_binary_O0", "new_package_binary_O2", ...],
  "val": [...],
  "test": [...]
}
```

### Important considerations when adding packages

- **GNU packages work best** -- they share gnulib utility functions with existing training data
- **Non-GNU packages** (sqlite, lua, strace) can cause cross-contamination if the model is too small. See experiment log for Exp 34 analysis.
- **Large packages** (gdb, firefox) produce 10K+ functions and can dominate training. Consider capping or weighting.
- **O0 binaries** have ENDBR64 indirect jump wrapper issues where BAP lifts `endbr64; jmp addr` as separate tiny wrapper functions. `build_dataset.py` resolves these for training.

---

## 4. Running the Preprocessing Pipeline

### Full pipeline (all packages)

```bash
source ~/cs785-project/activate.sh
bash scripts/02_compile_dataset.sh    # Compile all packages
bash scripts/03_preprocess.sh          # Full preprocessing
```

### Individual stages (for debugging or re-running)

```bash
# Step 3.1: Extract labels only
for sym in data/raw/*_sym; do
    name=$(basename "$sym" _sym)
    python3 -m src.preprocessing.align_labels "$sym" "data/labels/${name}_labels.json"
done

# Step 3.2: Lift with BAP only
for stripped in data/stripped/*_stripped; do
    name=$(basename "$stripped" _stripped)
    bap "$stripped" --dump=bir:"data/bir/${name}.bir"
done

# Step 3.3: Parse BAP-IR into graphs
python3 -m src.preprocessing.parse_bap --input-dir data/bir/ --output-dir data/graphs/

# Step 3.4: Extract external calls
python3 -m src.preprocessing.extract_external --bir-dir data/bir/ --output-dir data/external_calls/

# Step 3.5: Match addresses
python3 -m src.preprocessing.build_dataset --build-index \
    --labels-dir data/labels/ --graphs-dir data/graphs/ \
    --output data/match_index.json

# Step 3.7: Build external vocab
python3 -m src.preprocessing.extract_external --build-vocab \
    --input-dir data/external_calls/ \
    --output data/external_calls/external_vocab.json
```

---

## 5. Data Directory Structure

```
data/
├── raw/                         # Debug binaries with symbols
│   └── {pkg}_{bin}_O{level}_sym
│
├── stripped/                    # Stripped binaries (input to BAP)
│   └── {pkg}_{bin}_O{level}_stripped
│
├── bir/                         # BAP Intermediate Representation
│   └── {pkg}_{bin}_O{level}.bir
│
├── graphs/                      # Per-function CFG graphs (JSON)
│   └── {pkg}_{bin}_O{level}_{func_name}.json
│   # ~241K files, ~14GB total
│
├── labels/                      # Ground truth from debug symbols
│   └── {pkg}_{bin}_O{level}_labels.json
│
├── external_calls/              # External function calls per binary
│   ├── {pkg}_{bin}_O{level}_external.json
│   └── external_vocab.json      # Global ext call vocabulary
│
├── bpe_model/                   # BPE tokenizer (legacy, Votes used now)
│   ├── bpe.model
│   └── bpe.vocab
│
├── match_index.json             # CENTRAL: maps graph files -> ground truth names
├── split_assignments.json       # Train/val/test binary assignments
└── votes_vocab.json             # Votes sub-token vocabulary (current tokenizer)
```

---

## 6. Key File Formats

### match_index.json (the central data source)

This is the single most important file. It maps each graph JSON to its ground truth label:

```json
{
  "data/graphs/bash_bash_O0_sub_31d09.json": {
    "binary": "bash_bash_O0",
    "address": "0x31d09",
    "address_int": 204041,
    "bap_name": "sub_31d09",
    "real_name": "main"
  },
  ...
}
```

### Graph JSON (per-function CFG)

```json
{
  "binary": "coreutils_ls_O2",
  "function_name": "sub_4a30",
  "address": "0x00004a30",
  "blocks": [
    {
      "id": 0,
      "label": "00004a30",
      "tokens": ["ARG_SETUP", "STACK_STORE_64", "CALL_malloc", "RETVAL"],
      "num_tokens": 4,
      "has_external_call": true,
      "external_call_name": "malloc"
    },
    {
      "id": 1,
      "label": "00004a58",
      "tokens": ["MEM_READ_64", "COMPARE", "COND_BRANCH_ZF"],
      "num_tokens": 3
    }
  ],
  "edges": [[0, 1], [1, 2], [1, 3]],
  "num_blocks": 4,
  "num_edges": 3,
  "internal_callees": ["sub_3f20", "sub_5100"]
}
```

### Labels JSON (per-binary ground truth)

```json
{
  "binary": "coreutils_ls_O2",
  "num_functions": 347,
  "functions": {
    "main": "0x0000000000005c40",
    "usage": "0x0000000000004a30",
    ...
  },
  "addr_to_name": {
    "0x0000000000005c40": "main",
    "0x0000000000004a30": "usage",
    ...
  }
}
```

### External calls JSON (per-binary)

```json
{
  "binary": "coreutils_ls_O2",
  "functions": [
    {
      "function_name": "sub_4a30",
      "external_calls": [
        {"name": "malloc", "call_order": 0},
        {"name": "fprintf", "call_order": 1}
      ],
      "num_external_calls": 2
    }
  ]
}
```

---

## 7. How Each Stage Works

### Stage 3.1: Label Extraction

Uses `nm --defined-only -n` on debug binaries. Filters out:
- Libc internals (`__libc_*`, `__cxa_*`)
- Runtime stubs (`_start`, `frame_dummy`, `register_tm_clones`)
- PLT/GOT entries (`.plt`, `.init`, `.fini`)
- Stack protection (`__stack_chk_fail`)

### Stage 3.2: BAP Lifting

BAP translates x86-64 machine code to BAP-IR (BIR), a human-readable intermediate representation:

```
0000168e:                              # block label (hex address)
0000178d: #12582911 := RSP             # register assignment
00001791: RSP := RSP - 8               # stack pointer update
000017a1: call @malloc:external        # external function call
000017fe: return #12582905             # function return
```

BAP names functions as `sub_XXXX` (hex address). This is why address matching (step 3.5) is needed.

### Stage 3.3: Instruction-Type Tokenization (V3)

The key innovation. Raw BAP-IR instructions are classified into ~1,510 semantic types:

| Category | Examples | What it captures |
|----------|----------|-----------------|
| Calls | `CALL_malloc`, `CALL_INTERNAL`, `CALL_INDIRECT` | Function call behavior |
| Memory | `MEM_READ_32`, `MEM_WRITE_ARG_64` | Memory access patterns |
| Stack | `STACK_LOAD_64`, `STACK_STORE_32` | Stack frame operations |
| Flags | `FLAG_CF`, `FLAG_ZF`, `COND_BRANCH_ZF` | Conditional logic |
| Registers | `ARG_SETUP`, `RETVAL`, `ARG_LOAD_ADDR` | Calling convention |
| Arithmetic | `ARITH_ADD`, `ARITH_SHIFT`, `ARITH_XOR` | Computation type |
| Constants | `ASSIGN_ZERO`, `ASSIGN_POW2`, `ASSIGN_ADDR` | Immediate values |
| Control | `BRANCH`, `RETURN`, `COMPARE` | Control flow |

This reduces 32K+ raw BAP tokens to ~1,510 types -- a +1,750% F1 improvement over raw tokens.

Implementation: `src/preprocessing/parse_bap.py` function `classify_instruction()`.

### Stage 3.4: External Call Extraction

Identifies external/library calls from BAP-IR patterns:
- Explicit: `call @malloc:external`
- Implicit: matches against 56 known library functions

### Stage 3.5: Address Matching

The trickiest step. Aligns two address formats:
- **Labels (nm):** `0x0000000000031d09` (16-digit padded hex)
- **BAP graphs:** `sub_31d09` stored as `0x31d09` (variable length)

Both are converted to integer addresses for matching. Unmatched functions are discarded.

---

## 8. Common Tasks

### Task: Add a new GNU package to the dataset

```bash
# 1. Edit configs/packages.conf -- add the package line
# 2. Compile
bash scripts/02_compile_dataset.sh
# 3. Preprocess
bash scripts/03_preprocess.sh
# 4. Verify
python3 -c "
import json
with open('data/match_index.json') as f:
    idx = json.load(f)
pkg_funcs = {k:v for k,v in idx.items() if 'newpkg' in k}
print(f'Matched {len(pkg_funcs)} functions from newpkg')
"
```

### Task: Check data quality for a specific binary

```bash
# How many functions were extracted by nm?
python3 -c "
import json
with open('data/labels/coreutils_ls_O2_labels.json') as f:
    labels = json.load(f)
print(f'Functions: {labels[\"num_functions\"]}')
"

# How many graphs were created?
ls data/graphs/coreutils_ls_O2_*.json | wc -l

# How many were matched?
python3 -c "
import json
with open('data/match_index.json') as f:
    idx = json.load(f)
matched = [k for k in idx if 'coreutils_ls_O2' in k]
print(f'Matched: {len(matched)}')
"
```

### Task: Inspect a specific function's graph

```bash
python3 -c "
import json
with open('data/graphs/coreutils_ls_O2_sub_4a30.json') as f:
    g = json.load(f)
print(f'Blocks: {g[\"num_blocks\"]}, Edges: {g[\"num_edges\"]}')
for b in g['blocks'][:3]:
    print(f'  Block {b[\"id\"]}: {b[\"tokens\"][:5]}...')
"
```

### Task: Re-run BAP on a single binary

```bash
bap data/stripped/coreutils_ls_O2_stripped --dump=bir:"data/bir/coreutils_ls_O2.bir"
```

### Task: Re-run graph parsing on a single .bir file

```bash
python3 -m src.preprocessing.parse_bap \
    --input data/bir/coreutils_ls_O2.bir \
    --output-dir data/graphs/
```

### Task: Check dataset split balance

```bash
python3 -c "
import json
with open('data/split_assignments.json') as f:
    splits = json.load(f)
for split, bins in splits.items():
    print(f'{split}: {len(bins)} binaries')
"
```

---

## 9. Debugging & Troubleshooting

### BAP fails on a binary

```bash
# Try with --no-byteweight flag
bap data/stripped/problematic_stripped --no-byteweight --dump=bir:"output.bir"

# Check if the binary is valid ELF
file data/stripped/problematic_stripped
readelf -h data/stripped/problematic_stripped
```

Common BAP failures:
- **Out of memory**: Large binaries (gdb, firefox). Try increasing ulimit.
- **Unsupported instructions**: Some SSE/AVX may not lift. BAP skips them.
- **Non-ELF binaries**: BAP only supports ELF. Windows PE requires cross-compilation pipeline.

### Address matching failures (low match rate)

```bash
# Check a specific binary
python3 -c "
import json
with open('data/labels/pkg_bin_O2_labels.json') as f:
    labels = json.load(f)

import glob
graphs = glob.glob('data/graphs/pkg_bin_O2_*.json')
print(f'Labels: {labels[\"num_functions\"]}, Graphs: {len(graphs)}')

# Check address format
sample_label_addr = list(labels['addr_to_name'].keys())[0]
with open(graphs[0]) as f:
    g = json.load(f)
sample_graph_addr = g['address']
print(f'Label addr format: {sample_label_addr}')
print(f'Graph addr format: {sample_graph_addr}')
"
```

Common match issues:
- **Leading zeros**: nm gives 16-digit hex, BAP gives variable length
- **Address offset**: Some compilers add offsets. Check with `readelf -S` for .text section base address
- **O0 indirect jump wrappers**: BAP lifts `endbr64; jmp addr+4` as separate functions, creating tiny 2-block wrapper stubs. These have wrong addresses. `build_dataset.py` resolves them for training.

### Tokenization produces unexpected tokens

```bash
# Test tokenization on a specific BAP-IR line
python3 -c "
from src.preprocessing.parse_bap import classify_instruction
result = classify_instruction('00001791: RSP := RSP - 8')
print(result)  # Should print something like 'STACK_OP'
"
```

---

## 10. Critical Rules

1. **NEVER overwrite `data/external_calls/external_vocab.json` during inference/evaluation.** The vocab is saved inside the model checkpoint. Only regenerate during preprocessing.

2. **Use deterministic sort `(-count, name)` for token vocab building.** Non-deterministic sort caused 1,232 token ID mismatches (bug found 2026-03-23). Always sort by count descending, then name ascending.

3. **`data/split_assignments.json` is the source of truth for splits.** New binaries default to train. Do not randomly reassign existing binaries.

4. **Do not modify `match_index.json` by hand.** Always regenerate via the pipeline.

5. **BAP lifting is the bottleneck** (~1-4 hours for the full dataset). Plan accordingly and avoid unnecessary re-runs.

6. **O0 binaries have the ENDBR64 indirect jump wrapper problem.** ~96% of O0 functions from BAP are tiny wrapper stubs (compiler-inserted `endbr64; jmp target` sequences). `build_dataset.py` resolves these by following the wrapper to its real callee. If you see O0 functions with only 1-2 blocks, this is expected before resolution.

7. **Keep packages.conf entries stable.** Changing a package name or binary list invalidates all downstream data for that package.

8. **Large packages need careful handling.** sqlite3 (10K+ functions), binutils (multiple large binaries), and gdb can dominate the dataset. Consider per-package function caps or weighted sampling.

---

## Quick Reference

```bash
# Full pipeline from scratch
source ~/cs785-project/activate.sh
bash scripts/02_compile_dataset.sh
bash scripts/03_preprocess.sh

# Check dataset stats
python3 -c "
import json
with open('data/match_index.json') as f:
    idx = json.load(f)
print(f'Total matched functions: {len(idx)}')

# Count per package
from collections import Counter
pkgs = Counter()
for v in idx.values():
    pkg = v['binary'].rsplit('_', 1)[0].split('_')[0]
    pkgs[pkg] += 1
for pkg, count in pkgs.most_common(10):
    print(f'  {pkg}: {count}')
"

# Train the model (after preprocessing)
python3 -m src.training.train --config configs/optimized.yaml --seed 42
```

---

## Files You'll Work With Most

| File | Purpose |
|------|---------|
| `configs/packages.conf` | Package list for compilation |
| `scripts/02_compile_dataset.sh` | Compilation pipeline |
| `scripts/03_preprocess.sh` | Full preprocessing orchestration |
| `src/preprocessing/parse_bap.py` | BAP-IR parser + V3 tokenization |
| `src/preprocessing/extract_external.py` | External call extractor |
| `src/preprocessing/align_labels.py` | Ground truth label extraction |
| `src/preprocessing/build_dataset.py` | Dataset loader (PyTorch) |
| `src/preprocessing/build_votes.py` | Votes name tokenizer |
| `data/match_index.json` | Central: graph -> label mapping |
| `data/split_assignments.json` | Train/val/test splits |
