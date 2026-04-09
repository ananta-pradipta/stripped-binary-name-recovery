# Task: Add More Packages to Dataset

**For:** Robert Blacha
**Project:** CS785 Binary Function Name Recovery
**Date:** 2026-04-09
**Deadline:** April 22, 2026
**Estimated time:** 1 day
**Where:** Local machine (BAP required) → sync to Wulver

---

## Goal

Add diverse packages to our 300K training dataset. More packages = better generalization. Target: **18 new packages** from SYMGEN's published dataset (already downloaded on Wulver).

---

## Prerequisites

### 1. Local machine setup

BAP is **not on Wulver** — all preprocessing is local.

```bash
cd ~/cs785-project
git pull origin v3
source ~/cs785-project/activate.sh
bap --version   # Must be 2.5.0
```

If BAP not installed:
```bash
bash scripts/01_setup_environment.sh   # ~30 min
```

### 2. Wulver access (for syncing results)

```bash
ssh wulver   # Duo 2FA, then connection persists 24h
ls /project/hz79/_shared/cs785/data/match_index.json   # Verify access
```

---

## What We Have (77 packages, 300K functions)

```bash
cd ~/cs785-project
python3 -c "
import json
from collections import Counter
with open('data/match_index.json') as f:
    mi = json.load(f)
pkgs = Counter(v['binary'].split('_')[0] for v in mi.values())
print(f'Total: {len(mi)} functions, {len(pkgs)} packages')
for p, c in pkgs.most_common(10):
    print(f'  {p}: {c}')
"
```

---

## What to Add (18 new packages)

These packages are in SYMGEN's published dataset (already on Wulver) but **not in our training data**:

| Package | Domain | Expected size |
|---------|--------|---------------|
| ncurses | Terminal UI | Large (~50K fns) |
| openssl | Crypto | Very large (~36K fns) — BAP may OOM |
| gettext | i18n | Medium (~4K fns) |
| readline | CLI input | Medium (~2K fns) |
| gmp | Math/bignum | Medium (~5K fns) |
| libiconv | Encoding | Small (~1K fns) |
| libidn2 | DNS/IDN | Small (~1K fns) |
| libunistring | Unicode | Medium (~3K fns) |
| libmicrohttpd | HTTP server | Small (~1K fns) |
| libredwg | CAD format | Medium (~3K fns) |
| libtool | Build tools | Small (~500 fns) |
| wget2 | HTTP client | Medium (~3K fns) |
| adns | DNS resolver | Small (~1K fns) |
| cflow | Call graph | Small (~500 fns) |
| dico | Dictionary | Medium (~5K fns) |
| freeipmi | IPMI mgmt | Medium (~3K fns) |
| gss | Security | Small (~500 fns) |
| poke | Binary editor | Medium (~3K fns) |

**Start with the easy/medium ones.** Skip openssl (BAP will OOM on the 7MB binary).

---

## Step-by-Step for Each Package

### 1. Copy binaries from Wulver to local

The SYMGEN dataset has unstripped binaries (with debug symbols):
```bash
# On local machine — copy ONE package at a time
scp -r wulver:/project/hz79/_shared/cs785/baselines/SymGen/zenodo/extracted_bins/x86_64/O2/gettext-0.21/ ~/cs785-project/data/symgen_bins/gettext/
```

Repeat for each package you're adding. Start with O2 (one opt level).

### 2. Prepare stripped + debug binaries

```bash
cd ~/cs785-project

# For each binary in the package:
for bin in data/symgen_bins/gettext/*; do
    [ -f "$bin" ] || continue
    name=$(basename "$bin")
    
    # Skip shared libraries for now (focus on executables)
    file "$bin" | grep -q "shared object" && continue
    
    pkg_name="gettext_${name}_O2"
    
    # Save debug copy
    cp "$bin" "data/raw/${pkg_name}_sym"
    
    # Create stripped version
    strip -s "$bin" -o "data/stripped/${pkg_name}_stripped"
    
    echo "Prepared: $pkg_name"
done
```

### 3. Extract ground truth labels

```bash
for f in data/raw/gettext_*_sym; do
    [ -f "$f" ] || continue
    name=$(basename "$f" _sym)
    label_file="data/labels/${name}_labels.json"
    
    # Skip if already exists
    [ -f "$label_file" ] && continue
    
    echo -n "Labels: $name... "
    nm --defined-only "$f" 2>/dev/null | awk '$2 ~ /[tT]/ {printf "{\"0x%s\": \"%s\"}\n", $1, $3}' | python3 -c "
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
with open('$label_file', 'w') as f:
    json.dump(output, f, indent=2)
print(f'{len(labels)} labels')
"
done
```

**CRITICAL FORMAT RULE:** The `functions` field MUST be `{name: addr}`:
```json
{
  "functions": {
    "main": "0x5c40",
    "usage": "0x4a30"
  }
}
```
**NOT** `{addr: name}`. The evaluation script will break otherwise.

### 4. BAP lift stripped binaries

```bash
source ~/cs785-project/activate.sh

for f in data/stripped/gettext_*_stripped; do
    [ -f "$f" ] || continue
    name=$(basename "$f" _stripped)
    bir="data/bir/${name}.bir"
    
    # Skip if already done
    [ -f "$bir" ] && continue
    
    echo -n "BAP: $name... "
    if timeout 600 bap "$f" --dump=bir:"$bir" 2>/dev/null; then
        echo "OK"
    elif timeout 600 bap "$f" --no-byteweight --dump=bir:"$bir" 2>/dev/null; then
        echo "OK (fallback)"
    else
        echo "FAILED — skip this binary"
    fi
done
```

**Notes:**
- Each binary takes 1-5 min
- If BAP takes >10 min or OOMs, skip that binary
- The `--no-byteweight` fallback handles some binaries BAP struggles with

### 5. Parse BAP-IR into graphs + extract external calls

```bash
for bir in data/bir/gettext_*.bir; do
    [ -f "$bir" ] || continue
    name=$(basename "$bir" .bir)
    
    # Skip if graphs already exist
    ls data/graphs/${name}_sub_*.json 2>/dev/null | head -1 | grep -q . && continue
    
    echo "Parse: $name"
    python3 -m src.preprocessing.parse_bap \
        --bir "$bir" --binary-name "$name" --output-dir data/graphs
    python3 -m src.preprocessing.extract_external \
        --bir "$bir" --binary-name "$name" --output-dir data/external_calls
done
```

### 6. Update match_index (incremental)

```bash
bash scripts/expand_bap_pipeline.sh
```

This automatically:
- Finds new graphs not yet in match_index
- Matches them against labels by address
- Appends to `data/match_index.json`

### 7. Verify the new package

```bash
python3 -c "
import json
with open('data/match_index.json') as f:
    mi = json.load(f)
pkg = sum(1 for v in mi.values() if v['binary'].startswith('gettext'))
print(f'gettext: {pkg} functions matched')
print(f'Total dataset: {len(mi)} functions')
"
```

### 8. Sync to Wulver

```bash
REMOTE="wulver:/project/hz79/_shared/cs785"

# Sync new data
rsync -az data/graphs/gettext_*.json "$REMOTE/data/graphs/"
rsync -az data/labels/gettext_*_labels.json "$REMOTE/data/labels/"
rsync -az data/external_calls/gettext_*.json "$REMOTE/data/external_calls/"

# Sync updated index
rsync -avz data/match_index.json "$REMOTE/data/"
```

### 9. Repeat for each package

Replace `gettext` with the next package name and repeat steps 1-8.

**Batch approach:** You can process multiple packages before syncing — just sync everything at the end.

---

## After All Packages: Rebuild Vocabularies

```bash
# Rebuild votes vocab (name tokenizer)
python3 -m src.preprocessing.build_votes \
    --match-index data/match_index.json \
    --output data/votes_vocab.json \
    --min-count 2

# Sync to Wulver
rsync -avz data/votes_vocab.json wulver:/project/hz79/_shared/cs785/data/
```

---

## Important Rules

1. **BAP is LOCAL ONLY** — not on Wulver
2. **`functions` must be `{name: addr}` format** — NOT `{addr: name}`
3. **Compile with `-no-pie`** if compiling from source (prevents address mismatch)
4. **Don't add these to training:** tengine, angie, nginx118, recutils (cross-project)
5. **Skip binaries >5MB** — BAP will likely OOM (openssl, gdb)
6. **Coordinate with Ananta** before changing `split_assignments.json`

---

## File Locations

| Location | What |
|----------|------|
| `data/raw/*_sym` | Debug binaries (local only) |
| `data/stripped/*_stripped` | Stripped binaries (local only) |
| `data/bir/*.bir` | BAP IR files (local only) |
| `data/graphs/*.json` | Per-function CFG graphs (sync to Wulver) |
| `data/labels/*_labels.json` | Ground truth labels (sync to Wulver) |
| `data/external_calls/*.json` | Library calls (sync to Wulver) |
| `data/match_index.json` | Central mapping (sync to Wulver) |
| SYMGEN binaries on Wulver | `/project/hz79/_shared/cs785/baselines/SymGen/zenodo/extracted_bins/x86_64/` |

---

## Questions?

Ask Ananta on Discord.
