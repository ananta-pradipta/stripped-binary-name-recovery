#!/bin/bash
# ============================================================
# Preprocess only NEW binaries (O1/O3 additions)
# Steps: Labels → BAP lift → Parse graphs → Extract ext calls → Update match_index
# ============================================================
source ~/bfnr-project/activate.sh
cd ~/bfnr-project

DATA_RAW="data/raw"
DATA_STRIPPED="data/stripped"
DATA_BIR="data/bir"
DATA_GRAPHS="data/graphs"
DATA_LABELS="data/labels"
DATA_EXT="data/external_calls"

mkdir -p "$DATA_BIR" "$DATA_GRAPHS" "$DATA_LABELS" "$DATA_EXT"

echo "═══════════════════════════════════════════════"
echo " Preprocessing New O1/O3 Binaries"
echo "═══════════════════════════════════════════════"

# ── 1. Extract labels from debug (raw) binaries ──
echo ""
echo "══ Step 1: Extract labels ══"
for raw_bin in "$DATA_RAW"/*_O1 "$DATA_RAW"/*_O3; do
    [ -f "$raw_bin" ] || continue
    name=$(basename "$raw_bin")
    labels_file="$DATA_LABELS/${name}.json"
    if [ -f "$labels_file" ]; then
        continue
    fi
    echo -n "  Labels: $name ... "
    # Extract all defined symbols (nm without -g to get static functions too)
    nm --defined-only "$raw_bin" 2>/dev/null | awk '$2 ~ /[tT]/ {printf "{\"0x%s\": \"%s\"}\n", $1, $3}' | python3 -c "
import sys, json
labels = {}
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        labels.update(d)
    except: pass
with open('$labels_file', 'w') as f:
    json.dump(labels, f, indent=2)
print(f'{len(labels)} labels')
"
done

# ── 2. BAP lift stripped binaries ──
echo ""
echo "══ Step 2: BAP lifting ══"
BAP_OK=0
BAP_FAIL=0
for stripped_bin in "$DATA_STRIPPED"/*_O1_stripped "$DATA_STRIPPED"/*_O3_stripped; do
    [ -f "$stripped_bin" ] || continue
    name=$(basename "$stripped_bin" _stripped)
    bir_file="$DATA_BIR/${name}.bir"
    if [ -f "$bir_file" ]; then
        BAP_OK=$((BAP_OK + 1))
        continue
    fi
    echo -n "  BAP: $name ... "
    if bap "$stripped_bin" --dump=bir:"$bir_file" 2>/dev/null; then
        echo "OK"
        BAP_OK=$((BAP_OK + 1))
    elif bap "$stripped_bin" --no-byteweight --dump=bir:"$bir_file" 2>/dev/null; then
        echo "OK (fallback)"
        BAP_OK=$((BAP_OK + 1))
    else
        echo "FAILED"
        BAP_FAIL=$((BAP_FAIL + 1))
    fi
done
echo "  BAP: $BAP_OK success, $BAP_FAIL failed"

# ── 3. Parse BAP-IR into CFG graphs ──
echo ""
echo "══ Step 3: Parse CFG graphs ══"
for bir_file in "$DATA_BIR"/*_O1.bir "$DATA_BIR"/*_O3.bir; do
    [ -f "$bir_file" ] || continue
    name=$(basename "$bir_file" .bir)
    # Check if graphs already exist
    existing=$(ls "$DATA_GRAPHS"/${name}_*.json 2>/dev/null | wc -l)
    if [ "$existing" -gt 0 ]; then
        continue
    fi
    echo "  Parse: $name"
    python3 -m src.preprocessing.parse_bap \
        --bir "$bir_file" \
        --binary-name "$name" \
        --output-dir "$DATA_GRAPHS"
done

# ── 4. Extract external calls ──
echo ""
echo "══ Step 4: Extract external calls ══"
for bir_file in "$DATA_BIR"/*_O1.bir "$DATA_BIR"/*_O3.bir; do
    [ -f "$bir_file" ] || continue
    name=$(basename "$bir_file" .bir)
    ext_file="$DATA_EXT/${name}_external.json"
    if [ -f "$ext_file" ]; then
        continue
    fi
    echo "  ExtCalls: $name"
    python3 -m src.preprocessing.extract_external \
        --bir "$bir_file" \
        --binary-name "$name" \
        --output-dir "$DATA_EXT" \
        --vocab-path "$DATA_EXT/external_vocab.json"
done

# ── 5. Rebuild match_index ──
echo ""
echo "══ Step 5: Rebuild match_index ══"
python3 -c "
import json, os, glob, re

DATA_GRAPHS = 'data/graphs'
DATA_LABELS = 'data/labels'

# Load existing match_index
mi_path = 'data/match_index.json'
if os.path.exists(mi_path):
    with open(mi_path) as f:
        match_index = json.load(f)
    print(f'  Existing match_index: {len(match_index)} entries')
else:
    match_index = {}

# Find all binaries from graph files
graph_files = glob.glob(os.path.join(DATA_GRAPHS, '*.json'))
binaries = set()
for gf in graph_files:
    name = os.path.basename(gf)
    # Extract binary name: everything before the last _sub_ or _0x
    m = re.match(r'^(.+?)_(sub_|0x)', name)
    if m:
        binaries.add(m.group(1))

# For each binary, load labels and match
new_matches = 0
for binary in sorted(binaries):
    labels_file = os.path.join(DATA_LABELS, f'{binary}.json')
    if not os.path.exists(labels_file):
        continue

    with open(labels_file) as f:
        labels = json.load(f)

    # Get all graph files for this binary
    bin_graphs = glob.glob(os.path.join(DATA_GRAPHS, f'{binary}_*.json'))
    for gf in bin_graphs:
        gf_name = os.path.basename(gf).replace('.json', '')
        if gf_name in match_index:
            continue

        # Extract address from graph filename
        # Format: binary_sub_XXXX or binary_0xXXXX
        m = re.search(r'(sub_|0x)([0-9a-fA-F]+)$', gf_name)
        if not m:
            continue
        addr_hex = m.group(2)
        addr_int = int(addr_hex, 16)

        # Try matching with labels
        for label_addr, label_name in labels.items():
            try:
                label_int = int(label_addr, 16)
            except:
                continue
            if label_int == addr_int or label_int == addr_int + 4:
                # Skip compiler-generated names
                if label_name.startswith('_') and not label_name.startswith('__'):
                    continue
                match_index[gf_name] = {
                    'binary': binary,
                    'address': hex(addr_int),
                    'address_int': addr_int,
                    'bap_name': gf_name.split(binary + '_', 1)[1] if binary + '_' in gf_name else gf_name,
                    'real_name': label_name,
                }
                new_matches += 1
                break

print(f'  New matches: {new_matches}')
print(f'  Total match_index: {len(match_index)} entries')

with open(mi_path, 'w') as f:
    json.dump(match_index, f, indent=2)
print(f'  Saved to {mi_path}')
"

echo ""
echo "═══════════════════════════════════════════════"
echo " Done! New binaries preprocessed."
echo "═══════════════════════════════════════════════"
