#!/bin/bash
# ============================================================
# BAP Pipeline for ALL new/updated binaries
# ============================================================
# Runs: labels (nm) → BAP lift → parse → ext calls → match
# Processes ALL stripped binaries that don't have .bir files yet
# Incrementally updates match_index.json
# ============================================================
source ~/bfnr-project/activate.sh
cd ~/bfnr-project
set +e  # Don't exit on individual failures

DATA_RAW="data/raw"
DATA_STRIPPED="data/stripped"
DATA_BIR="data/bir"
DATA_GRAPHS="data/graphs"
DATA_LABELS="data/labels"
DATA_EXT="data/external_calls"

mkdir -p "$DATA_BIR" "$DATA_GRAPHS" "$DATA_LABELS" "$DATA_EXT"

echo "═══════════════════════════════════════════════"
echo " BAP Pipeline: Processing all new binaries"
echo "═══════════════════════════════════════════════"

# ══ Step 1: Extract labels from debug (raw) binaries via nm ══
echo ""
echo "══ Step 1: Extract labels via nm ══"
LABEL_COUNT=0
LABEL_SKIP=0
for raw_bin in "$DATA_RAW"/*_sym; do
    [ -f "$raw_bin" ] || continue
    name=$(basename "$raw_bin" _sym)
    # Check both naming conventions
    if [ -f "$DATA_LABELS/${name}_labels.json" ] || [ -f "$DATA_LABELS/${name}.json" ]; then
        LABEL_SKIP=$((LABEL_SKIP + 1))
        continue
    fi
    echo -n "  Labels: $name ... "
    labels_file="$DATA_LABELS/${name}_labels.json"
    nm --defined-only "$raw_bin" 2>/dev/null | awk '$2 ~ /[tT]/ {printf "{\"0x%s\": \"%s\"}\n", $1, $3}' | python3 -c "
import sys, json
binary = '$name'
labels = {}
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        labels.update(d)
    except: pass
# Save in the format expected by 03_preprocess.sh matching
addr_to_name = labels
name_to_addr = {v: k for k, v in labels.items()}
output = {
    'binary': binary,
    'num_functions': len(labels),
    'functions': labels,
    'name_to_addr': name_to_addr,
    'addr_to_name': addr_to_name,
}
with open('$labels_file', 'w') as f:
    json.dump(output, f, indent=2)
print(f'{len(labels)} labels')
"
    if [ $? -eq 0 ]; then
        LABEL_COUNT=$((LABEL_COUNT + 1))
    fi
done
echo "✓ Labels: $LABEL_COUNT new, $LABEL_SKIP skipped"

# ══ Step 2: BAP lift stripped binaries ══
echo ""
echo "══ Step 2: BAP lifting stripped binaries ══"
BAP_OK=0
BAP_FAIL=0
BAP_SKIP=0
for stripped_bin in "$DATA_STRIPPED"/*_stripped; do
    [ -f "$stripped_bin" ] || continue
    name=$(basename "$stripped_bin" _stripped)
    bir_file="$DATA_BIR/${name}.bir"
    if [ -f "$bir_file" ]; then
        BAP_SKIP=$((BAP_SKIP + 1))
        continue
    fi
    echo -n "  BAP: $name ... "
    if timeout 600 bap "$stripped_bin" --dump=bir:"$bir_file" 2>/dev/null; then
        echo "OK"
        BAP_OK=$((BAP_OK + 1))
    elif timeout 600 bap "$stripped_bin" --no-byteweight --dump=bir:"$bir_file" 2>/dev/null; then
        echo "OK (fallback)"
        BAP_OK=$((BAP_OK + 1))
    else
        echo "FAILED"
        BAP_FAIL=$((BAP_FAIL + 1))
    fi
done
echo "✓ BAP: $BAP_OK new, $BAP_SKIP skipped, $BAP_FAIL failed"

# ══ Step 3: Parse BAP-IR into CFG graphs ══
echo ""
echo "══ Step 3: Parse BAP-IR into CFG graphs ══"
PARSE_COUNT=0
for bir_file in "$DATA_BIR"/*.bir; do
    [ -f "$bir_file" ] || continue
    name=$(basename "$bir_file" .bir)
    # Skip if we already have graphs for this binary
    existing=$(ls "$DATA_GRAPHS"/${name}_sub_*.json 2>/dev/null | head -1)
    if [ -n "$existing" ]; then
        continue
    fi
    echo "  Parse: $name"
    python3 -m src.preprocessing.parse_bap \
        --bir "$bir_file" \
        --binary-name "$name" \
        --output-dir "$DATA_GRAPHS" 2>/dev/null || {
        echo "    ⚠ Parse failed for $name"
        continue
    }
    PARSE_COUNT=$((PARSE_COUNT + 1))
done
echo "✓ Parsed $PARSE_COUNT new binaries"

# ══ Step 4: Extract external calls ══
echo ""
echo "══ Step 4: Extract external calls ══"
EXT_COUNT=0
for bir_file in "$DATA_BIR"/*.bir; do
    [ -f "$bir_file" ] || continue
    name=$(basename "$bir_file" .bir)
    if [ -f "$DATA_EXT/${name}_external.json" ]; then
        continue
    fi
    echo "  ExtCalls: $name"
    python3 -m src.preprocessing.extract_external \
        --bir "$bir_file" \
        --binary-name "$name" \
        --output-dir "$DATA_EXT" 2>/dev/null || {
        echo "    ⚠ ExtCalls failed for $name"
        continue
    }
    EXT_COUNT=$((EXT_COUNT + 1))
done
echo "✓ External calls for $EXT_COUNT new binaries"

# ══ Step 5: Incremental address matching ══
echo ""
echo "══ Step 5: Incremental address matching ══"

python3 << 'PYEOF'
import json, glob, os

# Load existing match_index
mi_path = 'data/match_index.json'
if os.path.exists(mi_path):
    with open(mi_path) as f:
        match_index = json.load(f)
    print(f"  Existing match_index: {len(match_index)} entries")
else:
    match_index = {}
    print("  No existing match_index, starting fresh")

existing_keys = set(match_index.keys())

# Load ALL labels (both naming conventions)
labels_by_binary = {}
for lf in sorted(glob.glob('data/labels/*_labels.json') + glob.glob('data/labels/*.json')):
    with open(lf) as f:
        data = json.load(f)

    # Determine binary name
    if 'binary' in data:
        binary = data['binary']
    else:
        binary = os.path.basename(lf).replace('_labels.json', '').replace('.json', '')

    int_to_name = {}

    # Try addr_to_name format
    addr_to_name = data.get('addr_to_name', {})
    if addr_to_name:
        for addr_str, name in addr_to_name.items():
            try:
                addr_int = int(addr_str, 16)
                int_to_name[addr_int] = name
            except ValueError:
                pass

    # Try functions format (could be addr→name dict)
    if not int_to_name:
        funcs = data.get('functions', data)
        if isinstance(funcs, dict):
            for k, v in funcs.items():
                try:
                    addr_int = int(k, 16)
                    if isinstance(v, str):
                        int_to_name[addr_int] = v
                except (ValueError, TypeError):
                    try:
                        addr_int = int(v, 16)
                        int_to_name[addr_int] = k
                    except (ValueError, TypeError):
                        pass

    if int_to_name:
        labels_by_binary[binary] = int_to_name

total_labels = sum(len(v) for v in labels_by_binary.values())
print(f"  Loaded {total_labels} labels across {len(labels_by_binary)} binaries")

# Match NEW graph files only
graph_files = sorted(glob.glob('data/graphs/*.json'))
new_matched = 0
new_unmatched = 0

for gf in graph_files:
    if gf in existing_keys:
        continue

    try:
        with open(gf) as f:
            graph = json.load(f)
    except (json.JSONDecodeError, IOError):
        continue

    binary = graph.get('binary', '')
    address_str = graph.get('address', '')

    try:
        addr_int = int(address_str, 16)
    except (ValueError, TypeError):
        new_unmatched += 1
        continue

    label_map = labels_by_binary.get(binary, {})
    real_name = label_map.get(addr_int)

    if real_name:
        match_index[gf] = {
            'binary': binary,
            'address': address_str,
            'address_int': addr_int,
            'bap_name': graph.get('function_name', ''),
            'real_name': real_name,
        }
        new_matched += 1
    else:
        new_unmatched += 1

# Save
with open(mi_path, 'w') as f:
    json.dump(match_index, f, indent=2)

print(f"\n  ════════════════════════════════")
print(f"  New matched:   {new_matched}")
print(f"  New unmatched: {new_unmatched}")
print(f"  Total index:   {len(match_index)}")
print(f"  ════════════════════════════════")
PYEOF

echo ""
echo "═══════════════════════════════════════════════"
echo " BAP PIPELINE COMPLETE"
echo "═══════════════════════════════════════════════"
echo ""
echo "Next: sync to HPC and train"
