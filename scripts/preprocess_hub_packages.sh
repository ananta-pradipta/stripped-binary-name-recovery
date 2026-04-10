#!/bin/bash
# ============================================================
# Preprocess hub packages (incremental — no existing data deleted)
# Steps: Extract labels → BAP lift → Parse to graphs → Extract ext calls
# ============================================================

set -e
source ~/cs785-project/activate.sh

PROJECT_ROOT="$HOME/cs785-project"
DATA_DEBUG="$PROJECT_ROOT/data/debug"
DATA_STRIPPED="$PROJECT_ROOT/data/stripped"
DATA_BIR="$PROJECT_ROOT/data/bir"
DATA_GRAPHS="$PROJECT_ROOT/data/graphs"
DATA_LABELS="$PROJECT_ROOT/data/labels"
DATA_EXT="$PROJECT_ROOT/data/external_calls"

# Hub packages to preprocess
HUB_PACKAGES="zlib xz zstd libxml2 libpng expat pcre2 lz4 libyaml libarchive libsodium openssl"

mkdir -p "$DATA_BIR" "$DATA_GRAPHS" "$DATA_LABELS" "$DATA_EXT"

echo "═══════════════════════════════════════════════"
echo " Preprocessing Hub Packages (incremental)"
echo " Packages: $HUB_PACKAGES"
echo " Existing data will NOT be touched"
echo "═══════════════════════════════════════════════"

# ══ Step 1: Extract labels from debug binaries ══
echo ""
echo "══ Step 1: Extracting ground truth labels ══"
label_count=0
for pkg in $HUB_PACKAGES; do
    for debug_bin in "$DATA_DEBUG"/${pkg}_*_O0 "$DATA_DEBUG"/${pkg}_*_O2; do
        if [ -f "$debug_bin" ]; then
            name=$(basename "$debug_bin")
            label_file="$DATA_LABELS/${name}_labels.json"
            if [ -f "$label_file" ]; then
                echo "  Labels: $name (exists, skipping)"
                continue
            fi
            echo "  Labels: $name"
            python3 -c "
import subprocess, json, sys, os

debug_bin = '$debug_bin'
binary_name = '$name'
output_dir = '$DATA_LABELS'

# Run nm to get symbol table
result = subprocess.run(['nm', '-g', debug_bin], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')

functions = {}
addr_to_name = {}
name_to_addr = {}

for line in lines:
    parts = line.split()
    if len(parts) >= 3 and parts[1] in ('T', 't'):
        addr = parts[0]
        name = parts[2]
        # Skip internal/compiler names
        if name.startswith('_') and not name.startswith('__'):
            continue
        if name in ('_start', '_init', '_fini'):
            continue
        addr_hex = '0x' + addr.lstrip('0') if addr.lstrip('0') else '0x0'
        functions[name] = addr_hex
        addr_to_name[addr_hex] = name
        name_to_addr[name] = addr_hex

output = {
    'binary': binary_name,
    'num_functions': len(functions),
    'functions': functions,
    'name_to_addr': name_to_addr,
    'addr_to_name': addr_to_name,
}

out_path = os.path.join(output_dir, binary_name + '_labels.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f'    → {len(functions)} functions')
"
            label_count=$((label_count + 1))
        fi
    done
done
echo "✓ Labels: $label_count new binaries"

# ══ Step 2: BAP lift stripped binaries ══
echo ""
echo "══ Step 2: Lifting stripped binaries with BAP ══"
bap_count=0
for pkg in $HUB_PACKAGES; do
    for stripped_bin in "$DATA_STRIPPED"/${pkg}_*_O0 "$DATA_STRIPPED"/${pkg}_*_O2; do
        if [ -f "$stripped_bin" ]; then
            name=$(basename "$stripped_bin")
            bir_file="$DATA_BIR/${name}.bir"
            if [ -f "$bir_file" ]; then
                echo "  BAP: $name (exists, skipping)"
                continue
            fi
            echo "  BAP: $name"
            timeout 600 bap "$stripped_bin" -d > "$bir_file" 2>/dev/null || {
                echo "    WARNING: BAP failed for $name"
                rm -f "$bir_file"
                continue
            }
            bap_count=$((bap_count + 1))
            echo "    → $(wc -l < "$bir_file") lines"
        fi
    done
done
echo "✓ BAP lifted: $bap_count new binaries"

# ══ Step 3: Parse BAP-IR to graphs ══
echo ""
echo "══ Step 3: Parsing BAP-IR to graphs ══"
for pkg in $HUB_PACKAGES; do
    for bir_file in "$DATA_BIR"/${pkg}_*_O0.bir "$DATA_BIR"/${pkg}_*_O2.bir; do
        if [ -f "$bir_file" ]; then
            name=$(basename "$bir_file" .bir)
            # Check if graphs already exist
            existing=$(ls "$DATA_GRAPHS"/${name}_sub_*.json 2>/dev/null | wc -l)
            if [ "$existing" -gt 0 ]; then
                echo "  Parse: $name ($existing graphs exist, skipping)"
                continue
            fi
            echo "  Parse: $name"
            python3 -m src.preprocessing.parse_bap \
                --bir "$bir_file" \
                --binary-name "$name" \
                --output-dir "$DATA_GRAPHS" \
                2>/dev/null || echo "    WARNING: parse failed for $name"
        fi
    done
done
echo "✓ Graphs parsed"

# ══ Step 4: Extract external calls ══
echo ""
echo "══ Step 4: Extracting external calls ══"
for pkg in $HUB_PACKAGES; do
    for bir_file in "$DATA_BIR"/${pkg}_*_O0.bir "$DATA_BIR"/${pkg}_*_O2.bir; do
        if [ -f "$bir_file" ]; then
            name=$(basename "$bir_file" .bir)
            ext_file="$DATA_EXT/${name}_external.json"
            if [ -f "$ext_file" ]; then
                echo "  ExtCalls: $name (exists, skipping)"
                continue
            fi
            echo "  ExtCalls: $name"
            python3 -m src.preprocessing.extract_external \
                --bir "$bir_file" \
                --binary-name "$name" \
                --output-dir "$DATA_EXT" \
                2>/dev/null || echo "    WARNING: extract failed for $name"
        fi
    done
done
echo "✓ External calls extracted"

# ══ Summary ══
echo ""
echo "═══════════════════════════════════════════════"
echo " Hub Package Preprocessing Complete"
echo "═══════════════════════════════════════════════"
echo ""
echo "Labels:     $(ls $DATA_LABELS/*_labels.json 2>/dev/null | wc -l) total"
echo "BIR files:  $(ls $DATA_BIR/*.bir 2>/dev/null | wc -l) total"
echo "Graphs:     $(ls $DATA_GRAPHS/*.json 2>/dev/null | wc -l) total"
echo "Ext calls:  $(ls $DATA_EXT/*_external.json 2>/dev/null | wc -l) total"
echo ""
echo "New hub package files:"
for pkg in $HUB_PACKAGES; do
    graphs=$(ls "$DATA_GRAPHS"/${pkg}_*.json 2>/dev/null | wc -l)
    labels=$(ls "$DATA_LABELS"/${pkg}_*_labels.json 2>/dev/null | wc -l)
    echo "  $pkg: $labels label files, $graphs graph files"
done
echo ""
echo "Next: Update match_index.json, then retrain"
