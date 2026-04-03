#!/bin/bash
# ============================================================
# Preprocess ONLY new demo/cross-project packages
# Does NOT delete existing data or rebuild vocabs
# ============================================================

set -e
source ~/cs785-project/activate.sh

PROJECT_ROOT="$HOME/cs785-project"
DATA_RAW="$PROJECT_ROOT/data/raw"
DATA_STRIPPED="$PROJECT_ROOT/data/stripped"
DATA_BIR="$PROJECT_ROOT/data/bir"
DATA_GRAPHS="$PROJECT_ROOT/data/graphs"
DATA_LABELS="$PROJECT_ROOT/data/labels"
DATA_EXT="$PROJECT_ROOT/data/external_calls"

# Packages to preprocess (only new ones not yet in data/graphs)
NEW_PACKAGES="rcs tree bzip2 nginx dos2unix curl"

mkdir -p "$DATA_BIR" "$DATA_GRAPHS" "$DATA_LABELS" "$DATA_EXT"

echo "═══════════════════════════════════════════════"
echo " Preprocessing NEW demo packages only"
echo " Packages: $NEW_PACKAGES"
echo " Existing data will NOT be touched"
echo "═══════════════════════════════════════════════"

# ══ 3.1 Labels from DEBUG binaries ══
echo ""
echo "══ 3.1 Extracting ground truth labels ══"
for pkg in $NEW_PACKAGES; do
    for debug_bin in "$DATA_RAW"/${pkg}_*_sym; do
        if [ -f "$debug_bin" ]; then
            name=$(basename "$debug_bin" _sym)
            label_file="$DATA_LABELS/${name}_labels.json"
            if [ -f "$label_file" ]; then
                echo "  Labels: $name (exists, skipping)"
                continue
            fi
            echo "  Labels: $name"
            python3 -c "
import subprocess, json, sys

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
        if name.startswith('_') and not name.startswith('__') or name in ('main', '_start', '_init', '_fini'):
            if name in ('main',):
                pass  # keep main
            elif name.startswith('_'):
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

out_path = f'{output_dir}/{binary_name}_labels.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f'    {len(functions)} functions')
"
        fi
    done
done

# ══ 3.2 BAP on STRIPPED binaries ══
echo ""
echo "══ 3.2 Lifting with BAP (this is slow) ══"
BAP_SUCCESS=0
BAP_FAIL=0
for pkg in $NEW_PACKAGES; do
    for stripped_bin in "$DATA_STRIPPED"/${pkg}_*; do
        if [ -f "$stripped_bin" ]; then
            name=$(basename "$stripped_bin")
            # Skip if BIR already exists
            if [ -f "$DATA_BIR/${name}.bir" ]; then
                echo "  BAP: $name (exists, skipping)"
                BAP_SUCCESS=$((BAP_SUCCESS + 1))
                continue
            fi
            echo -n "  BAP: $name ... "
            if bap "$stripped_bin" --dump=bir:"$DATA_BIR/${name}.bir" 2>/dev/null; then
                echo "✓"
                BAP_SUCCESS=$((BAP_SUCCESS + 1))
            else
                if bap "$stripped_bin" --no-byteweight --dump=bir:"$DATA_BIR/${name}.bir" 2>/dev/null; then
                    echo "✓ (fallback)"
                    BAP_SUCCESS=$((BAP_SUCCESS + 1))
                else
                    echo "✗ FAILED"
                    BAP_FAIL=$((BAP_FAIL + 1))
                fi
            fi
        fi
    done
done
echo ""
echo "✓ BAP: $BAP_SUCCESS success, $BAP_FAIL failed"

# ══ 3.3 Parse into CFG graphs ══
echo ""
echo "══ 3.3 Parsing BAP-IR into CFG graphs ══"
NEW_GRAPHS=0
for pkg in $NEW_PACKAGES; do
    for bir_file in "$DATA_BIR"/${pkg}_*.bir; do
        if [ -f "$bir_file" ]; then
            name=$(basename "$bir_file" .bir)
            # Check if graphs already exist for this binary
            existing=$(ls "$DATA_GRAPHS"/${name}_*.json 2>/dev/null | wc -l)
            if [ "$existing" -gt 0 ]; then
                echo "  Parse: $name ($existing graphs exist, skipping)"
                NEW_GRAPHS=$((NEW_GRAPHS + existing))
                continue
            fi
            echo "  Parse: $name"
            python3 -m src.preprocessing.parse_bap \
                --bir "$bir_file" \
                --binary-name "$name" \
                --output-dir "$DATA_GRAPHS"
            count=$(ls "$DATA_GRAPHS"/${name}_*.json 2>/dev/null | wc -l)
            NEW_GRAPHS=$((NEW_GRAPHS + count))
        fi
    done
done
echo ""
echo "✓ $NEW_GRAPHS new function graphs"

# ══ 3.4 External calls ══
echo ""
echo "══ 3.4 Extracting external calls ══"
for pkg in $NEW_PACKAGES; do
    for bir_file in "$DATA_BIR"/${pkg}_*.bir; do
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
                --output-dir "$DATA_EXT"
        fi
    done
done

# ══ 3.5 Address matching (append to match_index) ══
echo ""
echo "══ 3.5 Matching functions by address ══"
python3 << 'PYEOF'
import json, glob, os

NEW_PACKAGES = "rcs tree bzip2 nginx dos2unix curl".split()

# Load existing match_index
match_index_path = 'data/match_index.json'
with open(match_index_path) as f:
    existing_index = json.load(f)

# Track existing binaries to avoid duplicates
existing_binaries = set()
for key, entry in existing_index.items():
    existing_binaries.add(entry.get('binary', ''))

print(f"  Existing match_index: {len(existing_index)} entries, {len(existing_binaries)} binaries")

# Process new packages
new_entries = 0
for pkg in NEW_PACKAGES:
    # Find label files
    for lf in sorted(glob.glob(f'data/labels/{pkg}_*_labels.json')):
        with open(lf) as f:
            data = json.load(f)
        binary = data['binary']

        if binary in existing_binaries:
            print(f"  {binary}: already in match_index, skipping")
            continue

        # Build address → name mapping
        int_to_name = {}
        addr_to_name = data.get('addr_to_name', {})
        funcs = data.get('functions', {})

        if addr_to_name:
            for addr_str, name in addr_to_name.items():
                try:
                    int_to_name[int(addr_str, 16)] = name
                except ValueError:
                    pass
        elif funcs:
            first_key = next(iter(funcs), '')
            if first_key.startswith('0x'):
                for addr_str, name in funcs.items():
                    try:
                        int_to_name[int(addr_str, 16)] = name
                    except ValueError:
                        pass
            else:
                for name, addr_str in funcs.items():
                    try:
                        if isinstance(addr_str, str):
                            int_to_name[int(addr_str, 16)] = name
                        elif isinstance(addr_str, int):
                            int_to_name[addr_str] = name
                    except (ValueError, TypeError):
                        pass

        if not int_to_name:
            print(f"  {binary}: no labels found, skipping")
            continue

        # Find matching graphs
        graph_files = sorted(glob.glob(f'data/graphs/{binary}_*.json'))
        matched = 0
        for gf in graph_files:
            with open(gf) as f:
                graph = json.load(f)

            # Extract address from BAP function name (sub_XXXX)
            bap_name = graph.get('function_name', '')
            if bap_name.startswith('sub_'):
                try:
                    bap_addr = int(bap_name[4:], 16)
                except ValueError:
                    continue
            else:
                continue

            if bap_addr in int_to_name:
                real_name = int_to_name[bap_addr]
                existing_index[gf] = {
                    'binary': binary,
                    'address': hex(bap_addr),
                    'address_int': bap_addr,
                    'bap_name': bap_name,
                    'real_name': real_name,
                }
                matched += 1
                new_entries += 1

        print(f"  {binary}: {matched} matched / {len(graph_files)} graphs / {len(int_to_name)} labels")

print(f"\n  New entries: {new_entries}")
print(f"  Total match_index: {len(existing_index)} entries")

# Save updated index
with open(match_index_path, 'w') as f:
    json.dump(existing_index, f, indent=2)
print(f"  Saved to {match_index_path}")
PYEOF

echo ""
echo "═══════════════════════════════════════════════"
echo " Done! New demo packages preprocessed."
echo " Existing training data is untouched."
echo " No vocab or split changes needed."
echo "═══════════════════════════════════════════════"
echo ""
echo "Note: The new packages are NOT added to split_assignments.json"
echo "(they're evaluated separately as cross-project demo)."
echo "The token vocab and ext call vocab remain unchanged"
echo "(new tokens map to <UNK>, which is expected for cross-project eval)."
