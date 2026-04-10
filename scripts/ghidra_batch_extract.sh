#!/bin/bash
# ============================================================
# Ghidra headless CFG extraction for all stripped binaries
# Replaces BAP for binaries that BAP can't handle
# ============================================================
PROJ=/course/2026/spring/cs/785/hz79/adp232/cs785
GHIDRA=$PROJ/baselines/ghidra_11.3.1_PUBLIC
JAVA_HOME=$PROJ/baselines/jdk-21.0.10
export PATH=$JAVA_HOME/bin:$PATH

DATA_STRIPPED=$PROJ/data/stripped
DATA_GRAPHS=$PROJ/data/graphs
DATA_LABELS=$PROJ/data/labels
DATA_EXT=$PROJ/data/external_calls
GHIDRA_PROJ=/tmp/ghidra_extract_$$

mkdir -p $DATA_GRAPHS $DATA_LABELS $DATA_EXT $GHIDRA_PROJ

echo "═══════════════════════════════════════════════"
echo " Ghidra Headless CFG Extraction"
echo "═══════════════════════════════════════════════"

# Process each stripped binary that doesn't have graphs yet
for stripped in $DATA_STRIPPED/*_stripped; do
    [ -f "$stripped" ] || continue
    name=$(basename "$stripped" _stripped)

    # Skip if graphs already exist
    existing=$(ls $DATA_GRAPHS/${name}_*.json 2>/dev/null | wc -l)
    if [ "$existing" -gt 0 ]; then
        echo "SKIP $name ($existing graphs exist)"
        continue
    fi

    # Skip if no labels (can't match without GT)
    raw=$PROJ/data/raw/$name
    labels=$DATA_LABELS/${name}.json
    if [ ! -f "$labels" ]; then
        # Try to extract labels
        if [ -f "$raw" ] && file "$raw" | grep -q "ELF.*not stripped"; then
            nm --defined-only "$raw" 2>/dev/null | awk '$2 ~ /[tT]/ {printf "{\"0x%s\": \"%s\"}\n", $1, $3}' | python3 -c "
import sys, json
labels = {}
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        labels.update(d)
    except: pass
with open('$labels', 'w') as f:
    json.dump(labels, f, indent=2)
print('  Labels: $name = ' + str(len(labels)))
" 2>/dev/null
        fi
    fi

    echo "Processing $name..."

    # Run Ghidra headless (must run analysis to discover functions)
    $GHIDRA/support/analyzeHeadless \
        $GHIDRA_PROJ ghidra_proj_$$ \
        -import "$stripped" \
        -postScript $PROJ/scripts/ghidra_extract_cfg.py "$name" "$DATA_GRAPHS" \
        -deleteProject \
        -processor x86:LE:64:default \
        2>&1 | grep -E "Extracted|ERROR|WARN|functions" | tail -5 || true

    echo "  Done: $name ($(ls $DATA_GRAPHS/${name}_*.json 2>/dev/null | wc -l) graphs)"
done

# Extract external calls from graph files (parse CALL_ tokens)
echo ""
echo "═══════════════════════════════════════════════"
echo " Extracting external calls from graphs"
echo "═══════════════════════════════════════════════"

python3 -c "
import json, glob, os
from collections import defaultdict

graphs_dir = '$DATA_GRAPHS'
ext_dir = '$DATA_EXT'

# Find all binaries that need ext call extraction
binaries = set()
for gf in glob.glob(os.path.join(graphs_dir, '*.json')):
    name = os.path.basename(gf).replace('.json', '')
    # Extract binary name (everything before _sub_)
    parts = name.rsplit('_sub_', 1)
    if len(parts) == 2:
        binaries.add(parts[0])

for binary in sorted(binaries):
    ext_file = os.path.join(ext_dir, f'{binary}_external.json')
    if os.path.exists(ext_file):
        continue

    # Collect all external calls from this binary's graphs
    ext_calls = defaultdict(list)
    for gf in glob.glob(os.path.join(graphs_dir, f'{binary}_sub_*.json')):
        with open(gf) as f:
            g = json.load(f)
        func_name = g.get('function_name', '')
        func_ext = []
        for block in g.get('blocks', []):
            for token in block.get('tokens', []):
                if token.startswith('CALL_') and token not in ('CALL_INTERNAL', 'CALL_INDIRECT'):
                    call_name = token.replace('CALL_', '')
                    func_ext.append({'name': call_name, 'address': '0x0'})
        if func_ext:
            ext_calls[func_name] = func_ext

    # Save
    result = {
        'binary': binary,
        'functions': [{'function': fn, 'external_calls': calls} for fn, calls in ext_calls.items()]
    }
    with open(ext_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  ExtCalls: {binary} ({len(ext_calls)} functions with calls)')
"

echo ""
echo "═══════════════════════════════════════════════"
echo " Done!"
echo "═══════════════════════════════════════════════"
