#!/bin/bash
# ============================================================
# STEP 6: Demo on Unseen Package
# ============================================================
# Saves every intermediate artifact so you can show the full
# pipeline in your presentation:
#   BIR → Graphs → External Calls → Labels → Prediction
#
# Usage: bash scripts/06_demo.sh
# ============================================================

set -e
source ~/bfnr-project/activate.sh
cd ~/bfnr-project

DEMO="demo"
mkdir -p "$DEMO/build" "$DEMO/raw" "$DEMO/stripped"
mkdir -p "$DEMO/bir" "$DEMO/graphs" "$DEMO/labels" "$DEMO/external_calls" "$DEMO/results"

echo "═══════════════════════════════════════════════════════════════"
echo " DEMO: Full Pipeline on Unseen Package (diffutils)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo " Training data:   coreutils, findutils, grep, gawk, sed, tar, + 11 more"
echo " Demo data:       diffutils (NEVER seen by model)"
echo " Model:           GAT + Option B (Gated Fusion)"
echo ""
echo " All intermediate files will be saved in demo/"
echo "═══════════════════════════════════════════════════════════════"

# Check model
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo "✗ No trained model. Run: bash scripts/04_train.sh"
    exit 1
fi

# ═══════════════════════════════════════
# Step 1: Download + Compile + Strip
# ═══════════════════════════════════════
echo ""
echo "══ Step 1: Download, Compile, Strip ══"

cd "$DEMO/build"
if [ ! -f "diffutils-3.10.tar.xz" ]; then
    echo "  Downloading diffutils-3.10..."
    wget -q --show-progress https://ftp.gnu.org/gnu/diffutils/diffutils-3.10.tar.xz
else
    echo "  Already downloaded."
fi

if [ ! -d "diffutils-3.10" ]; then
    echo "  Extracting..."
    tar xf diffutils-3.10.tar.xz
fi

cd diffutils-3.10
if [ ! -f "src/diff" ]; then
    echo "  Configuring..."
    ./configure CFLAGS="-g -O2" --quiet 2>/dev/null
    echo "  Compiling..."
    make -j$(nproc) --quiet 2>/dev/null
fi

cd ~/bfnr-project
for bin_name in diff cmp sdiff diff3; do
    src="$DEMO/build/diffutils-3.10/src/$bin_name"
    if [ -f "$src" ]; then
        cp "$src" "$DEMO/raw/diffutils_${bin_name}_sym"
        strip -s "$DEMO/raw/diffutils_${bin_name}_sym" -o "$DEMO/stripped/diffutils_${bin_name}_stripped"
        echo "  ✓ $bin_name → raw/ (debug) + stripped/ (no symbols)"
    fi
done

echo ""
echo "  Saved:"
echo "    $DEMO/raw/          — debug binaries (with function names)"
echo "    $DEMO/stripped/     — stripped binaries (names removed)"

# ═══════════════════════════════════════
# Step 2: Extract Labels from Debug Binary
# ═══════════════════════════════════════
echo ""
echo "══ Step 2: Extract Ground Truth Labels ══"

for debug_bin in $DEMO/raw/*_sym; do
    name=$(basename "$debug_bin" _sym)
    python3 -m src.preprocessing.align_labels \
        --debug-binary "$debug_bin" \
        --binary-name "$name" \
        --output-dir "$DEMO/labels"
done

LABEL_COUNT=$(ls $DEMO/labels/*.json 2>/dev/null | wc -l)
echo ""
echo "  Saved: $DEMO/labels/ ($LABEL_COUNT files)"
echo "  These contain address → real name mappings from debug binary."
echo "  Used ONLY for evaluation, NOT for prediction."

# ═══════════════════════════════════════
# Step 3: BAP Lifting (stripped binary)
# ═══════════════════════════════════════
echo ""
echo "══ Step 3: BAP Lifting (stripped → BAP-IR) ══"

for stripped_bin in $DEMO/stripped/*_stripped; do
    name=$(basename "$stripped_bin" _stripped)
    bir_file="$DEMO/bir/${name}.bir"
    if [ ! -f "$bir_file" ]; then
        echo -n "  BAP: $name ... "
        if bap "$stripped_bin" --dump=bir:"$bir_file" 2>/dev/null; then
            FUNC_COUNT=$(grep -c '^[0-9a-fA-F]*: sub ' "$bir_file" || echo 0)
            echo "✓ ($FUNC_COUNT functions)"
        else
            echo "✗ failed"
        fi
    else
        echo "  BAP: $name ... (already exists)"
    fi
done

echo ""
echo "  Saved: $DEMO/bir/ (BAP-IR files)"
echo "  These contain lifted assembly in BAP's intermediate representation."
echo "  Functions appear as sub_XXXX (no real names — binary is stripped)."

# Show sample
echo ""
echo "  Sample from diffutils_diff.bir:"
grep '^[0-9a-fA-F]*: sub sub_' "$DEMO/bir/diffutils_diff.bir" 2>/dev/null | head -5 | sed 's/^/    /'

# ═══════════════════════════════════════
# Step 4: Parse CFG Graphs
# ═══════════════════════════════════════
echo ""
echo "══ Step 4: Parse BAP-IR → CFG Graphs ══"

for bir_file in $DEMO/bir/*.bir; do
    name=$(basename "$bir_file" .bir)
    echo "  Parse: $name"
    python3 -m src.preprocessing.parse_bap \
        --bir "$bir_file" \
        --binary-name "$name" \
        --output-dir "$DEMO/graphs"
done

GRAPH_COUNT=$(ls $DEMO/graphs/*.json 2>/dev/null | wc -l)
echo ""
echo "  Saved: $DEMO/graphs/ ($GRAPH_COUNT function graph files)"
echo "  Each JSON contains: blocks (with tokens), edges (CFG structure)."

# Show sample
echo ""
echo "  Sample graph:"
python3 -c "
import json, glob
gf = sorted(glob.glob('$DEMO/graphs/*sub_*.json'))[0]
with open(gf) as f:
    g = json.load(f)
print(f'    File: {gf}')
print(f'    Function: {g[\"function_name\"]}')
print(f'    Address: {g[\"address\"]}')
print(f'    Blocks: {g[\"num_blocks\"]}, Edges: {g[\"num_edges\"]}')
print(f'    Block 0 tokens: {g[\"blocks\"][0][\"tokens\"][:10]}...')
"

# ═══════════════════════════════════════
# Step 5: Extract External Calls
# ═══════════════════════════════════════
echo ""
echo "══ Step 5: Extract External Function Calls ══"

for bir_file in $DEMO/bir/*.bir; do
    name=$(basename "$bir_file" .bir)
    echo "  ExtCalls: $name"
    python3 -m src.preprocessing.extract_external \
        --bir "$bir_file" \
        --binary-name "$name" \
        --output-dir "$DEMO/external_calls" \
        --vocab-path "$DEMO/external_calls/external_vocab.json"
done

echo ""
echo "  Saved: $DEMO/external_calls/"
echo "  These contain PLT/GOT function names (survive stripping)."
echo "  This is the key signal for Option B gated fusion."

# Show sample
python3 -c "
import json, glob
for ef in sorted(glob.glob('$DEMO/external_calls/*_external.json'))[:1]:
    with open(ef) as f:
        data = json.load(f)
    with_ext = sum(1 for func in data['functions'] if func['num_external_calls'] > 0)
    without = sum(1 for func in data['functions'] if func['num_external_calls'] == 0)
    print(f'    {data[\"binary\"]}: {with_ext} with ext calls, {without} without')
"

# ═══════════════════════════════════════
# Step 6: Predict with Option B Model
# ═══════════════════════════════════════
echo ""
echo "══ Step 6: Predict Function Names (Option B) ══"
echo ""
echo "  The model uses:"
echo "    ✓ Trained BPE model:     data/bpe_model/bpe.model (from training)"
echo "    ✓ External vocab:        data/external_calls/external_vocab.json (from training)"
echo "    ✓ Token vocab:           inside checkpoints/best_model.pt (from training)"
echo "    ✓ CFG + ext calls:       parsed on-the-fly from this demo binary"
echo ""

STRIPPED="$DEMO/stripped/diffutils_diff_stripped"
PRED_FILE="$DEMO/results/predictions_diff.json"

python3 scripts/predict.py \
    --binary "$STRIPPED" \
    --checkpoint checkpoints/best_model.pt \
    --output "$PRED_FILE" \
    --beam-width 5

# ═══════════════════════════════════════
# Step 7: Compare with Ground Truth
# ═══════════════════════════════════════
echo ""
echo ""
python3 << 'PYEOF'
import json, subprocess, re, sys
sys.path.insert(0, '.')
from src.evaluation.metrics import (
    compute_subtoken_f1, compute_exact_match,
    compute_char_ngram_similarity, compute_edit_distance_similarity,
    normalize_name, split_name,
)

debug_bin = 'demo/raw/diffutils_diff_sym'
pred_file = 'demo/results/predictions_diff.json'

# Extract ground truth
result = subprocess.run(['nm', '-n', '--defined-only', debug_bin], capture_output=True, text=True)
skip_re = [re.compile(p) for p in [r'^_', r'^\.', r'^frame_dummy', r'^register_tm_clones',
                                     r'^deregister_tm_clones', r'^__do_global', r'^__libc_csu']]
labels = {}
for line in result.stdout.strip().split('\n'):
    parts = line.split()
    if len(parts) >= 3 and parts[1] == 'T':
        name = parts[2]
        if len(name) > 1 and not any(r.search(name) for r in skip_re):
            labels[int(parts[0], 16)] = name

with open(pred_file) as f:
    preds = json.load(f)

rows = []
for p in preds:
    try:
        addr_int = int(p['address'], 16)
    except:
        continue
    true_name = labels.get(addr_int)
    if not true_name:
        continue
    pred_name = p['predicted_name']
    f1 = compute_subtoken_f1(pred_name, true_name)
    em = compute_exact_match(pred_name, true_name)
    ng = compute_char_ngram_similarity(pred_name, true_name)
    ed = compute_edit_distance_similarity(pred_name, true_name)
    rows.append({'address': p['address'], 'predicted': pred_name, 'true_name': true_name,
                 'f1': f1, 'em': em, 'ng': ng, 'ed': ed, 'ext': p.get('num_ext_calls', 0)})

rows.sort(key=lambda r: (-int(r['em']), -r['f1']))
total = len(rows)
exact = sum(r['em'] for r in rows)
avg_f1 = sum(r['f1'] for r in rows) / max(total, 1)
avg_ng = sum(r['ng'] for r in rows) / max(total, 1)
avg_ed = sum(r['ed'] for r in rows) / max(total, 1)

print("═══════════════════════════════════════════════════════════════════════════════════════════")
print("                    PREDICTION vs GROUND TRUTH  (diffutils — UNSEEN)")
print("═══════════════════════════════════════════════════════════════════════════════════════════")
print()
print(f"  {'Address':<10s} {'Predicted':<30s} {'True Name':<30s} {'F1':>5s} {'NgSim':>6s} {'EdSim':>6s} {'EM':>3s}")
print(f"  {'─'*10} {'─'*30} {'─'*30} {'─'*5} {'─'*6} {'─'*6} {'─'*3}")
for r in rows:
    em_str = '✓' if r['em'] else ' '
    print(f"  {r['address']:<10s} {r['predicted']:<30s} {r['true_name']:<30s} "
          f"{r['f1']:>5.2f} {r['ng']:>6.3f} {r['ed']:>6.3f} {em_str:>3s}")
print(f"  {'─'*92}")

print()
print("═══════════════════════════════════════════════════════════════════════════════════════════")
print("                                    SUMMARY")
print("═══════════════════════════════════════════════════════════════════════════════════════════")
print()
print(f"  Package:               diffutils (NOT in training data)")
print(f"  Functions predicted:   {len(preds)}")
print(f"  Functions with GT:     {total}")
print(f"  Exact matches:         {exact}/{total} ({100*exact/max(total,1):.1f}%)")
print(f"  Avg sub-token F1:      {avg_f1:.4f}")
print(f"  Avg char n-gram sim:   {avg_ng:.4f}")
print(f"  Avg edit distance sim: {avg_ed:.4f}")

perfect = [r for r in rows if r['em']]
close = [r for r in rows if not r['em'] and r['ed'] > 0.5]
poor = [r for r in rows if r['f1'] == 0 and r['ed'] < 0.3]

if perfect:
    print(f"\n  ✓ PERFECT predictions ({len(perfect)}):")
    for r in perfect:
        print(f"      {r['true_name']:<30s} (ext calls: {r['ext']})")
if close:
    print(f"\n  ~ CLOSE predictions ({len(close)}):")
    for r in close[:10]:
        print(f"      {r['true_name']:<25s} → {r['predicted']:<25s} EdSim={r['ed']:.3f}")
if poor:
    print(f"\n  ✗ POOR predictions ({len(poor)}):")
    for r in poor[:10]:
        print(f"      {r['true_name']:<25s} → {r['predicted']:<25s}")

with open('demo/results/comparison_diff.json', 'w') as f:
    json.dump({'package': 'diffutils', 'binary': 'diff', 'in_training': False,
               'total_predicted': len(preds), 'total_with_gt': total, 'exact_matches': exact,
               'avg_f1': avg_f1, 'avg_ngram_sim': avg_ng, 'avg_edit_sim': avg_ed,
               'functions': rows}, f, indent=2)

print(f"\n  Results saved to: demo/results/comparison_diff.json")
PYEOF

# ═══════════════════════════════════════
# Summary of all saved files
# ═══════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════════════════"
echo "                              SAVED FILES"
echo "═══════════════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  demo/"
echo "  ├── raw/                    Debug binaries (with real function names)"
ls $DEMO/raw/*_sym 2>/dev/null | while read f; do echo "  │   └── $(basename $f)"; done
echo "  ├── stripped/               Stripped binaries (NO function names — model input)"
ls $DEMO/stripped/*_stripped 2>/dev/null | while read f; do echo "  │   └── $(basename $f)"; done
echo "  ├── bir/                    BAP-IR (lifted from stripped binary)"
ls $DEMO/bir/*.bir 2>/dev/null | while read f; do echo "  │   └── $(basename $f)"; done
echo "  ├── labels/                 Ground truth: address → name (from debug binary)"
ls $DEMO/labels/*.json 2>/dev/null | while read f; do echo "  │   └── $(basename $f)"; done
echo "  ├── graphs/                 CFG graphs per function (blocks + tokens + edges)"
echo "  │   └── $(ls $DEMO/graphs/*.json 2>/dev/null | wc -l) files"
echo "  ├── external_calls/         External call lists per function (PLT names)"
ls $DEMO/external_calls/*.json 2>/dev/null | while read f; do echo "  │   └── $(basename $f)"; done
echo "  └── results/"
echo "      ├── predictions_diff.json     Model predictions"
echo "      └── comparison_diff.json      Predictions vs ground truth"
echo ""
echo "  Files from TRAINING used during prediction:"
echo "    data/bpe_model/bpe.model              BPE tokenizer (decode token IDs → text)"
echo "    data/external_calls/external_vocab.json  Ext call name → ID mapping"
echo "    checkpoints/best_model.pt             Model weights + token vocab"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════════════════"
echo " ✓ DEMO COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════════════════"
