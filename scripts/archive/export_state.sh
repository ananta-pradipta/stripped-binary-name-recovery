#!/bin/bash
# ============================================================
# Export Project State
# ============================================================
# Creates a state report summarizing the current project context.
#
# Usage: bash scripts/export_state.sh
# Output: results/project_state.md
# ============================================================
set -e
cd ~/cs785-project

OUT="results/project_state.md"
mkdir -p results

cat > "$OUT" << 'HEADER'
# Project State Report
Generated automatically — read this to understand the current project state.
HEADER

echo "" >> "$OUT"
echo "## Generated: $(date)" >> "$OUT"
echo "## Branch: $(git branch --show-current)" >> "$OUT"
echo "## Last commit: $(git log --oneline -1)" >> "$OUT"

# Dataset stats
echo "" >> "$OUT"
echo "## Dataset" >> "$OUT"
python3 -c "
import json, glob, os
mi = json.load(open('data/match_index.json'))
binaries = set(v['binary'] for v in mi.values())
ext_vocab = json.load(open('data/external_calls/external_vocab.json'))
bpe_vocab_lines = len(open('data/bpe_model/bpe.vocab').readlines())
print(f'- Functions: {len(mi)}')
print(f'- Binaries: {len(binaries)}')
print(f'- Ext vocab: {ext_vocab[\"vocab_size\"]} tokens')
print(f'- BPE vocab: {bpe_vocab_lines} tokens')
# Count by opt level
from collections import Counter
opt_counts = Counter()
for b in binaries:
    if '_O0' in b: opt_counts['O0'] += 1
    elif '_O1' in b: opt_counts['O1'] += 1
    elif '_O2' in b: opt_counts['O2'] += 1
    elif '_O3' in b: opt_counts['O3'] += 1
    else: opt_counts['unknown'] += 1
print(f'- By opt level: {dict(opt_counts)}')
" >> "$OUT" 2>/dev/null

# Checkpoint info
echo "" >> "$OUT"
echo "## Checkpoint" >> "$OUT"
python3 -c "
import torch
ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu')
print(f'- Epoch: {ckpt.get(\"epoch\", \"?\")}')
print(f'- Val F1: {ckpt.get(\"val_f1\", \"?\")}')
print(f'- Token vocab: {len(ckpt.get(\"token_vocab\", {}))}')
print(f'- Ext vocab: {len(ckpt.get(\"ext_vocab\", {}))}')
cfg = ckpt.get('config', {})
print(f'- LR: {cfg.get(\"training\", {}).get(\"learning_rate\", \"?\")}')
print(f'- Batch: {cfg.get(\"training\", {}).get(\"batch_size\", \"?\")}')
print(f'- Max blocks: {cfg.get(\"data\", {}).get(\"max_blocks_per_function\", \"?\")}')
print(f'- Max tokens: {cfg.get(\"data\", {}).get(\"max_tokens_per_block\", \"?\")}')
" >> "$OUT" 2>/dev/null

# Latest results
echo "" >> "$OUT"
echo "## Latest Results" >> "$OUT"
if [ -f results/ablation_table.json ]; then
    python3 -c "
import json
with open('results/ablation_table.json') as f:
    table = json.load(f)
for name, metrics in table.items():
    print(f'- {name}: F1={metrics[\"f1\"]:.4f} EM={metrics[\"exact_match\"]:.4f} EdSim={metrics[\"edit_sim\"]:.4f}')
" >> "$OUT" 2>/dev/null
fi

# Prediction diversity
echo "" >> "$OUT"
echo "## Prediction Diversity" >> "$OUT"
if [ -f results/predictions_test.json ]; then
    python3 -c "
import json
from collections import Counter
with open('results/predictions_test.json') as f:
    preds = json.load(f)
pred_names = Counter(p['predicted_name'] for p in preds)
print(f'- Total predictions: {len(preds)}')
print(f'- Unique predictions: {len(pred_names)}')
print(f'- Top-1 prediction: \"{pred_names.most_common(1)[0][0]}\" ({pred_names.most_common(1)[0][1]}x)')
# F1 by ext calls
from collections import defaultdict
by_ext = defaultdict(list)
for p in preds:
    n = p['num_ext_calls']
    bucket = '0' if n == 0 else '1-3' if n <= 3 else '4+'
    by_ext[bucket].append(p['f1'])
for bucket in ['0', '1-3', '4+']:
    vals = by_ext[bucket]
    print(f'- {bucket} ext calls: F1={sum(vals)/len(vals):.3f} ({len(vals)} functions)')
" >> "$OUT" 2>/dev/null
fi

# Gate analysis
echo "" >> "$OUT"
echo "## Gate Analysis" >> "$OUT"
if [ -f results/gate_analysis.json ]; then
    python3 -c "
import json
with open('results/gate_analysis.json') as f:
    g = json.load(f)
for group, data in g['groups'].items():
    print(f'- {group}: g={data[\"avg\"]:.3f} ± {data[\"std\"]:.3f} (n={data[\"count\"]})')
" >> "$OUT" 2>/dev/null
fi

# Config
echo "" >> "$OUT"
echo "## Current Config (configs/optimized.yaml)" >> "$OUT"
echo '```yaml' >> "$OUT"
cat configs/optimized.yaml >> "$OUT"
echo '```' >> "$OUT"

# File tree
echo "" >> "$OUT"
echo "## Project Structure" >> "$OUT"
echo '```' >> "$OUT"
find src/ scripts/ configs/ -name "*.py" -o -name "*.sh" -o -name "*.yaml" | sort >> "$OUT"
echo '```' >> "$OUT"

echo ""
echo "✓ Project state exported to: $OUT"
echo "  View with: cat results/project_state.md"
