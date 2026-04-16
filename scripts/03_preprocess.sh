#!/bin/bash
# ============================================================
# STEP 3: Full Preprocessing Pipeline (FIXED v3)
# ============================================================
# FIX: Address matching now normalizes both nm (0x00000000000026c5)
# and BAP sub_26c5 (0x26c5) to the same integer for comparison.
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
DATA_BPE="$PROJECT_ROOT/data/bpe_model"

rm -rf "$DATA_GRAPHS" "$DATA_LABELS" "$DATA_EXT" "$DATA_BPE"
mkdir -p "$DATA_BIR" "$DATA_GRAPHS" "$DATA_LABELS" "$DATA_EXT" "$DATA_BPE"

echo "═══════════════════════════════════════════════"
echo " Step 3: Full Preprocessing Pipeline"
echo "═══════════════════════════════════════════════"

# ══ 3.1 Labels from DEBUG binaries ══
echo ""
echo "══ 3.1 Extracting ground truth labels from DEBUG binaries ══"
echo ""
LABEL_COUNT=0
for debug_bin in "$DATA_RAW"/*_sym; do
    if [ -f "$debug_bin" ]; then
        name=$(basename "$debug_bin" _sym)
        echo "  Labels: $name"
        python3 -m src.preprocessing.align_labels \
            --debug-binary "$debug_bin" \
            --binary-name "$name" \
            --output-dir "$DATA_LABELS"
        LABEL_COUNT=$((LABEL_COUNT + 1))
    fi
done
echo ""
echo "✓ Labels extracted from $LABEL_COUNT binaries"

# ══ 3.2 BAP on STRIPPED binaries ══
echo ""
echo "══ 3.2 Lifting STRIPPED binaries with BAP ══"
echo ""
BAP_SUCCESS=0
BAP_FAIL=0
for stripped_bin in "$DATA_STRIPPED"/*_stripped; do
    if [ -f "$stripped_bin" ]; then
        name=$(basename "$stripped_bin" _stripped)
        # Skip if BIR already exists
        if [ -f "$DATA_BIR/${name}.bir" ]; then
            echo "  BAP: $name ... (already exists, skipping)"
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
		exit
            fi
        fi
    fi
done
echo ""
echo "✓ BAP: $BAP_SUCCESS success, $BAP_FAIL failed"

# ══ 3.3 Parse into CFG graphs ══
echo ""
echo "══ 3.3 Parsing BAP-IR into CFG graphs ══"
echo ""
for bir_file in "$DATA_BIR"/*.bir; do
    if [ -f "$bir_file" ]; then
        name=$(basename "$bir_file" .bir)
        echo "  Parse: $name"
        python3 -m src.preprocessing.parse_bap \
            --bir "$bir_file" \
            --binary-name "$name" \
            --output-dir "$DATA_GRAPHS"
    fi
done
TOTAL_GRAPHS=$(find "$DATA_GRAPHS" -name "*.json" | wc -l)
echo ""
echo "✓ $TOTAL_GRAPHS function graphs"

if [ "$TOTAL_GRAPHS" -eq 0 ]; then
    echo "  ✗ No graphs! Stopping."
    exit 1
fi

# ══ 3.4 External calls ══
echo ""
echo "══ 3.4 Extracting external function calls ══"
echo ""
for bir_file in "$DATA_BIR"/*.bir; do
    if [ -f "$bir_file" ]; then
        name=$(basename "$bir_file" .bir)
        echo "  ExtCalls: $name"
        python3 -m src.preprocessing.extract_external \
            --bir "$bir_file" \
            --binary-name "$name" \
            --output-dir "$DATA_EXT" \

    fi
done
echo "✓ External calls extracted"

# ══ 3.5 Address matching ══
echo ""
echo "══ 3.5 Matching functions by address ══"
echo ""
python3 -m src.preprocessing.match
echo ""
echo "✓ Address matching complete"

# ══ 3.6 BPE vocabulary ══
echo ""
echo "══ 3.6 Building BPE vocabulary ══"
echo ""

python3 << 'PYEOF'
import json, os, re
import sentencepiece as spm

with open('data/match_index.json') as f:
    idx = json.load(f)

names = set(v['real_name'] for v in idx.values())
print(f"  {len(names)} unique matched function names")

if len(names) == 0:
    print("  ✗ No matched names! Cannot train BPE.")
    print("    Fix the address matching first.")
    exit(1)

# Write corpus
corpus_path = 'data/bpe_model/corpus.txt'
os.makedirs('data/bpe_model', exist_ok=True)
with open(corpus_path, 'w') as f:
    for name in sorted(names):
        split = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        split = split.replace('_', ' _ ').replace('.', ' . ').lower()
        f.write(split + '\n')

# Safe vocab size
vocab_size = max(200, min(len(names) // 4, 2000))
print(f"  Vocab size: {vocab_size}")

spm.SentencePieceTrainer.train(
    input=corpus_path,
    model_prefix='data/bpe_model/bpe',
    vocab_size=vocab_size,
    model_type='unigram',
    pad_id=0, bos_id=1, eos_id=2, unk_id=3,
    pad_piece='<PAD>', bos_piece='<SOS>', eos_piece='<EOS>', unk_piece='<UNK>',
    character_coverage=1.0,
    max_sentence_length=200,
    num_threads=4,
)

sp = spm.SentencePieceProcessor(model_file='data/bpe_model/bpe.model')
print(f"  BPE trained: {sp.get_piece_size()} tokens")

print(f"\n  Encoding examples:")
for name in ['main', 'usage', 'hash_lookup', 'quotearg_buffer_restyled']:
    split = re.sub(r'([a-z])([A-Z])', r'\1 \2', name).replace('_', ' _ ').replace('.', ' . ').lower()
    tokens = sp.encode(split, out_type=str)
    print(f"    {name:35s} → {tokens}")
PYEOF

echo ""
echo "✓ BPE vocabulary trained"

# ══ FINAL VERIFICATION ══
echo ""
echo "═══════════════════════════════════════════════"
echo " FINAL VERIFICATION"
echo "═══════════════════════════════════════════════"

python3 << 'PYEOF'
import json, glob

labels = glob.glob('data/labels/*_labels.json')
graphs = glob.glob('data/graphs/*.json')
ext = glob.glob('data/external_calls/*_external.json')

with open('data/match_index.json') as f:
    mi = json.load(f)

import sentencepiece as spm
try:
    sp = spm.SentencePieceProcessor(model_file='data/bpe_model/bpe.model')
    bpe_size = sp.get_piece_size()
except:
    bpe_size = 'N/A'

print(f"  Label files:       {len(labels)}")
print(f"  Graph files:       {len(graphs)}")
print(f"  External call files: {len(ext)}")
print(f"  Matched functions: {len(mi)}")
print(f"  BPE vocab size:    {bpe_size}")

if len(mi) >= 100:
    print(f"\n  ✓ Ready for training! Run: bash scripts/04_train.sh")
elif len(mi) >= 20:
    print(f"\n  ⚠ Low match count but usable for testing.")
else:
    print(f"\n  ✗ Too few matches. Debug address alignment.")
PYEOF

echo ""
echo "═══════════════════════════════════════════════"
echo " ✓ PREPROCESSING COMPLETE"
echo " Next: bash scripts/04_train.sh"
echo "═══════════════════════════════════════════════"

# ══ 3.7 Rebuild global external vocabulary ══
echo ""
echo "══ 3.7 Rebuilding global external vocabulary ══"
python3 << 'PYEOF'
import json, glob

all_ext_names = set()
for ext_file in sorted(glob.glob('data/external_calls/*_external.json')):
    with open(ext_file) as f:
        data = json.load(f)
    for func in data.get('functions', []):
        for call in func.get('external_calls', []):
            name = call.get('name', '')
            if name:
                all_ext_names.add(name)
all_ext_names.discard('')

vocab = {'<NO_EXT>': 0}
for name in sorted(all_ext_names):
    vocab[name] = len(vocab)

with open('data/external_calls/external_vocab.json', 'w') as f:
    json.dump({'vocab_size': len(vocab), 'vocabulary': vocab}, f, indent=2)

print(f"  External vocabulary: {len(vocab)} tokens (from {len(glob.glob('data/external_calls/*_external.json'))} binaries)")
PYEOF
