#!/bin/bash
# ============================================================
# STEP 4: Train All Model Variants
# ============================================================
# Rerun-safe: clears old checkpoints, writes correct configs,
# trains all 5 variants, then runs evaluation + demo.
#
# Usage: bash scripts/04_train.sh
# Time:  ~1-3 hours (GPU), ~6-10 hours (CPU)
# ============================================================
set -e
source ~/bfnr-project/activate.sh
cd ~/bfnr-project

echo "═══════════════════════════════════════════════"
echo " Step 4: Training All Model Variants"
echo "═══════════════════════════════════════════════"

python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  ⚠ No GPU. Training will be slow.')
" 2>/dev/null || echo "  Checking GPU..."

# ── Clear old checkpoints ──
echo ""
echo "  Clearing old checkpoints..."
rm -rf checkpoints/variant2_gcn checkpoints/variant3_gat checkpoints/variant4_optA
rm -f checkpoints/best_model.pt
mkdir -p checkpoints results

# ══════════════════════════════════════════
# Variant 1: DeBin Baseline (ExtraTrees)
# ══════════════════════════════════════════
echo ""
#echo "═══ Variant 1/5: DeBin Baseline (ExtraTrees) ═══"
#python3 << 'EXTRATREES'
#import json, os, numpy as np
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.preprocessing import LabelEncoder
#
#with open('data/match_index.json') as f:
#    mi = json.load(f)
#
#features, names, binaries = [], [], []
#for gf, info in mi.items():
#    if not os.path.exists(gf): continue
#    with open(gf) as f:
#        g = json.load(f)
#    tokens = [t for b in g['blocks'] for t in b['tokens']]
#    from collections import Counter
#    c = Counter(tokens)
#    features.append([
#        g['num_blocks'], g['num_edges'], len(tokens),
#        sum(v for k,v in c.items() if k.startswith('CALL_') and k not in ('CALL_INTERNAL','CALL_INDIRECT')),
#        c.get('CALL_INTERNAL',0), c.get('COND_BRANCH',0),
#        c.get('MEM_READ',0)+c.get('MEM_WRITE',0), c.get('ARITH',0),
#    ])
#    names.append(info['real_name'])
#    binaries.append(info['binary'])
#
#X = np.array(features)
#le = LabelEncoder()
#y = le.fit_transform(names)
#
#unique_bins = sorted(set(binaries))
#np.random.seed(42)
#np.random.shuffle(unique_bins)
#split = int(0.8 * len(unique_bins))
#train_bins = set(unique_bins[:split])
#train_mask = np.array([b in train_bins for b in binaries])
#X_train, y_train = X[train_mask], y[train_mask]
#X_test, y_test = X[~train_mask], y[~train_mask]
#
#if len(X_test) > 0:
#    clf = ExtraTreesClassifier(n_estimators=20, random_state=42, n_jobs=1)
#    clf.fit(X_train, y_train)
#    acc = clf.score(X_test, y_test)
#    print(f"  Accuracy: {acc:.4f} ({len(X_train)} train, {len(X_test)} test)")
#    os.makedirs('results', exist_ok=True)
#    json.dump({'accuracy': acc}, open('results/baseline_extratrees.json','w'), indent=2)
#EXTRATREES

# # ══════════════════════════════════════════
# # Write optimized configs (correct hyperparams)
# # ══════════════════════════════════════════

# # Shared data block
# read -r -d '' DATA_BLOCK << 'DATAEOF' || true
# data:
#   graphs_dir: data/graphs
#   labels_dir: data/labels
#   external_calls_dir: data/external_calls
#   bpe_model_path: data/bpe_model/bpe.model
#   external_vocab_path: data/external_calls/external_vocab.json
#   train_split: 0.80
#   val_split: 0.10
#   test_split: 0.10
#   max_blocks_per_function: 50
#   max_tokens_per_block: 48
#   max_name_length: 16
# DATAEOF

# # Shared training block
# read -r -d '' TRAIN_BLOCK << 'TRAINEOF' || true
# training:
#   epochs: 70
#   batch_size: 128
#   learning_rate: 0.001
#   weight_decay: 0.0001
#   scheduler: cosine
#   warmup_steps: 200
#   gradient_clip: 1.0
#   early_stopping_patience: 7
# TRAINEOF

# # ── Variant 2: GCN + Mean Pool ──
# cat > configs/variant2_gcn_opt.yaml << EOF
# $DATA_BLOCK
# block_encoder:
#   type: mean
#   token_vocab_size: 3000
#   token_embed_dim: 128
#   num_layers: 2
#   num_heads: 4
#   dropout: 0.15
#   output_dim: 256
#   token_pooling: attention
#   use_block_features: true
# graph_encoder:
#   type: gcn
#   hidden_dim: 256
#   num_layers: 2
#   num_heads: 4
#   dropout: 0.15
#   pooling: mean
#   output_dim: 512
# external_encoder:
#   enabled: false
#   vocab_size: 1000
#   embed_dim: 256
#   hidden_dim: 256
#   num_layers: 1
#   dropout: 0.1
#   output_dim: 512
# fusion:
#   type: none
#   input_dim: 512
# decoder:
#   bpe_vocab_size: 3000
#   embed_dim: 256
#   hidden_dim: 512
#   num_layers: 1
#   dropout: 0.15
#   beam_width: 5
#   max_length: 16
# $TRAIN_BLOCK
#   checkpoint_dir: checkpoints/variant2_gcn
# EOF

# # ── Variant 3: GAT + Attn Pool ──
# cat > configs/variant3_gat_opt.yaml << EOF
# $DATA_BLOCK
# block_encoder:
#   type: transformer
#   token_vocab_size: 3000
#   token_embed_dim: 128
#   num_layers: 2
#   num_heads: 4
#   dropout: 0.15
#   output_dim: 256
#   token_pooling: attention
#   use_block_features: true
# graph_encoder:
#   type: gat
#   hidden_dim: 256
#   num_layers: 2
#   num_heads: 4
#   dropout: 0.15
#   pooling: attention
#   output_dim: 512
# external_encoder:
#   enabled: false
#   vocab_size: 1000
#   embed_dim: 256
#   hidden_dim: 256
#   num_layers: 1
#   dropout: 0.1
#   output_dim: 512
# fusion:
#   type: none
#   input_dim: 512
# decoder:
#   bpe_vocab_size: 3000
#   embed_dim: 256
#   hidden_dim: 512
#   num_layers: 1
#   dropout: 0.15
#   beam_width: 5
#   max_length: 16
# $TRAIN_BLOCK
#   checkpoint_dir: checkpoints/variant3_gat
# EOF

# # ── Variant 4: GAT + Option A ──
# cat > configs/variant4_optA_opt.yaml << EOF
# $DATA_BLOCK
# block_encoder:
#   type: transformer
#   token_vocab_size: 3000
#   token_embed_dim: 128
#   num_layers: 2
#   num_heads: 4
#   dropout: 0.15
#   output_dim: 256
#   token_pooling: attention
#   use_block_features: true
# graph_encoder:
#   type: gat
#   hidden_dim: 256
#   num_layers: 2
#   num_heads: 4
#   dropout: 0.15
#   pooling: attention
#   output_dim: 512
# external_encoder:
#   enabled: true
#   vocab_size: 1000
#   embed_dim: 256
#   hidden_dim: 256
#   num_layers: 1
#   dropout: 0.1
#   output_dim: 512
# fusion:
#   type: node_features
#   input_dim: 512
# decoder:
#   bpe_vocab_size: 3000
#   embed_dim: 256
#   hidden_dim: 512
#   num_layers: 1
#   dropout: 0.15
#   beam_width: 5
#   max_length: 16
# $TRAIN_BLOCK
#   checkpoint_dir: checkpoints/variant4_optA
# EOF

# # ── Variant 5: GAT + Option B (PRIMARY) ──
# cat > configs/optimized.yaml << EOF
# $DATA_BLOCK
# block_encoder:
#   type: transformer
#   token_vocab_size: 3000
#   token_embed_dim: 128
#   num_layers: 2
#   num_heads: 4
#   dropout: 0.15
#   output_dim: 256
#   token_pooling: attention
#   use_block_features: true
# graph_encoder:
#   type: gat
#   hidden_dim: 256
#   num_layers: 2
#   num_heads: 4
#   dropout: 0.15
#   pooling: attention
#   output_dim: 512
# external_encoder:
#   enabled: true
#   vocab_size: 1000
#   embed_dim: 256
#   hidden_dim: 256
#   num_layers: 1
#   dropout: 0.1
#   output_dim: 512
# fusion:
#   type: gated
#   input_dim: 512
# decoder:
#   bpe_vocab_size: 3000
#   embed_dim: 256
#   hidden_dim: 512
#   num_layers: 1
#   dropout: 0.15
#   beam_width: 5
#   max_length: 16
# $TRAIN_BLOCK
#   checkpoint_dir: checkpoints
# EOF

# ══════════════════════════════════════════
# Train DL variants
# ══════════════════════════════════════════

echo ""
echo "═══ Variant 2/5: GCN + Mean Pool ═══"
python3 -m src.training.train --config configs/variant2_gcn_opt.yaml || echo "⚠ Variant 2 failed"

echo ""
echo "═══ Variant 3/5: GAT + Attention Pool ═══"
python3 -m src.training.train --config configs/variant3_gat_opt.yaml || echo "⚠ Variant 3 failed"

echo ""
echo "═══ Variant 4/5: GAT + Option A (node features) ═══"
python3 -m src.training.train --config configs/variant4_optA_opt.yaml || echo "⚠ Variant 4 failed"

echo ""
echo "═══ Variant 5/5: GAT + Option B (gated fusion) ★ ═══"
python3 -m src.training.train --config configs/optimized.yaml || echo "⚠ Variant 5 failed"

# # ══════════════════════════════════════════
# # Evaluate + Demo
# # ══════════════════════════════════════════
# echo ""
# echo "═══ Running Evaluation ═══"
# bash scripts/05_evaluate.sh

# echo ""
# echo "═══ Running Demo on Unseen Package ═══"
# bash scripts/06_demo.sh

echo ""
echo "═══════════════════════════════════════════════"
echo " ✓ ALL COMPLETE"
echo ""
echo " Checkpoints:"
ls -la checkpoints/*/best_model.pt checkpoints/best_model.pt 2>/dev/null || echo "  (check dirs)"
echo ""
echo " Results:"
echo "   results/ablation_table.json"
echo "   results/predictions_test.json"
echo "   demo/results/comparison_diff.json"
echo "═══════════════════════════════════════════════"
