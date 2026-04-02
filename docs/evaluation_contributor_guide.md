# Evaluation & Metrics Contributor Guide

**For:** Zhihao Lin (NLP evaluation metrics implementation and semantic similarity analysis)
**Project:** CS785 Binary Function Name Recovery
**Last updated:** 2026-04-02

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Evaluation Overview](#2-evaluation-overview)
3. [Metrics in Detail](#3-metrics-in-detail)
4. [Sub-Token Splitting (Critical for F1)](#4-sub-token-splitting)
5. [Evaluation Scripts](#5-evaluation-scripts)
6. [Name Tokenization (Votes vs BPE)](#6-name-tokenization)
7. [Running Evaluations](#7-running-evaluations)
8. [Result File Formats](#8-result-file-formats)
9. [Common Tasks](#9-common-tasks)
10. [Metric Interpretation Guide](#10-metric-interpretation-guide)
11. [Known Issues & Edge Cases](#11-known-issues--edge-cases)

---

## 1. Environment Setup

```bash
source ~/cs785-project/activate.sh

# Key dependencies for evaluation:
# - PyTorch (model loading + inference)
# - PyTorch Geometric (graph data handling)
# - sentencepiece (BPE decoding, legacy)
# - No BAP needed for evaluation (only for predict.py on new binaries)
```

All evaluation scripts work locally or on Wulver. Demo evaluation on Wulver requires pre-processed graph files (already uploaded).

---

## 2. Evaluation Overview

### Three evaluation modes

| Mode | Script | What it measures | BAP needed? |
|------|--------|-----------------|-------------|
| **Test eval** | `scripts/eval_test.py` | Performance on held-out test split (8,973 functions) | No |
| **Demo eval** | `scripts/eval_demo_wulver.py` | Cross-project generalization (12,688 functions, 11 unseen packages) | No |
| **Stratified eval** | `scripts/eval_stratified.py` | Breakdown by difficulty (easy/hard-seen/hard-unseen) | No |
| **Predict** | `scripts/predict.py` | End-to-end on any stripped binary | **Yes** |

### Metrics computed

| Metric | Function | What it measures |
|--------|----------|-----------------|
| Sub-token F1 | `compute_subtoken_f1()` | Token-level precision/recall (primary metric) |
| Exact Match (EM) | `compute_exact_match()` | Exact string equality after normalization |
| N-gram Similarity | `compute_char_ngram_similarity()` | Character trigram overlap (Dice coefficient) |
| Edit Similarity | `compute_edit_distance_similarity()` | 1 - normalized Levenshtein distance |

---

## 3. Metrics in Detail

All metrics are in `src/evaluation/metrics.py`.

### Sub-token F1 (Primary Metric)

The most important metric. Measures how many sub-tokens in the predicted name match the ground truth.

```python
def compute_subtoken_f1(predicted: str, ground_truth: str) -> float:
    pred_tokens = split_name(predicted)    # "hash_table_lookup" → ["hash", "table", "lookup"]
    true_tokens = split_name(ground_truth)

    # Multiset intersection using Counter
    common = sum((Counter(pred_tokens) & Counter(true_tokens)).values())

    precision = common / len(pred_tokens)   # How many predicted tokens are correct?
    recall = common / len(true_tokens)      # How many true tokens were predicted?
    f1 = 2 * precision * recall / (precision + recall)

    return f1
```

**Examples:**
```
Predicted: "hash_table_find"
Truth:     "hash_table_lookup"
Tokens:    ["hash", "table", "find"] vs ["hash", "table", "lookup"]
Common: 2 (hash, table)
Precision: 2/3 = 0.667
Recall: 2/3 = 0.667
F1 = 0.667

Predicted: "xmalloc"
Truth:     "xmalloc"
F1 = 1.0 (exact match)

Predicted: "sqlite3_result"
Truth:     "hash_get_next"
Common: 0
F1 = 0.0
```

**Key detail:** Uses `Counter` for multiset intersection — duplicates matter. `["a", "a"]` vs `["a"]` gives precision=0.5, recall=1.0, F1=0.667.

### Exact Match (EM)

Simplest metric — binary 0 or 1 after normalization:

```python
def compute_exact_match(predicted: str, ground_truth: str) -> bool:
    return normalize_name(predicted) == normalize_name(ground_truth)
```

### Character N-gram Similarity

Dice coefficient on character trigrams — gives partial credit for near-misses:

```python
def compute_char_ngram_similarity(predicted: str, ground_truth: str, n: int = 3) -> float:
    pred_ngrams = set(ngrams(normalize_name(predicted), n))   # {"has", "ash", "sh_", ...}
    true_ngrams = set(ngrams(normalize_name(ground_truth), n))

    overlap = len(pred_ngrams & true_ngrams)
    return 2.0 * overlap / (len(pred_ngrams) + len(true_ngrams))
```

**Example:**
```
Predicted: "hash_lookup"
Truth:     "hash_find"
Shared trigrams: "has", "ash", "sh_"  (from "hash_" prefix)
N-gram sim ≈ 0.35
```

### Edit Distance Similarity

1 minus normalized Levenshtein distance — character-level partial credit:

```python
def compute_edit_distance_similarity(predicted: str, ground_truth: str) -> float:
    edit_dist = levenshtein(predicted, ground_truth)  # Dynamic programming
    max_len = max(len(predicted), len(ground_truth))
    return 1.0 - (edit_dist / max_len)
```

**Example:**
```
Predicted: "hash_table_find"
Truth:     "hash_table_lookup"
Edit distance: 6 (replace "find" with "lookup", adjust length)
Max length: 17
Edit sim: 1 - 6/17 = 0.647
```

### Unified computation

```python
def compute_all_metrics(predicted: str, ground_truth: str) -> dict:
    return {
        'precision': ...,
        'recall': ...,
        'f1': compute_subtoken_f1(predicted, ground_truth),
        'exact_match': compute_exact_match(predicted, ground_truth),
        'char_ngram_sim': compute_char_ngram_similarity(predicted, ground_truth),
        'edit_sim': compute_edit_distance_similarity(predicted, ground_truth),
    }
```

---

## 4. Sub-Token Splitting

The `split_name()` function is critical — it determines how function names are tokenized for F1 computation.

```python
def split_name(name: str) -> List[str]:
    # Step 1: Normalize
    name = normalize_name(name)  # lowercase, strip underscores/dots

    # Step 2: Insert _ at camelCase boundaries
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)          # handleClient → handle_Client
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)   # XMLParser → XML_Parser

    # Step 3: Split on separators
    tokens = re.split(r'[_.\-]+', name.lower())

    # Step 4: Filter empty tokens
    return [t for t in tokens if t]
```

**Examples:**
```python
split_name("hash_table_lookup")     → ["hash", "table", "lookup"]
split_name("handleClient")          → ["handle", "client"]
split_name("XMLParser")             → ["xml", "parser"]
split_name("x2nRealloc")            → ["x2n", "realloc"]
split_name("close_stdout")          → ["close", "stdout"]
split_name("__libc_start_main")     → ["libc", "start", "main"]
split_name("quotearg_n_custom_mem") → ["quotearg", "n", "custom", "mem"]
```

**This splitting aligns with the Votes tokenizer** — both split on `_` and camelCase boundaries. BPE may fragment differently, causing F1 mismatch.

---

## 5. Evaluation Scripts

### `scripts/eval_test.py` — Test Set Evaluation

Evaluates the model on train/val/test splits using greedy decode (argmax).

```bash
python3 scripts/eval_test.py checkpoints/best_model.pt [--splits val test] [--amp]
```

**Workflow:**
1. Load checkpoint (extracts `token_vocab`, `ext_vocab`, `config`)
2. Build dataset with `FunctionDataset`
3. Greedy decode: `pred_ids = logits.argmax(dim=-1)`
4. Decode IDs → name strings via Votes/BPE tokenizer
5. Compute all metrics per function, aggregate by split

**Output:** `results/test_eval_metrics.json`

### `scripts/eval_demo_wulver.py` — Cross-Project Evaluation

Evaluates on 11 unseen demo packages using beam search.

```bash
python3 scripts/eval_demo_wulver.py checkpoints/best_model.pt
```

**Key differences from test eval:**
- Uses **beam search** (k=5) instead of greedy decode
- Loads pre-processed graphs from disk (no BAP needed)
- Reports per-package and per-optimization-level breakdowns
- Handles address matching between predictions and ground truth

**Output:** `demo/results/expanded/expanded_eval.json`

### `scripts/eval_stratified.py` — Stratified Analysis

Deep breakdown by function difficulty:

```bash
python3 scripts/eval_stratified.py
```

**Strata:**
- **Easy**: 1+ external calls (strong naming signal)
- **Hard-Seen**: 0 ext calls, name exists in training set
- **Hard-Unseen**: 0 ext calls, name never seen in training

**Also computes:**
- Solvability analysis (theoretical EM ceiling)
- Duplicate function analysis
- Deduped metrics
- Failure pattern breakdown by name length
- Mode collapse detection

**Output:** `results/stratified_eval.json`

### `scripts/predict.py` — End-to-End Prediction

Predicts names for all functions in a stripped binary.

```bash
python3 scripts/predict.py --binary /path/to/binary --checkpoint checkpoints/best_model.pt
```

**Requires BAP** (local only). Full pipeline:
1. BAP lifts binary → BIR
2. Parse BIR → CFG graphs
3. Extract external calls
4. Resolve ENDBR64 indirect jump wrappers
5. Build callee/caller context
6. Beam search prediction
7. Output JSON with predictions + scores

---

## 6. Name Tokenization (Votes vs BPE)

### Votes Tokenizer (current, recommended)

`src/preprocessing/build_votes.py` — custom tokenizer that preserves semantic sub-tokens.

```python
# How it works:
# 1. Split on _ boundaries
# 2. Split camelCase
# 3. Keep tokens appearing ≥ min_count times
# 4. Character-level fallback for rare tokens

# Example:
"hash_table_lookup" → [SOS, "hash", "_", "table", "_", "lookup", EOS]
"handleClient"      → [SOS, "handle", "_", "client", EOS]
```

**Vocab:** 2,642 tokens (in `data/votes_vocab.json`)
**Special tokens:** `<PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3, _=4`

### BPE Tokenizer (legacy)

SentencePiece unigram model. May fragment tokens:
```
"hash_table_lookup" → ["▁hash", "_table", "_look", "up"]  # Fragments "lookup"
```

**Why Votes is better:** Sub-token boundaries align with `split_name()` in metrics.py. BPE fragments cause F1 mismatch between what the model generates and how the metric counts tokens.

### Impact on Evaluation

Both tokenizers ultimately produce a string that gets passed to `compute_all_metrics()`. The choice affects:
1. **What the model learns to generate** (Votes = whole sub-tokens, BPE = fragments)
2. **Decoder vocabulary size** (Votes: 2,642, BPE: ~1,095)
3. **OOV rate** (Votes: 5%, BPE: ~50% — huge difference)

---

## 7. Running Evaluations

### Test set (local or Wulver)

```bash
# Quick test eval
python3 scripts/eval_test.py checkpoints/best_model.pt

# With AMP (faster on GPU)
python3 scripts/eval_test.py checkpoints/best_model.pt --amp

# Specific splits only
python3 scripts/eval_test.py checkpoints/best_model.pt --splits test
```

### Cross-project demo (Wulver recommended)

```bash
# On Wulver
ssh wulver "cd /course/.../cs785 && python3 scripts/eval_demo_wulver.py checkpoints/best_model.pt"
```

### Stratified analysis

```bash
python3 scripts/eval_stratified.py
```

### Compare two models

```bash
# Run eval_test on both, then compare JSONs
python3 scripts/eval_test.py checkpoints/model_a.pt
cp results/test_eval_metrics.json results/model_a_metrics.json

python3 scripts/eval_test.py checkpoints/model_b.pt
cp results/test_eval_metrics.json results/model_b_metrics.json

# Compare
python3 -c "
import json
a = json.load(open('results/model_a_metrics.json'))
b = json.load(open('results/model_b_metrics.json'))
for split in ['val', 'test']:
    print(f'{split}: A={a[split][\"f1\"]:.4f} vs B={b[split][\"f1\"]:.4f} (delta={b[split][\"f1\"]-a[split][\"f1\"]:+.4f})')
"
```

---

## 8. Result File Formats

### `results/test_eval_metrics.json`

```json
{
  "val": {
    "split": "val",
    "count": 4695,
    "f1": 0.7341,
    "em": 0.6969,
    "ngsim": 0.7234,
    "edsim": 0.7668
  },
  "test": {
    "split": "test",
    "count": 8973,
    "f1": 0.7811,
    "em": 0.7233,
    "ngsim": 0.7850,
    "edsim": 0.8183
  }
}
```

### `demo/results/expanded/expanded_eval.json`

```json
{
  "overall": {
    "correct": 6156,
    "total": 12688,
    "em_pct": 48.5,
    "f1": 0.593,
    "ngsim": 0.625,
    "edsim": 0.688
  },
  "per_opt": {
    "O0": {"correct": ..., "total": ..., "em_pct": ..., "f1": ...},
    "O2": {"correct": ..., "total": ..., "em_pct": ..., "f1": ...}
  },
  "per_pkg": {
    "diffutils": {"correct": 300, "total": 438, "em_pct": 68.5, "f1": 0.588},
    "strace": {"correct": ..., ...},
    ...
  }
}
```

### `results/stratified_eval.json`

```json
{
  "overall": {"n": 8973, "f1": 0.795, "em": 0.743, ...},
  "stratified": {
    "easy": {"n": ..., "f1": ..., "em": ...},
    "hard_seen": {"n": ..., "f1": ..., "em": ...},
    "hard_unseen": {"n": ..., "f1": ..., "em": ...}
  },
  "solvability": {
    "theoretical_ceiling": 0.85,
    "actual_em_rate": 0.743,
    "gap_to_ceiling": 0.107
  }
}
```

---

## 9. Common Tasks

### Add a new metric

1. Implement in `src/evaluation/metrics.py`:
```python
def compute_new_metric(predicted: str, ground_truth: str) -> float:
    # Your implementation
    return score
```

2. Add to `compute_all_metrics()`:
```python
def compute_all_metrics(predicted, ground_truth):
    return {
        ...,
        'new_metric': compute_new_metric(predicted, ground_truth),
    }
```

3. Update evaluation scripts to report the new metric.

### Analyze prediction errors

```python
import json

# Load predictions (from eval scripts that save per-function results)
with open('results/predictions_test.json') as f:
    preds = json.load(f)

# Find worst predictions
worst = sorted(preds, key=lambda x: x['f1'])[:20]
for p in worst:
    print(f"True: {p['true_name']:30s} Pred: {p['pred_name']:30s} F1={p['f1']:.3f}")

# Find near-misses (high F1 but not exact match)
near_misses = [p for p in preds if 0.5 < p['f1'] < 1.0]
print(f"\n{len(near_misses)} near-misses (F1 between 0.5 and 1.0)")
```

### Compute metrics on custom predictions

```python
from src.evaluation.metrics import compute_all_metrics

predictions = [
    ("hash_table_find", "hash_table_lookup"),
    ("xmalloc", "xmalloc"),
    ("process_input", "handle_input"),
]

for pred, truth in predictions:
    m = compute_all_metrics(pred, truth)
    print(f"Pred={pred:25s} Truth={truth:25s} F1={m['f1']:.3f} EM={m['exact_match']} EdSim={m['edit_sim']:.3f}")
```

### Run the built-in metric tests

```bash
python3 -m src.evaluation.metrics
# Runs 8 test cases showing metric behavior on various string pairs
```

---

## 10. Metric Interpretation Guide

| Metric | Range | Best For | Interpretation |
|--------|-------|----------|----------------|
| **F1** | [0, 1] | Publication comparison | Token-level match quality. 0.8+ is strong. |
| **EM** | {0, 1} | Practical usefulness | All-or-nothing. Shows how often model is exactly right. |
| **N-gram Sim** | [0, 1] | Partial credit analysis | Character-level "closeness". Captures prefix/suffix matches. |
| **Edit Sim** | [0, 1] | Name similarity | How many character edits to fix. 0.9+ means nearly right. |

### When metrics disagree

| Scenario | EM | F1 | N-gram | Edit | Interpretation |
|----------|----|----|--------|------|---------------|
| Exact match | 1.0 | 1.0 | 1.0 | 1.0 | Perfect |
| One sub-token wrong | 0 | 0.67 | 0.7 | 0.8 | Close, useful for RE |
| Completely wrong | 0 | 0 | 0.1 | 0.2 | No value |
| Right tokens, wrong order | 0 | 1.0 | 0.6 | 0.5 | F1 overestimates (set-based) |

**F1 caveat:** F1 is a set-based metric — it doesn't penalize wrong token order. `"table_hash"` vs `"hash_table"` gives F1=1.0 but EM=0.

### Choosing metrics for the paper

- **Primary:** Sub-token F1 (standard in literature: BLens, SYMGEN, NERO all use it)
- **Secondary:** EM (shows practical accuracy)
- **Supporting:** Edit Similarity (shows how close wrong predictions are)
- **Analysis:** N-gram Similarity (for comparing across difficulty strata)

---

## 11. Known Issues & Edge Cases

### F1 edge cases

```python
# Both empty → F1 = 1.0 (both "correct" at being empty)
compute_subtoken_f1("", "") == 1.0

# One empty → F1 = 0.0
compute_subtoken_f1("hash", "") == 0.0

# Single common sub-token
compute_subtoken_f1("hash_find", "hash_lookup")
# → ["hash", "find"] vs ["hash", "lookup"], common=1, F1=0.5
```

### Normalization matters

```python
normalize_name("  _hash_TABLE_ ") → "hash_table"
normalize_name("close _ stdout")  → "close_stdout"  # BPE artifact handling
```

### Greedy vs beam search

- **Training validation:** Uses greedy decode (argmax) — faster but lower quality
- **Evaluation/demo:** Uses beam search (k=5) — ~3-5% higher F1
- **Important:** Val F1 in training logs is greedy. Published numbers should use beam search.

### The Val F1 paradox

Higher Val F1 doesn't always mean better cross-project performance. Val set is small GNU packages (coreutils, bash, gawk). A model can memorize these patterns while failing on diverse unseen packages. **Always check demo EM alongside Val F1.**

---

## Files You'll Work With Most

| File | Purpose |
|------|---------|
| `src/evaluation/metrics.py` | All metric implementations |
| `scripts/eval_test.py` | Test set evaluation |
| `scripts/eval_demo_wulver.py` | Cross-project demo evaluation |
| `scripts/eval_stratified.py` | Stratified difficulty analysis |
| `scripts/predict.py` | End-to-end prediction (needs BAP) |
| `src/preprocessing/build_votes.py` | Votes tokenizer (affects decoding) |
| `results/test_eval_metrics.json` | Test eval output |
| `demo/results/expanded/expanded_eval.json` | Demo eval output |
| `results/stratified_eval.json` | Stratified analysis output |
