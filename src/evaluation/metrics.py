"""
Evaluation metrics for function name prediction.

Includes:
  - Sub-token Precision / Recall / F1
  - Exact match
  - Character n-gram similarity (Dice coefficient)
  - Edit distance similarity (normalized Levenshtein)
  - Unified compute_all_metrics() function
"""
import re
from typing import List, Tuple, Set
from collections import Counter


def normalize_name(name: str) -> str:
    """
    Normalize a function name for fair comparison.
    Handles BPE decoding artifacts like extra spaces.

    "close _ stdout"  → "close_stdout"
    "quotearg _ n _"  → "quotearg_n"
    "digest _ file . constprop ." → "digest_file.constprop"
    """
    name = name.replace(' ', '')
    name = name.strip('_.')
    name = name.lower()
    return name


def split_name(name: str) -> List[str]:
    """
    Split function name into sub-tokens.

    "close_stdout"     → ["close", "stdout"]
    "handleClient"     → ["handle", "client"]
    "hash_get_next"    → ["hash", "get", "next"]
    """
    name = normalize_name(name)
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    tokens = re.split(r'[_.\-]+', name)
    tokens = [t.strip().lower() for t in tokens if t.strip()]
    return tokens


def compute_subtoken_f1(predicted: str, ground_truth: str) -> float:
    """Compute sub-token F1 between predicted and ground truth names."""
    pred_tokens = split_name(predicted)
    true_tokens = split_name(ground_truth)

    if not pred_tokens and not true_tokens:
        return 1.0
    if not pred_tokens or not true_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    true_counter = Counter(true_tokens)

    tp = sum((pred_counter & true_counter).values())
    fp = sum(pred_counter.values()) - tp
    fn = sum(true_counter.values()) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_subtoken_precision_recall(predicted: str, ground_truth: str) -> Tuple[float, float]:
    """Compute sub-token precision and recall."""
    pred_tokens = split_name(predicted)
    true_tokens = split_name(ground_truth)

    if not pred_tokens and not true_tokens:
        return 1.0, 1.0
    if not pred_tokens:
        return 0.0, 0.0
    if not true_tokens:
        return 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    true_counter = Counter(true_tokens)
    tp = sum((pred_counter & true_counter).values())

    precision = tp / sum(pred_counter.values()) if sum(pred_counter.values()) > 0 else 0
    recall = tp / sum(true_counter.values()) if sum(true_counter.values()) > 0 else 0
    return precision, recall


def compute_exact_match(predicted: str, ground_truth: str) -> bool:
    """Check exact string match after normalization."""
    return normalize_name(predicted) == normalize_name(ground_truth)


def compute_top_k_match(candidates: List[str], ground_truth: str, k: int) -> bool:
    """Check if ground truth appears in top-k candidates."""
    gt = normalize_name(ground_truth)
    return any(normalize_name(c) == gt for c in candidates[:k])


def compute_char_ngram_similarity(predicted: str, ground_truth: str, n: int = 3) -> float:
    """
    Compute character n-gram similarity (Dice coefficient).

    Gives partial credit for predictions that are close but not exact.
    "hash_lookup" vs "hash_find" share trigrams like "has", "ash", "sh_"

    Returns: float between 0.0 and 1.0
    """
    pred = normalize_name(predicted)
    true = normalize_name(ground_truth)

    if not pred or not true:
        return 0.0
    if pred == true:
        return 1.0

    def get_ngrams(text: str, n: int) -> Set[str]:
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    pred_ngrams = get_ngrams(pred, n)
    true_ngrams = get_ngrams(true, n)

    if not pred_ngrams or not true_ngrams:
        return 0.0

    overlap = len(pred_ngrams & true_ngrams)
    return 2.0 * overlap / (len(pred_ngrams) + len(true_ngrams))


def compute_edit_distance_similarity(predicted: str, ground_truth: str) -> float:
    """
    Compute normalized edit distance similarity.

    Returns 1.0 for identical strings, 0.0 for completely different.
    Based on Levenshtein distance.
    """
    pred = normalize_name(predicted)
    true = normalize_name(ground_truth)

    if pred == true:
        return 1.0
    if not pred or not true:
        return 0.0

    m, n = len(pred), len(true)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred[i-1] == true[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    max_len = max(m, n)
    return 1.0 - dp[m][n] / max_len


def compute_all_metrics(predicted: str, ground_truth: str) -> dict:
    """Compute all metrics at once."""
    p, r = compute_subtoken_precision_recall(predicted, ground_truth)
    f1 = compute_subtoken_f1(predicted, ground_truth)
    em = compute_exact_match(predicted, ground_truth)
    ngram_sim = compute_char_ngram_similarity(predicted, ground_truth)
    edit_sim = compute_edit_distance_similarity(predicted, ground_truth)

    return {
        'precision': p,
        'recall': r,
        'f1': f1,
        'exact_match': em,
        'char_ngram_sim': ngram_sim,
        'edit_sim': edit_sim,
    }


if __name__ == '__main__':
    print("Metric self-test:")
    print()

    tests = [
        ("usage", "usage", "perfect match"),
        ("print _ filename", "print_filename", "BPE spaces — should be perfect"),
        ("close _ stdout", "close_stdout", "BPE spaces — should be perfect"),
        ("quotearg _ n _ custom _", "quotearg_n_custom", "BPE spaces + trailing _"),
        ("quotearg _ n _ custom _", "quotearg_n_style_colon.cold", "partial match"),
        ("x2nrealloc", "filename_unescape", "no match"),
        ("hash_lookup", "hash_find", "partial (hash matches)"),
        ("read_config", "parse_config", "partial (config matches)"),
    ]

    print(f"  {'Predicted':<35s} {'True':<35s} {'F1':>5s} {'EM':>4s} {'NgSim':>6s} {'EdSim':>6s}")
    print(f"  {'─'*35} {'─'*35} {'─'*5} {'─'*4} {'─'*6} {'─'*6}")

    for pred, true, desc in tests:
        metrics = compute_all_metrics(pred, true)
        em_str = "Y" if metrics['exact_match'] else "N"
        print(f"  {pred:<35s} {true:<35s} {metrics['f1']:>5.2f} {em_str:>4s} "
              f"{metrics['char_ngram_sim']:>6.3f} {metrics['edit_sim']:>6.3f}  <- {desc}")
