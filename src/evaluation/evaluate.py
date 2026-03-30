"""
Full evaluation script.

Usage:
    python -m src.evaluation.evaluate --checkpoint checkpoints/best_model.pt
"""
import argparse
import json
import os
import yaml
import torch
from tqdm import tqdm

from src.evaluation.metrics import (
    compute_subtoken_f1, compute_subtoken_precision_recall,
    compute_exact_match, split_name,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help='Predictions JSON')
    parser.add_argument('--labels', required=True, help='Labels directory')
    parser.add_argument('--output', default='results/metrics.json')
    args = parser.parse_args()

    # Load predictions
    with open(args.predictions) as f:
        predictions = json.load(f)

    # Compute metrics
    all_f1, all_p, all_r = [], [], []
    top1_matches = 0
    total = 0

    for pred in predictions:
        gt = pred['ground_truth']
        predicted = pred['predicted']

        f1 = compute_subtoken_f1(predicted, gt)
        p, r = compute_subtoken_precision_recall(predicted, gt)
        em = compute_exact_match(predicted, gt)

        all_f1.append(f1)
        all_p.append(p)
        all_r.append(r)
        if em:
            top1_matches += 1
        total += 1

    results = {
        'num_functions': total,
        'subtoken_precision': sum(all_p) / max(total, 1),
        'subtoken_recall': sum(all_r) / max(total, 1),
        'subtoken_f1': sum(all_f1) / max(total, 1),
        'top1_exact_match': top1_matches / max(total, 1),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n═══ Evaluation Results ═══")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
