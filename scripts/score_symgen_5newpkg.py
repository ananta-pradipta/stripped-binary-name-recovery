#!/usr/bin/env python3
"""Score SymGen predictions on the 5-newpkg xproj set.
Parses predicted_function_name.json, matches to metadata by order,
computes per-package F1/EM using our sub-token F1 metric.
"""
import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, '/mmfs1<shared-project-root>')
from src.evaluation.metrics import compute_subtoken_f1

META = '<project-root>/baselines/SymGen/dataset/xproj_metadata_5newpkg.json'
PRED = '<project-root>/baselines/SymGen/results_full_ours_5newpkg/predicted_function_name.json'

meta = json.load(open(META))
preds = json.load(open(PRED))
print(f'Metadata entries: {len(meta)}')
print(f'Predictions:      {len(preds)}')
assert len(meta) == len(preds), f'Mismatch: meta={len(meta)} vs pred={len(preds)}'


def extract(text):
    m = re.match(r'The predicted function name is (.*?)(?:</s>|\.\s*$|\s*$)', text.strip())
    return m.group(1).strip().rstrip('.') if m else text.strip().rstrip('.')


pkg_stats = defaultdict(lambda: {'n': 0, 'em': 0, 'f1_sum': 0.0})
for m, p in zip(meta, preds):
    gt = m['gt_name']
    pred = extract(p['predicted_name'])
    pkg = m['package']
    pkg_stats[pkg]['n'] += 1
    if pred == gt:
        pkg_stats[pkg]['em'] += 1
    pkg_stats[pkg]['f1_sum'] += compute_subtoken_f1(pred, gt)

print('\n=== SymGen per-package F1 on 5 new xproj pkgs ===')
print(f"{'Pkg':<12} {'N':>6} {'F1':>8} {'EM':>8}")
total_n = total_em = 0
total_f1 = 0.0
for pkg in sorted(pkg_stats.keys()):
    s = pkg_stats[pkg]
    n = s['n']
    f1 = s['f1_sum'] / n
    em = s['em'] / n
    print(f"{pkg:<12} {n:>6} {f1:>8.4f} {em:>8.4f}")
    total_n += n
    total_f1 += s['f1_sum']
    total_em += s['em']

print(f"{'OVERALL':<12} {total_n:>6} {total_f1/total_n:>8.4f} {total_em/total_n:>8.4f}")

# Also combine with prior 4-pkg (from results_full_ours) to report 9-pkg total
META4 = '<project-root>/baselines/SymGen/dataset/xproj_metadata_v2.json'
PRED4 = '<project-root>/baselines/SymGen/results_full_ours/predicted_function_name.json'
import os
if os.path.exists(META4) and os.path.exists(PRED4):
    meta4 = json.load(open(META4))
    preds4 = json.load(open(PRED4))
    if len(meta4) == len(preds4):
        print('\n=== Combined with prior 4-pkg predictions (9-pkg total) ===')
        for m, p in zip(meta4, preds4):
            gt = m['gt_name']
            pred = extract(p['predicted_name'])
            pkg = m['package']
            pkg_stats[pkg]['n'] += 1
            if pred == gt:
                pkg_stats[pkg]['em'] += 1
            pkg_stats[pkg]['f1_sum'] += compute_subtoken_f1(pred, gt)
        print(f"{'Pkg':<12} {'N':>6} {'F1':>8} {'EM':>8}")
        total_n = total_em = 0
        total_f1 = 0.0
        for pkg in sorted(pkg_stats.keys()):
            s = pkg_stats[pkg]
            n = s['n']
            f1 = s['f1_sum'] / n
            em = s['em'] / n
            print(f"{pkg:<12} {n:>6} {f1:>8.4f} {em:>8.4f}")
            total_n += n
            total_f1 += s['f1_sum']
            total_em += s['em']
        print(f"{'9-PKG TOTAL':<12} {total_n:>6} {total_f1/total_n:>8.4f} {total_em/total_n:>8.4f}")

        # Subsets
        print('\n=== Aggregated subsets ===')
        contam = {'grep', 'sed'}
        clean7 = [p for p in pkg_stats if p not in contam]
        paper4 = ['tengine', 'angie', 'nginx118', 'recutils']
        low3 = ['dash', 'gettext', 'psmisc']
        for label, pkgs in [('All 9 pkgs', list(pkg_stats.keys())),
                             ('Clean 7 (no grep+sed)', clean7),
                             ('Paper 4 (t/a/n/r)', paper4),
                             ('Low-cov 3 (d/g/p)', low3)]:
            n = sum(pkg_stats[p]['n'] for p in pkgs)
            f1 = sum(pkg_stats[p]['f1_sum'] for p in pkgs) / n
            em = sum(pkg_stats[p]['em'] for p in pkgs) / n
            print(f"{label:<24} N={n:>6} F1={f1:.4f} EM={em:.4f}")
