#!/usr/bin/env python3
"""Test-tier results split by compiler (gcc vs clang builds) for our heads R / D / A4 / GBT union / oracle.
Same row sources and router recipe as score_blens_matched.py (C1 lambda1.0 encoder features, A4 run 1)."""
import argparse, collections, json, re, subprocess, sys
import numpy as np
sys.path.insert(0, '$WORKSPACE/dh2')
from src.evaluation.metrics import compute_subtoken_f1
WS = '$WORKSPACE/dh2'
ap = argparse.ArgumentParser()
ap.add_argument('--feat', default=f'{WS}/results/router_c1soft_l10_interim_features.tsv')
ap.add_argument('--regime', default=f'{WS}/results/c1soft_l10_interim_preds_greedy.tsv')
ap.add_argument('--a4', default=f'{WS}/results/a4_codet5p220m_v1/val_test_symgen_holdout_preds.tsv')
ap.add_argument('--out', default=f'{WS}/results/compiler_breakdown_c1l10_a4v1.json')
args = ap.parse_args()
def demangle_many(names):
    todo = sorted({n for n in names if n and n.startswith('_Z')}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]; r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out
def canon(n, dem):
    n = dem.get(n, n); n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]; return '_'.join(p for p in parts if p)
FEATS = ['sim1', 'margin', 'ext_jacc', 'str_jacc', 'd_conf', 'd_len', 'n_ext', 'n_str', 'n_blocks']
feat = {}; hdr = None
for l in open(args.feat):
    r = l.rstrip('\n').split('\t')
    if hdr is None: hdr = r; continue
    d = dict(zip(hdr, r))
    if d['bap'].startswith('sub_'): feat[(d['tier'], d['binary'], int(d['bap'][4:], 16))] = d
reg = {}; hdr = None
for l in open(args.regime):
    r = l.rstrip('\n').split('\t')
    if hdr is None: hdr = r; continue
    d = dict(zip(hdr, r))
    if d['bap_name'].startswith('sub_'): reg[(d['tier'], d['binary'], int(d['bap_name'][4:], 16))] = (d['regime'], d['name_seen'])
a4 = {}
for l in open(args.a4):
    r = l.rstrip('\n').split('\t')
    if r[0] in ('val', 'test'): a4[(r[0], r[1], int(r[2], 16))] = (r[4], float(r[9]) if len(r) > 9 else 0.0)
from sklearn.ensemble import GradientBoostingClassifier
def X(d, a4c): return [float(d[f]) for f in FEATS] + [a4c, float(d['sim1']) - a4c]
val_rows = [(d, *a4[k]) for k, d in feat.items() if k[0] == 'val' and k in a4]
dem = demangle_many([d['true'] for d, _, _ in val_rows] + [d['r_pred'] for d, _, _ in val_rows] + [p for _, p, _ in val_rows])
Xv, yv, wv = [], [], []
for d, p4, c4 in val_rows:
    ct = canon(d['true'], dem); fr = compute_subtoken_f1(canon(d['r_pred'], dem), ct); fa = compute_subtoken_f1(canon(p4, dem), ct) if p4 else 0.0
    Xv.append(X(d, c4)); yv.append(1 if fr >= fa else 0); wv.append(abs(fr - fa) + 1e-3)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(np.array(Xv), np.array(yv), sample_weight=np.array(wv))
rows = []
for k, d in feat.items():
    if k[0] != 'test' or k not in a4: continue
    rg = reg.get(k, ('?', '?')); b = k[1]
    rows.append({'bin': b, 'pkg': b.split('_')[0], 'compiler': 'clang' if '_clang_' in b else 'gcc', 'opt': b.rsplit('_', 1)[-1],
                 'true': d['true'], 'R': d['r_pred'], 'D': d['d_pred'], 'A4': a4[k][0], 'x': X(d, a4[k][1]), 'regime': rg[0], 'seen': rg[1]})
HEADS = ('R', 'D', 'A4')
dem = demangle_many([r['true'] for r in rows] + [r[h] for r in rows for h in HEADS])
pr = gb.predict_proba(np.array([r['x'] for r in rows]))[:, 1]
for r, p in zip(rows, pr):
    ct = canon(r['true'], dem)
    for h in HEADS: r['f_' + h] = compute_subtoken_f1(canon(r[h], dem), ct) if r[h] else 0.0
    r['f_gbt'] = r['f_R'] if p >= 0.5 else r['f_A4']; r['f_oracle'] = max(r['f_R'], r['f_A4'])
def table(rs, label):
    T = {}
    for h in ('R', 'D', 'A4', 'gbt', 'oracle'):
        key = 'f_' + h; g = collections.defaultdict(list)
        for r in rs: g[r['pkg']].append(r[key])
        T[h] = {'micro': round(sum(r[key] for r in rs) / len(rs), 4), 'macro': round(sum(sum(v)/len(v) for v in g.values()) / len(g), 4),
                'EM': round(sum(1 for r in rs if r[key] == 1.0) / len(rs), 4), 'n': len(rs), 'pkgs': len(g)}
    print(f'=== {label} (n={len(rs)}, {len(g)} pkgs) ===')
    for h, v in T.items(): print(f"  {h:<7} micro {v['micro']:.4f} macro {v['macro']:.4f} EM {v['EM']:.4f}")
    return T
report = {}
for c in ('gcc', 'clang'):
    sub = [r for r in rows if r['compiler'] == c]; report[c] = table(sub, f'compiler {c}')
    for rg in ('FT', 'NCT'):
        s2 = [r for r in sub if r['regime'] == rg]
        if s2: report[f'{c}_{rg}'] = table(s2, f'compiler {c} · regime {rg}')
# packages that have BOTH compilers in test: paired comparison on the same package set
pk = collections.defaultdict(set)
for r in rows: pk[r['pkg']].add(r['compiler'])
both = sorted(p for p, cs in pk.items() if len(cs) == 2); print('packages with both compilers in test:', both)
for c in ('gcc', 'clang'):
    sub = [r for r in rows if r['compiler'] == c and r['pkg'] in both]; report[f'paired_{c}'] = table(sub, f'PAIRED pkgs · {c}')
report['paired_per_pkg'] = {}
for p in both:
    e = {}
    for c in ('gcc', 'clang'):
        sub = [r for r in rows if r['compiler'] == c and r['pkg'] == p]
        e[c] = {'n': len(sub), 'R': round(sum(r['f_R'] for r in sub)/len(sub), 3), 'A4': round(sum(r['f_A4'] for r in sub)/len(sub), 3), 'gbt': round(sum(r['f_gbt'] for r in sub)/len(sub), 3), 'regime': sub[0]['regime']}
    report['paired_per_pkg'][p] = e; print(p, e)
report['train_note'] = 'train tier: 539 of 997 binaries are clang builds (mixed-compiler training)'
report['test_bins'] = {c: len({r['bin'] for r in rows if r['compiler'] == c}) for c in ('gcc', 'clang')}
json.dump(report, open(args.out, 'w'), indent=1); print('EFFECT: compiler breakdown ->', args.out, report['test_bins'])
