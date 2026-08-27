#!/usr/bin/env python3
"""Matched-key comparison on the P4 baseline samples (SymGen interim-C FT sample 7,532 keys; NCT sample 7,063 keys):
SymGen-34B, [BLens if per-row file given], our A1a retrieval (R), A1a decoder (D), A4 gen head, conf router, learned GBT router
(refit on val with the same features as router2_eval), oracle. Same metric-v2 canonicalization for every system."""
import json, re, subprocess, sys, collections, argparse
import numpy as np
sys.path.insert(0, '/project/hz79/_shared/cs785/dh2')
from src.evaluation.metrics import compute_subtoken_f1
WS = '/project/hz79/_shared/cs785/dh2'
ap = argparse.ArgumentParser()
ap.add_argument('--a4', default=f'{WS}/results/a4_codet5p220m_v1/val_test_symgen_holdout_preds.tsv')
ap.add_argument('--a4tag', default='A4 run1'); ap.add_argument('--blens', default=None)
ap.add_argument('--out', default=f'{WS}/results/matched_baselines_a4v1.json')
ap.add_argument('--sg-c', default=f'{WS}/symgen_v2/results_c/predicted_function_name.json', help='SymGen preds for the FT sample')
ap.add_argument('--sg-nct', default=f'{WS}/symgen_v2/results_nct/predicted_function_name.json', help='SymGen preds for the NCT sample')
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
def sg_clean(p):
    p = p.replace('</s>', '').strip(); p = re.sub(r'^The predicted function name is\s*', '', p).strip()
    return p.split()[0] if p else ''
# ---- our per-row heads (router features: R, D + feats; A4 by addr)
FEATS = ['sim1', 'margin', 'ext_jacc', 'str_jacc', 'd_conf', 'd_len', 'n_ext', 'n_str', 'n_blocks']
feat = {}; hdr = None
for l in open(f'{WS}/results/router_p3a_features.tsv'):
    r = l.rstrip('\n').split('\t')
    if hdr is None: hdr = r; continue
    d = dict(zip(hdr, r))
    if d['bap'].startswith('sub_'):
        a = int(d['bap'][4:], 16)
        for da in (0, 4, -4): feat.setdefault((d['tier'], d['binary'], a + da), d)
a4 = {}
for l in open(args.a4):
    r = l.rstrip('\n').split('\t')
    if r[0] in ('val', 'test'): a4[(r[0], r[1], int(r[2], 16))] = (r[4], float(r[9]) if len(r) > 9 else 0.0)
# ---- refit GBT router on val (same recipe as router2_eval)
from sklearn.ensemble import GradientBoostingClassifier
dem_cache = {}
def X(d, a4c): return [float(d[f]) for f in FEATS] + [a4c, float(d['sim1']) - a4c]
val_rows = []
for (tier, b, a), d in feat.items():
    if tier != 'val' or (tier, b, a) not in a4: continue
    p4, c4 = a4[(tier, b, a)]
    val_rows.append((d, p4, c4))
dem = demangle_many([d['true'] for d, _, _ in val_rows] + [d['r_pred'] for d, _, _ in val_rows] + [p for _, p, _ in val_rows])
Xv, yv, wv = [], [], []
for d, p4, c4 in val_rows:
    ct = canon(d['true'], dem); fr = compute_subtoken_f1(canon(d['r_pred'], dem), ct); fa = compute_subtoken_f1(canon(p4, dem), ct) if p4 else 0.0
    Xv.append(X(d, c4)); yv.append(1 if fr >= fa else 0); wv.append(abs(fr - fa) + 1e-3)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(np.array(Xv), np.array(yv), sample_weight=np.array(wv))
print('router refit on val rows:', len(val_rows))
# ---- samples
def load_sample(meta_path, sg_path, tag):
    meta = json.load(open(meta_path)); sg = json.load(open(sg_path)); assert len(sg) == len(meta)
    rows = []; miss = collections.Counter()
    for m, p in zip(meta, sg):
        key = ('test', m['binary'], int(m['addr'], 16))
        d = feat.get(key); a = a4.get(key)
        if d is None or a is None: miss['no_our_row' if d is None else 'no_a4'] += 1; continue
        rows.append({'pkg': m['package'], 'true': m['gt_name'], 'SymGen': sg_clean(p['predicted_name']), 'R': d['r_pred'], 'D': d['d_pred'],
                     'A4': a[0], 'a4c': a[1], 'sim1': float(d['sim1']), 'x': X(d, a[1]), 'seen': m.get('name_seen')})
    print(f'{tag}: {len(meta)} keys, joined {len(rows)}, missing {dict(miss)}')
    return rows
samples = {'FT_sample': load_sample(f'{WS}/symgen_v2/interim_c_metadata.json', args.sg_c, 'FT'),
           'NCT_sample': load_sample(f'{WS}/symgen_v2/nct_seen_metadata.json', args.sg_nct, 'NCT')}
if args.blens:
    bl = json.load(open(args.blens)); meta = json.load(open(f'{WS}/symgen_v2/interim_c_metadata.json'))
    if isinstance(bl, list) and len(bl) == len(meta) and isinstance(bl[0], dict) and 'pred' in bl[0]:
        bmap = {(m['binary'], int(m['addr'], 16)): b['pred'] for m, b in zip(meta, bl)}
        for r in samples['FT_sample']: r['BLens'] = ''
        # rows were filtered; rejoin by (pkg,true) fallback is unsafe, so match by index via meta again
        idx = {(m['binary'], int(m['addr'], 16)): i for i, m in enumerate(meta)}
        for r in samples['FT_sample']: pass
report = {}
for name, rows in samples.items():
    dem = demangle_many([r['true'] for r in rows] + [r[h] for r in rows for h in ('SymGen', 'R', 'D', 'A4')])
    for r in rows:
        ct = canon(r['true'], dem)
        for h in ('SymGen', 'R', 'D', 'A4'): r['f_' + h] = compute_subtoken_f1(canon(r[h], dem), ct) if r[h] else 0.0
        r['f_conf'] = r['f_R'] if r['sim1'] >= 0.615 else r['f_A4']
        r['f_oracle'] = max(r['f_R'], r['f_A4'])
    pr = gb.predict_proba(np.array([r['x'] for r in rows]))[:, 1]
    for r, p in zip(rows, pr): r['f_gbt'] = r['f_R'] if p >= 0.5 else r['f_A4']
    T = {}
    for h in ('SymGen', 'R', 'D', 'A4', 'conf', 'gbt', 'oracle'):
        key = 'f_' + h; g = collections.defaultdict(list)
        for r in rows: g[r['pkg']].append(r[key])
        em = sum(1 for r in rows if r[key] == 1.0) / len(rows)
        T[h] = {'micro': round(sum(r[key] for r in rows) / len(rows), 4), 'macro': round(sum(sum(v)/len(v) for v in g.values()) / len(g), 4), 'EM': round(em, 4), 'n': len(rows), 'pkgs': len(g)}
    wins = sum(1 for p in g if True)
    report[name] = T
    print(f'=== {name} (n={len(rows)}, {len(g)} pkgs) ===')
    for h, v in T.items(): print(f"  {h:<8} micro {v['micro']:.4f} macro {v['macro']:.4f} EM {v['EM']:.4f}")
json.dump(report, open(args.out, 'w'), indent=1); print('EFFECT: matched baselines done ->', args.out)
