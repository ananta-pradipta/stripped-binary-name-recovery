#!/usr/bin/env python3
"""P5: 3-head routed system (retrieval R / decoder D / generation A4) with a GBT multiclass router (fit on val),
plus per-package breakdown of the 2-head GBT union for the appendix. Inputs as router2_eval."""
import json, re, subprocess, sys, collections, argparse
import numpy as np
sys.path.insert(0, '$WORKSPACE/dh2')
from src.evaluation.metrics import compute_subtoken_f1
from sklearn.ensemble import GradientBoostingClassifier
WS = '$WORKSPACE/dh2'
ap = argparse.ArgumentParser(); ap.add_argument('--router', required=True); ap.add_argument('--regime-dump', required=True)
ap.add_argument('--a4', required=True); ap.add_argument('--out', required=True); args = ap.parse_args()
def demangle_many(names):
    todo = sorted({n for n in names if n and n.startswith('_Z')}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]; r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out
def canon(n, dem):
    n = dem.get(n, n); n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]; return '_'.join(p for p in parts if p)
regime = {}
for l in open(args.regime_dump):
    r = l.rstrip('\n').split('\t')
    if r[0] in ('val', 'test'): regime[(r[0], r[1], r[2])] = r[5]
a4_addr, a4_name = {}, {}
for l in open(args.a4):
    r = l.rstrip('\n').split('\t')
    if r[0] in ('val', 'test'):
        a4_addr[(r[0], r[1], int(r[2], 16))] = (r[4], float(r[9])); a4_name.setdefault((r[0], r[1], r[3]), (r[4], float(r[9])))
FEATS = ['sim1', 'margin', 'ext_jacc', 'str_jacc', 'd_conf', 'd_len', 'n_ext', 'n_str', 'n_blocks']
rows = []; hdr = None
for l in open(args.router):
    r = l.rstrip('\n').split('\t')
    if hdr is None: hdr = r; continue
    d = dict(zip(hdr, r)); tier = d['tier']
    if tier not in ('val', 'test'): continue
    a = int(d['bap'][4:], 16) if d['bap'].startswith('sub_') else None; p4 = None
    if a is not None:
        for da in (0, -4, 4):
            p4 = a4_addr.get((tier, d['binary'], a + da))
            if p4: break
    if p4 is None: p4 = a4_name.get((tier, d['binary'], d['true']))
    if p4 is None: continue
    x = {'tier': tier, 'pkg': d['binary'].split('_')[0], 'true': d['true'], 'R': d['r_pred'], 'D': d['d_pred'], 'A4': p4[0], 'a4c': p4[1],
         'regime': regime.get((tier, d['binary'], d['bap']), '?')}
    for f in FEATS: x[f] = float(d[f])
    rows.append(x)
dem = demangle_many([x['true'] for x in rows] + [x[h] for x in rows for h in ('R', 'D', 'A4')])
for x in rows:
    ct = canon(x['true'], dem)
    for h in ('R', 'D', 'A4'): x['f' + h] = compute_subtoken_f1(canon(x[h], dem), ct) if x[h] else 0.0
val = [x for x in rows if x['tier'] == 'val']; test = [x for x in rows if x['tier'] == 'test']
X = lambda rs: np.array([[x[f] for f in FEATS] + [x['a4c'], x['sim1'] - x['a4c']] for x in rs], dtype=np.float32)
def agg(rs, key):
    n = len(rs); g = collections.defaultdict(list)
    for r in rs: g[r['pkg']].append(r[key])
    ft = [r[key] for r in rs if r['regime'] == 'FT']; nct = [r[key] for r in rs if r['regime'] == 'NCT']
    return {'micro': round(sum(r[key] for r in rs)/n, 4), 'macro': round(sum(sum(v)/len(v) for v in g.values())/len(g), 4),
            'FT': round(sum(ft)/len(ft), 4), 'NCT': round(sum(nct)/len(nct), 4), 'n': n}
# 3-way target: argmax head (ties -> prefer A4, then R, then D — order of default usefulness)
def best_head(x):
    vals = {'A4': x['fA4'], 'R': x['fR'], 'D': x['fD']}; return max(vals, key=lambda h: (vals[h], h == 'A4', h == 'R'))
lab = {'A4': 0, 'R': 1, 'D': 2}
yv = np.array([lab[best_head(x)] for x in val]); wv = np.array([max(x['fR'], x['fD'], x['fA4']) - min(x['fR'], x['fD'], x['fA4']) + 1e-3 for x in val])
gb3 = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(X(val), yv, sample_weight=wv)
pred3 = gb3.predict(X(test)); inv = {v: k for k, v in lab.items()}
for x, p in zip(test, pred3): x['fU3'] = x['f' + inv[int(p)]]
# 2-way reference (R vs A4) for comparison and per-package table
y2 = np.array([1 if x['fR'] >= x['fA4'] else 0 for x in val]); w2 = np.abs(np.array([x['fR'] - x['fA4'] for x in val])) + 1e-3
gb2 = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(X(val), y2, sample_weight=w2)
u2 = gb2.predict_proba(X(test))[:, 1] >= 0.5
for x, u in zip(test, u2): x['fU2'] = x['fR'] if u else x['fA4']
for x in test: x['fO3'] = max(x['fR'], x['fD'], x['fA4']); x['fO2'] = max(x['fR'], x['fA4'])
rep = {'test': {k: agg(test, key) for k, key in (('retrieval', 'fR'), ('decoder', 'fD'), ('A4', 'fA4'), ('gbt_2head', 'fU2'), ('gbt_3head', 'fU3'), ('oracle_2head', 'fO2'), ('oracle_3head', 'fO3'))}}
rep['test']['route_share_3head'] = {inv[i]: round(float(np.mean(pred3 == i)), 3) for i in range(3)}
# per-package (2-head GBT union), sorted by n
g = collections.defaultdict(list)
for x in test: g[x['pkg']].append(x)
per = {p: {'n': len(v), 'regime': v[0]['regime'], 'R': round(sum(x['fR'] for x in v)/len(v), 3), 'A4': round(sum(x['fA4'] for x in v)/len(v), 3),
           'union': round(sum(x['fU2'] for x in v)/len(v), 3), 'oracle': round(sum(x['fO2'] for x in v)/len(v), 3)} for p, v in g.items()}
rep['per_package_2head'] = dict(sorted(per.items(), key=lambda kv: -kv[1]['n']))
json.dump(rep, open(args.out, 'w'), indent=1)
for k, v in rep['test'].items():
    if isinstance(v, dict) and 'micro' in v: print(f"  {k:<14} micro {v['micro']:.4f} macro {v['macro']:.4f} FT {v['FT']:.4f} NCT {v['NCT']:.4f}")
print('  route share 3-head:', rep['test']['route_share_3head'])
print('  per-package (top 12 by n):'); [print(f"    {p:<12} {d['regime']:<4} n={d['n']:>6} R {d['R']:.3f} A4 {d['A4']:.3f} union {d['union']:.3f} oracle {d['oracle']:.3f}") for p, d in list(rep['per_package_2head'].items())[:12]]
print('EFFECT: router3 done ->', args.out)
