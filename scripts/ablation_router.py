#!/usr/bin/env python3
"""Router feature ablation (P5): refit the GBT router with feature groups removed; report test micro/macro/FT/NCT,
routing accuracy on contested rows, regret, and selective F1@10/20%. Router/abstainer fit on VAL only.
Groups: retrieval-sim (sim1, margin), overlap (ext_jacc, str_jacc), decoder (d_conf, d_len), size (n_ext, n_str, n_blocks),
        a4 (a4_conf, sim1-a4_conf)."""
import json, re, subprocess, sys, collections, argparse
import numpy as np
sys.path.insert(0, '/project/hz79/_shared/cs785/dh2')
from src.evaluation.metrics import compute_subtoken_f1
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
WS = '/project/hz79/_shared/cs785/dh2'
ap = argparse.ArgumentParser(); ap.add_argument('--a4', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--router', default=f'{WS}/results/router_p3a_features.tsv'); ap.add_argument('--regime-dump', default=f'{WS}/results/p3a_preds_greedy.tsv')
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
regime = {}
for l in open(args.regime_dump):
    r = l.rstrip('\n').split('\t')
    if r[0] in ('val', 'test'): regime[(r[0], r[1], r[2])] = r[5]
a4_addr, a4_name = {}, {}
for l in open(args.a4):
    r = l.rstrip('\n').split('\t')
    if r[0] in ('val', 'test'):
        a4_addr[(r[0], r[1], int(r[2], 16))] = (r[4], float(r[9])); a4_name.setdefault((r[0], r[1], r[3]), (r[4], float(r[9])))
GROUPS = {'retrieval-sim': ['sim1', 'margin'], 'overlap': ['ext_jacc', 'str_jacc'], 'decoder': ['d_conf', 'd_len'],
          'size': ['n_ext', 'n_str', 'n_blocks'], 'a4': ['a4_conf', 'sim1-a4_conf']}
ALL = [f for g in GROUPS.values() for f in g]
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
    if p4 is None: p4 = a4_name.get((tier, d['binary'], d['true']), ('', 0.0))
    x = {'tier': tier, 'pkg': d['binary'].split('_')[0], 'true': d['true'], 'R': d['r_pred'], 'A4': p4[0], 'regime': regime.get((tier, d['binary'], d['bap']), '?')}
    for f in ['sim1', 'margin', 'ext_jacc', 'str_jacc', 'd_conf', 'd_len', 'n_ext', 'n_str', 'n_blocks']: x[f] = float(d[f])
    x['a4_conf'] = p4[1]; x['sim1-a4_conf'] = x['sim1'] - p4[1]
    rows.append(x)
dem = demangle_many([x['true'] for x in rows] + [x['R'] for x in rows] + [x['A4'] for x in rows])
for x in rows:
    ct = canon(x['true'], dem); x['fR'] = compute_subtoken_f1(canon(x['R'], dem), ct) if x['R'] else 0.0
    x['fA'] = compute_subtoken_f1(canon(x['A4'], dem), ct) if x['A4'] else 0.0
val = [x for x in rows if x['tier'] == 'val']; test = [x for x in rows if x['tier'] == 'test']
yv = np.array([1 if x['fR'] >= x['fA'] else 0 for x in val]); wv = np.abs(np.array([x['fR'] - x['fA'] for x in val])) + 1e-3
def agg(rs, key):
    n = len(rs); g = collections.defaultdict(list)
    for r in rs: g[r['pkg']].append(r[key])
    ft = [r[key] for r in rs if r['regime'] == 'FT']; nct = [r[key] for r in rs if r['regime'] == 'NCT']
    return {'micro': round(sum(r[key] for r in rs)/n, 4), 'macro': round(sum(sum(v)/len(v) for v in g.values())/len(g), 4), 'FT': round(sum(ft)/len(ft), 4), 'NCT': round(sum(nct)/len(nct), 4)}
def run(feats, tag):
    Xv = np.array([[x[f] for f in feats] for x in val], dtype=np.float32); Xt = np.array([[x[f] for f in feats] for x in test], dtype=np.float32)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(Xv, yv, sample_weight=wv)
    ut = gb.predict_proba(Xt)[:, 1] >= 0.5; uv = gb.predict_proba(Xv)[:, 1] >= 0.5
    for r, u in zip(test, ut): r['fU'] = r['fR'] if u else r['fA']
    contested = [(r, u) for r, u in zip(test, ut) if r['fR'] != r['fA']]
    acc = sum(1 for r, u in contested if (u and r['fR'] > r['fA']) or ((not u) and r['fA'] > r['fR'])) / len(contested)
    regret = sum(max(r['fR'], r['fA']) - r['fU'] for r in test) / len(test)
    # abstention regressor with the same feature set
    Fv = np.hstack([Xv, uv[:, None]]); Ft = np.hstack([Xt, ut[:, None]])
    yF = np.array([x['fR'] if u else x['fA'] for x, u in zip(val, uv)])
    reg = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(Fv, yF)
    sc = reg.predict(Ft); order = np.argsort(-sc)
    sel = {c: round(float(np.mean([test[i]['fU'] for i in order[:int(c * len(test))]])), 4) for c in (0.1, 0.2)}
    res = dict(agg(test, 'fU'), routing_acc_contested=round(acc, 3), regret=round(regret, 4), R_rate=round(float(ut.mean()), 3), sel10=sel[0.1], sel20=sel[0.2], n_feats=len(feats))
    print(f"  {tag:<28} micro {res['micro']:.4f} macro {res['macro']:.4f} FT {res['FT']:.4f} NCT {res['NCT']:.4f} | acc {res['routing_acc_contested']:.3f} regret {res['regret']:.4f} R% {res['R_rate']:.3f} | sel@10 {res['sel10']:.3f} @20 {res['sel20']:.3f}")
    return res
rep = {}
rep['all'] = run(ALL, 'all features')
for g, fs in GROUPS.items():
    rep[f'minus_{g}'] = run([f for f in ALL if f not in fs], f'- {g}')
for g, fs in GROUPS.items():
    rep[f'only_{g}'] = run(fs, f'only {g}')
rep['only_a4_conf'] = run(['a4_conf'], 'only a4_conf')
rep['only_sim1'] = run(['sim1'], 'only sim1')
json.dump(rep, open(args.out, 'w'), indent=1); print('EFFECT: router ablation done ->', args.out)
