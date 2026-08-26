#!/usr/bin/env python3
"""Learned 2-head router (retrieval R vs A4 generation) + selective prediction for the union system.
Inputs : router features TSV (per-row R/D preds + sim1, margin, ext_jacc, str_jacc, d_conf, d_len, n_ext, n_str, n_blocks)
         A4 preds TSV with `conf` column (a4_predict.py >= db771a46)
Train on VAL only (targets: which head has higher metric-v2 F1; abstention score = predicted F1 of the chosen head),
apply to TEST. Reports micro/macro/FT/NCT for: conf_sim1 baseline, learned router (logistic + gradient boosting),
oracle; and selective F1 @ 5/10/20/30/50/100% coverage for the routed system vs retrieval-only.
"""
import json, re, subprocess, sys, collections, argparse, math
import numpy as np
sys.path.insert(0, '/project/hz79/_shared/cs785/dh2')
from src.evaluation.metrics import compute_subtoken_f1
WS = '/project/hz79/_shared/cs785/dh2'
ap = argparse.ArgumentParser(); ap.add_argument('--router', default=f'{WS}/results/router_p3a_features.tsv')
ap.add_argument('--a4', required=True); ap.add_argument('--regime-dump', default=f'{WS}/results/p3a_preds_greedy.tsv')
ap.add_argument('--out', required=True); args = ap.parse_args()

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
    if r[0] not in ('val', 'test'): continue
    conf = float(r[9]) if len(r) > 9 else 0.0
    a4_addr[(r[0], r[1], int(r[2], 16))] = (r[4], conf); a4_name.setdefault((r[0], r[1], r[3]), (r[4], conf))
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
    if p4 is None: p4 = a4_name.get((tier, d['binary'], d['true']), ('', 0.0))
    x = {'tier': tier, 'pkg': d['binary'].split('_')[0], 'true': d['true'], 'R': d['r_pred'], 'A4': p4[0], 'a4_conf': p4[1],
         'regime': regime.get((tier, d['binary'], d['bap']), '?')}
    for f in FEATS: x[f] = float(d[f])
    rows.append(x)
dem = demangle_many([x['true'] for x in rows] + [x['R'] for x in rows] + [x['A4'] for x in rows])
for x in rows:
    ct = canon(x['true'], dem); x['fR'] = compute_subtoken_f1(canon(x['R'], dem), ct) if x['R'] else 0.0
    x['fA'] = compute_subtoken_f1(canon(x['A4'], dem), ct) if x['A4'] else 0.0
val = [x for x in rows if x['tier'] == 'val']; test = [x for x in rows if x['tier'] == 'test']
def X(rs): return np.array([[x[f] for f in FEATS] + [x['a4_conf'], x['sim1'] - x['a4_conf']] for x in rs], dtype=np.float32)
def agg(rs, key):
    n = len(rs); g = collections.defaultdict(list)
    for r in rs: g[r['pkg']].append(r[key])
    ft = [r[key] for r in rs if r['regime'] == 'FT']; nct = [r[key] for r in rs if r['regime'] == 'NCT']
    return {'n': n, 'micro': round(sum(r[key] for r in rs)/n, 4), 'macro': round(sum(sum(v)/len(v) for v in g.values())/len(g), 4),
            'FT': round(sum(ft)/len(ft), 4) if ft else None, 'NCT': round(sum(nct)/len(nct), 4) if nct else None}
def route(rs, useR):
    for r, u in zip(rs, useR): r['fU'] = r['fR'] if u else r['fA']; r['useR'] = bool(u)
    return agg(rs, 'fU')
def selective(rs, score, key='fU'):
    order = np.argsort(-np.asarray(score)); out = {}
    for c in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        k = max(1, int(round(c * len(rs)))); out[f'{int(c*100)}%'] = round(float(np.mean([rs[i][key] for i in order[:k]])), 4)
    return out
rep = {'val': {}, 'test': {}}
for tier, rs in (('val', val), ('test', test)):
    rep[tier]['retrieval'] = agg(rs, 'fR'); rep[tier]['A4'] = agg(rs, 'fA')
    for r in rs: r['fO'] = max(r['fR'], r['fA'])
    rep[tier]['oracle'] = agg(rs, 'fO')
# baseline conf_sim1 (tau on val)
taus = sorted({round(x['sim1'], 3) for x in val})[::max(1, len(val)//60)]
tau = max(taus, key=lambda t: route(val, [x['sim1'] >= t for x in val])['micro'])
rep['val']['conf_sim1'] = dict(route(val, [x['sim1'] >= tau for x in val]), tau=tau)
rep['test']['conf_sim1'] = dict(route(test, [x['sim1'] >= tau for x in test]), tau=tau)
rep['test']['conf_sim1_selective_R_only'] = selective(test, [x['sim1'] for x in test], key='fR')
rep['test']['conf_sim1_selective'] = selective(test, [max(x['sim1'], x['a4_conf']) for x in test])
# learned routers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
yv = np.array([1 if x['fR'] >= x['fA'] else 0 for x in val]); Xv, Xt = X(val), X(test)
wv = np.abs(np.array([x['fR'] - x['fA'] for x in val])) + 1e-3   # weight rows by how much the choice matters
lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xv, yv, sample_weight=wv)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05).fit(Xv, yv, sample_weight=wv)
def routing_stats(rs, useR):
    # oracle preference per row: R if fR > fA, A4 if fA > fR, tie otherwise (either choice is optimal)
    n = len(rs); pref_R = sum(1 for r in rs if r['fR'] > r['fA']); pref_A = sum(1 for r in rs if r['fA'] > r['fR']); ties = n - pref_R - pref_A
    correct = sum(1 for r, u in zip(rs, useR) if (r['fR'] == r['fA']) or (u and r['fR'] > r['fA']) or ((not u) and r['fA'] > r['fR']))
    contested = [(r, u) for r, u in zip(rs, useR) if r['fR'] != r['fA']]
    acc_contested = sum(1 for r, u in contested if (u and r['fR'] > r['fA']) or ((not u) and r['fA'] > r['fR'])) / max(1, len(contested))
    regret = sum(max(r['fR'], r['fA']) - (r['fR'] if u else r['fA']) for r, u in zip(rs, useR)) / n
    return {'oracle_prefers_R': round(pref_R / n, 3), 'oracle_prefers_A4': round(pref_A / n, 3), 'ties': round(ties / n, 3),
            'routing_acc_all': round(correct / n, 3), 'routing_acc_contested': round(acc_contested, 3), 'mean_regret_F1': round(regret, 4)}
rep['test']['routing_conf_sim1'] = routing_stats(test, [x['sim1'] >= tau for x in test])
for name, m in (('logreg', lr), ('gbt', gb)):
    pv = m.predict_proba(Xv)[:, 1]; pt = m.predict_proba(Xt)[:, 1]
    rep['val'][f'learned_{name}'] = route(val, pv >= 0.5); rep['test'][f'learned_{name}'] = route(test, pt >= 0.5)
    rep['test'][f'learned_{name}_R_rate'] = round(float(np.mean(pt >= 0.5)), 3)
    rep['test'][f'routing_{name}'] = routing_stats(test, pt >= 0.5)
# abstention: regress expected F1 of the routed answer (GBT) on val, rank test by it
useR_t = gb.predict_proba(Xt)[:, 1] >= 0.5; useR_v = gb.predict_proba(Xv)[:, 1] >= 0.5
Fv = np.hstack([Xv, useR_v[:, None]]); Ft = np.hstack([Xt, useR_t[:, None]])
yF = np.array([x['fR'] if u else x['fA'] for x, u in zip(val, useR_v)])
reg = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05).fit(Fv, yF)
route(test, useR_t); rep['test']['learned_gbt_selective'] = selective(test, reg.predict(Ft))
rep['test']['router_features'] = FEATS + ['a4_conf', 'sim1-a4_conf']
json.dump(rep, open(args.out, 'w'), indent=1)
for tier in ('val', 'test'):
    print(f'=== {tier} ===')
    for k, v in rep[tier].items():
        if isinstance(v, dict) and 'micro' in v: print(f"  {k:<26} micro {v['micro']:.4f} macro {v['macro']:.4f} FT {v['FT']} NCT {v['NCT']}" + (f" tau={v['tau']}" if 'tau' in v else ''))
        elif isinstance(v, dict): print(f"  {k:<26} {v}")
        else: print(f"  {k:<26} {v}")
print('EFFECT: router2 done ->', args.out)
