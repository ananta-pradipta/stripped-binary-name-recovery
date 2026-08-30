#!/usr/bin/env python3
"""Full-test-tier comparison on identical keys: SymGen (34B+LoRA retrained on our tier; sharded preds in
symgen_v2/results_fulltest/shard_K/) vs BLens (LORD inference log) vs our heads R / D / A4 / GBT union / oracle.
Same row sources and router recipe as score_blens_matched.py. Reports all / FT / NCT / seen-name / novel-name.
Partial shard sets are allowed (--allow-partial) for interim reads; the JSON records coverage."""
import argparse, collections, glob, json, os, pickle, re, subprocess, sys
import numpy as np
sys.path.insert(0, '/project/hz79/_shared/cs785/dh2')
from src.evaluation.metrics import compute_subtoken_f1
WS = '/project/hz79/_shared/cs785/dh2'
ap = argparse.ArgumentParser()
ap.add_argument('--shards', default=f'{WS}/symgen_v2/fulltest_shards')
ap.add_argument('--preds', default=f'{WS}/symgen_v2/results_fulltest')
ap.add_argument('--blens-log', default=f'{WS}/blens_ours_v2/xp/ours-v2/LORD-inference-logs-test-67.txt')
ap.add_argument('--nlp', default=f'{WS}/blens_ours_v2/xflBlensXProjectData')
ap.add_argument('--feat', default=f'{WS}/results/router_c1soft_l10_interim_features.tsv')
ap.add_argument('--regime', default=f'{WS}/results/c1soft_l10_interim_preds_greedy.tsv')
ap.add_argument('--a4', default=f'{WS}/results/a4_codet5p220m_v1/val_test_symgen_holdout_preds.tsv')
ap.add_argument('--extra-head', nargs='*', default=[], help='NAME=preds.tsv extra generation heads scored as additional columns (e.g. A4bap=results/a4_codet5p220m_baptext_v1/val_test_preds.tsv)')
ap.add_argument('--out', default=f'{WS}/results/fulltest_all_systems.json')
ap.add_argument('--allow-partial', action='store_true')
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
# ---- SymGen preds by key
sg = {}; shards_done, shards_all = 0, len(glob.glob(f'{args.shards}/meta_*.json'))
for mp in sorted(glob.glob(f'{args.shards}/meta_*.json'), key=lambda p: int(re.findall(r'meta_(\d+)', p)[0])):
    k = int(re.findall(r'meta_(\d+)', mp)[0]); pp = f'{args.preds}/shard_{k}/predicted_function_name.json'
    meta = json.load(open(mp))
    if not os.path.exists(pp): continue
    preds = json.load(open(pp))
    if len(preds) != len(meta):
        if not args.allow_partial: raise SystemExit(f'shard {k}: {len(preds)} preds vs {len(meta)} meta — incomplete (use --allow-partial for interim)')
        print(f'shard {k}: PARTIAL {len(preds)}/{len(meta)}')
    else: shards_done += 1
    for m, p in zip(meta, preds): sg[('test', m['binary'], int(m['addr'], 16))] = sg_clean(p['predicted_name'])
print(f'SymGen: {len(sg)} preds from {shards_done}/{shards_all} complete shards')
# ---- BLens preds by key (log order == BLens test rows)
blens = {}
if os.path.exists(args.blens_log) and os.path.exists(args.nlp):
    L = open(args.blens_log, errors='ignore').read().splitlines()
    pairs = [(l[8:].strip(), L[i+1][8:].strip()) for i, l in enumerate(L) if l.startswith('target: ') and i+1 < len(L) and L[i+1].startswith('output: ')]
    test = pickle.load(open(args.nlp, 'rb'))[2]; assert len(pairs) == len(test)
    for (t, o), e in zip(pairs, test):
        b = os.path.basename(e[0]); b = b[:-9] if b.endswith('_stripped') else b
        blens[('test', b, int(e[1]))] = o
print(f'BLens: {len(blens)} preds')
# ---- our heads
FEATS = ['sim1', 'margin', 'ext_jacc', 'str_jacc', 'd_conf', 'd_len', 'n_ext', 'n_str', 'n_blocks']
feat = {}; hdr = None
for l in open(args.feat):
    r = l.rstrip('\n').split('\t')
    if hdr is None: hdr = r; continue
    d = dict(zip(hdr, r))
    if d['bap'].startswith('sub_'):
        a = int(d['bap'][4:], 16)
        for da in (0, 4, -4): feat.setdefault((d['tier'], d['binary'], a + da), d)
reg = {}; hdr = None
for l in open(args.regime):
    r = l.rstrip('\n').split('\t')
    if hdr is None: hdr = r; continue
    d = dict(zip(hdr, r))
    if d['bap_name'].startswith('sub_'):
        a = int(d['bap_name'][4:], 16)
        for da in (0, 4, -4): reg.setdefault((d['tier'], d['binary'], a + da), (d['regime'], d['name_seen']))
def load_a4(path):
    m = {}
    for l in open(path):
        r = l.rstrip('\n').split('\t')
        if r[0] in ('val', 'test'): m[(r[0], r[1], int(r[2], 16))] = (r[4], float(r[9]) if len(r) > 9 else 0.0)
    return m
a4 = load_a4(args.a4); extra = {}
for spec in args.extra_head:
    name, path = spec.split('=', 1); extra[name] = load_a4(path); print(f'extra head {name}: {len(extra[name])} preds')
from sklearn.ensemble import GradientBoostingClassifier
def X(d, a4c): return [float(d[f]) for f in FEATS] + [a4c, float(d['sim1']) - a4c]
val_rows = [(d, *a4[k]) for k, d in feat.items() if k[0] == 'val' and k in a4]
dem = demangle_many([d['true'] for d, _, _ in val_rows] + [d['r_pred'] for d, _, _ in val_rows] + [p for _, p, _ in val_rows])
Xv, yv, wv = [], [], []
for d, p4, c4 in val_rows:
    ct = canon(d['true'], dem); fr = compute_subtoken_f1(canon(d['r_pred'], dem), ct); fa = compute_subtoken_f1(canon(p4, dem), ct) if p4 else 0.0
    Xv.append(X(d, c4)); yv.append(1 if fr >= fa else 0); wv.append(abs(fr - fa) + 1e-3)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(np.array(Xv), np.array(yv), sample_weight=np.array(wv))
# ---- rows = SymGen-covered test keys that have our row + A4 (identical functions for every system)
rows = []; miss = collections.Counter()
for key, sp in sg.items():
    d = feat.get(key); a = a4.get(key)
    if d is None: miss['no_our_row'] += 1; continue
    if a is None: miss['no_a4'] += 1; continue
    if blens and key not in blens: miss['no_blens'] += 1; continue
    rg = reg.get(key, ('?', '?'))
    r = {'pkg': key[1].split('_')[0], 'true': d['true'], 'SymGen': sp, 'BLens': blens.get(key, ''), 'R': d['r_pred'], 'D': d['d_pred'], 'A4': a[0],
         'x': X(d, a[1]), 'regime': rg[0], 'seen': rg[1]}
    for name, m in extra.items(): r[name] = m.get(key, ('', 0.0))[0]
    rows.append(r)
print(f'rows {len(rows)} (SymGen keys {len(sg)}; missing {dict(miss)})')
HEADS = ['SymGen'] + (['BLens'] if blens else []) + ['R', 'D', 'A4'] + list(extra)
dem = demangle_many([r['true'] for r in rows] + [r[h] for r in rows for h in HEADS])
pr = gb.predict_proba(np.array([r['x'] for r in rows]))[:, 1]
for r, p in zip(rows, pr):
    ct = canon(r['true'], dem)
    for h in HEADS: r['f_' + h] = compute_subtoken_f1(canon(r[h], dem), ct) if r[h] else 0.0
    r['f_gbt'] = r['f_R'] if p >= 0.5 else r['f_A4']; r['f_oracle'] = max(r['f_R'], r['f_A4'])
COLS = HEADS + ['gbt', 'oracle']
def table(rs, label):
    T = {}
    for h in COLS:
        key = 'f_' + h; g = collections.defaultdict(list)
        for r in rs: g[r['pkg']].append(r[key])
        T[h] = {'micro': round(sum(r[key] for r in rs) / len(rs), 4), 'macro': round(sum(sum(v)/len(v) for v in g.values()) / len(g), 4),
                'EM': round(sum(1 for r in rs if r[key] == 1.0) / len(rs), 4), 'n': len(rs), 'pkgs': len(g)}
    print(f'=== {label} (n={len(rs)}, {len(g)} pkgs) ===')
    for h, v in T.items(): print(f"  {h:<8} micro {v['micro']:.4f} macro {v['macro']:.4f} EM {v['EM']:.4f}")
    return T
report = {'all': table(rows, 'ALL SymGen-covered test keys')}
for rg in ('FT', 'NCT'):
    sub = [r for r in rows if r['regime'] == rg]
    if sub: report[rg] = table(sub, f'regime {rg}')
for sn, lab in (('1', 'seen_name'), ('0', 'novel_name')):
    sub = [r for r in rows if r['seen'] == sn]
    if sub: report[lab] = table(sub, lab)
report['coverage'] = {'shards_done': shards_done, 'shards_all': shards_all, 'symgen_keys': len(sg), 'rows': len(rows), 'missing': dict(miss),
                      'symgen_empty_preds': sum(1 for r in rows if not r['SymGen'])}
json.dump(report, open(args.out, 'w'), indent=1); print('EFFECT: full-tier all-systems scoring ->', args.out, report['coverage'])
