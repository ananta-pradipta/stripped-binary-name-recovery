#!/usr/bin/env python3
"""Score SymGen-34B (retrained on our train tier) on the FULL test tier (sharded inference outputs)
against our heads on the same joined keys: R/D from the router features tsv, A4 from the A4 preds tsv,
GBT router refit on val (same recipe as matched_baselines.py / router2_eval), oracle(R,A4), and
optionally BLens (LORD log) and extra generation heads (--extra-head NAME=preds.tsv, e.g. the BAP-text
control). Same metric-v2 canonicalization (c++filt demangle, template/arg stripping) for every system.
Strata: all / FT / NCT (meta regime) and seen/novel name (meta name_seen). Feeds FINAL_TABLES T3d.

Shards: symgen_v2/fulltest_shards/meta_K.json (row metadata, index-aligned with the shard input) and
symgen_v2/results_fulltest/shard_K/predicted_function_name.json (index-aligned list). --partial scores
whatever rows exist so far (interim read); the final run must see every shard complete."""
import argparse, collections, json, os, pickle, re, subprocess, sys
import numpy as np
sys.path.insert(0, '/project/hz79/_shared/cs785/dh2')
from src.evaluation.metrics import compute_subtoken_f1, split_name
WS = '/project/hz79/_shared/cs785/dh2'
ap = argparse.ArgumentParser()
ap.add_argument('--shards', type=int, default=34)
ap.add_argument('--meta-dir', default=f'{WS}/symgen_v2/fulltest_shards')
ap.add_argument('--pred-dir', default=f'{WS}/symgen_v2/results_fulltest')
ap.add_argument('--feat', default=f'{WS}/results/router_c1soft_l10_interim_features.tsv')
ap.add_argument('--a4', default=f'{WS}/results/a4_codet5p220m_v1/val_test_symgen_holdout_preds.tsv')
ap.add_argument('--blens-log', default=None, help='LORD-inference-logs-test-*.txt to add a BLens column')
ap.add_argument('--nlp', default=f'{WS}/blens_ours_v2/xflBlensXProjectData')
ap.add_argument('--extra-head', action='append', default=[], help='NAME=preds.tsv in a4_predict format')
ap.add_argument('--partial', action='store_true', help='score incomplete shards (interim read)')
ap.add_argument('--word-cluster', default=None,
                help='SymLM CodeWordNet word_cluster.json; adds a semantic-F1 column (predicted token '
                     'counts as correct when it shares a cluster with a target token, per SymLM CCS22 eval)')
ap.add_argument('--out', default=f'{WS}/results/score_symgen_full.json')
args = ap.parse_args()
WC = json.load(open(args.word_cluster)) if args.word_cluster else None

def semantic_f1(cp, ct):
    """SymLM-style cluster-replacement F1 on canonicalized names (same split as compute_subtoken_f1)."""
    pt, tt = split_name(cp), split_name(ct)
    if not pt and not tt: return 1.0
    if not pt or not tt: return 0.0
    tset = set(tt)
    rep = []
    for p in pt:
        if p not in tset and p in WC:
            pc = set(WC[p])
            for t in tt:
                if t in WC and pc.intersection(WC[t]): p = t; break
        rep.append(p)
    pc_, tc_ = collections.Counter(rep), collections.Counter(tt)
    tp = sum((pc_ & tc_).values())
    prec = tp / sum(pc_.values()); rec = tp / sum(tc_.values())
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

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

# ---- SymGen shards -> per-key pred
sg = {}; meta_by_key = {}; gt_mismatch = 0; incomplete = []
for k in range(args.shards):
    mp = f'{args.meta_dir}/meta_{k}.json'; pp = f'{args.pred_dir}/shard_{k}/predicted_function_name.json'
    meta = json.load(open(mp))
    try:
        preds = json.load(open(pp)) if os.path.exists(pp) else []
    except json.JSONDecodeError:  # shard being rewritten by a running job
        preds = []
    if len(preds) < len(meta):
        incomplete.append((k, len(preds), len(meta)))
        if not args.partial: continue
    for m, p in zip(meta, preds):
        if p['ground_truth'] != m['gt_name']: gt_mismatch += 1; continue
        key = ('test', m['binary'], int(m['addr'], 16))
        sg[key] = sg_clean(p['predicted_name']); meta_by_key[key] = m
if incomplete:
    print(f'incomplete shards: {incomplete}')
    if not args.partial: sys.exit(f'FATAL: {len(incomplete)} shards incomplete; rerun array or pass --partial')
print(f'SymGen rows: {len(sg)} (gt mismatches skipped: {gt_mismatch})')

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
def load_a4_tsv(path):
    m = {}
    for l in open(path):
        r = l.rstrip('\n').split('\t')
        if r[0] in ('val', 'test'): m[(r[0], r[1], int(r[2], 16))] = (r[4], float(r[9]) if len(r) > 9 else 0.0)
    return m
a4 = load_a4_tsv(args.a4)
extra = {}
for spec in args.extra_head:
    name, path = spec.split('=', 1); extra[name] = load_a4_tsv(path)

# ---- optional BLens per-key preds (same parse as score_blens_matched.py)
blens = None
if args.blens_log:
    pairs = []; lines = open(args.blens_log, errors='ignore').read().splitlines()
    for i, l in enumerate(lines):
        if l.startswith('target: ') and i + 1 < len(lines) and lines[i+1].startswith('output: '):
            pairs.append((l[8:].strip(), lines[i+1][8:].strip()))
    test = pickle.load(open(args.nlp, 'rb'))[2]
    assert len(pairs) == len(test), f'BLens log pairs {len(pairs)} != test rows {len(test)}'
    blens = {}
    for (t, o), e in zip(pairs, test):
        b = os.path.basename(e[0]); b = b[:-9] if b.endswith('_stripped') else b
        blens[('test', b, int(e[1]))] = o
    print(f'BLens keys: {len(blens)}')

# ---- refit GBT router on val (same recipe as matched_baselines.py)
from sklearn.ensemble import GradientBoostingClassifier
def X(d, a4c): return [float(d[f]) for f in FEATS] + [a4c, float(d['sim1']) - a4c]
val_rows = [(d, *a4[k]) for k, d in feat.items() if k[0] == 'val' and k in a4]
dem = demangle_many([d['true'] for d, _, _ in val_rows] + [d['r_pred'] for d, _, _ in val_rows] + [p for _, p, _ in val_rows])
Xv, yv, wv = [], [], []
for d, p4, c4 in val_rows:
    ct = canon(d['true'], dem); fr = compute_subtoken_f1(canon(d['r_pred'], dem), ct); fa = compute_subtoken_f1(canon(p4, dem), ct) if p4 else 0.0
    Xv.append(X(d, c4)); yv.append(1 if fr >= fa else 0); wv.append(abs(fr - fa) + 1e-3)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(np.array(Xv), np.array(yv), sample_weight=np.array(wv))
print('router refit on val rows:', len(val_rows))

# ---- join full test
rows = []; miss = collections.Counter()
for key, p in sg.items():
    m = meta_by_key[key]; d = feat.get(key); a = a4.get(key)
    if d is None or a is None: miss['no_our_row' if d is None else 'no_a4'] += 1; continue
    r = {'pkg': m['package'], 'true': m['gt_name'], 'regime': m['regime'], 'seen': bool(m.get('name_seen')),
         'SymGen': p, 'R': d['r_pred'], 'D': d['d_pred'], 'A4': a[0], 'x': X(d, a[1])}
    if blens is not None: r['BLens'] = blens.get(key, '')
    for name, mp in extra.items(): r[name] = mp.get(key, ('', 0.0))[0]
    rows.append(r)
print(f'joined {len(rows)} / {len(sg)} SymGen rows; missing {dict(miss)}')

HEADS = ['SymGen', 'R', 'D', 'A4'] + (['BLens'] if blens is not None else []) + list(extra)
dem = demangle_many([r['true'] for r in rows] + [r[h] for r in rows for h in HEADS])
for r in rows:
    ct = canon(r['true'], dem)
    for h in HEADS:
        cp = canon(r[h], dem) if r[h] else ''
        r['f_' + h] = compute_subtoken_f1(cp, ct) if r[h] else 0.0
        if WC: r['s_' + h] = semantic_f1(cp, ct) if r[h] else 0.0
pr = gb.predict_proba(np.array([r['x'] for r in rows]))[:, 1]
for r, p in zip(rows, pr):
    r['f_gbt'] = r['f_R'] if p >= 0.5 else r['f_A4']; r['f_oracle'] = max(r['f_R'], r['f_A4'])
    if WC:
        r['s_gbt'] = r['s_R'] if p >= 0.5 else r['s_A4']; r['s_oracle'] = max(r['s_R'], r['s_A4'])
ALL = HEADS + ['gbt', 'oracle']

def table(rs):
    T = {}
    for h in ALL:
        key = 'f_' + h; g = collections.defaultdict(list)
        for r in rs: g[r['pkg']].append(r[key])
        n = len(rs)
        T[h] = {'micro': round(sum(r[key] for r in rs) / n, 4) if n else None,
                'macro': round(sum(sum(v)/len(v) for v in g.values()) / len(g), 4) if g else None,
                'EM': round(sum(1 for r in rs if r[key] == 1.0) / n, 4) if n else None, 'n': n, 'pkgs': len(g)}
        if WC and n:
            sg_ = collections.defaultdict(list)
            for r in rs: sg_[r['pkg']].append(r['s_' + h])
            T[h]['sem_micro'] = round(sum(r['s_' + h] for r in rs) / n, 4)
            T[h]['sem_macro'] = round(sum(sum(v)/len(v) for v in sg_.values()) / len(sg_), 4)
    return T
report = {'partial': bool(incomplete), 'incomplete_shards': incomplete, 'joined': len(rows), 'missing': dict(miss),
          'strata': {}}
strata = [('all', rows), ('FT', [r for r in rows if r['regime'] == 'FT']), ('NCT', [r for r in rows if r['regime'] == 'NCT']),
          ('seen_name', [r for r in rows if r['seen']]), ('novel_name', [r for r in rows if not r['seen']])]
for name, rs in strata:
    report['strata'][name] = table(rs)
    print(f'=== {name} (n={len(rs)}) ===')
    for h, v in report['strata'][name].items():
        if v['n']:
            sem = f" | sem {v['sem_micro']:.4f}/{v['sem_macro']:.4f}" if 'sem_micro' in v else ''
            print(f"  {h:<8} micro {v['micro']:.4f} macro {v['macro']:.4f} EM {v['EM']:.4f} ({v['pkgs']} pkgs){sem}")
json.dump(report, open(args.out, 'w'), indent=1)
print('EFFECT: score_symgen_full done ->', args.out)
