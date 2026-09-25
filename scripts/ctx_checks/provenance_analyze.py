#!/usr/bin/env python3
"""Provenance analysis (advisor request #2). Wulver, CPU. Joins per-function predictions of the no-context head,
the adopted ±10 head, and the inference arms (random / same-unit-outside-window / different-unit) on the functions
with source-file labels, and reports:
  T1  F1 by context source on the identical subset (fn / pkg) incl. SU and DU
  T2  gain of address context over no context, binned by same-file share of the window (functions) and of the digest (tokens)
  T3  prefix vs rest-of-name decomposition of the gain
  T4  gain conditional on where the ground-truth prefix in the digest came from (same file / other file / absent)
Output: results/ctx_layout/provenance_analysis.json"""
import csv, json, os, re, random
from collections import defaultdict
WS = '/project/hz79/_shared/cs785/dh2'
SPLIT_RE1 = re.compile(r'([a-z0-9])([A-Z])'); SPLIT_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')


def subtoks(name):
    n = re.sub(r'\(.*\)$', '', name); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]; n = '_'.join(p for p in parts if p)
    s = SPLIT_RE1.sub(r'\1_\2', n); s = SPLIT_RE2.sub(r'\1_\2', s)
    return [t for t in re.split(r'[^A-Za-z0-9]+', s.lower()) if t]


def read(path, tier='test'):
    out = {}
    for r in csv.DictReader(open(path), delimiter='\t'):
        if r['tier'] != tier: continue
        out[f"{r['binary']}_{r['entry_addr']}"] = (float(r['f1_v2']), r['true'], r['pred'])
    return out


def f1(a, b):
    from collections import Counter
    ca, cb = Counter(a), Counter(b); tp = sum((ca & cb).values())
    if tp == 0: return 0.0
    p = tp / sum(ca.values()); r = tp / sum(cb.values()); return 2 * p * r / (p + r)


def mean(xs): return round(sum(xs) / len(xs), 4) if xs else None


def main():
    prov = json.load(open(f'{WS}/results/ctx_layout/provenance_test.json'))
    arms = {'none': read(f'{WS}/results/a4_codet5p220m_v1/val_test_symgen_holdout_preds.tsv'),
            'address': read(f'{WS}/results/a4_modctx_dm_sens_win10/test_preds.tsv'),
            'random': read(f'{WS}/results/a4_modctx_dm_sens_random/test_preds.tsv'),
            'callgraph_swap': read(f'{WS}/results/a4_modctx_dm_sens_callgraph/test_preds.tsv'),
            'same_unit_outside_window': read(f'{WS}/results/a4_modctx_dm_sens_su/test_preds.tsv'),
            'different_unit': read(f'{WS}/results/a4_modctx_dm_sens_du/test_preds.tsv')}
    for n in ('callgraph_trained', 'random_trained', 'union_trained'):
        tag = {'callgraph_trained': 'codet5p220m_ctxcallgraph_dm_v1', 'random_trained': 'codet5p220m_ctxrandom_dm_v1', 'union_trained': 'codet5p220m_ctxaddrcall_dm_v1'}[n]
        p = f'{WS}/results/a4_{tag}/val_test_preds.tsv'
        if os.path.exists(p): arms[n] = read(p)
    keys = [k for k in prov if all(k in a for a in arms.values())]
    out = {'n_functions': len(keys), 'n_binaries': len({prov[k]['binary'] for k in keys}), 'n_packages': len({prov[k]['package'] for k in keys})}
    print(f"subset: {len(keys)} functions, {out['n_binaries']} binaries, {out['n_packages']} packages")
    # T1
    t1 = {}
    for n, a in arms.items():
        g = defaultdict(list)
        for k in keys: g[prov[k]['package']].append(a[k][0])
        t1[n] = {'f1_fn': mean([a[k][0] for k in keys]), 'f1_pkg': mean([sum(v) / len(v) for v in g.values()])}
        print(f"T1 {n:26s} fn {t1[n]['f1_fn']} pkg {t1[n]['f1_pkg']}")
    su_nonempty = [k for k in keys if prov[k]['su_digest_n'] > 0]; du_nonempty = [k for k in keys if prov[k]['du_digest_n'] > 0]
    both = [k for k in keys if prov[k]['su_digest_n'] > 0 and prov[k]['du_digest_n'] > 0]
    t1['subset_su_and_du_nonempty'] = {n: mean([arms[n][k][0] for k in both]) for n in arms}; t1['n_su_and_du_nonempty'] = len(both)
    print(f"T1 (functions with both SU and DU digests non-empty, n={len(both)}): " + ' | '.join(f"{n} {v}" for n, v in t1['subset_su_and_du_nonempty'].items()))
    # T2
    def binned(share_of):
        bins = {'0': [], '(0,0.5)': [], '[0.5,1)': [], '1': []}
        for k in keys:
            s = share_of(k); b = '0' if s == 0 else '1' if s >= 1 else '(0,0.5)' if s < 0.5 else '[0.5,1)'
            bins[b].append(arms['address'][k][0] - arms['none'][k][0])
        return {b: {'n': len(v), 'delta_address_minus_none': mean(v)} for b, v in bins.items()}
    t2 = {'by_window_same_file_function_share': binned(lambda k: prov[k]['win_same_file_fns'] / max(1, prov[k]['win_n'])),
          'by_digest_same_file_token_share': binned(lambda k: prov[k]['digest_same_file_tokens'] / max(1, prov[k]['digest_n']))}
    for name, d in t2.items():
        print('T2', name, {b: (v['n'], v['delta_address_minus_none']) for b, v in d.items()})
    # T3 prefix vs rest
    t3 = {}
    for n in ('none', 'address', 'random', 'same_unit_outside_window', 'different_unit'):
        pre = []; rest = []
        for k in keys:
            _, tr, pr = arms[n][k]; gt = subtoks(tr); pd = subtoks(pr)
            if not gt: continue
            pre.append(1.0 if gt[0] in pd else 0.0)
            if len(gt) > 1: rest.append(f1(pd, gt[1:]))
        t3[n] = {'prefix_recall': mean(pre), 'rest_f1': mean(rest)}
        print(f"T3 {n:26s} prefix recall {t3[n]['prefix_recall']} rest F1 {t3[n]['rest_f1']}")
    t3['delta_address_minus_none'] = {'prefix_recall': round(t3['address']['prefix_recall'] - t3['none']['prefix_recall'], 4), 'rest_f1': round(t3['address']['rest_f1'] - t3['none']['rest_f1'], 4)}
    print('T3 Δ(address−none):', t3['delta_address_minus_none'])
    # T4 gain by prefix provenance
    t4 = {}
    for src in ('same_file', 'other_file', 'absent'):
        ks = [k for k in keys if prov[k]['prefix_source'] == src]
        t4[src] = {'n': len(ks), 'none': mean([arms['none'][k][0] for k in ks]), 'address': mean([arms['address'][k][0] for k in ks]),
                   'delta': mean([arms['address'][k][0] - arms['none'][k][0] for k in ks])}
        print(f"T4 prefix {src:10s} n={len(ks)} none {t4[src]['none']} address {t4[src]['address']} Δ {t4[src]['delta']}")
    # package-level paired bootstrap: SU vs DU, address vs SU, on the both-non-empty subset
    rng = random.Random(0); boot = {}
    g = defaultdict(lambda: defaultdict(list))
    for k in both:
        for n in ('address', 'same_unit_outside_window', 'different_unit', 'random'): g[n][prov[k]['package']].append(arms[n][k][0])
    pkgs = sorted(g['address'])
    for a, b in (('same_unit_outside_window', 'different_unit'), ('address', 'same_unit_outside_window'), ('same_unit_outside_window', 'random'), ('different_unit', 'random')):
        d = [sum(g[a][p]) / len(g[a][p]) - sum(g[b][p]) / len(g[b][p]) for p in pkgs]
        bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
        boot[f'{a} - {b}'] = {'delta_pkg': round(sum(d) / len(d), 4), 'ci95': [round(bs[250], 4), round(bs[9749], 4)], 'first_higher': f'{sum(1 for x in d if x > 0)}/{len(d)}'}
        print(f"BOOT {a} - {b}: pkg {sum(d)/len(d):+.4f} CI [{bs[250]:+.4f},{bs[9749]:+.4f}] first higher {sum(1 for x in d if x>0)}/{len(d)}")
    out.update({'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4, 'bootstrap_both_nonempty': boot,
                'su_nonempty': len(su_nonempty), 'du_nonempty': len(du_nonempty)})
    json.dump(out, open(f'{WS}/results/ctx_layout/provenance_analysis.json', 'w'), indent=1)
    print('EFFECT: provenance analysis written')


if __name__ == '__main__':
    main()
