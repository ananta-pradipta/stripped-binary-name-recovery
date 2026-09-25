#!/usr/bin/env python3
"""Punstrip continuation (plan §8–§11, §13, §20): trained-variant comparison table, package-level paired bootstrap,
P2/P3 complementarity, confidence-based selection, digest statistics, cross-benchmark table. Runs on Wulver dh2 after
the four predict jobs finish. Outputs punstrip/data_ctx/punstrip_ctx_{trained_comparison,bootstrap,complementarity,
confidence_select,digest_stats}.json and punstrip_ctx_results_table.csv.
Package identity = the Punstrip package (row field 'package' = binary id before '__')."""
import csv, json, os, random
from collections import defaultdict
WS = '/project/hz79/_shared/cs785/dh2'; P = '/project/hz79/_shared/cs785/punstrip'; OUT = f'{P}/data_ctx'
HEADS = {'P0 none': 'punstrip_ctxnone_v1', 'P1 random (window-excluded, seed 1)': 'punstrip_ctxrandom_excl_v1',
         'P2 callers+callees': 'punstrip_ctxcallgraph_v1', 'P3 ±10 address (adopted)': 'punstrip_modctx_v1',
         'P4 address ∪ callers/callees': 'punstrip_ctxaddrcall_v1'}
LB = {'P0 none': (0.184, 0.360), 'P1 random (window-excluded, seed 1)': (0.1927, 0.3891), 'P2 callers+callees': (0.2138, 0.4030),
      'P3 ±10 address (adopted)': (0.2125, 0.4000), 'P4 address ∪ callers/callees': (0.2165, 0.4052)}


def read_preds(tag, tier):
    out = {}
    p = f'{WS}/results/a4_{tag}/val_test_preds.tsv'
    if not os.path.exists(p): return None
    for r in csv.DictReader(open(p), delimiter='\t'):
        if r['tier'] != tier: continue
        out[(r['binary'], r['entry_addr'])] = (float(r['f1_v2']), float(r['conf']), r['binary'].split('__')[0], r['pred'])
    return out


def main():
    table = {}; preds = {}
    for name, tag in HEADS.items():
        ev = f'{WS}/results/a4_{tag}/val_test_eval.json'
        if not os.path.exists(ev): print('missing', name, ev); continue
        d = json.load(open(ev))['tiers']; te = d['test']; va = d.get('val', {})
        table[name] = {'tag': tag, 'val_f1': round(va.get('scored', {}).get('f1', float('nan')), 4), 'test_f1_fn': round(te['scored']['f1'], 4),
                       'test_f1_pkg': round(te['macro_pkg']['f1'], 4), 'test_em': round(te['scored']['em'], 4), 'n_test': te['scored']['n']}
        preds[name] = {'val': read_preds(tag, 'val'), 'test': read_preds(tag, 'test')}
        print(f"{name:38s} val {table[name]['val_f1']:.4f} | test fn {table[name]['test_f1_fn']:.4f} pkg {table[name]['test_f1_pkg']:.4f} EM {table[name]['test_em']:.4f} n {te['scored']['n']}")
    json.dump(table, open(f'{OUT}/punstrip_ctx_trained_comparison.json', 'w'), indent=1)
    with open(f'{OUT}/punstrip_ctx_results_table.csv', 'w') as fh:
        fh.write('context_source,lineagebench_f1_fn,lineagebench_f1_pkg,punstrip_f1_fn,punstrip_f1_pkg,punstrip_em\n')
        for name in HEADS:
            if name in table: fh.write(f"{name},{LB[name][0]},{LB[name][1]},{table[name]['test_f1_fn']},{table[name]['test_f1_pkg']},{table[name]['test_em']}\n")
    # bootstrap
    have = [n for n in HEADS if n in preds and preds[n]['test']]
    keys = set.intersection(*[set(preds[n]['test']) for n in have]) if have else set()
    def pk(n):
        g = defaultdict(list)
        for k in keys: g[preds[n]['test'][k][2]].append(preds[n]['test'][k][0])
        return {p: sum(v) / len(v) for p, v in g.items()}
    pkm = {n: pk(n) for n in have}; pkgs = sorted(next(iter(pkm.values()))) if pkm else []
    rng = random.Random(0); boot = {}
    pairs = [('P3 ±10 address (adopted)', 'P1 random (window-excluded, seed 1)'), ('P2 callers+callees', 'P1 random (window-excluded, seed 1)'),
             ('P3 ±10 address (adopted)', 'P2 callers+callees'), ('P4 address ∪ callers/callees', 'P3 ±10 address (adopted)'),
             ('P4 address ∪ callers/callees', 'P2 callers+callees'), ('P3 ±10 address (adopted)', 'P0 none'), ('P1 random (window-excluded, seed 1)', 'P0 none')]
    for a, b in pairs:
        if a not in pkm or b not in pkm: continue
        d = [pkm[a][p] - pkm[b][p] for p in pkgs]; bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
        fn = sum(preds[a]['test'][k][0] - preds[b]['test'][k][0] for k in keys) / len(keys)
        boot[f'{a} - {b}'] = {'delta_fn': round(fn, 4), 'delta_pkg': round(sum(d) / len(d), 4), 'ci95_pkg': [round(bs[250], 4), round(bs[9749], 4)],
                              'first_higher_packages': f'{sum(1 for x in d if x > 0)}/{len(d)}'}
        print(f"BOOT {a} - {b}: fn {fn:+.4f} pkg {sum(d)/len(d):+.4f} CI [{bs[250]:+.4f},{bs[9749]:+.4f}] first higher {sum(1 for x in d if x>0)}/{len(d)}")
    boot['n_functions'] = len(keys); boot['n_packages'] = len(pkgs)
    json.dump(boot, open(f'{OUT}/punstrip_ctx_bootstrap.json', 'w'), indent=1)
    # complementarity + confidence selection
    A, C, U = 'P3 ±10 address (adopted)', 'P2 callers+callees', 'P4 address ∪ callers/callees'
    if A in preds and C in preds and preds[A]['test'] and preds[C]['test']:
        comp = {}; sel = {}
        for tier in ('val', 'test'):
            X, Y = preds[A][tier], preds[C][tier]
            if not X or not Y: continue
            ks = sorted(set(X) & set(Y)); n = len(ks)
            same = sum(1 for k in ks if X[k][3] == Y[k][3]) / n
            a1 = sum(1 for k in ks if X[k][0] == 1) / n; c1 = sum(1 for k in ks if Y[k][0] == 1) / n; both = sum(1 for k in ks if X[k][0] == 1 and Y[k][0] == 1) / n
            orac = sum(max(X[k][0], Y[k][0]) for k in ks) / n
            pick = sum((X[k][0] if X[k][1] >= Y[k][1] else Y[k][0]) for k in ks) / n
            share_c = sum(1 for k in ks if Y[k][1] > X[k][1]) / n
            g = defaultdict(list)
            for k in ks: g[X[k][2]].append(X[k][0] if X[k][1] >= Y[k][1] else Y[k][0])
            pick_pkg = sum(sum(v) / len(v) for v in g.values()) / len(g)
            comp[tier] = {'n': n, 'same_prediction': round(same, 4), 'P3_exact': round(a1, 4), 'P2_exact': round(c1, 4), 'both_exact': round(both, 4), 'oracle_max_P2_P3': round(orac, 4)}
            sel[tier] = {'f1_fn': round(pick, 4), 'f1_pkg': round(pick_pkg, 4), 'share_P2': round(share_c, 4), 'share_P3': round(1 - share_c, 4)}
            if U in preds and preds[U][tier]:
                Z = preds[U][tier]; ks3 = [k for k in ks if k in Z]; m = len(ks3)
                comp[tier]['oracle_max_P2_P3_P4'] = round(sum(max(X[k][0], Y[k][0], Z[k][0]) for k in ks3) / m, 4)
                sel[tier]['f1_fn_P2_P3_P4'] = round(sum(max([(X[k][1], X[k][0]), (Y[k][1], Y[k][0]), (Z[k][1], Z[k][0])])[1] for k in ks3) / m, 4)
            print(f"COMP {tier}: same {same:.3f} P3 exact {a1:.3f} P2 exact {c1:.3f} both {both:.3f} oracle {orac:.4f} | pick fn {pick:.4f} pkg {pick_pkg:.4f} P2 share {share_c:.3f}")
        json.dump(comp, open(f'{OUT}/punstrip_ctx_complementarity.json', 'w'), indent=1)
        json.dump(sel, open(f'{OUT}/punstrip_ctx_confidence_select.json', 'w'), indent=1)
    # digest stats
    ds = {}
    for m in ('none', 'random_excl', 'callgraph', 'addrcall', 'win10'):
        p = f'{OUT}/{m}/stats.json'
        if os.path.exists(p): ds[m] = json.load(open(p))
    json.dump(ds, open(f'{OUT}/punstrip_ctx_digest_stats.json', 'w'), indent=1)
    print('EFFECT: punstrip diagnostics written to', OUT)


if __name__ == '__main__':
    main()
