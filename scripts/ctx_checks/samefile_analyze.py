#!/usr/bin/env python3
"""Matched-population mechanism table (follow-up plan §14, §15, §22, §9). Wulver, CPU, after the four matched heads are
scored. Population = results/ctx_layout/samefile_population.json (tier == test). Heads trained on the identical matched
training rows: nonem (no context), addrm (±10 address), sfo (same-source-file outside ±10), dff (different-file outside
±10); plus inference arms on the adopted head restricted to the population (random, callgraph swap) and the full-data
trained heads (address, callgraph, random, union) for reference.
Output: results/ctx_layout/samefile_analysis.json"""
import csv, json, os, random, re
from collections import Counter, defaultdict
WS = '/project/hz79/_shared/cs785/dh2'
SPLIT_RE1 = re.compile(r'([a-z0-9])([A-Z])'); SPLIT_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')


def subtoks(name):
    n = re.sub(r'\(.*\)$', '', name); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]; n = '_'.join(p for p in parts if p)
    s = SPLIT_RE1.sub(r'\1_\2', n); s = SPLIT_RE2.sub(r'\1_\2', s)
    return [t for t in re.split(r'[^A-Za-z0-9]+', s.lower()) if t]


def f1(a, b):
    ca, cb = Counter(a), Counter(b); tp = sum((ca & cb).values())
    if tp == 0: return 0.0
    p = tp / sum(ca.values()); r = tp / sum(cb.values()); return 2 * p * r / (p + r)


def read(path, tier='test'):
    out = {}
    if not os.path.exists(path): return None
    for r in csv.DictReader(open(path), delimiter='\t'):
        if r['tier'] != tier: continue
        out[f"{r['binary']}_{r['entry_addr']}"] = (float(r['f1_v2']), r['true'], r['pred'], float(r['conf']))
    return out


def mean(xs): return round(sum(xs) / len(xs), 4) if xs else None


def main():
    pop = {k: v for k, v in json.load(open(f'{WS}/results/ctx_layout/samefile_population.json')).items() if v['tier'] == 'test'}
    heads = {'NONE-M (no context, matched-trained)': f'{WS}/results/a4_codet5p220m_ctxnonem_dm_v1/val_test_preds.tsv',
             'DFF (different file, matched-trained)': f'{WS}/results/a4_codet5p220m_ctxdff_dm_v1/val_test_preds.tsv',
             'SFO (same file outside ±10, matched-trained)': f'{WS}/results/a4_codet5p220m_ctxsfo_dm_v1/val_test_preds.tsv',
             'ADDR-M (±10 address, matched-trained)': f'{WS}/results/a4_codet5p220m_ctxaddrm_dm_v1/val_test_preds.tsv',
             'random (inference arm, adopted head)': f'{WS}/results/a4_modctx_dm_sens_random/test_preds.tsv',
             'callers+callees (inference arm, adopted head)': f'{WS}/results/a4_modctx_dm_sens_callgraph/test_preds.tsv',
             'address (full-data head)': f'{WS}/results/a4_codet5p220m_modctx_dm_v1/val_test_preds.tsv',
             'callers+callees (full-data head)': f'{WS}/results/a4_codet5p220m_ctxcallgraph_dm_v1/val_test_preds.tsv',
             'union (full-data head)': f'{WS}/results/a4_codet5p220m_ctxaddrcall_dm_v1/val_test_preds.tsv',
             'none (full-data head)': f'{WS}/results/a4_codet5p220m_v1/val_test_symgen_holdout_preds.tsv'}
    P = {n: read(p) for n, p in heads.items()}; P = {n: v for n, v in P.items() if v}
    keys = [k for k in pop if all(k in v for v in P.values())]
    print(f"population: {len(keys)} test functions, {len({pop[k]['binary'] for k in keys})} binaries, {len({pop[k]['package'] for k in keys})} packages; heads: {list(P)}")
    table = {}
    for n, v in P.items():
        g = defaultdict(list); pre = []; rest = []
        for k in keys:
            g[pop[k]['package']].append(v[k][0]); gt = subtoks(v[k][1]); pd = subtoks(v[k][2])
            if gt:
                pre.append(1.0 if gt[0] in pd else 0.0)
                if len(gt) > 1: rest.append(f1(pd, gt[1:]))
        table[n] = {'f1_pkg': mean([sum(x) / len(x) for x in g.values()]), 'f1_fn': mean([v[k][0] for k in keys]), 'prefix_recall': mean(pre), 'remainder_f1': mean(rest)}
        print(f"T14 {n:48s} pkg {table[n]['f1_pkg']} fn {table[n]['f1_fn']} prefix {table[n]['prefix_recall']} remainder {table[n]['remainder_f1']}")
    pk = {n: defaultdict(list) for n in P}
    for k in keys:
        for n in P: pk[n][pop[k]['package']].append(P[n][k][0])
    pm = {n: {p: sum(x) / len(x) for p, x in d.items()} for n, d in pk.items()}; pkgs = sorted(next(iter(pm.values())))
    rng = random.Random(0); boot = {}
    S, D, A, N = 'SFO (same file outside ±10, matched-trained)', 'DFF (different file, matched-trained)', 'ADDR-M (±10 address, matched-trained)', 'NONE-M (no context, matched-trained)'
    for a, b in ((S, D), (S, 'random (inference arm, adopted head)'), (A, S), (A, D), (S, N), (D, N), (A, N)):
        if a not in pm or b not in pm: continue
        d = [pm[a][p] - pm[b][p] for p in pkgs]; bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
        boot[f'{a} - {b}'] = {'delta_pkg': round(sum(d) / len(d), 4), 'ci95': [round(bs[250], 4), round(bs[9749], 4)], 'first_higher': f'{sum(1 for x in d if x > 0)}/{len(d)}',
                              'delta_fn': round(sum(P[a][k][0] - P[b][k][0] for k in keys) / len(keys), 4)}
        print(f"BOOT {a} - {b}: pkg {sum(d)/len(d):+.4f} CI [{bs[250]:+.4f},{bs[9749]:+.4f}] first higher {sum(1 for x in d if x>0)}/{len(d)}")
    # locality bins within the population (matched ADDR-M vs NONE-M)
    bins = {'0%': [], '1-25%': [], '25-50%': [], '50-75%': [], '75-100%': []}
    if A in P and N in P:
        for k in keys:
            s = pop[k]['win_same_file'] / max(1, pop[k]['win_n'])
            b = '0%' if s == 0 else '1-25%' if s <= 0.25 else '25-50%' if s <= 0.5 else '50-75%' if s <= 0.75 else '75-100%'
            bins[b].append((P[A][k][0], P[N][k][0]))
    t16 = {b: {'n': len(v), 'address_f1': mean([x for x, _ in v]), 'none_f1': mean([y for _, y in v]), 'delta': mean([x - y for x, y in v])} for b, v in bins.items()}
    print('T16', {b: (d['n'], d['delta']) for b, d in t16.items()})
    json.dump({'n_functions': len(keys), 'n_packages': len(pkgs), 'T14': table, 'bootstrap': boot, 'T16_matched': t16}, open(f'{WS}/results/ctx_layout/samefile_analysis.json', 'w'), indent=1)
    print('EFFECT: samefile analysis written')


if __name__ == '__main__':
    main()
