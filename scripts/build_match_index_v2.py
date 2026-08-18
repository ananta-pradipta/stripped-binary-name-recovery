#!/usr/bin/env python3
"""Matcher v2 (defect B3a/B6/B9/B10): join parse_bir_v3 graphs with labels_v2 BY ADDRESS.

Input : data/graphs_v3/<id>.index.json (+ per-function graphs), data/labels_v2/<id>.json
Output: data/match_index_v2.json  — list of records, one per KEPT (graph, label) pair:
          {binary, graph, bap_name, name_kind, entry_addr, real_name, aliases, binding,
           in_dynsym, size, match_kind (exact|plus4), n_tokens, num_blocks, tok_hash,
           n_import_calls, corpus}
        data/match_index_v2_report.tsv — per-binary: labels, graphs, matched, exact, plus4,
          unmatched_graphs, unmatched_labels, thunk_dups_dropped, named_mismatch, cov
Policy:
  * a graph matches a label if entry_addr == label.addr, else if entry_addr == label.addr+4
    (BAP started after ENDBR64).
  * several labels at one address are ALIASES; canonical real_name = best by
    (binding GLOBAL < WEAK < LOCAL, no leading underscore, shorter, lexicographic);
    all aliases kept — evaluation should score against the best alias.
  * several graphs matching one label (thunk sub_X + body sub_X+4): keep the graph with the
    most tokens, drop the rest as thunk duplicates (old corpus had 58K of these).
  * `named` graphs (BAP named the sub from .dynsym) must agree with a label alias at that
    address; disagreement is reported (BAP mis-naming, old defect B6) and the label wins.
  * crt junk / section subs / stubs never reach here (parser drops them).
"""
import argparse, csv, hashlib, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

GRAPHS = 'data/graphs_v3'; LABELS = 'data/labels_v2'
BIND_RANK = {'GLOBAL': 0, 'WEAK': 1, 'LOCAL': 2}


def canonical(aliases):
    """aliases: list of (name, binding). Deterministic best."""
    return sorted(aliases, key=lambda a: (BIND_RANK.get(a[1], 3), a[0].startswith('_'), len(a[0]), a[0]))[0][0]


def process(bid):
    idx_p = os.path.join(GRAPHS, bid + '.index.json'); lab_p = os.path.join(LABELS, bid + '.json')
    if not (os.path.exists(idx_p) and os.path.exists(lab_p)):
        return None, None
    idx = json.load(open(idx_p)); lab = json.load(open(lab_p))
    corpus = lab.get('corpus', '')
    by_addr = {}
    for n, f in lab['functions'].items():
        by_addr.setdefault(int(f['addr'], 16), []).append((n, f))
    # candidate matches
    cands = []   # (label_addr, graph_rec, match_kind)
    unmatched_graphs = 0
    for g in idx['functions']:
        if not g['entry_addr']:
            unmatched_graphs += 1; continue
        a = int(g['entry_addr'], 16)
        if a in by_addr:
            cands.append((a, g, 'exact'))
        elif a - 4 in by_addr:
            cands.append((a - 4, g, 'plus4'))
        else:
            unmatched_graphs += 1
    # dedup: one graph per label address (max tokens; tie -> exact over plus4, then name)
    best = {}; per_addr = {}
    for a, g, k in cands:
        per_addr.setdefault(a, []).append(g['name'])
        key = (-g['n_tokens'], 0 if k == 'exact' else 1, g['name'])
        if a not in best or key < best[a][0]:
            best[a] = (key, g, k)
    thunk_dups = len(cands) - len(best)
    recs = []; named_mismatch = 0
    for a, (_, g, k) in sorted(best.items()):
        aliases = by_addr[a]
        names = [(n, f['binding']) for n, f in aliases]
        cname = canonical(names)
        cf = dict(aliases)[cname]
        if g['name_kind'] == 'named' and g['name'] not in {n for n, _ in names}:
            named_mismatch += 1
        gpath = os.path.join(GRAPHS, g['file'])
        gd = json.load(open(gpath))
        toks = '|'.join(t for b in gd['blocks'] for t in b['tokens'])
        recs.append({
            'binary': bid, 'corpus': corpus, 'graph': gpath, 'bap_name': g['name'], 'name_kind': g['name_kind'],
            'entry_addr': g['entry_addr'], 'label_addr': '0x%x' % a, 'match_kind': k,
            'real_name': cname, 'aliases': sorted(n for n, _ in names if n != cname),
            'binding': cf['binding'], 'in_dynsym': bool(cf['in_dynsym']) or any(f['in_dynsym'] for _, f in aliases),
            'size': cf['size'], 'n_tokens': g['n_tokens'], 'num_blocks': g['num_blocks'],
            'n_import_calls': g['n_import_calls'], 'tok_hash': hashlib.md5(toks.encode()).hexdigest(),
            'dropped_graphs': sorted(n for n in per_addr[a] if n != g['name']),  # thunk aliases -> resolve callee refs
        })
    n_lab = len(by_addr)
    rep = {'binary': bid, 'corpus': corpus, 'labels': n_lab, 'graphs': idx['n_functions'], 'matched': len(recs),
           'exact': sum(1 for r in recs if r['match_kind'] == 'exact'),
           'plus4': sum(1 for r in recs if r['match_kind'] == 'plus4'),
           'unmatched_graphs': unmatched_graphs, 'unmatched_labels': n_lab - len(recs),
           'thunk_dups_dropped': thunk_dups, 'named_mismatch': named_mismatch,
           'in_dynsym': sum(1 for r in recs if r['in_dynsym']),
           'cov': round(len(recs) / max(1, n_lab), 4)}
    return recs, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/match_index_v2.json')
    ap.add_argument('--report', default='data/match_index_v2_report.tsv')
    ap.add_argument('--ids', nargs='*')
    ap.add_argument('--workers', type=int, default=1)  # per-record graph reads dominate; parallelism is I/O relief on GPFS
    args = ap.parse_args()
    ids = sorted(f[:-11] for f in os.listdir(GRAPHS) if f.endswith('.index.json'))
    if args.ids:
        ids = [i for i in ids if i in set(args.ids)]
    all_recs = []; reps = []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pairs = {}
        with ProcessPoolExecutor(args.workers) as ex:
            for k, (bid, (recs, rep)) in enumerate(zip(ids, ex.map(process, ids, chunksize=4)), 1):
                if recs is not None:
                    pairs[bid] = (recs, rep)
                if k % 50 == 0:
                    print(f'{k}/{len(ids)} binaries', flush=True)
        for bid in ids:  # deterministic order regardless of worker scheduling
            if bid in pairs:
                all_recs.extend(pairs[bid][0]); reps.append(pairs[bid][1])
    else:
        for k, bid in enumerate(ids, 1):
            recs, rep = process(bid)
            if recs is None:
                continue
            all_recs.extend(recs); reps.append(rep)
            if k % 50 == 0:
                print(f'{k}/{len(ids)} binaries, {len(all_recs)} records', flush=True)
    json.dump(all_recs, open(args.out, 'w'))
    with open(args.report, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(reps[0].keys()), delimiter='\t'); w.writeheader(); w.writerows(reps)
    covs = sorted(r['cov'] for r in reps)
    med = covs[len(covs) // 2] if covs else 0
    low = [(r['binary'], r['cov']) for r in reps if r['cov'] < 0.9]
    tot_lab = sum(r['labels'] for r in reps)
    print(f"\n{len(reps)} binaries, {tot_lab} labels, {len(all_recs)} matched records "
          f"({sum(r['plus4'] for r in reps)} plus4, {sum(r['thunk_dups_dropped'] for r in reps)} thunk dups dropped, "
          f"{sum(r['named_mismatch'] for r in reps)} named mismatches, {sum(r['in_dynsym'] for r in reps)} in_dynsym)")
    print(f"EFFECT CHECK: median per-binary label coverage {med:.4f}; <0.90: {len(low)} {sorted(low, key=lambda x: x[1])[:12]}")
    if med < 0.95:
        sys.exit(1)


if __name__ == '__main__':
    main()
