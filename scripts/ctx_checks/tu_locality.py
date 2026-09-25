#!/usr/bin/env python3
"""Translation-unit locality of the address window vs. the call graph (advisor check, 2026-09-25).

For every scored test function whose binary has a debug ELF under --root, map every Ghidra function
address to its source file with addr2line (DWARF line table of the unstripped build), then measure
how much of the +-5 / +-10 / +-20 address window, and of the direct caller/callee set, lies in the
SAME source file as the target. A uniform-random source set is the baseline (expected same-file share
= (n_same_file - 1) / (n - 1)). Functions addr2line cannot place (no DWARF: statically linked library
code) count as 'unknown'. Aggregates by compiler, optimisation level and ELF type.

Usage: tu_locality.py --layout test_layout.json --root <dir where debug_elf paths resolve> --out out.json
Run once per machine that holds a share of the debug ELFs, then merge with --merge a.json b.json.
"""
import argparse, json, os, subprocess, sys
from collections import defaultdict

WINDOWS = (5, 10, 20)


def addr2file(elf, addrs):
    inp = '\n'.join(f'0x{int(a, 16):x}' for a in addrs)
    r = subprocess.run(['addr2line', '-e', elf, '-a'], input=inp, capture_output=True, text=True)
    files = {}
    cur = None
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith('0x'):
            cur = s; continue
        if cur is None: continue
        f = s.split(':')[0]
        files[int(cur, 16)] = None if f.startswith('?') else os.path.normpath(f)
        cur = None
    return [files.get(int(a, 16)) for a in addrs]


def share(files, i, src):
    """(same, other, unknown) shares of src relative to target i."""
    if not src: return None
    t = files[i]
    if t is None: return None
    same = sum(1 for j in src if files[j] == t); unk = sum(1 for j in src if files[j] is None)
    n = len(src)
    return (same / n, (n - same - unk) / n, unk / n)


def analyse(layout, root):
    per_row = []; files_map = {}
    n_bins = 0; skipped = defaultdict(int)
    for b, L in layout.items():
        elf = L['debug_elf'] or ''
        path = elf if os.path.isabs(elf) else os.path.join(root, elf)
        if not os.path.exists(path): skipped['no_debug_elf'] += 1; continue
        if not L['rows']: skipped['no_rows'] += 1; continue
        addrs = L['addrs']; n = len(addrs); idx = {a: i for i, a in enumerate(addrs)}
        files = addr2file(path, addrs)
        if sum(f is not None for f in files) < 0.2 * n: skipped['no_dwarf'] += 1; continue
        n_bins += 1; files_map[b] = files
        comp = 'clang' if 'clang' in b else 'gcc'
        opt = 'O' + b.rsplit('_O', 1)[-1][:1] if '_O' in b else '?'
        callers = defaultdict(set)
        for i, cs in enumerate(L['callees']):
            for j in cs: callers[j].add(i)
        by_file = defaultdict(int)
        for f in files:
            if f is not None: by_file[f] += 1
        for r in L['rows']:
            i = idx.get(r['addr'])
            if i is None: skipped['row_addr_missing'] += 1; continue
            t = files[i]
            if t is None: skipped['row_no_dwarf'] += 1; continue
            rec = {'binary': b, 'addr': r['addr'], 'package': L['package'], 'compiler': comp, 'opt': opt, 'elf_type': L['elf_type'],
                   'regime': r.get('regime'), 'seen': r.get('seen'), 'n_fns': n, 'n_same_file': by_file[t]}
            for k in WINDOWS:
                src = [j for j in range(max(0, i - k), min(n, i + 1 + k)) if j != i]
                rec[f'win{k}'] = share(files, i, src)
            cg = sorted(set(L['callees'][i]) | callers[i])[:40]
            rec['callgraph'] = share(files, i, cg); rec['callgraph_n'] = len(cg)
            rec['random_expected'] = (by_file[t] - 1) / (n - 1) if n > 1 else 0.0
            per_row.append(rec)
    return per_row, n_bins, dict(skipped), files_map


def aggregate(rows):
    def mean(xs): return round(sum(xs) / len(xs), 4) if xs else None
    def summ(sub):
        o = {'n_rows': len(sub), 'n_binaries': len({r['binary'] for r in sub}), 'n_packages': len({r['package'] for r in sub})}
        for key in [f'win{k}' for k in WINDOWS] + ['callgraph']:
            v = [r[key] for r in sub if r[key] is not None]
            o[key] = {'same_file': mean([x[0] for x in v]), 'other_file': mean([x[1] for x in v]), 'unknown': mean([x[2] for x in v]),
                      'frac_majority_same': mean([1.0 if x[0] >= 0.5 else 0.0 for x in v]),
                      'frac_any_same': mean([1.0 if x[0] > 0 else 0.0 for x in v]), 'n': len(v)}
        o['callgraph_mean_n'] = mean([r['callgraph_n'] for r in sub])
        o['random_expected_same_file'] = mean([r['random_expected'] for r in sub])
        return o
    out = {'all': summ(rows), 'by_compiler': {}, 'by_opt': {}, 'by_elf_type': {}, 'by_regime': {}, 'by_compiler_opt': {}}
    for key, field in (('by_compiler', 'compiler'), ('by_opt', 'opt'), ('by_elf_type', 'elf_type'), ('by_regime', 'regime')):
        for v in sorted({r[field] for r in rows}, key=str):
            out[key][str(v)] = summ([r for r in rows if r[field] == v])
    for c in ('gcc', 'clang'):
        for o in ('O0', 'O1', 'O2', 'O3'):
            sub = [r for r in rows if r['compiler'] == c and r['opt'] == o]
            if sub: out['by_compiler_opt'][f'{c}_{o}'] = summ(sub)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layout'); ap.add_argument('--root', default='.'); ap.add_argument('--out', required=True)
    ap.add_argument('--merge', nargs='*', default=[], help='per-machine row dumps to merge instead of analysing')
    args = ap.parse_args()
    if args.merge:
        rows = []
        for p in args.merge: rows += json.load(open(p))['rows']
        seen = set(); uniq = []
        for r in rows:
            k = (r['binary'], r['addr'])
            if k in seen: continue
            seen.add(k); uniq.append(r)
        agg = aggregate(uniq)
        json.dump({'rows_n': len(uniq), 'aggregate': agg}, open(args.out, 'w'), indent=1)
        print(f"EFFECT: merged {len(uniq)} rows from {len(args.merge)} dumps: "
              f"{agg['all']['n_binaries']} binaries, {agg['all']['n_packages']} packages; "
              f"win10 same-file {agg['all']['win10']['same_file']} | callgraph {agg['all']['callgraph']['same_file']} | random {agg['all']['random_expected_same_file']}")
        return
    layout = json.load(open(args.layout))
    rows, n_bins, skipped, files_map = analyse(layout, args.root)
    agg = aggregate(rows)
    json.dump({'rows': rows, 'n_binaries': n_bins, 'skipped': skipped, 'aggregate': agg}, open(args.out, 'w'))
    json.dump(files_map, open(args.out.replace('.json', '') + '_files.json', 'w'))
    a = agg['all']
    print(f"EFFECT: tu_locality: {n_bins} binaries analysed, {len(rows)} rows, skipped {skipped}")
    if rows:
        print(f"  same-file share: win5 {a['win5']['same_file']} win10 {a['win10']['same_file']} win20 {a['win20']['same_file']} "
              f"| callgraph {a['callgraph']['same_file']} (mean {a['callgraph_mean_n']} fns) | random-expected {a['random_expected_same_file']}")
        for k, v in agg['by_compiler_opt'].items():
            print(f"  {k:9s} n={v['n_rows']:6d} win10 same {v['win10']['same_file']} majority {v['win10']['frac_majority_same']} | cg same {v['callgraph']['same_file']}")


if __name__ == '__main__':
    main()
