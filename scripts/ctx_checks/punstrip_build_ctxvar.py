#!/usr/bin/env python3
"""Punstrip counterpart of a4_build_ctxvar.py (advisor checks, 2026-09-25): same digest recipe as
punstrip/scripts/punstrip_build_modctx.py (K=10 address neighbours among the decompiled functions, TOP=40
tokens from string literals + named calls, ranked by distinct-neighbour count then total count, comment
prefix '/* module context: ... */', ALL-occurrence masking, raw-name targets), with the SOURCE SET swapped:
win5 / win10 / win20 / callgraph (direct callees + callers, cap 40) / random (20 uniform, seeded) / empty.
win10 must reproduce punstrip/data/<split>.jsonl byte-for-byte (checked; exit 3 otherwise).
Also writes the per-binary layout (addresses, callees, scored rows, debug ELF path) for tu_locality.py.
Output: punstrip/data_ctx/<mode>/<split>.jsonl (+ stats.json), punstrip/data_ctx/layout_<split>.json
"""
import argparse, hashlib, json, os, random, re, sys
from collections import Counter, defaultdict

P = '/project/hz79/_shared/cs785/punstrip'
TOP = 40; CALL_FN_CAP = 40; RANDOM_N = 20
STR_RE = re.compile(r'"((?:[^"\\\n]|\\.){3,200})"')
CALL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(')
REF_RE = re.compile(r'\b(?:FUN_|sub_)([0-9a-fA-F]{4,})')
GNAME_RE = re.compile(r'([0-9a-fA-F]+)$')
SPLIT_RE1 = re.compile(r'([a-z0-9])([A-Z])'); SPLIT_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
STOP = set('''if else while for return switch case break continue goto sizeof do
void int char long short float double unsigned signed bool true false null nullptr
uint ulong ushort byte undefined undefined1 undefined2 undefined4 undefined8 code
param local stack var unaff extraout in out ram fun sub ptr concat sext zext
memcpy memset strlen strcmp strcpy strncpy strncmp malloc calloc realloc free printf fprintf sprintf
the and for with not this that from func file line error warning failed invalid cannot could unable
halt baddata warning'''.split())
MODE_SPEC = {'win5': ('window', 5), 'win10': ('window', 10), 'win20': ('window', 20),
             'callgraph': ('callgraph', CALL_FN_CAP), 'random': ('random', RANDOM_N), 'empty': ('empty', 0),
             'addrcall': ('addrcall', 10), 'random_excl': ('random_excl', RANDOM_N)}


def toks(s):
    s = SPLIT_RE1.sub(r'\1_\2', s); s = SPLIT_RE2.sub(r'\1_\2', s)
    return [t for t in re.split(r'[^A-Za-z0-9]+', s.lower()) if 2 < len(t) < 25 and not t.isdigit() and t not in STOP]


def fn_evidence(code):
    ev = Counter()
    for m in STR_RE.findall(code): ev.update(toks(m))
    for c in CALL_RE.findall(code):
        if not c.startswith(('FUN_', 'sub_', '_')) and c.lower() not in STOP: ev.update(toks(c))
    return ev


def rank(ev, src):
    df = Counter(); tot = Counter()
    for j in src:
        for t in ev[j]: df[t] += 1; tot[t] += ev[j][t]
    return ' '.join(sorted(df, key=lambda t: (-df[t], -tot[t], t))[:TOP])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', default='win10,win5,win20,callgraph,random,empty')
    ap.add_argument('--splits', default='test')
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--limit-bins', type=int, default=None)
    ap.add_argument('--out-root', default=f'{P}/data_ctx')
    ap.add_argument('--suffix', default='', help='output dir suffix, e.g. _s1 for a second random seed')
    args = ap.parse_args()
    modes = args.modes.split(','); splits = args.splits.split(',')
    ref = {}
    if 'win10' in modes:
        for sp in splits:
            for line in open(f'{P}/data/{sp}.jsonl'):
                r = json.loads(line); ref[r['key']] = r['code']
    fhs = {}
    for m in modes:
        os.makedirs(f'{args.out_root}/{m}{args.suffix}', exist_ok=True)
        for sp in splits: fhs[(m, sp)] = open(f'{args.out_root}/{m}{args.suffix}/{sp}.jsonl', 'w')
    stats = {m: {} for m in modes}; repro = {'checked': 0, 'mismatch': 0}
    for sp in splits:
        rows_by_bin = defaultdict(list); n_in = 0
        for line in open(f'{P}/rows/{sp}.jsonl'):
            r = json.loads(line); rows_by_bin[r['binary']].append(r); n_in += 1
        bins = list(rows_by_bin)
        if args.limit_bins: bins = bins[:args.limit_bins]
        acc = {m: {'rows_out': 0, 'empty_ctx': 0, 'src': 0, 'with_src': 0, 'skipped': Counter()} for m in modes}
        layout = {}
        for b in bins:
            rs = rows_by_bin[b]; p = f'{P}/decomp/json/{b}.json'
            if not os.path.exists(p):
                for m in modes: acc[m]['skipped']['no_decomp_file'] += len(rs)
                continue
            dec = json.load(open(p))
            addrs = sorted(dec.keys(), key=lambda a: int(a, 16)); idx = {a: i for i, a in enumerate(addrs)}
            ev = [fn_evidence(dec[a].get('code', '')) for a in addrs]
            g2i = {}
            for a in addrs:
                mm = GNAME_RE.search(dec[a].get('ghidra_name', '') or '')
                if mm: g2i[int(mm.group(1), 16)] = idx[a]
            callees = defaultdict(set); callers = defaultdict(set)
            for a in addrs:
                i = idx[a]
                for h in REF_RE.findall(dec[a].get('code', '')):
                    j = g2i.get(int(h, 16))
                    if j is not None and j != i: callees[i].add(j); callers[j].add(i)
            n = len(addrs)
            layout[b] = {'package': rs[0]['package'], 'debug_elf': rs[0]['binpath'].lstrip('/') + '.debug', 'elf_type': None,
                         'addrs': addrs, 'callees': [sorted(callees[i]) for i in range(n)],
                         'rows': [{'addr': r['entry_addr'], 'name': r['name'], 'regime': r.get('regime'), 'seen': r.get('name_seen_in_train')} for r in rs]}
            for r in rs:
                a = r['entry_addr']; e = dec.get(a)
                if not e or 'code' not in e:
                    for m in modes: acc[m]['skipped']['addr_missing_or_failed'] += 1
                    continue
                code = e['code'].replace(e['ghidra_name'], '[MASK]')  # ALL occurrences, as in punstrip_build_modctx.py
                if '[MASK]' not in code:
                    for m in modes: acc[m]['skipped']['mask_not_applied'] += 1
                    continue
                i = idx[a]
                for m in modes:
                    kind, par = MODE_SPEC[m]
                    if kind == 'window': src = [j for j in range(max(0, i - par), min(n, i + 1 + par)) if j != i]
                    elif kind == 'callgraph': src = sorted(callees[i] | callers[i])[:par]
                    elif kind == 'addrcall':
                        win = [j for j in range(max(0, i - par), min(n, i + 1 + par)) if j != i]
                        src = sorted(set(win) | set(sorted(callees[i] | callers[i])[:CALL_FN_CAP]))
                    elif kind in ('random', 'random_excl'):
                        seed = int(hashlib.sha1(f'{b}:{a}:{args.seed}'.encode()).hexdigest()[:12], 16)
                        excl = set(range(max(0, i - 10), min(n, i + 11))) if kind == 'random_excl' else {i}
                        pool = [j for j in range(n) if j not in excl] or [j for j in range(n) if j != i]
                        src = random.Random(seed).sample(pool, min(par, len(pool)))
                    else: src = []
                    ctx = rank(ev, src) if src else ''
                    text = f'/* module context: {ctx} */\n{code}'
                    nii = len(r['name']) >= 4 and re.search(r'(?<![A-Za-z0-9_])' + re.escape(r['name']) + r'(?![A-Za-z0-9_])', text) is not None
                    row = {'key': f"{b}_{a}", 'binary': b, 'binpath': r['binpath'], 'package': r['package'], 'addr': a,
                           'name': r['name'], 'code': text, 'regime': r['regime'], 'name_seen': r['name_seen_in_train'],
                           'dynsym_visible': r['in_dynsym'], 'name_in_input': nii}
                    if m == 'win10' and ref:
                        rr = ref.get(row['key'])
                        if rr is not None:
                            repro['checked'] += 1
                            if rr != text: repro['mismatch'] += 1
                    fhs[(m, sp)].write(json.dumps(row) + '\n')
                    acc[m]['rows_out'] += 1; acc[m]['empty_ctx'] += (not ctx); acc[m]['src'] += len(src); acc[m]['with_src'] += bool(src)
        json.dump(layout, open(f'{args.out_root}/layout_{sp}{args.suffix}.json', 'w'))
        for m in modes:
            a_ = acc[m]; k = max(1, a_['rows_out'])
            st = {'rows_in': n_in, 'rows_out': a_['rows_out'], 'skipped': dict(a_['skipped']), 'empty_ctx': a_['empty_ctx'],
                  'mean_src_fns': round(a_['src'] / k, 2), 'frac_with_src': round(a_['with_src'] / k, 4)}
            stats[m][sp] = st
            print(f"EFFECT: punstrip ctx {m} {sp}: {st['rows_out']}/{n_in} rows, empty_ctx {st['empty_ctx']}, mean_src_fns {st['mean_src_fns']}, "
                  f"frac_with_src {st['frac_with_src']}, skipped={st['skipped']}", flush=True)
        print(f"EFFECT: layout_{sp}: {len(layout)} binaries", flush=True)
    for fh in fhs.values(): fh.close()
    for m in modes: json.dump(stats[m], open(f'{args.out_root}/{m}{args.suffix}/stats.json', 'w'), indent=1)
    if 'win10' in modes:
        print(f"EFFECT: win10 reproduction vs punstrip/data: checked {repro['checked']} mismatch {repro['mismatch']}", flush=True)
        if repro['checked'] and repro['mismatch']: print('FATAL: win10 does not reproduce punstrip/data'); sys.exit(3)
    rc = 0
    if 'callgraph' in modes and any(stats['callgraph'][sp]['frac_with_src'] < 0.5 for sp in splits):
        print('FATAL: callgraph resolved for <50% of rows'); rc = 2
    sys.exit(rc)


if __name__ == '__main__':
    main()
