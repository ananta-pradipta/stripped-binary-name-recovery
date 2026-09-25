#!/usr/bin/env python3
"""Context-variant builder for the generation head (advisor checks, 2026-09-25).

Same digest recipe as a4_build_modctx.py (identifier tokens from string literals + named calls of a
SOURCE SET of functions in the same stripped binary, ranked by number of distinct source functions
containing them, then total count, TOP=40, same comment prefix, same first-occurrence masking), but
the source set differs per mode:

  win5 / win10 / win20 : the K address-adjacent functions on either side (win10 == modctx; used as a
                         reproduction check against results/a4_modctx)
  callgraph            : direct callees + callers of the target (Ghidra FUN_ references, both
                         directions, cap 40 functions, address order)
  random               : N=20 functions sampled uniformly from the same binary, excluding the target,
                         deterministic per (binary, addr, seed)
  empty                : no context tokens (control for the comment prefix itself)

Call references are resolved through each function's ghidra_name address (Ghidra's image base and the
BAP +4 entry offset differ per binary) — a4_build_poolctx.py resolved them by string-matching the row
key and silently found 0 callees on PIE (DYN) binaries.

Targets are canonized exactly as scripts/a4_build_modctx_dm.py does (c++filt, strip args/templates,
last two :: qualifiers), so every output dir is directly trainable/predictable with the adopted dm
recipe. Output: results/a4_ctx_<mode>_dm/{tier}.jsonl (+ stats.json).
"""
import argparse, hashlib, json, os, random, re, subprocess, sys
from collections import Counter, defaultdict

WS = '/project/hz79/_shared/cs785/dh2'
PROTO = f'{WS}/results/baseline_protocol_v2'
TOP = 40
CALL_FN_CAP = 40
RANDOM_N = 20

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


def toks(s):
    s = SPLIT_RE1.sub(r'\1_\2', s); s = SPLIT_RE2.sub(r'\1_\2', s)
    return [t for t in re.split(r'[^A-Za-z0-9]+', s.lower()) if 2 < len(t) < 25 and not t.isdigit() and t not in STOP]


def fn_evidence(code):
    ev = Counter()
    for m in STR_RE.findall(code): ev.update(toks(m))
    for c in CALL_RE.findall(code):
        if not c.startswith(('FUN_', 'sub_', '_')) and c.lower() not in STOP: ev.update(toks(c))
    return ev


def rank(ev, src_idx):
    df = Counter(); tot = Counter()
    for j in src_idx:
        for t in ev[j]: df[t] += 1; tot[t] += ev[j][t]
    return ' '.join(sorted(df, key=lambda t: (-df[t], -tot[t], t))[:TOP])


# ---- target canonization (verbatim from a4_build_modctx_dm.py)
def demangle_many(names):
    todo = sorted({n for n in names if n and n.startswith('_Z')}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]; r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out


def canon(n, dem):
    n = dem.get(n, n); n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]; return '_'.join(p for p in parts if p)


MODE_SPEC = {'win5': ('window', 5), 'win10': ('window', 10), 'win20': ('window', 20),
             'callgraph': ('callgraph', CALL_FN_CAP), 'random': ('random', RANDOM_N), 'empty': ('empty', 0),
             'addrcall': ('addrcall', 10), 'random_excl': ('random_excl', RANDOM_N)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', default='win5,win20,callgraph,random,empty')
    ap.add_argument('--tiers', default='train,val,test')
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--limit-bins', type=int, default=None, help='smoke: only this many binaries per tier')
    ap.add_argument('--out-root', default=f'{WS}/results')
    ap.add_argument('--suffix', default='', help='output dir suffix, e.g. _s1 for a second random seed')
    ap.add_argument('--check-modctx', default=f'{WS}/results/a4_modctx', help='if win10 is built, its digests must equal this dir (reproduction check)')
    args = ap.parse_args()
    modes = args.modes.split(','); tiers = args.tiers.split(',')
    for m in modes: assert m in MODE_SPEC, m
    which = subprocess.run(['which', 'c++filt'], capture_output=True).returncode
    if which != 0: print('FATAL: no c++filt'); sys.exit(1)

    stats = {m: {} for m in modes}
    fhs = {}
    for m in modes:
        od = f'{args.out_root}/a4_ctx_{m}{args.suffix}_dm'; os.makedirs(od, exist_ok=True)
        for t in tiers: fhs[(m, t)] = open(f'{od}/{t}.jsonl', 'w')
    repro = {'checked': 0, 'mismatch': 0}
    ref_modctx = {}
    if 'win10' in modes and args.check_modctx:
        for t in tiers:
            p = f'{args.check_modctx}/{t}.jsonl'
            if os.path.exists(p):
                for line in open(p):
                    r = json.loads(line); ref_modctx[r['key']] = r['code']

    for tier in tiers:
        rows_by_bin = defaultdict(list); n_in = 0
        for line in open(f'{PROTO}/{tier}.jsonl'):
            r = json.loads(line); rows_by_bin[r['binary']].append(r); n_in += 1
        bins = list(rows_by_bin)
        if args.limit_bins: bins = bins[:args.limit_bins]
        acc = {m: {'rows_out': 0, 'empty_ctx': 0, 'src_sizes': 0, 'resolved_any': 0, 'skipped': Counter(), 'canonized': 0} for m in modes}
        all_names = []
        for b in bins:
            all_names += [r['name'] for r in rows_by_bin[b]]
        dem = demangle_many(all_names)
        for b in bins:
            rs = rows_by_bin[b]
            p = f'{WS}/symgen_v2/decomp/{b}.json'
            if not os.path.exists(p):
                for m in modes: acc[m]['skipped']['no_decomp_file'] += len(rs)
                continue
            dec = json.load(open(p))
            addrs = sorted(dec.keys(), key=lambda a: int(a, 16))
            idx = {a: i for i, a in enumerate(addrs)}
            ev = [fn_evidence(dec[a].get('code', '')) for a in addrs]
            # call graph through ghidra_name addresses
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
            for r in rs:
                a = r['entry_addr']; e = dec.get(a)
                if not e or 'code' not in e:
                    for m in modes: acc[m]['skipped']['addr_missing_or_failed'] += 1
                    continue
                code = e['code'].replace(e['ghidra_name'], '[MASK]', 1)
                if '[MASK]' not in code:
                    for m in modes: acc[m]['skipped']['mask_not_applied'] += 1
                    continue
                i = idx[a]
                cname = canon(r['name'], dem)
                for m in modes:
                    kind, par = MODE_SPEC[m]
                    if kind == 'window':
                        src = [j for j in range(max(0, i - par), min(n, i + 1 + par)) if j != i]
                    elif kind == 'callgraph':
                        src = sorted(callees[i] | callers[i])[:par]
                    elif kind == 'addrcall':
                        win = [j for j in range(max(0, i - par), min(n, i + 1 + par)) if j != i]
                        src = sorted(set(win) | set(sorted(callees[i] | callers[i])[:CALL_FN_CAP]))
                    elif kind in ('random', 'random_excl'):
                        seed = int(hashlib.sha1(f'{b}:{a}:{args.seed}'.encode()).hexdigest()[:12], 16)
                        excl = set(range(max(0, i - 10), min(n, i + 11))) if kind == 'random_excl' else {i}
                        pool = [j for j in range(n) if j not in excl] or [j for j in range(n) if j != i]
                        src = random.Random(seed).sample(pool, min(par, len(pool)))
                    else:
                        src = []
                    ctx = rank(ev, src) if src else ''
                    if not ctx: acc[m]['empty_ctx'] += 1
                    acc[m]['src_sizes'] += len(src); acc[m]['resolved_any'] += (1 if src else 0)
                    text = f'/* module context: {ctx} */\n{code}'
                    if m == 'win10' and ref_modctx:
                        ref = ref_modctx.get(f"{b}_{a}")
                        if ref is not None:
                            repro['checked'] += 1
                            if ref != text: repro['mismatch'] += 1
                    row = {'key': f"{b}_{a}", 'binary': b, 'package': r['package'], 'addr': a, 'name': cname, 'code': text}
                    if cname != r['name']: acc[m]['canonized'] += 1
                    if 'regime' in r: row['regime'] = r['regime']
                    if 'name_seen_in_train' in r: row['name_seen_in_train'] = r['name_seen_in_train']
                    if 'in_dynsym' in r: row['in_dynsym'] = r['in_dynsym']
                    fhs[(m, tier)].write(json.dumps(row) + '\n'); acc[m]['rows_out'] += 1
        for m in modes:
            a_ = acc[m]; k = max(1, a_['rows_out'])
            st = {'rows_in': n_in, 'rows_out': a_['rows_out'], 'skipped': dict(a_['skipped']), 'empty_ctx': a_['empty_ctx'],
                  'mean_src_fns': round(a_['src_sizes'] / k, 2), 'frac_with_src': round(a_['resolved_any'] / k, 4),
                  'targets_canonized': a_['canonized']}
            stats[m][tier] = st
            print(f"EFFECT: a4_ctx_{m} {tier}: {st['rows_out']}/{n_in} rows, empty_ctx {st['empty_ctx']}, "
                  f"mean_src_fns {st['mean_src_fns']}, frac_with_src {st['frac_with_src']}, canonized {st['targets_canonized']}, "
                  f"skipped={st['skipped']}", flush=True)
    for fh in fhs.values(): fh.close()
    for m in modes:
        json.dump(stats[m], open(f'{args.out_root}/a4_ctx_{m}{args.suffix}_dm/stats.json', 'w'), indent=1)
    if 'win10' in modes:
        print(f"EFFECT: win10 reproduction vs a4_modctx: checked {repro['checked']} mismatch {repro['mismatch']}", flush=True)
        if repro['checked'] and repro['mismatch']:
            print('FATAL: win10 does not reproduce the adopted modctx digests'); sys.exit(3)
    # tripwires (effect, not configuration)
    rc = 0
    if 'callgraph' in modes:
        for t in tiers:
            f = stats['callgraph'][t]['frac_with_src']
            if f < 0.5: print(f'FATAL: callgraph {t}: only {f:.3f} of rows have a resolved caller/callee'); rc = 2
    if 'random' in modes:
        for t in tiers:
            if stats['random'][t]['empty_ctx'] > 0.05 * max(1, stats['random'][t]['rows_out']):
                print(f"FATAL: random {t}: {stats['random'][t]['empty_ctx']} empty digests"); rc = 2
    for m in modes:
        for l, _ in zip(open(f'{args.out_root}/a4_ctx_{m}{args.suffix}_dm/{tiers[-1]}.jsonl'), range(1)):
            r = json.loads(l); print(f'SAMPLE {m}', r['name'], '|', r['code'][:300].replace('\n', ' '))
    sys.exit(rc)


if __name__ == '__main__':
    main()
