#!/usr/bin/env python3
"""BAP-text control for the Ghidra-vs-BAP ablation: linearise exactly what the BAP encoder sees (V3 instruction-type
tokens per block + CFG edges, PLT calls, referenced .rodata strings, token signatures of <=5 callees / <=5 callers)
into text, so the SAME CodeT5+ 220M recipe (scripts/a4_train_codet5p.py --data-dir results/a4_baptext) can be
fine-tuned on it. Rows mirror results/a4_ft/*.jsonl: key, binary, package, addr, name, code (+regime/name_seen for eval).
Output: results/a4_baptext/{train,val,test}.jsonl + stats.json."""
import json, os, sys, collections, multiprocessing as mp
WS = '$WORKSPACE/dh2'; PROTO = f'{WS}/results/baseline_protocol_v2'
G = '$WORKSPACE/relift_ws/data/graphs_v3'; S = '$WORKSPACE/relift_ws/data/string_refs_v2'
OUT = f'{WS}/results/a4_baptext'
MAX_BLOCKS, MAX_TOK, SIG, MAX_CTX, MAX_STR, STR_LEN = 30, 20, 10, 5, 8, 60

def linearise(g, strs, callee_sigs, caller_sigs):
    parts = []
    ext = [c for c in (g.get('external_calls') or []) if c][:16]
    parts.append('EXT: ' + (' '.join(ext) if ext else 'none'))
    if strs: parts.append('STR: ' + ' '.join(json.dumps(s[:STR_LEN]) for s in strs[:MAX_STR]))
    if callee_sigs: parts.append('CALLEES: ' + ' ; '.join(' '.join(s) for s in callee_sigs))
    if caller_sigs: parts.append('CALLERS: ' + ' ; '.join(' '.join(s) for s in caller_sigs))
    blocks = sorted(g.get('blocks') or [], key=lambda b: int(b.get('addr', '0x0'), 16))[:MAX_BLOCKS]
    ids = {b['id']: i for i, b in enumerate(blocks)}
    succ = collections.defaultdict(list)
    for e in g.get('edges') or []:
        a, b = (e[0], e[1]) if isinstance(e, (list, tuple)) else (e.get('src', e.get('from')), e.get('dst', e.get('to')))
        if a in ids and b in ids: succ[ids[a]].append(ids[b])
    for i, b in enumerate(blocks):
        toks = (b.get('tokens') or [])[:MAX_TOK]
        nxt = ' -> ' + ' '.join(f'B{j}' for j in sorted(set(succ[i]))) if succ[i] else ''
        parts.append(f'B{i}: ' + ' '.join(toks) + nxt)
    return '\n'.join(parts)

def sig(g):
    toks = []
    for b in sorted(g.get('blocks') or [], key=lambda b: int(b.get('addr', '0x0'), 16)):
        toks.extend(b.get('tokens') or [])
        if len(toks) >= SIG: break
    return toks[:SIG]

def do_binary(args):
    binary, rows = args
    try: idx = json.load(open(f'{G}/{binary}.index.json'))['functions']
    except Exception as e: return binary, [], {'no_index': len(rows)}
    by_addr = {f['entry_addr']: f for f in idx}
    graphs = {}
    for f in idx:
        try: graphs[f['name']] = json.load(open(f'{G}/{f["file"]}'))
        except Exception: pass
    callers = collections.defaultdict(list)
    for name, g in graphs.items():
        for c in g.get('internal_callees') or []:
            cn = c if isinstance(c, str) else c.get('name')
            if cn: callers[cn].append(name)
    strs = {}
    try: strs = json.load(open(f'{S}/{binary}.json'))['functions']
    except Exception: pass
    out, miss = [], collections.Counter()
    for r in rows:
        f = by_addr.get(r['entry_addr'])
        if not f or f['name'] not in graphs: miss['no_graph'] += 1; continue
        g = graphs[f['name']]
        cal = [(c if isinstance(c, str) else c.get('name')) for c in (g.get('internal_callees') or [])]
        callee_sigs = [sig(graphs[c]) for c in cal if c in graphs][:MAX_CTX]
        caller_sigs = [sig(graphs[c]) for c in callers.get(f['name'], []) if c in graphs][:MAX_CTX]
        code = linearise(g, strs.get(f['name']) or [], callee_sigs, caller_sigs)
        out.append({'key': f"{binary}_{r['entry_addr']}", 'binary': binary, 'package': r['package'], 'addr': r['entry_addr'],
                    'name': r['name'], 'code': code, 'regime': r.get('regime'), 'name_seen_in_train': r.get('name_seen_in_train'), 'in_dynsym': False})
    return binary, out, dict(miss)

def main():
    os.makedirs(OUT, exist_ok=True); stats = {}
    for tier in ('train', 'val', 'test'):
        rows = [json.loads(l) for l in open(f'{PROTO}/{tier}.jsonl')]
        by_bin = collections.defaultdict(list)
        for r in rows: by_bin[r['binary']].append(r)
        n, miss, lens = 0, collections.Counter(), []
        with mp.Pool(int(os.environ.get('NPROC', 16))) as pool, open(f'{OUT}/{tier}.jsonl', 'w') as fh:
            for binary, out, m in pool.imap_unordered(do_binary, sorted(by_bin.items()), chunksize=1):
                miss.update(m)
                for o in out: fh.write(json.dumps(o) + '\n'); n += 1; lens.append(len(o['code']))
        lens.sort(); pct = lambda p: lens[min(len(lens)-1, int(p*len(lens)))] if lens else 0
        stats[tier] = {'rows_in': len(rows), 'rows_out': n, 'skipped': dict(miss), 'chars_p50': pct(.5), 'chars_p90': pct(.9), 'chars_max': lens[-1] if lens else 0}
        print(f'EFFECT: {tier}: {n}/{len(rows)} rows written, skipped {dict(miss)}, chars p50 {pct(.5)} p90 {pct(.9)}', flush=True)
    json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1)
if __name__ == '__main__': main()
