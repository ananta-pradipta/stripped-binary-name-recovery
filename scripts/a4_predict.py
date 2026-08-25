#!/usr/bin/env python3
"""A4 generation head: predict names for protocol tier rows (default: test) from masked Ghidra
decompilation and score with metric v2 (split_name camel fix + C++ demangle canonicalization,
same as scripts/rescore_metric_v2.py). Emits an eval_v2-style TSV + JSON summary.

TSV columns: tier binary entry_addr true pred regime name_seen f1_raw f1_v2
"""
import argparse, json, os, re, subprocess, sys, time
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, '/project/hz79/_shared/cs785/dh2')
from src.evaluation.metrics import compute_subtoken_f1
WS = '/project/hz79/_shared/cs785/dh2'
PROTO = f'{WS}/results/baseline_protocol_v2'

def demangle_many(names):
    todo = sorted({n for n in names if n.startswith('_Z')}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]
        r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out

def canon(name, dem):
    n = dem.get(name, name)
    n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]
    return '_'.join(p for p in parts if p)

def agg(rows, key):
    n = len(rows)
    return {'n': n, 'f1': sum(r[key] for r in rows) / n if n else 0.0,
            'em': sum(r['em'] for r in rows) / n if n else 0.0}

def macro(rows, key):
    g = defaultdict(list)
    for r in rows: g[r['pkg']].append(r)
    per = {p: agg(v, key) for p, v in g.items()}
    return ({'n_groups': len(per), 'f1': sum(d['f1'] for d in per.values()) / len(per) if per else 0.0,
             'em': sum(d['em'] for d in per.values()) / len(per) if per else 0.0}, per)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--tiers', nargs='+', default=['test'])
    ap.add_argument('--tag', required=True)
    ap.add_argument('--max-src', type=int, default=1024)
    ap.add_argument('--max-tgt', type=int, default=24)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--beams', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--bf16', action='store_true')
    args = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    device = 'cuda'
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt).to(device).eval()
    outdir = f'{WS}/results/a4_{args.tag}'; os.makedirs(outdir, exist_ok=True)
    report = {'ckpt': args.ckpt, 'beams': args.beams, 'max_src': args.max_src, 'tiers': {}}
    dump = []
    dec_cache = {}
    def decomp(b):
        if b not in dec_cache:
            p = f'{WS}/symgen_v2/decomp/{b}.json'
            dec_cache[b] = json.load(open(p)) if os.path.exists(p) else {}
        return dec_cache[b]
    for tier in args.tiers:
        rows = [json.loads(l) for l in open(f'{PROTO}/{tier}.jsonl')]
        if args.limit: rows = rows[:args.limit]
        items, miss = [], 0
        for r in rows:
            e = decomp(r['binary']).get(r['entry_addr'])
            if not e or 'code' not in e: miss += 1; items.append((r, None)); continue
            code = e['code'].replace(e['ghidra_name'], '[MASK]', 1)
            items.append((r, code.strip()))
        dec_cache.clear()
        # sort by length for efficient batching, keep original order via index
        order = sorted(range(len(items)), key=lambda i: len(items[i][1] or ''))
        preds = [''] * len(items); t0 = time.time()
        todo = [i for i in order if items[i][1] is not None]
        for s in range(0, len(todo), args.bs):
            idx = todo[s:s+args.bs]
            enc = tok([items[i][1] for i in idx], max_length=args.max_src, truncation=True, padding=True, return_tensors='pt')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16 if args.bf16 else torch.float16):
                gen = model.generate(input_ids=enc.input_ids.to(device), attention_mask=enc.attention_mask.to(device),
                                     max_new_tokens=args.max_tgt, num_beams=args.beams)
            for i, p in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                preds[i] = '_'.join(p.strip().split())
            if (s // args.bs) % 200 == 0:
                print(f'{tier}: {s}/{len(todo)} {(time.time()-t0)/60:.1f}m', flush=True)
        dem = demangle_many([r['name'] for r, _ in items] + preds)
        scored = []
        for (r, code), p in zip(items, preds):
            t = r['name']; cp, ct = canon(p, dem), canon(t, dem)
            scored.append({'tier': tier, 'binary': r['binary'], 'addr': r['entry_addr'], 'pkg': r['package'],
                           'true': t, 'pred': p, 'regime': r.get('regime', '?'), 'name_seen': bool(r.get('name_seen_in_train')),
                           'f1_raw': compute_subtoken_f1(p, t), 'f1_v2': compute_subtoken_f1(cp, ct), 'em': 1.0 if cp == ct else 0.0,
                           'has_decomp': code is not None})
        mac, per_pkg = macro(scored, 'f1_v2')
        out = {'n_rows': len(rows), 'no_decomp': miss, 'scored': agg(scored, 'f1_v2'), 'scored_raw_f1': agg(scored, 'f1_raw')['f1'],
               'macro_pkg': mac, 'per_pkg': per_pkg, 'by_regime': {}, 'by_name_stratum': {},
               'pred_uniqueness': len({x['pred'] for x in scored}) / max(1, len(scored))}
        for reg in ('FT', 'NCT'):
            sub = [x for x in scored if x['regime'] == reg]
            out['by_regime'][reg] = {'micro': agg(sub, 'f1_v2'), 'macro_pkg': macro(sub, 'f1_v2')[0]}
        for name, cond in (('seen_name', True), ('novel_name', False)):
            out['by_name_stratum'][name] = agg([x for x in scored if x['name_seen'] == cond], 'f1_v2')
        report['tiers'][tier] = out; dump += scored
        print(f"=== {tier} A4 {args.tag} === n {len(scored)} (no_decomp {miss}) uniq {out['pred_uniqueness']:.3f}")
        print(f"  micro F1(v2) {out['scored']['f1']:.4f} EM {out['scored']['em']:.4f} | raw-F1 {out['scored_raw_f1']:.4f} | macro F1 {mac['f1']:.4f} over {mac['n_groups']} pkgs")
        for reg, d in out['by_regime'].items():
            print(f"  {reg:<4} micro F1 {d['micro']['f1']:.4f} EM {d['micro']['em']:.4f} n={d['micro']['n']} | macro {d['macro_pkg']['f1']:.4f}")
        for k, d in out['by_name_stratum'].items():
            print(f"  {k:<10} F1 {d['f1']:.4f} EM {d['em']:.4f} n={d['n']}")
    json.dump(report, open(f'{outdir}/{"_".join(args.tiers)}_eval.json', 'w'), indent=1)
    with open(f'{outdir}/{"_".join(args.tiers)}_preds.tsv', 'w') as fh:
        fh.write('tier\tbinary\tentry_addr\ttrue\tpred\tregime\tname_seen\tf1_raw\tf1_v2\n')
        for x in dump:
            fh.write(f"{x['tier']}\t{x['binary']}\t{x['addr']}\t{x['true']}\t{x['pred']}\t{x['regime']}\t{int(x['name_seen'])}\t{x['f1_raw']:.3f}\t{x['f1_v2']:.3f}\n")
    print('EFFECT: a4_predict done', {t: (d['scored']['n'], round(d['scored']['f1'], 4)) for t, d in report['tiers'].items()}, flush=True)

if __name__ == '__main__':
    main()
