#!/usr/bin/env python3
"""Parse every re-lifted binary (data/bir_v2/relift_manifest.tsv rows with rc=0)
with parse_bir_v3 into data/graphs_v3/.  Resumable (skips ids whose index.json
exists with the current PARSER_VERSION unless --force).  Writes
data/graphs_v3/parse_manifest.tsv and prints an effect check: per-binary label
coverage of kept graphs (entry addr == label addr or label addr+4).
"""
import argparse, csv, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from src.preprocessing.parse_bir_v3 import parse_bir, load_bap_syms, write_graphs, PARSER_VERSION  # noqa

BIR = 'data/bir_v2'; OUT = 'data/graphs_v3'; LAB = 'data/labels_v2'
FIELDS = ['id', 'parser_version', 'n_kept', 'n_skipped', 'n_labels', 'label_cov', 'n_named_kept', 'seconds', 'rc', 'note']


def one(bid, force):
    t0 = time.time()
    row = dict.fromkeys(FIELDS, ''); row.update(id=bid, parser_version=PARSER_VERSION)
    idx_p = os.path.join(OUT, bid + '.index.json')
    try:
        if not force and os.path.exists(idx_p):
            meta = json.load(open(idx_p))
            if meta.get('parser_version') == PARSER_VERSION:
                row.update(rc='skip'); return row
        fns = parse_bir(os.path.join(BIR, bid + '.bir'))
        syms = load_bap_syms(os.path.join(BIR, bid + '.syms'))
        meta = write_graphs(fns, bid, OUT, syms)
        row['n_kept'] = meta['n_functions']; row['n_skipped'] = meta['n_skipped']
        row['n_named_kept'] = sum(1 for r in meta['functions'] if r['name_kind'] == 'named')
        lp = os.path.join(LAB, bid + '.json')
        if os.path.exists(lp):
            lab = json.load(open(lp))['functions']
            laddr = {int(f['addr'], 16) for f in lab.values()}
            ent = {int(r['entry_addr'], 16) for r in meta['functions'] if r['entry_addr']}
            row['n_labels'] = len(laddr)
            row['label_cov'] = round(sum(1 for a in laddr if a in ent or a + 4 in ent) / max(1, len(laddr)), 4)
        row['rc'] = 0
    except Exception as e:  # noqa
        row.update(rc='EXC', note=repr(e)[:200])
    row['seconds'] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--ids', nargs='*')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ids = []
    with open(os.path.join(BIR, 'relift_manifest.tsv')) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['rc'] == '0' and os.path.exists(os.path.join(BIR, r['id'] + '.bir')):
                ids.append(r['id'])
    if args.ids:
        ids = [i for i in ids if i in set(args.ids)]
    ids = sorted(set(ids), key=lambda i: -os.path.getsize(os.path.join(BIR, i + '.bir')))
    print(f'{len(ids)} lifted binaries; workers={args.workers}', flush=True)
    mp = os.path.join(OUT, 'parse_manifest.tsv')
    rows = {}
    if os.path.exists(mp):
        with open(mp) as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                rows[r['id']] = r
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, i, args.force): i for i in ids}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r['rc'] != 'skip':
                rows[r['id']] = r
                print(f"[{k}/{len(ids)}] {r['id']:34s} kept={r['n_kept']} cov={r['label_cov']} {r['seconds']}s {r['note']}", flush=True)
    with open(mp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter='\t'); w.writeheader()
        for i in sorted(rows):
            w.writerow(rows[i])
    covs = sorted(float(r['label_cov']) for r in rows.values() if r.get('rc') in (0, '0') and r.get('label_cov'))
    if covs:
        med = covs[len(covs) // 2]
        low = [(r['id'], r['label_cov']) for r in rows.values() if r.get('rc') in (0, '0') and r.get('label_cov') and float(r['label_cov']) < 0.9]
        print(f'EFFECT CHECK: {len(covs)} parsed; median label coverage {med:.4f}; <0.90: {len(low)} {sorted(low, key=lambda x: float(x[1]))[:15]}')


if __name__ == '__main__':
    main()
