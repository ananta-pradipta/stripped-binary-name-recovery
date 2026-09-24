#!/usr/bin/env python3
"""B7 string channel: resolve each function's constant addresses (parse_bir_v3
``gref_addrs``) against the STRIPPED ELF's read-only data (the same file BAP
lifted, so addresses agree) and keep the ones that point at printable
NUL-terminated strings.  Deployable on stripped binaries by construction.

Output: data/string_refs_v2/<id>.json =
  {binary, n_functions, n_with_strings, functions: {bap_name: [str, ...]}}
plus data/string_refs_v2/report.tsv.
"""
import argparse, csv, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
from elftools.elf.elffile import ELFFile  # noqa

GRAPHS = 'data/graphs_v3'; STRIP = 'data/stripped_v2'; OUT = 'data/string_refs_v2'
RO_SECTIONS = ('.rodata', '.data.rel.ro', '.data', '.rodata1', '.data.rel.ro.local')
MIN_LEN, MAX_LEN = 3, 200


def load_ro(path):
    segs = []
    with open(path, 'rb') as fh:
        elf = ELFFile(fh)
        for s in elf.iter_sections():
            if s.name in RO_SECTIONS and s['sh_size'] > 0 and s['sh_type'] != 'SHT_NOBITS':
                segs.append((s['sh_addr'], s.data()))
    return segs


def string_at(segs, addr):
    for base, data in segs:
        if base <= addr < base + len(data):
            off = addr - base
            end = data.find(b'\x00', off, off + MAX_LEN + 1)
            if end == -1 or end - off < MIN_LEN:
                return None
            raw = data[off:end]
            try:
                s = raw.decode('ascii')
            except UnicodeDecodeError:
                return None
            if not all(32 <= ord(c) < 127 or c in '\t\n\r' for c in s):
                return None
            return s
    return None


def one(bid):
    idx_p = os.path.join(GRAPHS, bid + '.index.json'); ep = os.path.join(STRIP, bid)
    if not (os.path.exists(idx_p) and os.path.exists(ep)):
        return {'binary': bid, 'rc': 'missing'}
    try:
        segs = load_ro(ep)
        idx = json.load(open(idx_p))
        out = {}; n_str = 0
        for g in idx['functions']:
            gd = json.load(open(os.path.join(GRAPHS, g['file'])))
            strs = []
            for a in gd.get('gref_addrs', []):
                s = string_at(segs, int(a, 16))
                if s is not None and s not in strs:
                    strs.append(s)
            if strs:
                out[g['name']] = strs; n_str += len(strs)
        json.dump({'binary': bid, 'n_functions': idx['n_functions'], 'n_with_strings': len(out),
                   'functions': out}, open(os.path.join(OUT, bid + '.json'), 'w'), indent=1, sort_keys=True)
        return {'binary': bid, 'rc': 0, 'n_functions': idx['n_functions'], 'n_with_strings': len(out),
                'n_strings': n_str}
    except Exception as e:  # noqa
        return {'binary': bid, 'rc': 'EXC', 'note': repr(e)[:200]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--ids', nargs='*')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ids = sorted(f[:-11] for f in os.listdir(GRAPHS) if f.endswith('.index.json'))
    if args.ids:
        ids = [i for i in ids if i in set(args.ids)]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, f in enumerate(as_completed({ex.submit(one, i): i for i in ids}), 1):
            r = f.result(); rows.append(r)
            if k % 100 == 0:
                print(f'{k}/{len(ids)}', flush=True)
    keys = ['binary', 'rc', 'n_functions', 'n_with_strings', 'n_strings', 'note']
    with open(os.path.join(OUT, 'report.tsv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter='\t', extrasaction='ignore'); w.writeheader()
        for r in sorted(rows, key=lambda r: r['binary']):
            w.writerow(r)
    ok = [r for r in rows if r['rc'] == 0]
    tf = sum(r['n_functions'] for r in ok); ts = sum(r['n_with_strings'] for r in ok)
    print(f'{len(ok)}/{len(rows)} binaries; functions with >=1 string ref: {ts}/{tf} = {ts / max(1, tf):.3f}')


if __name__ == '__main__':
    main()
