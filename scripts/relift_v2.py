#!/usr/bin/env python3
"""Corpus re-lift v2 (defects B3a/B3b/B4/B10 of docs/DUALHEAD_HYDRA_PLAN.md).

For every binary that has a DEBUG (unstripped) ELF:
  1. stripped_v2/<id>        = `strip --strip-all` of the debug ELF (guarantees the
                               stripped input is the SAME build as the labels; B4)
  2. labels_v2/<id>.json     = FUNC symbols from the debug .symtab with addr/size/
                               binding/section + `in_dynsym` (name visible in the
                               stripped ELF's .dynsym → B10 symbol-visible stratum)
  3. bir_v2/<id>.starts      = function-start roots from the stripped ELF's .eh_frame
                               FDEs (deployable on stripped ELFs; B3b)
  4. bir_v2/<id>.bir + .syms = `bap <stripped> --read-symbols-from=<starts>
                               --dump=bir --dump-symbols` (BAP-IR + BAP's own function
                               table incl. dynsym-named subs; B3a matcher input)
  5. row in bir_v2/relift_manifest.tsv with build-id, counts, label coverage
     (label addr found as a BAP function start at addr or addr+4), seconds, rc.

Resumable: ids whose manifest row has rc=0 and whose outputs exist are skipped.
Effect assertion at the end (preflight discipline): prints median coverage and
every binary < 0.90; exits 1 if median < 0.95 unless --no-assert.

Usage:
  python3 scripts/relift_v2.py --workers 4                # main corpus (data/raw + cross_project/debug)
  python3 scripts/relift_v2.py --sources ftdomains clang   # extra corpora
  python3 scripts/relift_v2.py --ids bash_bash_O2 groff_troff_O2 --force
"""
import argparse, csv, json, os, re, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

OUT_STRIP = 'data/stripped_v2'
OUT_LABELS = 'data/labels_v2'
OUT_BIR = 'data/bir_v2'
MANIFEST = os.path.join(OUT_BIR, 'relift_manifest.tsv')
FIELDS = ['id', 'corpus', 'debug_elf', 'build_id', 'elf_type', 'n_labels', 'n_local', 'n_dynsym',
          'n_fde', 'n_bap_fns', 'n_bap_named', 'label_cov', 'label_cov_plain_hint',
          'seconds', 'rc', 'note']


# --------------------------------------------------------------------------- sources
def discover(sources):
    """Yield (id, corpus, debug_elf_path)."""
    seen = set()
    def emit(i, c, p):
        if i in seen:
            return
        seen.add(i)
        yield_list.append((i, c, p))
    yield_list = []
    if 'main' in sources:
        d = 'data/raw'
        for f in sorted(os.listdir(d)):
            i = f[:-4] if f.endswith('_sym') else f
            emit(i, 'local_main', os.path.join(d, f))
        d = 'data/cross_project/debug'
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                emit(f, 'local_xproj', os.path.join(d, f))
    for src, corpus in (('ftdomains', 'ftdomains'), ('ftdomains2', 'ftdomains2'),
                        ('clang_train', 'clang_train'), ('clang_o1o3', 'clang_o1o3')):
        if src in sources or ('clang' in sources and src.startswith('clang')):
            d = os.path.join(src, 'bins')
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith('.debug'):
                        emit(f[:-6], corpus, os.path.join(d, f))
    return yield_list


# --------------------------------------------------------------------------- helpers
def run(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def build_id(path):
    out = run(['readelf', '-n', path]).stdout
    m = re.search(r'Build ID:\s*([0-9a-f]+)', out)
    return m.group(1) if m else ''


def elf_type(path):
    out = run(['readelf', '-h', path]).stdout
    m = re.search(r'Type:\s+(\S+)', out)
    return m.group(1) if m else ''


def symtab_funcs(debug):
    """FUNC symbols from .symtab: name -> dict(addr,size,binding,section)."""
    out = run(['readelf', '-s', '-W', debug]).stdout
    funcs = {}
    in_symtab = False
    for line in out.splitlines():
        if line.startswith("Symbol table '"):
            in_symtab = '.symtab' in line
            continue
        if not in_symtab:
            continue
        parts = line.split()
        if len(parts) < 8 or parts[3] != 'FUNC':
            continue
        try:
            addr = int(parts[1], 16); size = int(parts[2], 0)
        except ValueError:
            continue
        name = parts[7]
        if addr == 0 or parts[6] == 'UND':
            continue
        # keep first (lowest-address / first-seen) entry per name; record aliases separately
        if name in funcs:
            continue
        funcs[name] = {'addr': addr, 'size': size, 'binding': parts[4], 'section': parts[6]}
    return funcs


def dynsym_names(stripped):
    out = run(['readelf', '--dyn-syms', '-W', stripped]).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[3] == 'FUNC' and parts[6] != 'UND':
            names.add(parts[7].split('@')[0])
    return names


def fde_starts(stripped):
    out = run(['readelf', '-wF', stripped]).stdout
    return sorted(set(int(m, 16) for m in re.findall(r'FDE cie=[0-9a-f]+ pc=([0-9a-f]+)\.\.', out)))


def parse_bap_syms(path):
    """dump-symbols file: one '(name start end)' per basic block (decimal). Return name -> min start."""
    fns = {}
    with open(path, errors='ignore') as fh:
        for line in fh:
            m = re.match(r'\((\S+) (\d+) (\d+)\)', line.strip())
            if m:
                n, s = m.group(1), int(m.group(2))
                if n not in fns or s < fns[n]:
                    fns[n] = s
    return fns


# --------------------------------------------------------------------------- worker
def process(item, force, bap_timeout):
    bid, corpus, debug = item
    t0 = time.time()
    row = dict.fromkeys(FIELDS, ''); row.update(id=bid, corpus=corpus, debug_elf=debug, rc='', note='')
    stripped = os.path.join(OUT_STRIP, bid)
    labels_p = os.path.join(OUT_LABELS, bid + '.json')
    starts_p = os.path.join(OUT_BIR, bid + '.starts')
    bir_p = os.path.join(OUT_BIR, bid + '.bir')
    syms_p = os.path.join(OUT_BIR, bid + '.syms')
    try:
        # 1. strip
        if force or not os.path.exists(stripped):
            r = run(['strip', '--strip-all', '-o', stripped, debug])
            if r.returncode != 0:
                row.update(rc=r.returncode, note='strip failed: ' + r.stderr.strip()[:200]); return row
        row['build_id'] = build_id(stripped)
        if build_id(debug) != row['build_id']:
            row['note'] += 'BUILDID_MISMATCH_AFTER_STRIP;'
        row['elf_type'] = elf_type(stripped)
        # 2. labels v2
        funcs = symtab_funcs(debug)
        dyn = dynsym_names(stripped)
        for n, f in funcs.items():
            f['in_dynsym'] = n in dyn
        row['n_labels'] = len(funcs)
        row['n_local'] = sum(1 for f in funcs.values() if f['binding'] == 'LOCAL')
        row['n_dynsym'] = sum(1 for f in funcs.values() if f['in_dynsym'])
        json.dump({'binary': bid, 'corpus': corpus, 'build_id': row['build_id'], 'elf_type': row['elf_type'],
                   'label_source': 'readelf -s .symtab FUNC (debug ELF); in_dynsym from stripped .dynsym',
                   'functions': {n: {'addr': hex(f['addr']), 'size': f['size'], 'binding': f['binding'],
                                     'section': f['section'], 'in_dynsym': f['in_dynsym']}
                                 for n, f in funcs.items()}},
                  open(labels_p, 'w'), indent=1)
        # 3. eh_frame roots
        starts = fde_starts(stripped)
        row['n_fde'] = len(starts)
        with open(starts_p, 'w') as fh:
            fh.write(';; function starts from .eh_frame FDEs of %s\n' % stripped)
            for a in starts:
                fh.write('0x%x\n' % a)
        # 4. bap
        if force or not (os.path.exists(bir_p) and os.path.exists(syms_p) and os.path.getsize(bir_p) > 0):
            cmd = ['bap', stripped, '--read-symbols-from=' + starts_p,
                   '--dump=bir:' + bir_p, '--dump-symbols', '--dump-symbols-file=' + syms_p]
            try:
                r = run(cmd, timeout=bap_timeout)
            except subprocess.TimeoutExpired:
                row.update(rc='TIMEOUT', note='bap timeout %ds' % bap_timeout, seconds=round(time.time() - t0, 1)); return row
            if r.returncode != 0 or not os.path.exists(bir_p):
                row.update(rc=r.returncode, note='bap failed: ' + (r.stderr.strip()[-200:] if r.stderr else ''),
                           seconds=round(time.time() - t0, 1)); return row
        # 5. coverage
        bap_fns = parse_bap_syms(syms_p)          # name -> start
        starts_set = set(bap_fns.values())
        row['n_bap_fns'] = len(bap_fns)
        row['n_bap_named'] = sum(1 for n in bap_fns if not n.startswith('sub_') and not n.startswith('.'))
        if funcs:
            hit = sum(1 for f in funcs.values() if f['addr'] in starts_set or f['addr'] + 4 in starts_set)
            row['label_cov'] = round(hit / len(funcs), 4)
            fde_set = set(starts)
            row['label_cov_plain_hint'] = round(sum(1 for f in funcs.values() if f['addr'] in fde_set or f['addr'] + 4 in fde_set) / len(funcs), 4)
        row['rc'] = 0
    except Exception as e:  # noqa
        row.update(rc='EXC', note=repr(e)[:200])
    row['seconds'] = round(time.time() - t0, 1)
    return row


# --------------------------------------------------------------------------- main
def load_manifest():
    rows = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                rows[r['id']] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', default=['main'], help='main ftdomains ftdomains2 clang')
    ap.add_argument('--ids', nargs='*', help='restrict to these binary ids')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--bap-timeout', type=int, default=3600)
    ap.add_argument('--no-assert', action='store_true')
    args = ap.parse_args()
    for d in (OUT_STRIP, OUT_LABELS, OUT_BIR):
        os.makedirs(d, exist_ok=True)
    items = discover(args.sources)
    if args.ids:
        want = set(args.ids); items = [it for it in items if it[0] in want]
    done = load_manifest()
    todo = [it for it in items if args.force or done.get(it[0], {}).get('rc') != '0'
            or not os.path.exists(os.path.join(OUT_BIR, it[0] + '.bir'))]
    print(f'{len(items)} binaries discovered, {len(items) - len(todo)} already done, {len(todo)} to process, '
          f'{args.workers} workers', flush=True)
    write_header = not os.path.exists(MANIFEST)
    n_ok = 0
    with open(MANIFEST, 'a', newline='') as fh, ProcessPoolExecutor(args.workers) as ex:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter='\t')
        if write_header:
            w.writeheader()
        futs = {ex.submit(process, it, args.force, args.bap_timeout): it for it in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            done[row['id']] = row
            w.writerow(row); fh.flush()
            if row['rc'] == 0:
                n_ok += 1
            print(f"[{k}/{len(todo)}] {row['id']:36s} rc={row['rc']} cov={row['label_cov']} "
                  f"labels={row['n_labels']} fde={row['n_fde']} bapfns={row['n_bap_fns']} {row['seconds']}s {row['note']}", flush=True)
    # effect assertion
    covs = sorted(float(r['label_cov']) for r in done.values() if r.get('rc') in (0, '0') and r.get('label_cov') not in ('', None))
    if covs:
        med = covs[len(covs) // 2]
        low = [(r['id'], r['label_cov']) for r in done.values() if r.get('rc') in (0, '0') and r.get('label_cov') not in ('', None) and float(r['label_cov']) < 0.9]
        print(f'\nEFFECT CHECK: {len(covs)} lifted; median label coverage {med:.4f}; <0.90: {len(low)}')
        for i, c in sorted(low, key=lambda x: float(x[1]))[:40]:
            print(f'   LOW {i} {c}')
        if med < 0.95 and not args.no_assert:
            print('EFFECT CHECK FAILED (median < 0.95)'); sys.exit(1)
    print('done')


if __name__ == '__main__':
    main()
