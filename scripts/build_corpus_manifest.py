#!/usr/bin/env python3
"""Build the single-source-of-truth CORPUS MANIFEST (one row per binary id).

READ-ONLY on all data. Writes only to results/phase0/manifest/:
  corpus_manifest.tsv, corpus_manifest.json, MANIFEST.md,
  elf_cache.json (per-ELF readelf facts, keyed by path+size+mtime; speeds re-runs),
  HPC_inventory.json (cached output of HPC_inventory_remote.py; refresh with --HPC).

Usage:
  source activate.sh
  python3 scripts/build_corpus_manifest.py            # local + cached HPC inventory
  python3 scripts/build_corpus_manifest.py --HPC   # re-pull HPC inventory over ssh first
  python3 scripts/build_corpus_manifest.py --no-elf   # skip readelf pass (reuse cache only)

Binary id = <package>_<tool>_<opt>; opt in {O0,O1,O2,O3} or absent (=default);
package = prefix before first '_', tool = middle (may contain '_' or '-').
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = 'results/phase0/manifest'
os.makedirs(OUT, exist_ok=True)
OPTS = ('O0', 'O1', 'O2', 'O3')
ELFCHECK_ARCHIVE = f'{OUT}/corpus_elfcheck_archive_unified.tsv'   # git show archive/unified:results/rcdg/corpus_elfcheck.tsv
HPC_INV = f'{OUT}/HPC_inventory.json'
HPC_REMOTE_PY = f'{OUT}/HPC_inventory_remote.py'
ELF_CACHE = f'{OUT}/elf_cache.json'
AUDIT = 'results/phase0/audit.json'


# ----------------------------------------------------------------------------- id helpers
def parse_id(bid):
    parts = bid.split('_')
    pkg = parts[0]
    if parts[-1] in OPTS:
        opt = parts[-1]
        tool = '_'.join(parts[1:-1])
    else:
        opt = 'default'
        tool = '_'.join(parts[1:])
    return pkg, tool, opt


def strip_suffix(name, sufs):
    for s in sufs:
        if s and name.endswith(s):
            return name[:-len(s)]
    return None


# ----------------------------------------------------------------------------- ELF facts
def readelf_facts(path):
    """Return dict: elf_type, build_id, has_symtab, has_debug_info, has_eh_frame, n_fde (eh_frame only)."""
    facts = {'path': path, 'elf_type': None, 'build_id': None, 'has_symtab': False,
             'has_debug_info': False, 'has_eh_frame': False, 'n_fde': None, 'err': None}
    try:
        h = subprocess.run(['readelf', '-h', '-S', '-n', path], capture_output=True, text=True, timeout=120)
        txt = h.stdout
        m = re.search(r'^\s*Type:\s+(\w+)', txt, re.M)
        if m:
            facts['elf_type'] = m.group(1)
        m = re.search(r'Build ID:\s*([0-9a-f]+)', txt)
        if m:
            facts['build_id'] = m.group(1)
        facts['has_symtab'] = bool(re.search(r'\]\s+\.symtab\s', txt))
        facts['has_debug_info'] = bool(re.search(r'\]\s+\.debug_info\s', txt))
        facts['has_eh_frame'] = bool(re.search(r'\]\s+\.eh_frame\s', txt))
        if not facts['elf_type']:
            facts['err'] = 'not-elf'
            return facts
        # FDE count restricted to .eh_frame section (readelf -wF also dumps .debug_frame for debug ELFs)
        f = subprocess.run(['readelf', '-wF', path], capture_output=True, text=True, timeout=300, errors='replace')
        n = 0
        in_eh = False
        for line in f.stdout.split('\n'):
            if line.startswith('Contents of the '):
                in_eh = '.eh_frame' in line
                continue
            if in_eh and ' FDE ' in line:
                n += 1
        facts['n_fde'] = n
    except Exception as e:  # noqa
        facts['err'] = str(e)[:80]
    return facts


def elf_key(path):
    st = os.stat(path)
    return f'{path}|{st.st_size}|{int(st.st_mtime)}'


def collect_elf_facts(paths, cache, workers=8, skip=False):
    todo = [p for p in paths if elf_key(p) not in cache]
    if skip or not todo:
        return
    print(f'[elf] readelf on {len(todo)} files (cached {len(paths) - len(todo)})', file=sys.stderr)
    with ThreadPoolExecutor(workers) as ex:
        for i, facts in enumerate(ex.map(readelf_facts, todo)):
            cache[elf_key(facts['path'])] = facts
            if i % 500 == 0:
                print(f'[elf] {i}/{len(todo)}', file=sys.stderr)
    json.dump(cache, open(ELF_CACHE, 'w'))


# ----------------------------------------------------------------------------- labels
def count_labels(path):
    """Handle both schemas:
    (a) {'binary','num_functions','functions','name_to_addr','addr_to_name'} — 'functions' direction flips per file
    (b) flat {addr: name}  (data/labels/<id>.json without _labels suffix)
    Returns (n_unique_names, n_unique_addrs, schema)."""
    try:
        d = json.load(open(path))
    except Exception as e:
        return None, None, f'ERR:{str(e)[:40]}'
    if not isinstance(d, dict):
        return None, None, 'ERR:notdict'
    if 'name_to_addr' in d or 'addr_to_name' in d or 'functions' in d:
        n2a = d.get('name_to_addr') or {}
        a2n = d.get('addr_to_name') or {}
        fn = d.get('functions') or {}
        names = set(n2a.keys()) | set(a2n.values())
        addrs = set(a2n.keys()) | set(n2a.values())
        if isinstance(fn, dict) and fn:
            k0 = next(iter(fn))
            if str(k0).startswith('0x'):
                addrs |= set(fn.keys()); names |= set(fn.values()); direction = 'addr->name'
            else:
                names |= set(fn.keys()); addrs |= set(fn.values()); direction = 'name->addr'
        else:
            direction = 'none'
        return len(names), len(addrs), f'schemaA:functions={direction}'
    # flat
    k0 = next(iter(d), None)
    if k0 is None:
        return 0, 0, 'flat:empty'
    if str(k0).startswith('0x'):
        return len(set(d.values())), len(d), 'flat:addr->name'
    return len(d), len(set(d.values())), 'flat:name->addr'


def count_external(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if isinstance(d, dict) and 'functions' in d:
        return len(d['functions'])
    if isinstance(d, dict):
        return len(d)
    if isinstance(d, list):
        return len(d)
    return None


# ----------------------------------------------------------------------------- graphs
def graph_counts_from_dir(gdir, known_ids):
    """Per-binary counts of <id>_sub_XXXX.json (sub) and other <id>_<name>.json (nonsub) via os.listdir."""
    sub = collections.Counter(); non = collections.Counter(); unres = 0
    if not os.path.isdir(gdir):
        return sub, non, unres
    names = os.listdir(gdir)
    ids_sorted = sorted(known_ids, key=len, reverse=True)
    for n in names:
        if not n.endswith('.json'):
            continue
        n = n[:-5]
        m = re.match(r'^(.*?)_sub_[0-9a-f]+$', n)
        if m:
            sub[m.group(1)] += 1
            continue
        hit = None
        for i in ids_sorted:
            if n.startswith(i + '_'):
                hit = i; break
        if hit:
            non[hit] += 1
        else:
            unres += 1
    return sub, non, unres


# ----------------------------------------------------------------------------- elfcheck (ported from rcdg_stage0_elfcheck.py)
RE_SUB = re.compile(rb'^[0-9a-f]+: sub sub_([0-9a-f]+)\(')
PROLOGUE = {0xF3, 0x55, 0x41, 0x53, 0x48, 0x31, 0xE9, 0xC3, 0x8B, 0xB8, 0x66, 0x0F, 0xFF, 0x89, 0x85, 0x40, 0x39,
            0x83, 0x80, 0xEB, 0x33, 0x4C, 0x49, 0xB9, 0xBA, 0xBF, 0xBE, 0x45, 0x44, 0x4D, 0x50, 0x51, 0x52, 0x56,
            0x57, 0x90, 0xE8, 0x64}
N_CHECK = 300


def elfcheck(strip, bir):
    from elftools.elf.elffile import ELFFile
    segs = []
    with open(strip, 'rb') as fh:
        elf = ELFFile(fh)
        for s in elf.iter_segments():
            if s['p_type'] == 'PT_LOAD' and s['p_flags'] & 1:
                segs.append((s['p_vaddr'], s['p_filesz'], s['p_offset']))
    data = open(strip, 'rb').read()
    ok = tot = oob = 0
    seen = set()
    with open(bir, 'rb') as fh:
        for line in fh:
            m = RE_SUB.match(line)
            if not m:
                continue
            a = int(m.group(1), 16)
            if a in seen:
                continue
            seen.add(a)
            b = None
            for va, sz, off in segs:
                if va <= a < va + sz:
                    o = off + (a - va); b = data[o:o + 4]; break
            tot += 1
            if b is None or len(b) < 4:
                oob += 1
            elif b[:4] == b'\xf3\x0f\x1e\xfa' or b[0] in PROLOGUE:
                ok += 1
            if tot >= N_CHECK:
                break
    return tot, ok, oob, (ok / tot if tot else 0.0)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--HPC', action='store_true', help='re-pull HPC inventory via ssh')
    ap.add_argument('--no-elf', action='store_true', help='skip readelf pass; use cache only')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    rows = collections.defaultdict(lambda: {
        'id': None, 'package': None, 'tool': None, 'opt_tag': None, 'corpora': set(),
        'debug_elf_paths': [], 'stripped_elf_paths': [], 'other_elf_paths': [],
        'bir_paths': [], 'label_paths': [], 'external_paths': [],
        'n_sub_graphs': None, 'n_nonsub_graphs': None, 'n_matched': None,
        'local_split': None, 'notes': []})

    def R(bid, corpus):
        r = rows[bid]
        if r['id'] is None:
            r['id'] = bid; r['package'], r['tool'], r['opt_tag'] = parse_id(bid)
        r['corpora'].add(corpus)
        return r

    elf_paths = set()

    # ---- local_main ELFs
    for f in sorted(os.listdir('data/raw')):
        p = f'data/raw/{f}'
        if not os.path.isfile(p) or f.startswith('.'):
            continue
        bid = strip_suffix(f, ['_sym']) or f
        R(bid, 'local_main')['other_elf_paths'].append(p); elf_paths.add(p)
    for d in ['data/stripped'] + sorted(glob.glob('data/stripped/_p0_0_6_*')):
        for f in sorted(os.listdir(d)):
            p = f'{d}/{f}'
            if not os.path.isfile(p) or f.startswith('.'):
                continue
            bid = strip_suffix(f, ['_stripped']) or f
            corpus = 'local_main' if d == 'data/stripped' else 'local_main:' + os.path.basename(d)
            R(bid, corpus)['other_elf_paths'].append(p); elf_paths.add(p)
    for sub in ['stripped', 'debug', 'candidates']:
        d = f'data/cross_project/{sub}'
        for f in sorted(os.listdir(d)):
            p = f'{d}/{f}'
            if not os.path.isfile(p) or f.startswith('.'):
                continue
            bid = strip_suffix(f, ['_stripped']) or f
            R(bid, 'local_main:cross_project')['other_elf_paths'].append(p); elf_paths.add(p)
    # ---- local_main bir / labels / external
    for d in ['data/bir'] + sorted(glob.glob('data/bir/_p0_0_6_*')):
        for p in sorted(glob.glob(f'{d}/*.bir')):
            bid = os.path.basename(p)[:-4]
            corpus = 'local_main' if d == 'data/bir' else 'local_main:' + os.path.basename(d)
            R(bid, corpus)['bir_paths'].append(p)
    for p in sorted(glob.glob('data/labels/*.json')):
        f = os.path.basename(p)
        bid = strip_suffix(f, ['_labels.json', '.json'])
        R(bid, 'local_main')['label_paths'].append(p)
    for p in sorted(glob.glob('data/external_calls/*_external.json')):
        bid = os.path.basename(p)[:-len('_external.json')]
        R(bid, 'local_main')['external_paths'].append(p)

    # cross_project ground truth (per-binary dict of function labels for held-out packages)
    gt_path = 'data/cross_project/ground_truth/ground_truth.json'
    if os.path.exists(gt_path):
        gt = json.load(open(gt_path))
        for bid, v in gt.items():
            r = R(bid, 'local_main:cross_project')
            fns = v.get('functions') if isinstance(v, dict) else None
            r['n_xproj_ground_truth'] = len(fns) if hasattr(fns, '__len__') else (len(v) if hasattr(v, '__len__') else None)
    # ---- extra corpora
    extra = {}
    for c in ['ftdomains', 'ftdomains2', 'clang_train', 'clang_o1o3']:
        if not os.path.isdir(c):
            continue
        extra[c] = {}
        for f in sorted(os.listdir(f'{c}/bins')):
            p = f'{c}/bins/{f}'
            if not os.path.isfile(p) or f.startswith('.'):
                continue
            bid = strip_suffix(f, ['.debug', '_stripped']) or f
            R(bid, c)['other_elf_paths'].append(p); elf_paths.add(p)
        for p in sorted(glob.glob(f'{c}/bir/*.bir')):
            R(os.path.basename(p)[:-4], c)['bir_paths'].append(p)
        for p in sorted(glob.glob(f'{c}/labels/*.json')):
            R(strip_suffix(os.path.basename(p), ['_labels.json', '.json']), c)['label_paths'].append(p)
        for p in sorted(glob.glob(f'{c}/external/*_external.json')):
            R(os.path.basename(p)[:-len('_external.json')], c)['external_paths'].append(p)

    # ---- ELF facts
    cache = json.load(open(ELF_CACHE)) if os.path.exists(ELF_CACHE) else {}
    collect_elf_facts(sorted(elf_paths), cache, args.workers, args.no_elf)

    def facts(p):
        return cache.get(elf_key(p))

    for bid, r in rows.items():
        for p in r['other_elf_paths']:
            fc = facts(p)
            if fc is None:
                r['notes'].append(f'no-elf-facts:{p}')
                continue
            if fc['err'] == 'not-elf':
                r['notes'].append(f'not-elf:{p}')
                continue
            (r['debug_elf_paths'] if fc['has_symtab'] else r['stripped_elf_paths']).append(p)

    # ---- labels / external counts
    for bid, r in rows.items():
        r['n_labels'] = None; r['n_label_addrs'] = None; r['label_schema'] = None
        for p in r['label_paths']:
            n, na, schema = count_labels(p)
            if r['n_labels'] is None or (n or 0) > (r['n_labels'] or 0):
                r['n_labels'], r['n_label_addrs'], r['label_schema'] = n, na, schema
        r['n_external_fns'] = None
        for p in r['external_paths']:
            n = count_external(p)
            if n is not None and (r['n_external_fns'] is None or n > r['n_external_fns']):
                r['n_external_fns'] = n

    # ---- graphs: audit.json for local_main; listdir for extras
    audit = json.load(open(AUDIT))
    b2 = audit['B2']['per_binary']; b3 = audit['B3']['per_binary']
    ids_local = {b for b, r in rows.items() if any(c.startswith('local_main') for c in r['corpora'])}
    for bid in ids_local:
        r = rows[bid]
        if bid in b3:
            r['n_sub_graphs'] = b3[bid].get('n_sub_graphs'); r['n_matched'] = b3[bid].get('n_matched')
        elif bid in b2:
            r['n_sub_graphs'] = b2[bid].get('n_sub_graphs'); r['n_matched'] = b2[bid].get('n_matched')
    # nonsub graph counts for local_main are not in audit.json per-binary -> compute via listdir (fast, ~300K names)
    sub_l, non_l, unres_l = graph_counts_from_dir('data/graphs', set(rows))
    for bid in set(sub_l) | set(non_l):
        r = R(bid, 'local_main')
        if r['n_sub_graphs'] is None:
            r['n_sub_graphs'] = sub_l.get(bid, 0)
        elif r['n_sub_graphs'] != sub_l.get(bid, 0):
            r['notes'].append(f'audit_sub={r["n_sub_graphs"]}!=ls_sub={sub_l.get(bid, 0)}')
        r['n_nonsub_graphs'] = non_l.get(bid, 0)
    for c in extra:
        sub_c, non_c, unres_c = graph_counts_from_dir(f'{c}/graphs', set(rows))
        extra[c]['graph_unresolved'] = unres_c
        for bid in set(sub_c) | set(non_c):
            r = R(bid, c)
            r['n_sub_graphs'] = sub_c.get(bid, 0); r['n_nonsub_graphs'] = non_c.get(bid, 0)
    # ---- match_index counts
    def mi_counts(path, check_exists=False):
        c = collections.Counter(); dangling = collections.Counter()
        mi = json.load(open(path))
        for k, v in mi.items():
            b = v.get('binary') if isinstance(v, dict) else None
            if not b:
                b = re.sub(r'_sub_[0-9a-f]+$', '', os.path.basename(k)[:-5])
            c[b] += 1
            if check_exists and not os.path.exists(k):
                dangling[b] += 1
        return c, len(mi), dangling
    mi_local, n_mi_local, mi_dangling = mi_counts('data/match_index.json', check_exists=True)
    for bid, n in mi_local.items():
        r = R(bid, 'local_main')
        r['n_matched_dangling'] = mi_dangling.get(bid, 0)
        if r['n_matched'] is None:
            r['n_matched'] = n
        elif r['n_matched'] != n:
            r['notes'].append(f'audit_matched={r["n_matched"]}!=mi={n}')
            r['n_matched'] = n
        if r['n_matched_dangling']:
            r['notes'].append(f'match_index_dangling={r["n_matched_dangling"]}/{n}')
    n_mi_clang = 0
    if os.path.exists('clang_train/match_index.json'):
        mi_clang, n_mi_clang, _ = mi_counts('clang_train/match_index.json')
        for bid, n in mi_clang.items():
            r = R(bid, 'clang_train'); r['n_matched_clang_mi'] = n
    # ---- local split
    sp = json.load(open('data/split_assignments.json'))
    for k, v in sp.items():
        for bid in v:
            R(bid, 'local_main:split_only' if bid not in rows else next(iter(rows[bid]['corpora'])))['local_split'] = k
    # ---- elfcheck (archive + compute missing)
    elfc = {}
    if os.path.exists(ELFCHECK_ARCHIVE):
        for line in open(ELFCHECK_ARCHIVE).read().splitlines()[1:]:
            b, strip, tot, ok, oob, ratio = line.split('\t')
            elfc[b] = {'src': 'archive/unified:results/rcdg/corpus_elfcheck.tsv', 'stripped': strip,
                       'n': int(tot), 'ok': int(ok), 'oob': int(oob), 'ratio': float(ratio)}
    n_computed = 0
    for bid, r in rows.items():
        r['elfcheck_ratio'] = None; r['elfcheck_src'] = None; r['elfcheck_best_stripped'] = None
        if bid in elfc and elfc[bid]['stripped'] != 'MISSING':
            r['elfcheck_ratio'] = elfc[bid]['ratio']; r['elfcheck_src'] = 'archive'
            r['elfcheck_best_stripped'] = elfc[bid]['stripped']
            continue
        if r['stripped_elf_paths'] and r['bir_paths']:
            best = None
            for s in r['stripped_elf_paths']:
                try:
                    tot, ok, oob, ratio = elfcheck(s, r['bir_paths'][0])
                except Exception as e:
                    r['notes'].append(f'elfcheck-err:{str(e)[:40]}'); continue
                if best is None or ratio > best[0]:
                    best = (ratio, s, tot, ok, oob)
            if best:
                r['elfcheck_ratio'] = best[0]; r['elfcheck_src'] = 'computed'; r['elfcheck_best_stripped'] = best[1]
                n_computed += 1
    print(f'[elfcheck] archive={len(elfc)} computed_now={n_computed}', file=sys.stderr)

    # ---- HPC inventory
    if args.HPC or not os.path.exists(HPC_INV):
        print('[HPC] pulling inventory via ssh ...', file=sys.stderr)
        try:
            with open(HPC_REMOTE_PY) as fin, open(HPC_INV + '.tmp', 'w') as fout:
                subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HPC', 'cat > /tmp/adp_inv.py; python3 /tmp/adp_inv.py'],
                               stdin=fin, stdout=fout, timeout=900, check=True)
            os.replace(HPC_INV + '.tmp', HPC_INV)
        except Exception as e:
            print(f'[HPC] FAILED ({e}); using cached inventory if present', file=sys.stderr)
    wul = json.load(open(HPC_INV)) if os.path.exists(HPC_INV) else None
    wul_ids = set()
    if wul:
        wsplit = {}
        for tag in ['ccs_split_v1', 'ccs_split_paper_clean', 'ccs_split_paper_clean_strict_idx', 'data_split']:
            m = {}
            v = wul.get(tag)
            if isinstance(v, dict):
                for k, ids in v.items():
                    for b in ids:
                        m[b] = k
            wsplit[tag] = m
        w_bir = wul['data_bir']; w_lab = set(wul['data_labels']) | set(wul['ccs_labels'])
        w_ext = set(wul['data_external']) | set(wul['ccs_external'])
        w_gsub = wul['data_graphs']['sub']; w_gnon = wul['data_graphs']['nonsub']
        w_mi = wul['data_match_index'] if isinstance(wul['data_match_index'], dict) else {}
        w_debug = {x for x in wul['data_debug'] if not x.startswith('.')}; w_raw = {strip_suffix(x, ['_sym']) or x for x in wul['data_raw'] if not x.startswith('.')}
        w_strip = {strip_suffix(x, ['_stripped']) or x for x in wul['data_stripped'] if not x.startswith('.')}
        w_xp = {k: {strip_suffix(x, ['_stripped']) or x for x in v} for k, v in wul['data_cross_project'].items()}
        wul_ids = set(w_bir) | w_lab | w_ext | set(w_gsub) | set(w_mi) | w_debug | w_raw | w_strip
        for k in ['stripped', 'stripped_unseen', 'debug_unseen']:
            wul_ids |= w_xp.get(k, set())
        for m in wsplit.values():
            wul_ids |= set(m)
        for bid in wul_ids:
            r = R(bid, 'HPC')
            r['HPC_has_bir'] = bid in w_bir; r['HPC_bir_size'] = w_bir.get(bid)
            r['HPC_has_labels'] = bid in w_lab; r['HPC_has_external'] = bid in w_ext
            r['HPC_n_sub_graphs'] = w_gsub.get(bid); r['HPC_n_nonsub_graphs'] = w_gnon.get(bid)
            r['HPC_n_matched'] = w_mi.get(bid)
            r['HPC_has_debug_elf'] = bid in w_debug or bid in w_raw or bid in w_xp.get('debug_unseen', set())
            r['HPC_has_stripped_elf'] = bid in w_strip or bid in w_xp.get('stripped', set()) or bid in w_xp.get('stripped_unseen', set())
            r['HPC_split_v1'] = wsplit['ccs_split_v1'].get(bid)
            r['HPC_split_paper_clean'] = wsplit['ccs_split_paper_clean'].get(bid)
            r['HPC_split_paper_clean_strict_idx'] = wsplit['ccs_split_paper_clean_strict_idx'].get(bid)
            r['HPC_data_split'] = wsplit['data_split'].get(bid)

    # ---- duplicates from audit B5
    dup_partner = collections.defaultdict(list)
    for pr in audit['B5']['all_pairs']:
        ident = max(pr.get('identity_raw') or 0, pr.get('identity_thunk_resolved') or 0)
        if ident >= 0.9:
            dup_partner[pr['bin1']].append((pr['bin2'], round(ident, 3)))
            dup_partner[pr['bin2']].append((pr['bin1'], round(ident, 3)))

    # ---- finalize per-row derived fields + eligibility
    def bid_of(paths):
        for p in paths:
            fc = facts(p)
            if fc and fc.get('build_id'):
                return fc['build_id']
        return None

    final = []
    for bid in sorted(rows):
        r = rows[bid]
        if r['id'] is None:
            r['id'] = bid; r['package'], r['tool'], r['opt_tag'] = parse_id(bid)
        for k in ['n_labels', 'n_label_addrs', 'label_schema', 'n_external_fns', 'elfcheck_ratio', 'elfcheck_src',
                  'elfcheck_best_stripped', 'n_matched_dangling', 'n_xproj_ground_truth']:
            r.setdefault(k, None)
        # prefer data/raw and *_stripped for the canonical paths
        dbg = sorted(r['debug_elf_paths'], key=lambda p: (not p.startswith('data/raw/'), p))
        stp = sorted(r['stripped_elf_paths'], key=lambda p: (not p.endswith('_stripped'), not p.startswith('data/stripped/'), p))
        # if elfcheck says a specific stripped copy is the consistent one, prefer it
        if r.get('elfcheck_best_stripped') in stp:
            stp.remove(r['elfcheck_best_stripped']); stp.insert(0, r['elfcheck_best_stripped'])
        r['has_debug_elf'] = dbg[0] if dbg else ''
        r['has_stripped_elf'] = stp[0] if stp else ''
        r['n_debug_elfs'] = len(dbg); r['n_stripped_elfs'] = len(stp)
        r['has_bir'] = r['bir_paths'][0] if r['bir_paths'] else ''
        r['bir_size'] = os.path.getsize(r['bir_paths'][0]) if r['bir_paths'] else None
        r['has_labels'] = bool(r['label_paths'])
        r['label_path'] = r['label_paths'][0] if r['label_paths'] else ''
        r['build_id_stripped'] = bid_of(stp); r['build_id_debug'] = bid_of(dbg)
        r['build_ids_stripped_all'] = sorted({facts(p)['build_id'] for p in stp if facts(p) and facts(p)['build_id']})
        r['build_ids_debug_all'] = sorted({facts(p)['build_id'] for p in dbg if facts(p) and facts(p)['build_id']})
        r['build_id_match'] = (r['build_id_stripped'] == r['build_id_debug']) if (r['build_id_stripped'] and r['build_id_debug']) else None
        if r['build_id_match'] is False:
            # is ANY stripped copy consistent with ANY debug copy?
            r['build_id_any_match'] = bool(set(r['build_ids_stripped_all']) & set(r['build_ids_debug_all']))
        else:
            r['build_id_any_match'] = r['build_id_match']
        fs = facts(stp[0]) if stp else None
        fd = facts(dbg[0]) if dbg else None
        src = fs or fd
        r['elf_type'] = src['elf_type'] if src else None
        r['has_eh_frame'] = src['has_eh_frame'] if src else None
        r['n_fde'] = fs['n_fde'] if fs else None
        r['n_fde_debug'] = fd['n_fde'] if fd else None
        r['duplicate_of'] = dup_partner.get(bid, [])
        r['corpora'] = sorted(r['corpora'])
        # eligibility (conservative)
        reasons = []
        if not r['has_debug_elf'] and not r.get('HPC_has_debug_elf'):
            reasons.append('no_debug_elf')
        if not r['has_stripped_elf'] and not r.get('HPC_has_stripped_elf'):
            reasons.append('no_stripped_elf')
        if not r['has_bir'] and not r.get('HPC_has_bir'):
            reasons.append('no_bir')
        if not r['has_labels'] and not r.get('HPC_has_labels') and not r.get('n_xproj_ground_truth'):
            reasons.append('no_labels')
        elif r['n_labels'] is not None and r['n_labels'] == 0:
            reasons.append('empty_labels')
        if r['build_id_any_match'] is False:
            reasons.append('build_id_mismatch_stripped_vs_debug')
        if r['elfcheck_ratio'] is not None and r['elfcheck_ratio'] < 0.8:
            reasons.append(f'elfcheck_ratio<0.8({r["elfcheck_ratio"]:.2f})')
        if r['duplicate_of']:
            # keep the lowest opt of a duplicate group; flag the others
            partners = [p for p, _ in r['duplicate_of']]
            keep = min([bid] + partners)
            if bid != keep:
                reasons.append(f'duplicate_build_of:{keep}')
            else:
                r['notes'].append('dup_group_keeper:' + ','.join(sorted(partners)))
        if r.get('n_sub_graphs') in (None, 0) and r.get('HPC_n_sub_graphs') in (None, 0):
            reasons.append('no_sub_graphs')
        if r.get('n_matched') in (None, 0) and r.get('HPC_n_matched') in (None, 0) and r.get('n_matched_clang_mi') in (None, 0):
            reasons.append('no_matched_functions')
        if r.get('n_matched_dangling') and r['n_matched_dangling'] == r.get('n_matched'):
            reasons.append('match_index_entries_dangling(graph_files_missing)')
        # role hint (NOT an eligibility blocker): held-out cross-project packages / excluded-in-v1
        role = []
        if r.get('HPC_split_paper_clean') == 'xproject' or r.get('HPC_split_v1') == 'xproject':
            role.append('xproject_holdout(HPC_paper_clean)')
        if 'local_main:cross_project' in r['corpora']:
            role.append('local_cross_project_dir')
        if r.get('HPC_split_v1') == 'excluded' and r.get('HPC_split_paper_clean') != 'xproject':
            role.append('excluded_in_HPC_v1(not in paper_clean)')
        if r['local_split'] == 'excluded':
            role.append('excluded_in_local_split')
        r['role_hint'] = role
        r['eligible_v2'] = not reasons
        r['ineligible_reasons'] = reasons
        final.append(r)

    # ---- write JSON + TSV
    cols = ['id', 'package', 'tool', 'opt_tag', 'corpora', 'has_debug_elf', 'has_stripped_elf', 'n_debug_elfs', 'n_stripped_elfs',
            'has_bir', 'bir_size', 'has_labels', 'n_labels', 'label_schema', 'n_xproj_ground_truth', 'n_external_fns', 'n_sub_graphs', 'n_nonsub_graphs',
            'n_matched', 'n_matched_dangling', 'n_matched_clang_mi', 'local_split', 'HPC_split_v1', 'HPC_split_paper_clean',
            'HPC_split_paper_clean_strict_idx', 'HPC_data_split', 'HPC_has_bir', 'HPC_bir_size', 'HPC_has_labels',
            'HPC_has_external', 'HPC_n_sub_graphs', 'HPC_n_nonsub_graphs', 'HPC_n_matched', 'HPC_has_debug_elf',
            'HPC_has_stripped_elf', 'build_id_stripped', 'build_id_debug', 'build_id_match', 'build_id_any_match', 'elf_type',
            'has_eh_frame', 'n_fde', 'n_fde_debug', 'elfcheck_ratio', 'elfcheck_src', 'duplicate_of', 'eligible_v2',
            'ineligible_reasons', 'role_hint', 'notes']

    def cell(v):
        if v is None:
            return ''
        if isinstance(v, bool):
            return '1' if v else '0'
        if isinstance(v, (list, tuple, set)):
            return ';'.join(cell(x) if not isinstance(x, (list, tuple)) else ':'.join(map(str, x)) for x in v)
        return str(v)

    with open(f'{OUT}/corpus_manifest.tsv', 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in final:
            f.write('\t'.join(cell(r.get(c)) for c in cols) + '\n')
    slim = []
    for r in final:
        d = {c: r.get(c) for c in cols}
        for k in ['debug_elf_paths', 'stripped_elf_paths', 'bir_paths', 'label_paths', 'external_paths',
                  'build_ids_stripped_all', 'build_ids_debug_all', 'elfcheck_best_stripped']:
            d[k] = r.get(k)
        slim.append(d)
    json.dump({'n_rows': len(slim), 'generated_by': 'scripts/build_corpus_manifest.py', 'rows': slim},
              open(f'{OUT}/corpus_manifest.json', 'w'), indent=0)

    write_md(final, wul, audit, extra, n_mi_local, n_mi_clang, cache, elfc, unres_l)
    print(f'[done] {len(final)} ids -> {OUT}/corpus_manifest.{{tsv,json}}, MANIFEST.md', file=sys.stderr)


# ----------------------------------------------------------------------------- MANIFEST.md
def write_md(final, wul, audit, extra, n_mi_local, n_mi_clang, cache, elfc, unres_l):
    L = []
    P = L.append
    by_id = {r['id']: r for r in final}
    P('# CORPUS MANIFEST (single source of truth)\n')
    P('Generated by `scripts/build_corpus_manifest.py` (re-runnable; READ-ONLY on data). Row-level detail: '
      '`corpus_manifest.tsv` / `corpus_manifest.json` in this directory. ELF facts cached in `elf_cache.json`; '
      'HPC inventory cached in `HPC_inventory.json` (from `HPC_inventory_remote.py`, run on the login node).\n')
    P('## Method\n')
    P('- **Binary id** = `<package>_<tool>_<opt>`; opt in {O0,O1,O2,O3} else `default`; package = prefix before first `_`, tool = middle.')
    P('- **Local sources**: `data/raw` (`<id>_sym` and bare `<id>`), `data/stripped` (+`_p0_0_6_{siblings,round2,round3,xrep}`), '
      '`data/cross_project/{stripped,debug,candidates}`, `data/bir` (+subdirs), `data/labels` (`<id>_labels.json` schema A with '
      '`name_to_addr`/`addr_to_name`/`functions` whose direction flips per file, and bare `<id>.json` flat `addr->name`), '
      '`data/external_calls`, `data/graphs` (audit.json B2/B3 per_binary for sub-graph + matched counts; `os.listdir` for non-sub counts), '
      '`data/match_index.json`, `data/split_assignments.json`. Extra corpora `ftdomains`, `ftdomains2`, `clang_train`, `clang_o1o3` '
      '(bins/`<id>` + `<id>.debug`, bir, labels, external, graphs, clang_train/match_index.json).')
    P('- **debug vs stripped** decided per ELF by `readelf -S` (`.symtab` present => debug/unstripped), NOT by filename. '
      '`build_id` from `readelf -n`; `elf_type` from `readelf -h`; `n_fde` = count of ` FDE ` lines inside the `.eh_frame` dump of '
      '`readelf -wF` (`.debug_frame` excluded so debug and stripped copies are comparable). Computed for ALL local ELFs.')
    P('- **elfcheck_ratio**: fraction of the first 300 `sub sub_X(` entries in the .bir whose vaddr X lies in an executable PT_LOAD '
      'segment of the stripped ELF and starts with a plausible prologue byte (port of `archive/unified:experiments_semantic/rcdg_stage0_elfcheck.py`). '
      'Reused from `archive/unified:results/rcdg/corpus_elfcheck.tsv` where present, computed here otherwise. <0.8 => the .bir was lifted from a different build.')
    P('- **HPC** (`$WORKSPACE`): `data/{bir,labels,external_calls,graphs,debug,raw,stripped,cross_project,match_index.json,split_assignments.json}` and '
      '`ccs/data/{labels,external_calls,graphs->data/graphs,match_index.json->data/match_index.json,split_assignments*.json}`. Splits recorded from '
      '`ccs/data/split_assignments.json` (v1), `split_assignments_paper_clean.json`, `split_assignments_paper_clean_strict_idx.json`, `data/split_assignments.json`.')
    P('- **Duplicate builds**: `results/phase0/audit.json` B5.all_pairs, `max(identity_raw, identity_thunk_resolved) >= 0.9` => duplicate opt pair; the lexicographically lowest id in a duplicate group is kept.')
    P('- **eligible_v2** (conservative): requires debug ELF (local or HPC), stripped ELF, .bir, labels (non-empty), sub-graphs, matched functions, '
      'stripped/debug build-id agreement (any copy), elfcheck >= 0.8, not a duplicate build, no dangling match_index entries. '
      'Split membership (`excluded`/`xproject`) is NOT a blocker; it is reported in `role_hint` so held-out packages can be kept out of the train pool.\n')

    # (1) totals
    P('## 1. Totals per corpus / location\n')
    ccount = collections.Counter()
    for r in final:
        for c in r['corpora']:
            ccount[c] += 1
    P('| corpus tag | n ids |'); P('|---|---|')
    for c, n in sorted(ccount.items()):
        P(f'| {c} | {n} |')
    P(f'| **all distinct ids** | **{len(final)}** |\n')
    tot = lambda pred: sum(1 for r in final if pred(r))
    P('| local asset | n ids | files/notes |'); P('|---|---|---|')
    n_raw = len([p for p in os.listdir('data/raw') if os.path.isfile(f'data/raw/{p}')])
    raw_sym = sum(1 for p in os.listdir('data/raw') if p.endswith('_sym'))
    raw_ids = {(p[:-4] if p.endswith('_sym') else p) for p in os.listdir('data/raw')}
    raw_facts = [cache.get(elf_key(f'data/raw/{p}')) for p in os.listdir('data/raw')]
    raw_symtab = sum(1 for f in raw_facts if f and f['has_symtab'])
    P(f'| data/raw | {len(raw_ids)} | {n_raw} files: {raw_sym} `*_sym` + {n_raw - raw_sym} bare; {raw_symtab}/{n_raw} have .symtab '
      f'(bare files are ALSO unstripped debug builds; {n_raw - len(raw_ids)} ids present in both forms) |')
    ns = [p for p in os.listdir('data/stripped') if os.path.isfile(f'data/stripped/{p}')]
    ns_f = [cache.get(elf_key(f'data/stripped/{p}')) for p in ns]
    P(f'| data/stripped (top) | {len({(p[:-9] if p.endswith("_stripped") else p) for p in ns})} | {len(ns)} files: '
      f'{sum(1 for p in ns if p.endswith("_stripped"))} `*_stripped` + {sum(1 for p in ns if not p.endswith("_stripped"))} bare; '
      f'{sum(1 for f in ns_f if f and f["has_symtab"])} have .symtab |')
    for d in sorted(glob.glob('data/stripped/_p0_0_6_*')):
        P(f'| {d} | {len(os.listdir(d))} | all `*_stripped` |')
    for sub in ['stripped', 'debug', 'candidates']:
        d = f'data/cross_project/{sub}'; fl = os.listdir(d)
        fs = [cache.get(elf_key(f'{d}/{p}')) for p in fl]
        P(f'| {d} | {len({(p[:-9] if p.endswith("_stripped") else p) for p in fl})} | {len(fl)} files; {sum(1 for f in fs if f and f["has_symtab"])} with .symtab |')
    notelf = sorted({n.split(':', 1)[1] for r in final for n in r['notes'] if n.startswith('not-elf:')})
    P(f'| NOT-ELF files found among ELF dirs | {len(notelf)} | libtool wrapper shell scripts etc.: ' + ', '.join(notelf) + ' |')
    P(f'| data/bir (+subdirs) | {tot(lambda r: bool(r["has_bir"]) and any(c.startswith("local_main") for c in r["corpora"]))} | '
      f'{len(glob.glob("data/bir/*.bir"))} top + {len(glob.glob("data/bir/_p0_0_6_*/*.bir"))} in subdirs |')
    P(f'| data/labels | {tot(lambda r: bool(r["label_paths"]) and any(p.startswith("data/labels") for p in r["label_paths"]))} | '
      f'{len(glob.glob("data/labels/*_labels.json"))} `_labels.json` + {len(glob.glob("data/labels/*.json")) - len(glob.glob("data/labels/*_labels.json"))} bare `.json` |')
    P(f'| data/external_calls | {len(glob.glob("data/external_calls/*_external.json"))} | |')
    P(f'| data/graphs | {tot(lambda r: (r.get("n_sub_graphs") or 0) > 0 and "local_main" in r["corpora"])} ids with sub-graphs | '
      f'{audit["scan"]["n_files"]} files (audit.json scan); {unres_l} non-sub files unresolvable to an id |')
    P(f'| data/match_index.json | {tot(lambda r: (r.get("n_matched") or 0) > 0)} ids | {n_mi_local} matched functions |')
    sp = json.load(open('data/split_assignments.json'))
    P(f'| data/split_assignments.json | {sum(len(v) for v in sp.values())} | ' + ', '.join(f'{k}={len(v)}' for k, v in sp.items()) + ' |')
    for c in extra:
        nb = len(os.listdir(f'{c}/bins'))
        P(f'| {c} | {tot(lambda r: c in r["corpora"])} | bins={nb} files, bir={len(glob.glob(c + "/bir/*.bir"))}, labels={len(glob.glob(c + "/labels/*.json"))}, '
          f'external={len(glob.glob(c + "/external/*.json"))}, graphs={len(os.listdir(c + "/graphs"))} files' +
          (f', match_index={n_mi_clang} fns' if c == 'clang_train' else '') + ' |')
    if wul:
        P('\n**HPC** (`$WORKSPACE`):\n')
        P('| location | count |'); P('|---|---|')
        P(f'| data/bir | {len(wul["data_bir"])} .bir |')
        P(f'| data/labels (= ccs/data/labels) | {len(wul["data_labels"])} / {len(wul["ccs_labels"])} ids |')
        P(f'| data/external_calls (= ccs) | {len(wul["data_external"])} / {len(wul["ccs_external"])} |')
        P(f'| data/graphs (ccs/data/graphs symlinks here) | {len(wul["data_graphs"]["sub"])} ids; {sum(wul["data_graphs"]["sub"].values())} sub + {sum(wul["data_graphs"]["nonsub"].values())} non-sub files |')
        P(f'| data/match_index.json (ccs symlink) | {len(wul["data_match_index"])} ids; {wul.get("data_match_index_n")} fns |')
        P(f'| data/debug | {len(wul["data_debug"])} | ')
        P(f'| data/raw | {len(wul["data_raw"])} |')
        P(f'| data/stripped | {len([x for x in wul["data_stripped"] if x != ".gitkeep"])} |')
        P(f'| data/cross_project | ' + ', '.join(f'{k}={len(v)}' for k, v in wul['data_cross_project'].items()) + ' |')
        for tag in ['data_split', 'ccs_split_v1', 'ccs_split_paper_clean', 'ccs_split_paper_clean_strict_idx']:
            v = wul.get(tag)
            if isinstance(v, dict):
                P(f'| {tag} | ' + ', '.join(f'{k}={len(x)}' for k, x in v.items()) + ' |')
        P(f'| corpus_stage | {", ".join(wul["corpus_stage"])} (source tarballs only, no binaries) |')
        P(f'| build_tmp | {len(wul["build_tmp"])} entries: source trees + tarballs (' + ', '.join(x for x in wul['build_tmp'] if not x.endswith(('.xz', '.gz', '.bz2', '.lz')))[:300] + ') |')

    # (2) Venn
    P('\n## 2. Coverage Venn (local unless stated)\n')
    loc = lambda r: any(c.startswith('local_main') for c in r['corpora'])
    sets = {
        'bir but no stripped ELF': [r['id'] for r in final if r['has_bir'] and not r['has_stripped_elf']],
        'stripped ELF but no bir': [r['id'] for r in final if r['has_stripped_elf'] and not r['has_bir']],
        'labels but no graphs': [r['id'] for r in final if r['has_labels'] and not (r.get('n_sub_graphs') or 0)],
        'graphs but no labels': [r['id'] for r in final if (r.get('n_sub_graphs') or 0) and not r['has_labels']],
        'debug ELF but no labels': [r['id'] for r in final if r['has_debug_elf'] and not r['has_labels']],
        'labels but no debug ELF (local)': [r['id'] for r in final if r['has_labels'] and not r['has_debug_elf']],
        'graphs but not in local match_index': [r['id'] for r in final if (r.get('n_sub_graphs') or 0) and not (r.get('n_matched') or 0) and 'local_main' in r['corpora']],
        'HPC-only ids (no local trace at all)': [r['id'] for r in final if r['corpora'] == ['HPC']],
        'local-only ids (absent from HPC) [all corpora]': [r['id'] for r in final if 'HPC' not in r['corpora']],
        'local-only ids, local_main* only': [r['id'] for r in final if 'HPC' not in r['corpora'] and all(c.startswith('local_main') for c in r['corpora'])],
        'in local match_index but not in local split file': [r['id'] for r in final if (r.get('n_matched') or 0) and not r['local_split']],
        'in local split file but no local sub-graphs': [r['id'] for r in final if r['local_split'] and not (r.get('n_sub_graphs') or 0)],
        'HPC bir but no local bir': [r['id'] for r in final if r.get('HPC_has_bir') and not r['has_bir']],
        'local bir but no HPC bir': [r['id'] for r in final if r['has_bir'] and not r.get('HPC_has_bir') and 'HPC' in r['corpora']],
        'HPC labels but no local labels': [r['id'] for r in final if r.get('HPC_has_labels') and not r['has_labels']],
        'HPC graphs but no local graphs': [r['id'] for r in final if (r.get('HPC_n_sub_graphs') or 0) and not (r.get('n_sub_graphs') or 0)],
        'multiple stripped ELF copies with DIFFERENT build-ids': [r['id'] for r in final if len(r.get('build_ids_stripped_all') or []) > 1],
        'multiple debug ELF copies with DIFFERENT build-ids': [r['id'] for r in final if len(r.get('build_ids_debug_all') or []) > 1],
    }
    P('| set | n | examples (full list: filter TSV) |'); P('|---|---|---|')
    for k, v in sets.items():
        pk = collections.Counter(parse_id(x)[0] for x in v)
        ex = ', '.join(v[:6]) + (' ...' if len(v) > 6 else '')
        P(f'| {k} | {len(v)} | {len(pk)} pkgs; {ex} |')
    # (3) build consistency
    P('\n## 3. Build consistency\n')
    mism = [r for r in final if r['build_id_any_match'] is False]
    mism_primary = [r for r in final if r['build_id_match'] is False]
    P(f'- ids where the canonical stripped ELF build-id != canonical debug ELF build-id: **{len(mism_primary)}**; '
      f'of these, NO stripped copy matches ANY debug copy: **{len(mism)}**.')
    if mism_primary:
        P('\n| id | canonical stripped (build-id) | canonical debug (build-id) | resolvable |'); P('|---|---|---|---|')
        for r in mism_primary:
            P(f'| {r["id"]} | {r["has_stripped_elf"]} ({(r["build_id_stripped"] or "")[:8]}) | {r["has_debug_elf"]} ({(r["build_id_debug"] or "")[:8]}) | '
              f'{"yes" if r["build_id_any_match"] else "NO"} |')
        P('')
    low = [r for r in final if r['elfcheck_ratio'] is not None and r['elfcheck_ratio'] < 0.8]
    nchk = sum(1 for r in final if r['elfcheck_ratio'] is not None)
    P(f'- elfcheck computed for {nchk} ids ({sum(1 for r in final if r["elfcheck_src"] == "archive")} from archive tsv, '
      f'{sum(1 for r in final if r["elfcheck_src"] == "computed")} computed now); **{len(low)}** with ratio < 0.8:')
    for r in low:
        P(f'  - `{r["id"]}` ratio={r["elfcheck_ratio"]:.2f} ({r["elfcheck_src"]}; stripped={r["has_stripped_elf"]}; copies={r["n_stripped_elfs"]}; build_id_match={r["build_id_match"]})')
    noelfchk = [r['id'] for r in final if r['has_bir'] and r['has_stripped_elf'] and r['elfcheck_ratio'] is None]
    P(f'- ids with bir+stripped ELF but no elfcheck value: {len(noelfchk)} {noelfchk[:10]}')
    # (4) FDE coverage
    P('\n## 4. FDE coverage (n_fde of canonical stripped ELF vs n_labels)\n')
    P('| opt | n ids (both) | median n_fde | median n_labels | median n_fde/n_labels | frac ids with .eh_frame | ids n_fde==0 |')
    P('|---|---|---|---|---|---|---|')
    for opt in list(OPTS) + ['default']:
        rr = [r for r in final if r['opt_tag'] == opt and r['n_fde'] is not None and r['n_labels']]
        allr = [r for r in final if r['opt_tag'] == opt and r['n_fde'] is not None]
        if not allr:
            continue
        ratios = [r['n_fde'] / r['n_labels'] for r in rr]
        P(f'| {opt} | {len(rr)} | {statistics.median([r["n_fde"] for r in rr]) if rr else "-"} | '
          f'{statistics.median([r["n_labels"] for r in rr]) if rr else "-"} | {statistics.median(ratios):.2f} | '
          f'{sum(1 for r in allr if r["has_eh_frame"]) / len(allr):.2f} | {sum(1 for r in allr if r["n_fde"] == 0)} |'
          if rr else f'| {opt} | 0 | - | - | - | {sum(1 for r in allr if r["has_eh_frame"]) / len(allr):.2f} | {sum(1 for r in allr if r["n_fde"] == 0)} |')
    P('\nn_labels counts unique names in the label file (includes crt/libc stubs like `_start`, `deregister_tm_clones`); '
      'FDEs cover every function with unwind info, including PLT/crt entries, so ratios around 1 are expected; >>1 or <<1 flags something odd.')
    et = collections.Counter(r['elf_type'] for r in final if r['elf_type'])
    P(f'\nELF type distribution (canonical ELF): {dict(et)}. Per opt: ' + '; '.join(
        f'{opt}: ' + str(dict(collections.Counter(r['elf_type'] for r in final if r['opt_tag'] == opt and r['elf_type'])))
        for opt in list(OPTS) + ['default']))
    # (5) per package
    P('\n## 5. Per package: tools x opts, duplicate opt-pairs (audit B5, identity >= 0.9)\n')
    pk = collections.defaultdict(lambda: {'tools': set(), 'opts': set(), 'ids': [], 'corp': set()})
    for r in final:
        p = pk[r['package']]; p['tools'].add(r['tool']); p['opts'].add(r['opt_tag']); p['ids'].append(r['id']); p['corp'] |= set(r['corpora'])
    pairs = audit['B5']['all_pairs']
    dup_pairs = [x for x in pairs if max(x.get('identity_raw') or 0, x.get('identity_thunk_resolved') or 0) >= 0.9]
    dp_by_pkg = collections.Counter(parse_id(x['bin1'])[0] for x in dup_pairs)
    ap_by_pkg = collections.Counter(parse_id(x['bin1'])[0] for x in pairs)
    P(f'Total opt pairs audited: {len(pairs)}; duplicate pairs (identity>=0.9): **{len(dup_pairs)}**; '
      f'by opt-pair: {dict(collections.Counter(x["opt1"] + "-" + x["opt2"] for x in dup_pairs))}. '
      f'ids flagged as non-keeper duplicates: {sum(1 for r in final if any(s.startswith("duplicate_build_of") for s in r["ineligible_reasons"]))}.\n')
    with open(f'{OUT}/per_package.tsv', 'w') as f:
        f.write('package\tn_ids\tn_tools\topts\tpairs_audited\tdup_pairs\tcorpora\tn_eligible_v2\tids\n')
        for name in sorted(pk):
            p = pk[name]
            f.write('\t'.join([name, str(len(p['ids'])), str(len(p['tools'])), ','.join(sorted(p['opts'])), str(ap_by_pkg.get(name, 0)),
                                str(dp_by_pkg.get(name, 0)), ','.join(sorted(p['corp'])), str(sum(1 for i in p['ids'] if by_id[i]['eligible_v2'])),
                                ';'.join(sorted(p['ids']))]) + '\n')
    full = [n for n in sorted(pk) if sum(1 for i in pk[n]['ids'] if by_id[i]['eligible_v2']) == len(pk[n]['ids']) and dp_by_pkg.get(n, 0) == 0]
    P(f'{len(pk)} packages total (full table: `per_package.tsv`). {len(full)} packages are fully eligible with no duplicate pairs: ' + ', '.join(full) + '.\n')
    P('Packages with duplicate opt-pairs or non-eligible ids:\n')
    P('| package | n ids | tools | opts | pairs | dup pairs | corpora | eligible | top ineligibility reasons |'); P('|---|---|---|---|---|---|---|---|---|')
    for name in sorted(pk):
        if name in full:
            continue
        p = pk[name]
        elig = sum(1 for i in p['ids'] if by_id[i]['eligible_v2'])
        rs = collections.Counter(x.split('(')[0].split(':')[0] for i in p['ids'] for x in by_id[i]['ineligible_reasons'])
        P(f'| {name} | {len(p["ids"])} | {len(p["tools"])} | {",".join(sorted(p["opts"]))} | {ap_by_pkg.get(name, 0)} | {dp_by_pkg.get(name, 0)} | '
          f'{",".join(sorted(c.replace("local_main", "L") for c in p["corp"]))} | {elig} | {", ".join(f"{k}={v}" for k, v in rs.most_common(3))} |')
    # (6) eligibility
    P('\n## 6. Eligibility for dataset v2\n')
    ne = sum(1 for r in final if r['eligible_v2'])
    P(f'- eligible_v2 = **{ne}** / {len(final)} ids. Reason histogram (an id can carry several):')
    rc = collections.Counter()
    for r in final:
        for s in r['ineligible_reasons']:
            rc[s.split('(')[0].split(':')[0]] += 1
    for k, v in rc.most_common():
        P(f'  - {k}: {v}')
    rh = collections.Counter(x.split('(')[0] for r in final for x in r.get('role_hint', []))
    P('- role hints (not blockers): ' + ', '.join(f'{k}={v}' for k, v in rh.most_common()))
    P('- eligible AND role_hint contains xproject_holdout (keep OUT of train pool): ' + str(sum(1 for r in final if r['eligible_v2'] and any(x.startswith('xproject') for x in r['role_hint']))))
    P('- eligible by opt: ' + ', '.join(f'{o}={sum(1 for r in final if r["eligible_v2"] and r["opt_tag"] == o)}' for o in list(OPTS) + ['default']))
    P('- eligible by corpus: ' + ', '.join(f'{c}={n}' for c, n in sorted(collections.Counter(c for r in final if r['eligible_v2'] for c in r['corpora']).items())))
    P('- eligible with local debug ELF (labels re-derivable locally): ' + str(sum(1 for r in final if r['eligible_v2'] and r['has_debug_elf'])) +
      '; eligible only via HPC assets: ' + str(sum(1 for r in final if r['eligible_v2'] and not r['has_debug_elf'])))
    nd = sum(r.get('n_matched_dangling') or 0 for r in final)
    P(f'- local data/match_index.json entries whose graph file is missing on disk: {nd} (ids: ' + ', '.join(r['id'] for r in final if r.get('n_matched_dangling')) + ')')
    P('\nCaveats: `no_matched_functions` for ftdomains/ftdomains2/cross_project ids means "not in any match_index" (those corpora are matched at eval time '
      'via match_local.py / ground_truth.json), not "unmatchable"; they need a re-match step before entering v2. Eligibility trusts existing labels/graphs where debug ELF is only on HPC; `no_debug_elf` ids can still be used '
      'if their labels are trusted, but names cannot be re-derived. Duplicate-group keeper = lexicographically lowest id (usually the lower opt).')
    open(f'{OUT}/MANIFEST.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
