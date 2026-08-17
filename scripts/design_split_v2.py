#!/usr/bin/env python3
"""Dataset v2 split designer (plan §3; defects B5/B8/B9/B10).

Input : data/match_index_v2.json (matcher v2 records), results/phase0/manifest/corpus_manifest.tsv
        (duplicate-build keeper flags), package role config below.
Output: data/split_v2.json  (schema v2, consumed by FunctionDataset.get_splits):
          train, val_indist, val_xproj, test, xproject, excluded  (lists of binary ids)
          + "meta": regime per xproject package, family map, dedup counts, generation args
        docs/DATASET_V2_CARD.md  (stats + leakage table per held-out package)

Assignment unit = PROGRAM = (package, tool): all optimisation levels of a program move
together (otherwise O0/O2 builds of the same source leak across splits).

Tiers
  train        : programs of training packages
  val_indist   : held-out programs of training packages   (~VAL_INDIST_FRAC of train fns)
  test         : held-out programs of training packages   (~TEST_INDIST_FRAC)
  val_xproj    : whole packages, disjoint from train AND from xproject; regime-mixed
                 (this is the dev set every threshold / router / early-stopping uses)
  xproject     : whole packages, the reported cross-project test; each package tagged with a
                 regime: NCT (near-clone: version/fork of a training package, family overlap
                 >= NCT_NAME_OVERLAP) or FT (far transfer)
  excluded     : duplicate builds (B5), programs with < MIN_FNS functions, ids without labels,
                 anything the caller lists in EXCLUDE

Families: packages whose function-name sets overlap >= FAMILY_OVERLAP (Jaccard on names of
functions with >= 5 tokens) are one family (e.g. coreutils/coreutils2/3/4, nginx/nginx118/
angie/tengine/openresty, gawk/gawk2).  A family is never split between train and val/test
UNLESS one member is deliberately designated NCT test (that is the point of the NCT tier).

Leakage table (per held-out package): verbatim-name overlap with train (% of functions whose
canonical name or any alias appears as a train name), body-duplicate rate (tok_hash seen in
train), sub-token composability (% of names whose Votes sub-tokens are all in the train
name vocabulary — computed later by the vocab builder), in_dynsym share.
"""
import argparse, csv, json, os, random, re, sys
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ------------------------------------------------------------------ configuration
XPROJECT_FT = ['dash', 'gettext', 'psmisc', 'recutils',            # CCS far-transfer set (comparability)
               'expat', 'jansson', 'lmdb', 'mbedtls', 'libsodium',  # new domains (parser/crypto/db)
               'cvs', 'lighttpd', 'tinycc', 'fossil']              # hard/unseen domains
XPROJECT_NCT = ['nginx118', 'angie', 'tengine', 'openresty', 'nginx114', 'nginx126',
                'gawk2', 'grep2', 'sed2', 'gzip2', 'tar2', 'units2', 'which2', 'patch2',
                'findutils2', 'diffutils2', 'inetutils2', 'coreutils4']
VAL_XPROJ = ['rush', 'cppi', 'direvent', 'csplit2', 'wdiff', 'spell',   # small FT-like
             'coreutils3',                                              # near-clone of coreutils (version)
             'zstd', 'tig', 'iotop']                                    # non-GNU domains
EXCLUDE_PKGS = ['libtool', 'combinatorics']
VAL_INDIST_FRAC, TEST_INDIST_FRAC = 0.05, 0.10
FAMILY_OVERLAP = 0.35
MIN_FNS = 20
SEED = 20260817


def pkg_of(bid):
    return bid.split('_')[0]


def program_of(bid):
    return re.sub(r'_O[0-3]$', '', bid)


def opt_of(bid):
    m = re.search(r'_O([0-3])$', bid)
    return 'O' + m.group(1) if m else 'default'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--match-index', default='data/match_index_v2.json')
    ap.add_argument('--manifest', default='results/phase0/manifest/corpus_manifest.tsv')
    ap.add_argument('--out', default='data/split_v2.json')
    ap.add_argument('--card', default='docs/DATASET_V2_CARD.md')
    ap.add_argument('--corpora', nargs='*', default=None, help='restrict to these corpus tags')
    args = ap.parse_args()
    random.seed(SEED)

    recs = json.load(open(args.match_index))
    if args.corpora:
        recs = [r for r in recs if r.get('corpus') in set(args.corpora)]
    by_bin = defaultdict(list)
    for r in recs:
        by_bin[r['binary']].append(r)
    bins = sorted(by_bin)
    print(f'{len(recs)} records, {len(bins)} binaries')

    # ---- duplicate builds from manifest (keeper flag) -------------------------------------
    dup_excluded = set()
    if os.path.exists(args.manifest):
        with open(args.manifest) as fh:
            for row in csv.DictReader(fh, delimiter='\t'):
                if row.get('duplicate_build', '').lower() in ('1', 'true', 'yes') and row.get('id') in by_bin:
                    dup_excluded.add(row['id'])
    # in-corpus duplicate detection as a second net: two opt builds of one program whose
    # multiset of tok_hash is >= 95% identical -> keep the lower opt tag
    prog_bins = defaultdict(list)
    for b in bins:
        prog_bins[program_of(b)].append(b)
    for prog, bl in prog_bins.items():
        bl = sorted(bl, key=lambda b: opt_of(b))
        kept_hashes = []
        for b in bl:
            hs = Counter(r['tok_hash'] for r in by_bin[b] if r['n_tokens'] >= 5)
            dup = False
            for kh in kept_hashes:
                inter = sum((hs & kh).values()); tot = max(1, sum(hs.values()))
                if tot >= MIN_FNS and inter / tot >= 0.95:
                    dup = True; break
            if dup:
                dup_excluded.add(b)
            else:
                kept_hashes.append(hs)
    small = {b for b in bins if len(by_bin[b]) < MIN_FNS}
    excluded = set(dup_excluded) | small | {b for b in bins if pkg_of(b) in EXCLUDE_PKGS}
    live = [b for b in bins if b not in excluded]

    # ---- families ------------------------------------------------------------------------
    pkg_names = defaultdict(set)
    for b in live:
        for r in by_bin[b]:
            if r['n_tokens'] >= 5:
                pkg_names[pkg_of(b)].add(r['real_name'])
    pkgs = sorted(pkg_names)
    parent = {p: p for p in pkgs}
    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]; p = parent[p]
        return p
    fam_pairs = []
    for i in range(len(pkgs)):
        for j in range(i + 1, len(pkgs)):
            a, b = pkg_names[pkgs[i]], pkg_names[pkgs[j]]
            if min(len(a), len(b)) < MIN_FNS:
                continue
            ov = len(a & b) / min(len(a), len(b))
            if ov >= FAMILY_OVERLAP:
                fam_pairs.append((pkgs[i], pkgs[j], round(ov, 3)))
                parent[find(pkgs[i])] = find(pkgs[j])
    families = defaultdict(list)
    for p in pkgs:
        families[find(p)].append(p)
    families = {k: sorted(v) for k, v in families.items() if len(v) > 1}

    # ---- roles ---------------------------------------------------------------------------
    role = {}
    ft = set(XPROJECT_FT); nct = set(XPROJECT_NCT); vx = set(VAL_XPROJ)
    for p in pkgs:
        if p in ft:
            role[p] = 'xproject_FT'
        elif p in nct:
            role[p] = 'xproject_NCT'
        elif p in vx:
            role[p] = 'val_xproj'
        else:
            role[p] = 'train_pool'
    # family consistency: a train_pool package in a family with an FT package would leak → move
    # it to val_xproj? No: FT means far — if a family member is in train the package is not FT.
    warnings = []
    for fam, members in families.items():
        roles = {role[m] for m in members}
        if 'xproject_FT' in roles and 'train_pool' in roles:
            warnings.append(f'family {members}: FT package shares a family with training package(s) — reclassify as NCT')
            for m in members:
                if role[m] == 'xproject_FT':
                    role[m] = 'xproject_NCT'
        if 'val_xproj' in roles and 'xproject_FT' in roles:
            warnings.append(f'family {members}: val_xproj and xproject_FT in one family')

    # ---- in-distribution split of the train pool by program --------------------------------
    train_progs = sorted({program_of(b) for b in live if role[pkg_of(b)] == 'train_pool'})
    # stratify by package: shuffle programs within package, take last k% for val/test
    prog_by_pkg = defaultdict(list)
    for pr in train_progs:
        prog_by_pkg[pkg_of(pr)].append(pr)
    val_progs, test_progs = set(), set()
    n_fn = lambda pr: sum(len(by_bin[b]) for b in prog_bins[pr] if b in set(live))
    for p, prs in sorted(prog_by_pkg.items()):
        prs = sorted(prs); random.shuffle(prs)
        if len(prs) < 3:
            continue           # single-program packages stay entirely in train
        tot = sum(n_fn(pr) for pr in prs)
        acc = 0
        for pr in prs:
            if acc < tot * TEST_INDIST_FRAC:
                test_progs.add(pr)
            elif acc < tot * (TEST_INDIST_FRAC + VAL_INDIST_FRAC):
                val_progs.add(pr)
            else:
                break
            acc += n_fn(pr)
    split = {'train': [], 'val_indist': [], 'val_xproj': [], 'test': [], 'xproject': [], 'excluded': sorted(excluded)}
    for b in live:
        p, pr = pkg_of(b), program_of(b)
        r = role[p]
        if r == 'train_pool':
            split['test' if pr in test_progs else 'val_indist' if pr in val_progs else 'train'].append(b)
        elif r == 'val_xproj':
            split['val_xproj'].append(b)
        else:
            split['xproject'].append(b)

    # ---- leakage table -------------------------------------------------------------------
    train_names, train_hashes, train_name_hash = set(), set(), set()
    for b in split['train']:
        for r in by_bin[b]:
            train_names.add(r['real_name']); train_names.update(r.get('aliases', []))
            if r['n_tokens'] >= 10:
                train_hashes.add(r['tok_hash'])
            train_name_hash.add((r['real_name'], r['tok_hash']))
    def leak(pkg_bins):
        rs = [r for b in pkg_bins for r in by_bin[b]]
        n = len(rs)
        verb = sum(1 for r in rs if r['real_name'] in train_names or any(a in train_names for a in r.get('aliases', [])))
        big = [r for r in rs if r['n_tokens'] >= 10]
        body = sum(1 for r in big if r['tok_hash'] in train_hashes)
        nb = sum(1 for r in rs if (r['real_name'], r['tok_hash']) in train_name_hash)
        dyn = sum(1 for r in rs if r.get('in_dynsym'))
        return {'n_fns': n, 'n_bins': len(pkg_bins), 'verbatim_name_pct': round(100 * verb / max(1, n), 1),
                'body_dup_ge10_pct': round(100 * body / max(1, len(big)), 1),
                'name_and_body_dup_pct': round(100 * nb / max(1, n), 1),
                'in_dynsym_pct': round(100 * dyn / max(1, n), 1)}
    tiers = {}
    for tier in ('val_indist', 'test', 'val_xproj', 'xproject'):
        per_pkg = defaultdict(list)
        for b in split[tier]:
            per_pkg[pkg_of(b)].append(b)
        tiers[tier] = {p: dict(leak(bl), regime=role[p].replace('xproject_', '') if tier == 'xproject' else tier)
                       for p, bl in sorted(per_pkg.items())}

    meta = {'seed': SEED, 'families': families, 'family_pairs': fam_pairs, 'roles': role,
            'warnings': warnings, 'n_dup_excluded': len(dup_excluded), 'n_small_excluded': len(small),
            'tiers': tiers,
            'counts': {k: {'binaries': len(v), 'functions': sum(len(by_bin[b]) for b in v)} for k, v in split.items()}}
    split['meta'] = meta
    json.dump(split, open(args.out, 'w'), indent=1, sort_keys=True)

    # ---- card ----------------------------------------------------------------------------
    L = ['# Dataset v2 — split card (generated by scripts/design_split_v2.py)', '',
         f'Records: {len(recs)} functions / {len(bins)} binaries; excluded {len(excluded)} binaries '
         f'({len(dup_excluded)} duplicate builds, {len(small)} with < {MIN_FNS} functions).', '',
         '| tier | binaries | functions |', '|---|---|---|']
    for k, v in meta['counts'].items():
        L.append(f"| {k} | {v['binaries']} | {v['functions']} |")
    L += ['', '## Families (name overlap >= %.2f)' % FAMILY_OVERLAP]
    for fam, mem in sorted(families.items()):
        L.append(f'- {mem}')
    if warnings:
        L += ['', '## Warnings'] + [f'- {w}' for w in warnings]
    for tier in ('xproject', 'val_xproj', 'test', 'val_indist'):
        L += ['', f'## {tier} — leakage table (vs train)', '',
              '| package | regime | bins | fns | verbatim-name % | body-dup (>=10 tok) % | name+body dup % | in_dynsym % |', '|---|---|---|---|---|---|---|---|']
        for p, d in tiers[tier].items():
            L.append(f"| {p} | {d['regime']} | {d['n_bins']} | {d['n_fns']} | {d['verbatim_name_pct']} | {d['body_dup_ge10_pct']} | {d['name_and_body_dup_pct']} | {d['in_dynsym_pct']} |")
    os.makedirs(os.path.dirname(args.card), exist_ok=True)
    open(args.card, 'w').write('\n'.join(L) + '\n')
    print(json.dumps(meta['counts'], indent=1))
    for w in warnings:
        print('WARNING:', w)
    print(f'wrote {args.out} and {args.card}')


if __name__ == '__main__':
    main()
