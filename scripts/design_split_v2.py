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

Tiers (policy v3, agreed 2026-08-23: three package-disjoint tiers, no in-distribution eval)
  train        : all binaries of training package families
  val_xproj    : whole packages, disjoint from train AND test — the ONLY dev set (model
                 selection, thresholds, router). Written under the v2 key `val_xproj`;
                 `val_indist` is emitted EMPTY so the v2 loader falls back to val_xproj.
  test         : whole held-out packages = former xproject (FT + NCT) + the reserve pool.
                 Each package carries a regime tag in meta.test_regime: NCT (shares a
                 family with a training package) or FT (far transfer). One test set,
                 reported as one headline + two regime sub-rows.
  excluded     : duplicate builds (B5), binaries with < MIN_FNS functions, EXCLUDE_PKGS
Record-level policy (applied by FunctionDatasetV2.apply_split_policy, stats in meta.policy):
  train  : one sample per (tok_hash, name)
  val/test: drop samples whose tok_hash occurs in train; drop in_dynsym (symbol-visible)

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
XPROJECT_RESERVE = ['atop', 'bdb', 'bsdtar', 'byacc', 'diffutils3', 'entr', 'file', 'gdbm',
                    'gperf2', 'icu', 'lsof', 'mawk', 'mksh', 'mutt', 'procps', 'pv', 'sbase',
                    'sysstat', 'tcsh', 'tdb',                         # 2026-08 Wulver harvest, never trained on
                    'sggmp', 'sglibpng', 'sglibmicrohttpd', 'sgpoke', 'sglibredwg']  # SymGen-corpus external holdout (B1, confirmed 2026-08-26)
VAL_XPROJ = ['rush', 'cppi', 'direvent', 'csplit2', 'wdiff', 'spell',   # small FT-like
             'coreutils3',                                              # near-clone of coreutils (version)
             'zstd', 'tig', 'iotop']                                    # non-GNU domains
EXCLUDE_PKGS = ['libtool', 'combinatorics']
FAMILY_OVERLAP = 0.35
NCT_NAME_OVERLAP = 60.0   # % verbatim-name overlap with train at/above which a held-out package is NCT
UBIQ_PKGS = 3          # a name in >= this many packages is 'ubiquitous' and ignored for family detection
MIN_FNS = 20          # applies to train-pool binaries; held-out binaries need >= MIN_FNS_HELDOUT
MIN_FNS_HELDOUT = 5
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
    heldout_pkgs = set(XPROJECT_FT) | set(XPROJECT_NCT) | set(XPROJECT_RESERVE) | set(VAL_XPROJ)
    small = {b for b in bins if len(by_bin[b]) < (MIN_FNS_HELDOUT if pkg_of(b) in heldout_pkgs else MIN_FNS)}
    excluded = set(dup_excluded) | small | {b for b in bins if pkg_of(b) in EXCLUDE_PKGS}
    live = [b for b in bins if b not in excluded]

    # ---- families ------------------------------------------------------------------------
    pkg_names = defaultdict(set)
    for b in live:
        for r in by_bin[b]:
            if r['n_tokens'] >= 5:
                pkg_names[pkg_of(b)].add(r['real_name'])
    pkgs = sorted(pkg_names)
    # names shared by >= UBIQ_PKGS packages (gnulib/libc-style helpers) must not define families:
    # otherwise every gnulib user chains into one 40-package family. Their leakage is still
    # measured per package in the leakage table (verbatim-name %).
    name_pkg_count = Counter(n for p in pkgs for n in pkg_names[p])
    ubiquitous = {n for n, c in name_pkg_count.items() if c >= UBIQ_PKGS}
    pkg_names = {p: pkg_names[p] - ubiquitous for p in pkgs}
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
    ft = set(XPROJECT_FT) | set(XPROJECT_RESERVE); nct = set(XPROJECT_NCT); vx = set(VAL_XPROJ)
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
        if 'val_xproj' in roles and ('xproject_FT' in roles or 'xproject_NCT' in roles):
            warnings.append(f'family {members}: val_xproj shares a family with a test package')

    # ---- three package-disjoint tiers (v2 loader schema; val_indist / xproject empty) -----
    split = {'train': [], 'val_indist': [], 'val_xproj': [], 'test': [], 'xproject': [],
             'excluded': sorted(excluded)}
    for b in live:
        r = role[pkg_of(b)]
        split['train' if r == 'train_pool' else 'val_xproj' if r == 'val_xproj' else 'test'].append(b)

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
    for tier in ('val_xproj', 'test'):
        per_pkg = defaultdict(list)
        for b in split[tier]:
            per_pkg[pkg_of(b)].append(b)
        tiers[tier] = {p: dict(leak(bl), regime=role[p].replace('xproject_', '') if tier == 'test' else
                               ('NCT' if any(p in m for m in families.values()) else 'FT'))
                       for p, bl in sorted(per_pkg.items())}
    # data-driven regime: a package whose names are mostly already training names is a
    # near-clone regardless of family detection (catches single-binary siblings, e.g. diffutils3)
    for tier in tiers.values():
        for p, d in tier.items():
            if d['regime'] == 'FT' and d['verbatim_name_pct'] >= NCT_NAME_OVERLAP:
                d['regime'] = 'NCT'; warnings.append(f'{p}: verbatim-name {d["verbatim_name_pct"]}% >= {NCT_NAME_OVERLAP} -> NCT')
    test_regime = {p: d['regime'] for p, d in tiers['test'].items()}
    # record-level policy preview (the loader applies the same rules; numbers must agree)
    seen = set(); n_train = 0; n_train_kept = 0
    all_train_hashes = set()
    for b in split['train']:
        for r in by_bin[b]:
            n_train += 1; all_train_hashes.add(r['tok_hash'])
            k = (r['tok_hash'], r['real_name'])
            if k not in seen:
                seen.add(k); n_train_kept += 1
    policy = {'train_raw': n_train, 'train_dedup_hash_name': n_train_kept}
    for tier in ('val_xproj', 'test'):
        rs = [r for b in split[tier] for r in by_bin[b]]
        body = sum(1 for r in rs if r['tok_hash'] in all_train_hashes)
        dyn = sum(1 for r in rs if r.get('in_dynsym') and r['tok_hash'] not in all_train_hashes)
        policy[tier] = {'raw': len(rs), 'drop_body_in_train': body, 'drop_in_dynsym': dyn,
                        'scored': len(rs) - body - dyn}

    meta = {'schema': 'v2', 'policy_version': 'v3-2026-08-23', 'seed': SEED, 'families': families,
            'family_pairs': fam_pairs, 'roles': role, 'test_regime': test_regime, 'policy': policy,
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
    L += ['', '## Record-level policy (train dedup; eval drops body-in-train and in_dynsym)', '', '```', json.dumps(policy, indent=1), '```']
    for tier in ('test', 'val_xproj'):
        L += ['', f'## {tier} — leakage table (vs train, before record-level drops)', '',
              '| package | regime | bins | fns | verbatim-name % | body-dup (>=10 tok) % | name+body dup % | in_dynsym % |', '|---|---|---|---|---|---|---|---|']
        for p, d in tiers[tier].items():
            L.append(f"| {p} | {d['regime']} | {d['n_bins']} | {d['n_fns']} | {d['verbatim_name_pct']} | {d['body_dup_ge10_pct']} | {d['name_and_body_dup_pct']} | {d['in_dynsym_pct']} |")
    os.makedirs(os.path.dirname(args.card) or '.', exist_ok=True)
    open(args.card, 'w').write('\n'.join(L) + '\n')
    print(json.dumps(meta['counts'], indent=1))
    for w in warnings:
        print('WARNING:', w)
    print(f'wrote {args.out} and {args.card}')


if __name__ == '__main__':
    main()
