#!/usr/bin/env python3
"""B1: build the SymGen-corpus relift source for relift_v2.py.
Creates relift_ws/symgen/bins/<id>.debug -> symgen_corpus/binaries/x86_64/<opt>/<proj>/<file>
for the 24 ingestible projects (9 eval-colliding projects EXCLUDED, see docs/B1_SYMGEN_CORPUS_INGEST.md).
id = sg<proj>_<tool>_<opt>   (pkg prefix 'sg' keeps SymGen builds distinguishable from our own packages;
                              '_' inside tool names -> '-' so pkg_of/opt_of split stays valid)
Writes relift_ws/symgen/ingest_manifest.tsv and prints EFFECT counts."""
import os, re, struct, csv
SRC = '$WORKSPACE/symgen_corpus/binaries/x86_64'
DST = '$WORKSPACE/relift_ws/symgen/bins'
EXCLUDE = {'coreutils', 'diffutils', 'gettext', 'gawk', 'grep', 'gzip', 'inetutils', 'tar', 'units'}
def norm(p):
    p = p.lower().replace('openssl-openssl', 'openssl')
    return re.sub(r'[-_]\d[\d.]*[a-z]?$', '', p)
def is_elf(path):
    try:
        with open(path, 'rb') as fh: return fh.read(4) == b'\x7fELF'
    except Exception: return False
os.makedirs(DST, exist_ok=True)
rows = []; n_link = n_skip_proj = n_nonelf = 0
for opt in sorted(os.listdir(SRC)):
    for proj in sorted(os.listdir(f'{SRC}/{opt}')):
        pkg = norm(proj)
        if pkg in EXCLUDE: n_skip_proj += 1; continue
        for f in sorted(os.listdir(f'{SRC}/{opt}/{proj}')):
            path = f'{SRC}/{opt}/{proj}/{f}'
            if not os.path.isfile(path): continue
            elf = is_elf(path)
            if not elf: n_nonelf += 1; continue
            tool = f.replace('_', '-')
            bid = f'sg{pkg}_{tool}_{opt}'
            link = f'{DST}/{bid}.debug'
            if os.path.islink(link) or os.path.exists(link): os.remove(link)
            os.symlink(path, link); n_link += 1
            rows.append({'id': bid, 'pkg': f'sg{pkg}', 'proj': proj, 'tool': tool, 'opt': opt, 'path': path,
                         'size_mb': round(os.path.getsize(path) / 2**20, 2)})
with open('$WORKSPACE/relift_ws/symgen/ingest_manifest.tsv', 'w') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t'); w.writeheader(); w.writerows(rows)
pk = sorted({r['pkg'] for r in rows})
print(f'EFFECT: symgen ingest source: {n_link} ELF links, {len(pk)} packages, skipped {n_skip_proj} excluded proj-dirs, {n_nonelf} non-ELF files; total {sum(r["size_mb"] for r in rows)/1024:.1f} GB')
print('packages:', pk)
big = sorted(rows, key=lambda r: -r['size_mb'])[:5]; print('largest:', [(r['id'], r['size_mb']) for r in big])
