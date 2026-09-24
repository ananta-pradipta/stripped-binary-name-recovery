#!/usr/bin/env python3
"""B1 x A4: build (masked STRIPPED Ghidra decomp -> name) rows from the SymGen Zenodo corpus (x86_64).
Join: unstripped JSON (name -> function_address.start) with stripped JSON (addr -> decomp_code, func_name) at the SAME
address (verified delta 0). Mask = stripped func_name -> [MASK] (first occurrence), exactly as our own A4 rows.
Output (results/a4_ft/):
  train_symgen.jsonl      19 ingestible non-holdout packages
  symgen_holdout.jsonl    external holdout: sggmp sglibpng sglibmicrohttpd sgpoke sglibredwg (regime FT)
Row: key binary package addr name code in_dynsym opt. `in_dynsym` = stripped name was not FUN_* (exported/PLT-named):
scored rows of the holdout must drop in_dynsym (split policy v3), training keeps them (as our protocol does).
"""
import json, os, re, collections
SRC_S = '$WORKSPACE/symgen_corpus/decomp_stripped/x86_64'
SRC_U = '/course/2026/spring/cs/785/ACCOUNT/USER/cs785/baselines/SymGen/zenodo/extracted/x86_64'
OUT = '$WORKSPACE/dh2/results/a4_ft'
EXCLUDE = {'coreutils', 'diffutils', 'gettext', 'gawk', 'grep', 'gzip', 'inetutils', 'tar', 'units'}
HOLDOUT = {'gmp', 'libpng', 'libmicrohttpd', 'poke', 'libredwg'}
CRT = {'_init', '_fini', '_start', 'deregister_tm_clones', 'register_tm_clones', '__do_global_dtors_aux',
       'frame_dummy', '_DT_INIT', '_DT_FINI', '__libc_csu_init', '__libc_csu_fini', '_dl_relocate_static_pie'}
def norm(p):
    p = p.lower().replace('openssl-openssl', 'openssl')
    return re.sub(r'[-_]\d[\d.]*[a-z]?$', '', p)
stats = collections.Counter(); per_pkg = collections.Counter(); miss = collections.Counter()
ft = open(f'{OUT}/train_symgen.jsonl', 'w'); fh = open(f'{OUT}/symgen_holdout.jsonl', 'w')
for opt in sorted(os.listdir(SRC_S)):
    for proj in sorted(os.listdir(f'{SRC_S}/{opt}')):
        pkg = norm(proj)
        if pkg in EXCLUDE: stats['skip_excluded_bins'] += 1; continue
        for f in sorted(os.listdir(f'{SRC_S}/{opt}/{proj}')):
            up = f'{SRC_U}/{opt}/{proj}/{f}'
            if not os.path.exists(up): miss['no_unstripped_json'] += 1; continue
            try:
                S = json.load(open(f'{SRC_S}/{opt}/{proj}/{f}')); U = json.load(open(up))
            except Exception as e:
                miss['bad_json'] += 1; continue
            tool = f[:-5].replace('_', '-'); binary = f'sg{pkg}_{tool}_{opt}'
            stats['bins'] += 1
            for name, v in U.items():
                if name in CRT or not isinstance(v, dict) or 'function_address' not in v: miss['crt_or_malformed'] += 1; continue
                addr = v['function_address']['start']; s = S.get(addr)
                if s is None:
                    a = int(addr, 16); s = S.get(format(a, '08x')) or S.get(format(a, 'x'))
                if s is None or 'decomp_code' not in s: miss['no_stripped_match'] += 1; continue
                sn = s.get('func_name', ''); code = s['decomp_code']
                if sn and sn in code: code = code.replace(sn, '[MASK]', 1)
                else: miss['mask_not_applied'] += 1; continue
                row = {'key': f'{binary}_0x{int(addr, 16):x}', 'binary': binary, 'package': f'sg{pkg}', 'addr': f'0x{int(addr, 16):x}',
                       'name': name, 'code': code.strip(), 'in_dynsym': not sn.startswith('FUN_'), 'opt': opt}
                if pkg in HOLDOUT:
                    row['regime'] = 'FT'; row['name_seen'] = None; fh.write(json.dumps(row) + '\n'); stats['holdout_rows'] += 1
                else:
                    ft.write(json.dumps(row) + '\n'); stats['train_rows'] += 1
                per_pkg[f'sg{pkg}'] += 1
                if row['in_dynsym']: stats['in_dynsym_rows'] += 1
ft.close(); fh.close()
json.dump({'stats': dict(stats), 'miss': dict(miss), 'per_pkg': dict(per_pkg)}, open(f'{OUT}/symgen_stats.json', 'w'), indent=1)
print(f'EFFECT: a4 symgen rows: train {stats["train_rows"]} holdout {stats["holdout_rows"]} bins {stats["bins"]} in_dynsym {stats["in_dynsym_rows"]} miss {dict(miss)}')
for p, n in per_pkg.most_common(): print(f'  {p:<18} {n:>7}')
