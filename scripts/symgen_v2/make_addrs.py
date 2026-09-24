#!/usr/bin/env python3
"""Build per-binary .addrs lists + elf bases from the exported protocol jsonl."""
import json, os, sys
sys.path.insert(0, '/project/hz79/_shared/cs785/relift_ws/pylib')
from elftools.elf.elffile import ELFFile
WS = '/project/hz79/_shared/cs785/dh2/symgen_v2'
PROTO = '/project/hz79/_shared/cs785/dh2/results/baseline_protocol_v2'
STRIP = '/project/hz79/_shared/cs785/relift_ws/data/stripped_v2'
os.makedirs(WS + '/addrs', exist_ok=True); os.makedirs(WS + '/bases', exist_ok=True)
os.makedirs(WS + '/decomp', exist_ok=True); os.makedirs(WS + '/logs', exist_ok=True)
per = {}
for tier in ('val', 'test'):
    for line in open(f'{PROTO}/{tier}.jsonl'):
        r = json.loads(line)
        per.setdefault(r['binary'], set()).add(r['entry_addr'])
n = 0
for b, addrs in sorted(per.items()):
    p = f'{STRIP}/{b}'
    if not os.path.exists(p):
        print('MISSING', b); continue
    with open(p, 'rb') as fh:
        elf = ELFFile(fh)
        base = min((s.header.p_vaddr for s in elf.iter_segments() if s.header.p_type == 'PT_LOAD'), default=0)
    open(f'{WS}/bases/{b}', 'w').write(hex(base))
    open(f'{WS}/addrs/{b}.addrs', 'w').write('\n'.join(sorted(addrs)) + '\n')
    n += 1
open(f'{WS}/binaries.txt', 'w').write('\n'.join(sorted(per)) + '\n')
print('EFFECT: addrs for', n, 'binaries,', sum(len(v) for v in per.values()), 'functions')
