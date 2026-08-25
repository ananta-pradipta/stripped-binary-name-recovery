#!/usr/bin/env python3
"""A3+: match .rodata bytes at gref-referenced addresses against known algorithm
constant tables -> data/rodata_consts_v2/<bin>.json {gref_addr_hex: [CONST_* tags]}.
Also tags identifier-free but famous tables (CRC32, AES S-box, base64 alphabet)."""
import json, os, struct, sys, glob
sys.path.insert(0, '/project/hz79/_shared/cs785/relift_ws/pylib')
from elftools.elf.elffile import ELFFile

GR = '/project/hz79/_shared/cs785/relift_ws/data/graphs_v3'
ST = '/project/hz79/_shared/cs785/relift_ws/data/stripped_v2'
OUT = '/project/hz79/_shared/cs785/relift_ws/data/rodata_consts_v2'
SIGS = []
def u32s(*vals): return b''.join(struct.pack('<I', v) for v in vals)
SIGS.append(('CONST_SHA256', u32s(0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5)))
SIGS.append(('CONST_SHA256', u32s(0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a)))
SIGS.append(('CONST_SHA512', struct.pack('<Q',0x428a2f98d728ae22)+struct.pack('<Q',0x7137449123ef65cd)))
SIGS.append(('CONST_MD5', u32s(0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee)))
SIGS.append(('CONST_SHA1', u32s(0x5a827999,0x6ed9eba1,0x8f1bbcdc,0xca62c1d6)))
SIGS.append(('CONST_CRC32', u32s(0x00000000,0x77073096,0xee0e612c,0x990951ba)))
SIGS.append(('CONST_AES_SBOX', bytes([0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5])))
SIGS.append(('CONST_BASE64', b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'))
SIGS.append(('CONST_CHACHA', b'expand 32-byte k'))
SIGS.append(('CONST_BLOWFISH', u32s(0x243f6a88,0x85a308d3,0x13198a2e,0x03707344)))
SIGS.append(('CONST_DES_SBOX', bytes([0x0e,0x04,0x0d,0x01,0x02,0x0f,0x0b,0x08])))
SIGS.append(('CONST_ZLIB_LEN', struct.pack('<8H',3,4,5,6,7,8,9,10)))

def scan(binary):
    op = f'{OUT}/{binary}.json'
    if os.path.exists(op):
        return 0
    sp = f'{ST}/{binary}'
    if not os.path.exists(sp):
        return 0
    grefs = set()
    for gp in glob.glob(f'{GR}/{binary}/*.json'):
        try:
            g = json.load(open(gp))
        except Exception:
            continue
        grefs.update(g.get('gref_addrs', []))
    if not grefs:
        json.dump({}, open(op, 'w')); return 0
    with open(sp, 'rb') as fh:
        elf = ELFFile(fh)
        segs = [(s.header.p_vaddr, s.header.p_filesz, s.header.p_offset)
                for s in elf.iter_segments() if s.header.p_type == 'PT_LOAD']
        fh.seek(0); blob = fh.read()
    out = {}
    for a in grefs:
        try:
            va = int(a, 16)
        except Exception:
            continue
        off = None
        for v, sz, o in segs:
            if v <= va < v + sz:
                off = o + (va - v); break
        if off is None:
            continue
        window = blob[off:off + 256]
        tags = [name for name, sig in SIGS if sig in window]
        if tags:
            out[a] = sorted(set(tags))
    json.dump(out, open(op, 'w'))
    return len(out)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    bins = sorted(os.listdir(GR))
    bins = [b for b in bins if os.path.isdir(f'{GR}/{b}')]
    tot = 0
    for i, b in enumerate(bins):
        tot += scan(b)
    print(f'EFFECT: rodata consts done bins={len(bins)} tagged_addrs={tot}')
