"""Single-pass streaming scan of data/graphs/*.json.
Per file returns: (filename, binary, function_name, is_sub, has_ext_call_token,
                   n_tokens, n_blocks, token_stream_md5, is_thunk_candidate, address)
Writes a compact TSV to OUT.
"""
import os, sys, json, hashlib, time
from multiprocessing import Pool

GDIR = '/home/apradipta/cs785-project/data/graphs'
OUT = sys.argv[1]

def scan(fn):
    p = os.path.join(GDIR, fn)
    try:
        with open(p) as f:
            g = json.load(f)
    except Exception as e:
        return (fn, '', '', 0, 0, -1, -1, '', 0, '', 'ERR:'+type(e).__name__)
    binary = g.get('binary', '')
    fname = g.get('function_name', '')
    is_sub = 1 if fname.startswith('sub_') else 0
    blocks = g.get('blocks', [])
    toks = []
    for b in blocks:
        toks.extend(b.get('tokens', []))
    has_ext = 0
    for t in toks:
        if t.startswith('CALL_') and t != 'CALL_INTERNAL' and t != 'CALL_INDIRECT':
            has_ext = 1
            break
    h = hashlib.md5('|'.join(toks).encode()).hexdigest()[:16]
    thunk = 1 if (len(toks) <= 2 and 'CALL_INTERNAL' in toks) else 0
    callee0 = ''
    if thunk:
        ic = g.get('internal_callees', [])
        callee0 = ic[0] if ic else ''
    return (fn, binary, fname, is_sub, has_ext, len(toks), len(blocks), h, thunk, g.get('address',''), callee0)

if __name__ == '__main__':
    t0 = time.time()
    files = sorted(os.listdir(GDIR))
    print(f'{len(files)} files', flush=True)
    n = 0
    with Pool(6) as pool, open(OUT, 'w') as out:
        for r in pool.imap(scan, files, chunksize=400):
            out.write('\t'.join(str(x) for x in r) + '\n')
            n += 1
            if n % 50000 == 0:
                print(f'{n} files scanned, {time.time()-t0:.0f}s', flush=True)
    print(f'done {n} files in {time.time()-t0:.0f}s', flush=True)
