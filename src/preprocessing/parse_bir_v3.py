#!/usr/bin/env python3
"""parse_bir_v3 — deterministic, version-stamped BAP-IR → per-function graph parser.

Replaces parse_bap.py for dataset v2 (see docs/DUALHEAD_HYDRA_PLAN.md, defects
B2/B7/B11/B12).  Input is a .bir produced by scripts/relift_v2.py, i.e. dumped
with ``--print-bir-attr=address`` (real addresses on every term) after an
``--read-symbols-from`` (eh_frame roots) lift.

What is different from parse_bap.py
-----------------------------------
* ALL subs are parsed (``sub_<addr>`` and symbol-named ones); the entry address
  comes from the ``.address`` attribute, so matching is by address (B3a).
  PLT import stubs / section pseudo-subs / crt junk are recognised and skipped.
* Deterministic output: canonical flag ordering (``COND_BRANCH_CF_ZF``), sorted
  callee lists, ``json.dump(sort_keys=True)``, PARSER_VERSION stamped in every
  file.  Two runs on the same .bir are byte-identical.
* CFG edges come from jump statements only (``goto``, ``when … goto``,
  ``call … with return %T``); the old parser added a fall-through edge between
  every pair of consecutive blocks, which is wrong for BAP-IR (B11).
* ``ret`` is recognised: BAP 2.5 lifts it as ``#N := mem[RSP]; RSP := RSP+8;
  call #N with noreturn`` and parse_bap tokenised every return as
  ``CALL_INDIRECT`` (RETURN never occurred in the corpus).  v3 emits ``RETURN``
  for that idiom, ``CALL_INDIRECT`` for genuine indirect calls (``call #N with
  return``, ``call mem[…]``, ``call RAX``) and ``TAILJUMP_INDIRECT`` for
  ``call #N with noreturn`` that is not a return (B12).
* ``call @intrinsic:<op>`` becomes an ``FP_<class>``/``TRAP`` token instead of
  ``CALL_intrinsic``.
* Call kinds are recorded per call site: ``import`` (PLT stub / ``:external``),
  ``internal_named`` (function defined in the binary but exported through
  ``.dynsym`` — its name is visible in the stripped ELF, B10),
  ``internal_sub`` (``sub_<addr>``), ``indirect``, ``intrinsic``.  The token
  stream keeps ``CALL_<name>`` ONLY for imports; internal_named calls are
  tokenised ``CALL_INTERNAL`` and the name is kept in ``call_sites`` so the
  dataset builder can decide whether to expose it.
* Side channels for B7 (kept out of the main token stream so V3 vocab stays
  comparable): per-block ``lit_tokens`` (immediate buckets / magic constants),
  per-function ``gref_addrs`` (constant addresses ≥ 0x1000 that appear as
  operands — resolved to .rodata strings downstream), ``arg_regs_used`` (arg
  registers read before written near entry), BAP-inferred ``bap_args``.
* Output layout: ``<out>/<binary>/<safe_name>.json`` (one dir per binary) plus
  ``<out>/<binary>.index.json``.

Usage:
  python3 -m src.preprocessing.parse_bir_v3 --bir data/bir_v2/X.bir --syms data/bir_v2/X.syms \
      --binary-name X --out data/graphs_v3
"""
import argparse
import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.preprocessing.parse_bap import classify_instruction, hextester  # noqa: E402

PARSER_VERSION = 'v3.0'

RE_SUB = re.compile(r'^([0-9a-fA-F]+): sub (\S+?)\((.*)\)\s*$')
RE_PARAM = re.compile(r'^([0-9a-fA-F]+): (\S+) :: (in out|in|out) ')
RE_BLOCK = re.compile(r'^([0-9a-fA-F]+):\s*$')
RE_STMT = re.compile(r'^([0-9a-fA-F]+): (.*)$')
RE_ADDR = re.compile(r'^\.address 0x([0-9a-fA-F]+)\s*$')
RE_PROGRAM = re.compile(r'^[0-9a-fA-F]+:\s+program\s*$')
# BAP occasionally prints two terms on one line: "when ZF goto %001894ea001938da: goto %00189544"
RE_GLUED = re.compile(r'(%[0-9a-fA-F]{8})([0-9a-fA-F]{8}: )')
RE_JUMP_TARGET = re.compile(r'(?:goto|return)\s+%([0-9a-fA-F]+)')
RE_CALL = re.compile(r'\bcall\s+(@[^\s(]+|#\d+|mem\b[^\s]*|R[A-Z0-9]+|\S+)')
RE_HEX = re.compile(r'(?<![#\w])0x([0-9a-fA-F]+)\b')
FLAGS_ORDER = ('CF', 'OF', 'ZF', 'SF', 'PF', 'AF')
ARG_REGS = ('RDI', 'RSI', 'RDX', 'RCX', 'R8', 'R9')
SECTION_SUBS = {'.plt', '.init', '.fini', '.text', '.plt.sec', '.plt.got'}
CRT_JUNK = {'_start', '_init', '_fini', 'register_tm_clones', 'deregister_tm_clones',
            'frame_dummy', '__do_global_dtors_aux', '__do_global_ctors_aux', '__libc_csu_init',
            '__libc_csu_fini', '_dl_relocate_static_pie'}
MAGIC = {0x7fffffff: 'INT_MAX', 0xffffffff: 'U32_MAX', 0x7fffffffffffffff: 'I64_MAX',
         0xffffffffffffffff: 'U64_MAX', 0x1000: 'PAGE', 0xff: 'BYTE_MASK', 0x3f: 'MASK6',
         0x1f: 'MASK5', 0x0f: 'MASK4', 0x100: 'K256', 0x400: 'K1024', 0x10000: 'K64K'}
INTRINSIC_CLASS = [
    ('is_nan', 'FP_ISNAN'), ('forder', 'FP_CMP'), ('fadd', 'FP_ADD'), ('fsub', 'FP_SUB'),
    ('fmul', 'FP_MUL'), ('fdiv', 'FP_DIV'), ('fsqrt', 'FP_SQRT'), ('cast_sfloat', 'FP_CAST_TO_FLOAT'),
    ('cast_sint', 'FP_CAST_TO_INT'), ('cast_uint', 'FP_CAST_TO_INT'), ('fconvert', 'FP_CONVERT'),
    ('fround', 'FP_ROUND'), ('hlt', 'TRAP'), ('__ud2', 'TRAP'), ('ud2', 'TRAP'), ('int3', 'TRAP'),
]


def _intrinsic_token(name: str) -> str:
    low = name.lower()
    for key, tok in INTRINSIC_CLASS:
        if key in low:
            return tok
    return 'INTRINSIC_OTHER'


def _lit_tokens(stmt: str) -> List[str]:
    out = []
    for h in RE_HEX.findall(stmt):
        v = int(h, 16)
        if v in MAGIC:
            out.append('LIT_' + MAGIC[v])
        elif v == 0:
            out.append('LIT_0')
        elif v <= 16:
            out.append('LIT_%d' % v)
        elif 0x20 <= v <= 0x7e:
            out.append('LIT_CHR')
        elif v & (v - 1) == 0:
            out.append('LIT_POW2_%d' % (v.bit_length() - 1))
        elif v < 0x1000:
            out.append('LIT_SMALL12')
        else:
            out.append('LIT_HI%d' % min(len(h), 16))
    return out


def _cond_branch_token(stmt: str) -> str:
    cond = stmt.split('when', 1)[1].split('goto', 1)[0]
    flags = [f for f in FLAGS_ORDER if re.search(r'\b%s\b' % f, cond)]
    return 'COND_BRANCH_' + '_'.join(flags) if flags else 'COND_BRANCH'


class _Func:
    __slots__ = ('tid', 'name', 'params', 'entry_addr', 'blocks', 'edges', 'call_sites',
                 'gref_addrs', 'arg_regs_used', 'written_regs', 'n_stmts', 'bap_in_args',
                 'bap_has_result', 'first_block_addr')

    def __init__(self, tid, name, params, entry_addr):
        self.tid, self.name, self.params, self.entry_addr = tid, name, params, entry_addr
        self.blocks: List[dict] = []
        self.edges: List[Tuple[str, str]] = []      # (src_tid, dst_tid)
        self.call_sites: List[dict] = []
        self.gref_addrs = set()
        self.arg_regs_used = set()
        self.written_regs = set()
        self.n_stmts = 0
        self.bap_in_args = 0
        self.bap_has_result = False
        self.first_block_addr = None


def _split_glued(stmt: str) -> List[Tuple[Optional[str], str]]:
    """'when ZF goto %A<tid>: goto %B' -> [(None,'when ZF goto %A'), ('<tid>','goto %B')]."""
    parts = RE_GLUED.split(stmt)
    if len(parts) == 1:
        return [(None, stmt)]
    out = [(None, parts[0] + parts[1])]
    i = 2
    while i < len(parts):
        tid = parts[i].rstrip(': ')
        rest = parts[i + 1] if i + 1 < len(parts) else ''
        nxt = parts[i + 2] if i + 2 < len(parts) else ''
        out.append((tid, rest + nxt))
        i += 3
    return out


def parse_bir(bir_path: str, stub_names: Optional[set] = None) -> Dict[str, dict]:
    """Return {sub_name: function-dict}.  Two passes: (1) find PLT stub subs
    (a sub whose body contains ``call @<own name>:external`` or that lives
    entirely at .plt/.plt.sec addresses), (2) full parse."""
    def lines_iter():
        with open(bir_path, errors='replace') as fh:
            for raw in fh:
                yield raw.rstrip('\n')

    # ---------------------------------------------------------------- pass 1: stubs
    all_sub_names = set()
    if stub_names is None:
        stub_names = set()
    cur = None
    for line in lines_iter():
        m = RE_SUB.match(line)
        if m:
            cur = m.group(2); all_sub_names.add(cur); continue
        if cur and re.search(r'call @%s:external' % re.escape(cur), line):
            stub_names.add(cur)
    # ---------------------------------------------------------------- pass 2
    funcs: Dict[str, _Func] = {}
    cur: Optional[_Func] = None
    cur_block: Optional[dict] = None
    pending_addr: Optional[int] = None
    stmt_buf: Optional[Tuple[str, str, Optional[int]]] = None  # (tid, text, addr)
    order: List[str] = []

    def flush_stmt():
        nonlocal stmt_buf
        if stmt_buf is None or cur is None or cur_block is None:
            stmt_buf = None; return
        tid, text, addr = stmt_buf
        stmt_buf = None
        for sub_tid, piece in _split_glued(text.strip()):
            _handle_stmt(cur, cur_block, sub_tid or tid, piece, addr)

    def _handle_stmt(fn: _Func, blk: dict, tid: str, stmt: str, addr: Optional[int]):
        fn.n_stmts += 1
        blk['n_stmts'] += 1
        stmt = re.sub(r'\s+', ' ', stmt).strip()
        # ---- jumps / edges
        for t in RE_JUMP_TARGET.findall(stmt):
            fn.edges.append((blk['tid'], t.lower()))
        # ---- calls
        cm = RE_CALL.search(stmt) if stmt.startswith('call') else None
        toks: List[str] = []
        if cm:
            tgt = cm.group(1)
            noreturn = 'noreturn' in stmt
            if tgt.startswith('@intrinsic:'):
                toks = [_intrinsic_token(tgt[len('@intrinsic:'):])]
                fn.call_sites.append({'kind': 'intrinsic', 'name': tgt[1:], 'block': blk['tid'], 'addr': addr})
            elif tgt.startswith('@'):
                name = tgt[1:].split(':')[0]
                is_sub = name.startswith('sub_') and hextester(name.split('sub_'))
                if is_sub:
                    kind = 'internal_sub'; toks = ['CALL_INTERNAL']
                elif name == 'interrupt':
                    kind = 'syscall'; toks = ['SYSCALL']
                elif ':external' in tgt or name in stub_names or name not in all_sub_names:
                    # PLT stub, explicit external, or a symbol with no body in this
                    # binary (e.g. __libc_start_main through .plt.got) -> import
                    kind = 'import'; toks = ['CALL_' + name]
                else:
                    kind = 'internal_named'; toks = ['CALL_INTERNAL']
                fn.call_sites.append({'kind': kind, 'name': name, 'block': blk['tid'], 'addr': addr,
                                      'tail': noreturn})
                if kind == 'import' and name != 'interrupt':
                    blk['has_external_call'] = True
                    if blk['external_call_name'] is None:
                        blk['external_call_name'] = name
            elif tgt.startswith('#'):
                # ret idiom: '#N := mem[RSP, el]:u64' ; 'RSP := RSP + 8' ; 'call #N with noreturn'
                if noreturn and blk.get('_last_pop') == tgt:
                    toks = ['RETURN']
                elif noreturn:
                    toks = ['TAILJUMP_INDIRECT']
                    fn.call_sites.append({'kind': 'indirect', 'name': None, 'block': blk['tid'], 'addr': addr, 'tail': True})
                else:
                    toks = ['CALL_INDIRECT']
                    fn.call_sites.append({'kind': 'indirect', 'name': None, 'block': blk['tid'], 'addr': addr, 'tail': False})
            else:
                toks = ['TAILJUMP_INDIRECT' if noreturn else 'CALL_INDIRECT']
                fn.call_sites.append({'kind': 'indirect', 'name': None, 'block': blk['tid'], 'addr': addr, 'tail': noreturn})
            blk['_last_pop'] = None
        elif stmt.startswith('when '):
            toks = [_cond_branch_token(stmt)]
        else:
            # track the ret idiom
            mp = re.match(r'(#\d+) := mem\[RSP, el\]:u64$', stmt)
            if mp:
                blk['_last_pop'] = mp.group(1)
            elif stmt == 'RSP := RSP + 8' and blk.get('_last_pop'):
                pass  # keep pending
            elif not stmt.startswith('RSP := RSP + 8'):
                blk['_last_pop'] = None
            toks = classify_instruction('%s: %s' % (tid, stmt))
        blk['tokens'].extend(toks)
        # ---- side channels
        lits = _lit_tokens(stmt)
        if lits:
            blk['lit_tokens'].extend(lits)
        for h in RE_HEX.findall(stmt):
            v = int(h, 16)
            if v >= 0x1000 and 'goto' not in stmt:
                fn.gref_addrs.add(v)
        # arg registers: read before written (only in first ~40 statements)
        if fn.n_stmts <= 40:
            lhs = stmt.split(':=')[0].strip() if ':=' in stmt else ''
            for r in ARG_REGS:
                if re.search(r'\b%s\b' % r, stmt):
                    if lhs == r:
                        fn.written_regs.add(r)
                    elif r not in fn.written_regs:
                        fn.arg_regs_used.add(r)

    for line in lines_iter():
        if not line.strip():
            continue
        if RE_PROGRAM.match(line):
            continue
        ma = RE_ADDR.match(line)
        if ma:
            flush_stmt()
            pending_addr = int(ma.group(1), 16); continue
        ms = RE_SUB.match(line)
        if ms:
            flush_stmt()
            tid, name, params = ms.groups()
            cur = _Func(tid, name, params, pending_addr)
            funcs[name] = cur; order.append(name)
            cur_block = None
            continue
        if cur is None:
            continue
        mp = RE_PARAM.match(line)
        if mp:
            flush_stmt()
            if 'in' in mp.group(3).split():
                cur.bap_in_args += 1
            if mp.group(3) == 'out' or mp.group(2).endswith('_result'):
                cur.bap_has_result = True
            continue
        mb = RE_BLOCK.match(line)
        if mb:
            flush_stmt()
            cur_block = {'tid': mb.group(1).lower(), 'addr': pending_addr, 'tokens': [], 'lit_tokens': [],
                         'n_stmts': 0, 'has_external_call': False, 'external_call_name': None,
                         '_last_pop': None}
            cur.blocks.append(cur_block)
            if cur.first_block_addr is None:
                cur.first_block_addr = pending_addr
            continue
        mst = RE_STMT.match(line)
        if mst and cur_block is not None:
            flush_stmt()
            stmt_buf = (mst.group(1).lower(), mst.group(2), pending_addr)
            continue
        # continuation line (indented) of a multi-line statement
        if stmt_buf is not None and line[:1].isspace():
            stmt_buf = (stmt_buf[0], stmt_buf[1] + ' ' + line.strip(), stmt_buf[2])
            continue
    flush_stmt()

    # ---------------------------------------------------------------- finalise
    out: Dict[str, dict] = {}
    for name in order:
        fn = funcs[name]
        blocks = [b for b in fn.blocks if b['n_stmts'] > 0 or b['tokens']]
        if not blocks:
            continue
        tid2idx = {b['tid']: i for i, b in enumerate(blocks)}
        edges = []
        seen = set()
        for s, d in fn.edges:
            if s in tid2idx and d in tid2idx:
                e = (tid2idx[s], tid2idx[d])
                if e not in seen:
                    seen.add(e); edges.append(list(e))
        entry = fn.entry_addr if fn.entry_addr is not None else fn.first_block_addr
        is_sub = name.startswith('sub_') and hextester(name.split('sub_'))
        if name.startswith('intrinsic:'):
            kind = 'intrinsic'
        elif name in stub_names:
            kind = 'plt_stub'
        elif name in SECTION_SUBS or name.startswith('.'):
            kind = 'section'
        elif name in CRT_JUNK:
            kind = 'crt'
        elif is_sub:
            kind = 'sub_addr'
        else:
            kind = 'named'
        n_import = sum(1 for c in fn.call_sites if c['kind'] == 'import')
        for b in blocks:
            b.pop('_last_pop', None)
        out[name] = {
            'parser_version': PARSER_VERSION,
            'bap_name': name,
            'name_kind': kind,
            'entry_addr': ('0x%x' % entry) if entry is not None else None,
            'address': ('0x%x' % entry) if entry is not None else None,  # legacy field name
            'blocks': [{'id': i, 'label': b['tid'], 'addr': ('0x%x' % b['addr']) if b['addr'] is not None else None,
                        'tokens': b['tokens'], 'num_tokens': len(b['tokens']), 'lit_tokens': b['lit_tokens'],
                        'has_external_call': b['has_external_call'], 'external_call_name': b['external_call_name']}
                       for i, b in enumerate(blocks)],
            'edges': edges,
            'num_blocks': len(blocks),
            'num_edges': len(edges),
            'call_sites': [dict(c, addr=('0x%x' % c['addr']) if c.get('addr') is not None else None)
                           for c in fn.call_sites],
            'internal_callees': sorted({c['name'] for c in fn.call_sites if c['kind'] == 'internal_sub'}),
            'internal_named_callees': sorted({c['name'] for c in fn.call_sites if c['kind'] == 'internal_named'}),
            'external_calls': sorted({c['name'] for c in fn.call_sites if c['kind'] == 'import'}),
            'n_import_calls': n_import,
            'gref_addrs': sorted('0x%x' % a for a in fn.gref_addrs),
            'arg_regs_used': sorted(fn.arg_regs_used, key=ARG_REGS.index),
            'bap_in_args': fn.bap_in_args,
            'bap_has_result': fn.bap_has_result,
            'n_stmts': fn.n_stmts,
        }
    return out


def load_bap_syms(path: str) -> Dict[str, Tuple[int, int]]:
    """dump-symbols file → name -> (min_start, max_end)."""
    fns: Dict[str, Tuple[int, int]] = {}
    if not path or not os.path.exists(path):
        return fns
    with open(path, errors='ignore') as fh:
        for line in fh:
            m = re.match(r'\((\S+) (\d+) (\d+)\)', line.strip())
            if m:
                n, s, e = m.group(1), int(m.group(2)), int(m.group(3))
                if n in fns:
                    fns[n] = (min(fns[n][0], s), max(fns[n][1], e))
                else:
                    fns[n] = (s, e)
    return fns


def safe_name(name: str) -> str:
    s = re.sub(r'[^\w\-.]', '_', name)
    if len(s) > 120:
        s = s[:100] + '_' + hashlib.md5(name.encode()).hexdigest()[:12]
    return s


def write_graphs(functions: Dict[str, dict], binary: str, out_dir: str, syms: Dict[str, Tuple[int, int]],
                 keep_kinds=('sub_addr', 'named')) -> dict:
    bdir = os.path.join(out_dir, binary)
    os.makedirs(bdir, exist_ok=True)
    index = []
    kept = skipped = 0
    for name, fd in functions.items():
        if fd['name_kind'] not in keep_kinds:
            skipped += 1; continue
        if syms and name in syms and fd['entry_addr'] is None:
            fd['entry_addr'] = fd['address'] = '0x%x' % syms[name][0]
        fd['binary'] = binary
        fd['function_name'] = name
        if name in syms:
            fd['bap_sym_start'] = '0x%x' % syms[name][0]
            fd['bap_sym_end'] = '0x%x' % syms[name][1]
        fn = os.path.join(bdir, safe_name(name) + '.json')
        with open(fn, 'w') as fh:
            json.dump(fd, fh, indent=1, sort_keys=True)
        index.append({'name': name, 'file': os.path.relpath(fn, out_dir), 'entry_addr': fd['entry_addr'],
                      'name_kind': fd['name_kind'], 'num_blocks': fd['num_blocks'],
                      'n_tokens': sum(b['num_tokens'] for b in fd['blocks']),
                      'n_import_calls': fd['n_import_calls']})
        kept += 1
    index.sort(key=lambda r: (r['entry_addr'] or '', r['name']))
    meta = {'binary': binary, 'parser_version': PARSER_VERSION, 'n_functions': kept, 'n_skipped': skipped,
            'functions': index}
    with open(os.path.join(out_dir, binary + '.index.json'), 'w') as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bir', required=True)
    ap.add_argument('--syms', default=None, help='BAP dump-symbols file (optional)')
    ap.add_argument('--binary-name', required=True)
    ap.add_argument('--out', default='data/graphs_v3')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    functions = parse_bir(args.bir)
    syms = load_bap_syms(args.syms) if args.syms else {}
    meta = write_graphs(functions, args.binary_name, args.out, syms)
    if not args.quiet:
        kinds = {}
        for fd in functions.values():
            kinds[fd['name_kind']] = kinds.get(fd['name_kind'], 0) + 1
        print(f"{args.binary_name}: {meta['n_functions']} kept, {meta['n_skipped']} skipped; kinds={kinds}")


if __name__ == '__main__':
    main()
