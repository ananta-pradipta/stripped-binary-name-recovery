"""parse_bir_v3 regression tests on a synthetic BIR snippet: determinism,
no fall-through edges (B11), ret idiom -> RETURN (B12), import vs internal
call kinds, intrinsic -> FP token, glued statements, cond-branch flag order."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.parse_bir_v3 import parse_bir

BIR = """00000001: program
.address 0x402370
0000000a: sub free(free_ptr)
0000000b: free_ptr :: in out u64 = RDI

.address 0x402370
0000000c:
.address 0x402374
0000000d: call @free:external with return %0000000e
.address 0x401000
00000010: sub sub_401000(sub_401000_result)
00000011: sub_401000_result :: out u32 = low:32[RAX]

.address 0x401000
00000012:
.address 0x401000
00000013: RSP := RSP - 8
.address 0x401004
00000014: mem := mem with [RSP, el]:u64 <- 0x401009
.address 0x401004
00000015: call @free with return %00000016
.address 0x401009
00000016:
.address 0x401009
00000017: ZF := 0 = RAX
.address 0x40100c
00000018: when ZF | CF goto %0000001a00000019: goto %0000001b
.address 0x40100e
0000001a:
.address 0x40100e
0000001c: RAX := RAX + 0x7fffffff
.address 0x401012
0000001d: goto %0000001b
.address 0x401014
0000001b:
.address 0x401014
0000001e: call @intrinsic:fadd_rne_ieee754_binary with return %0000001f
.address 0x401018
0000001f:
.address 0x401018
00000020: #100 := mem[RSP, el]:u64
.address 0x401018
00000021: RSP := RSP + 8
.address 0x401018
00000022: call #100 with noreturn
.address 0x401100
00000030: sub helper(helper_result)
.address 0x401100
00000031:
.address 0x401100
00000032: call @sub_401000 with return %00000033
.address 0x401105
00000033:
.address 0x401105
00000034: RDI := mem[RSP, el]:u64
.address 0x401105
00000035: call RAX with return %00000036
.address 0x401107
00000036:
.address 0x401107
00000037: call @helper with noreturn
"""


def _parse():
    d = tempfile.mkdtemp(); p = os.path.join(d, 'x.bir')
    open(p, 'w').write(BIR)
    return parse_bir(p)


def test_all():
    f = _parse()
    g = f['sub_401000']
    assert g['entry_addr'] == '0x401000' and g['name_kind'] == 'sub_addr'
    assert f['free']['name_kind'] == 'plt_stub'
    assert g['num_blocks'] == 5
    # edges: b0->b1 (call return), b1->b2 (when), b1->b3 (glued goto), b2->b3, b3->b4 (intrinsic return); NO fall-through b4->? etc
    assert sorted(g['edges']) == [[0, 1], [1, 2], [1, 3], [2, 3], [3, 4]], g['edges']
    toks = [t for b in g['blocks'] for t in b['tokens']]
    assert 'CALL_free' in toks and 'RETURN' in toks and 'CALL_INDIRECT' not in toks, toks
    assert 'COND_BRANCH_CF_ZF' in toks, toks           # canonical flag order regardless of text order
    assert 'FP_ADD' in toks and 'CALL_intrinsic' not in toks
    assert any(b['lit_tokens'] and 'LIT_INT_MAX' in b['lit_tokens'] for b in g['blocks'])
    kinds = {c['kind'] for c in g['call_sites']}
    assert kinds == {'import', 'intrinsic'}, kinds
    h = f['helper']
    assert h['name_kind'] == 'named'
    hk = [c['kind'] for c in h['call_sites']]
    assert hk == ['internal_sub', 'indirect', 'internal_named'], hk
    ht = [t for b in h['blocks'] for t in b['tokens']]
    assert 'CALL_INDIRECT' in ht and 'CALL_INTERNAL' in ht and 'CALL_helper' not in ht, ht
    assert h['arg_regs_used'] == [] and 'RDI' not in h['arg_regs_used']  # RDI written before read
    # determinism
    assert json.dumps(_parse(), sort_keys=True) == json.dumps(f, sort_keys=True)


if __name__ == '__main__':
    test_all(); print('PASS')
