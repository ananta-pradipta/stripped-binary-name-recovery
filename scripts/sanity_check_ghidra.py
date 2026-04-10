"""Sanity check: compare Ghidra-extracted graphs with BAP format."""
import json, glob, os, sys

GRAPHS_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/graphs"

print("=" * 60)
print("SANITY CHECK: Ghidra-extracted graphs")
print("=" * 60)

# 1. Format check on diffutils (Ghidra-extracted)
ghidra_graphs = sorted(glob.glob(os.path.join(GRAPHS_DIR, "diffutils_cmp_O0_sub_*.json")))
print(f"\n1. Ghidra graphs for diffutils_cmp_O0: {len(ghidra_graphs)}")

if ghidra_graphs:
    with open(ghidra_graphs[5]) as f:
        g = json.load(f)
    fn = g.get("function_name", "?")
    bn = g.get("binary", "?")
    print(f"   Sample: {os.path.basename(ghidra_graphs[5])}")
    print(f"   Keys: {sorted(g.keys())}")
    print(f"   binary={bn}, function_name={fn}, address={g.get('address')}")
    print(f"   num_blocks={g.get('num_blocks')}, num_edges={g.get('num_edges')}")
    print(f"   callees: {g.get('internal_callees', [])[:5]}")
    if g.get("blocks"):
        b = g["blocks"][0]
        print(f"   Block 0 tokens: {b.get('tokens', [])[:15]}")
        print(f"   Block 0 keys: {sorted(b.keys())}")

# 2. OpenSSL check
ssl = sorted(glob.glob(os.path.join(GRAPHS_DIR, "openssl_openssl_O0_sub_*.json")))
print(f"\n2. OpenSSL O0 graphs: {len(ssl)}")
for gf in ssl[100:103]:
    with open(gf) as f:
        g = json.load(f)
    toks = g["blocks"][0]["tokens"][:8] if g.get("blocks") else []
    print(f"   {os.path.basename(gf)}: {g.get('num_blocks')} blocks, tokens={toks}")

# 3. Token vocabulary overlap with BAP V3
all_ghidra_tokens = set()
for gf in ghidra_graphs[:50] + ssl[:50]:
    with open(gf) as f:
        g = json.load(f)
    for b in g.get("blocks", []):
        for t in b.get("tokens", []):
            all_ghidra_tokens.add(t)

known_v3 = {
    "NOP", "RETURN", "BRANCH", "COMPARE", "ASSIGN", "LOAD_ADDR", "OTHER",
    "STACK_OP", "STACK_STORE", "STACK_LOAD_64", "SIGN_EXTEND", "XCHG",
    "STRING_OP", "SIMD_OP", "FLAG_OP", "CMOV", "SET_FLAG", "ASSIGN_ZERO",
}
v3_prefixes = ("CALL_", "MEM_", "ARITH_", "ARG_", "COND_BRANCH", "FLAG_")

matched = sum(1 for t in all_ghidra_tokens if t in known_v3 or t.startswith(v3_prefixes))
unknown = [t for t in sorted(all_ghidra_tokens) if t not in known_v3 and not t.startswith(v3_prefixes)]

print(f"\n3. Token vocabulary (100-func sample):")
print(f"   Unique tokens: {len(all_ghidra_tokens)}")
print(f"   V3-compatible: {matched}/{len(all_ghidra_tokens)} ({100*matched/max(len(all_ghidra_tokens),1):.1f}%)")
print(f"   Unknown tokens: {unknown}")

# 4. Compare with BAP reference
bap_graphs = sorted(glob.glob(os.path.join(GRAPHS_DIR, "coreutils_cat_O0_sub_*.json")))
print(f"\n4. BAP reference (coreutils_cat_O0): {len(bap_graphs)} graphs")
if bap_graphs:
    with open(bap_graphs[5]) as f:
        bg = json.load(f)
    print(f"   BAP keys: {sorted(bg.keys())}")
    print(f"   BAP Block 0 tokens: {bg['blocks'][0]['tokens'][:15]}")

    # Key-by-key comparison
    ghidra_keys = set(g.keys()) if ghidra_graphs else set()
    bap_keys = set(bg.keys())
    print(f"\n5. Key comparison:")
    print(f"   BAP keys:    {sorted(bap_keys)}")
    print(f"   Ghidra keys: {sorted(ghidra_keys)}")
    print(f"   Missing in Ghidra: {bap_keys - ghidra_keys}")
    print(f"   Extra in Ghidra:   {ghidra_keys - bap_keys}")

    # Block key comparison
    if g.get("blocks") and bg.get("blocks"):
        ghidra_bkeys = set(g["blocks"][0].keys())
        bap_bkeys = set(bg["blocks"][0].keys())
        print(f"   BAP block keys:    {sorted(bap_bkeys)}")
        print(f"   Ghidra block keys: {sorted(ghidra_bkeys)}")
        print(f"   Missing: {bap_bkeys - ghidra_bkeys}")

# 6. Address format check
print(f"\n6. Address format:")
if ghidra_graphs:
    addrs = []
    for gf in ghidra_graphs[:10]:
        with open(gf) as f:
            g = json.load(f)
        addrs.append(g.get("address", "?"))
    print(f"   Ghidra addresses: {addrs}")
if bap_graphs:
    addrs = []
    for gf in bap_graphs[:10]:
        with open(gf) as f:
            bg = json.load(f)
        addrs.append(bg.get("address", "?"))
    print(f"   BAP addresses:    {addrs}")

print(f"\n{'='*60}")
print("SANITY CHECK COMPLETE")
