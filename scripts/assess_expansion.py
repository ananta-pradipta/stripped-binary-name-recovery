"""Assess whether proposed packages benefit cross-project prediction."""
import json, os, glob, sys
from collections import Counter

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"

# Load match_index
with open(os.path.join(DATA, "match_index.json")) as f:
    mi = json.load(f)

# Current training function names
train_names = Counter()
train_pkgs = Counter()
for v in mi.values():
    train_names[v["real_name"]] += 1
    train_pkgs[v["binary"].split("_")[0]] += 1

print("=" * 60)
print("DATASET EXPANSION ASSESSMENT")
print("=" * 60)
print(f"Current: {len(mi):,} functions, {len(train_pkgs)} packages")
print(f"Unique function names: {len(train_names):,}")

# Cross-project function names (from labels)
labels_dir = os.path.join(DATA, "labels")
xproj_pkgs = ["tengine", "angie", "nginx118", "recutils", "curl"]
xproj_names = {}

for pkg in xproj_pkgs:
    names = set()
    for lf in glob.glob(os.path.join(labels_dir, pkg + "_*.json")):
        with open(lf) as f:
            labels = json.load(f)
        for addr, name in labels.items():
            if isinstance(name, str):
                names.add(name)
    xproj_names[pkg] = names

print(f"\n{'='*60}")
print("CROSS-PROJECT NAME OVERLAP WITH TRAINING")
print(f"{'='*60}")

all_xproj = set()
for pkg, names in xproj_names.items():
    overlap = names & set(train_names.keys())
    pct = 100 * len(overlap) / max(len(names), 1)
    print(f"  {pkg:12s}: {len(names):5d} names, {len(overlap):4d} in training ({pct:.1f}%)")
    all_xproj |= names

missing = all_xproj - set(train_names.keys())
print(f"\n  Total unique xproj names: {len(all_xproj):,}")
print(f"  Already covered by training: {len(all_xproj) - len(missing):,} ({100*(len(all_xproj)-len(missing))/len(all_xproj):.1f}%)")
print(f"  NOT in training: {len(missing):,}")

# Sample missing names by prefix
from collections import defaultdict
missing_prefixes = defaultdict(int)
for n in missing:
    parts = n.split("_")
    if len(parts) > 1:
        missing_prefixes[parts[0]] += 1
    else:
        missing_prefixes[n] += 1
print(f"\n  Top missing name prefixes: {dict(sorted(missing_prefixes.items(), key=lambda x: -x[1])[:15])}")

# Assess proposed packages
print(f"\n{'='*60}")
print("PROPOSED PACKAGE ASSESSMENT")
print(f"{'='*60}")

# What matters: does adding a package introduce function names that
# overlap with cross-project function names we're missing?
# Key insight: Our model is a recognizer — it can only predict names it's seen in training.
# Adding packages with names that match cross-project test set names directly improves recall.

# nginx/openssl/curl share many function name patterns:
# SSL_*, EVP_*, BIO_*, ngx_*, curl_*
# These won't help unless the cross-project set has them.

# GNU packages share gnulib functions: xmalloc, quotearg, close_stdout, etc.
# These DO help because cross-project packages (recutils, curl) also use gnulib.

print("\n  Key question: which missing xproj names could new packages provide?")
print()

# Check what types of names are missing
nginx_missing = [n for n in missing if n.startswith("ngx_")]
ssl_missing = [n for n in missing if n.startswith(("SSL_", "EVP_", "BIO_", "CRYPTO_", "OSSL_"))]
curl_missing = [n for n in missing if n.startswith(("curl_", "Curl_"))]
gnu_missing = [n for n in missing if not n.startswith(("ngx_", "SSL_", "EVP_", "BIO_", "curl_", "Curl_", "CRYPTO_", "OSSL_"))]

print(f"  nginx-specific missing (ngx_*): {len(nginx_missing)}")
print(f"  SSL-specific missing: {len(ssl_missing)}")
print(f"  curl-specific missing: {len(curl_missing)}")
print(f"  Generic/GNU missing: {len(gnu_missing)}")

print(f"\n  Proposed packages and their expected benefit:")
print()

proposals = [
    ("GDB", "~40K funcs", "Debugger internals, many unique names (gdb_*, tdep_*, dwarf_*). LOW overlap with xproj. Adds diversity but won't help cross-project directly."),
    ("Vim", "~20K funcs", "Editor internals (vim_*, buf_*, win_*). LOW overlap with xproj. Diversity only."),
    ("Git", "~12K funcs", "VCS internals (git_*, strbuf_*, hashmap_*). Has xmalloc/xstrdup from gnulib. MEDIUM benefit for recutils overlap."),
    ("nmap", "~12K funcs", "Network scanner (nmap_*, nse_*, pcap_*). SOME overlap with nginx networking patterns. MEDIUM."),
    ("coreutils v8.32", "~16K funcs", "Same gnulib as training but different version. HIGH overlap — fills missing gnulib variants."),
]

for name, size, analysis in proposals:
    print(f"  {name} ({size}):")
    print(f"    {analysis}")
    print()

# Category distribution
print(f"{'='*60}")
print("CURRENT CATEGORY DISTRIBUTION")
print(f"{'='*60}")

categories = {
    "GNU core utils": ["coreutils", "coreutils2", "findutils", "sed", "tar", "gzip", "patch", "which", "grep", "gawk", "diffutils"],
    "Text processing": ["m4", "make", "less", "bison", "flex", "enscript", "texinfo", "spell", "indent", "wdiff"],
    "Networking": ["wget", "inetutils"],
    "Crypto/Security": ["openssl", "nettle"],
    "System tools": ["binutils", "bash", "screen", "nano", "strace", "bc", "time", "cpio"],
    "Data/Libraries": ["sqlite", "lua", "jq", "lz4", "zlib", "gdbm"],
    "Games/Other": ["gnuchess", "groff", "gperf", "datamash", "gawk2"],
    "Embedded": ["busybox"],
}

total_funcs = sum(train_pkgs.values())
for cat, pkgs_list in sorted(categories.items()):
    cat_funcs = sum(train_pkgs.get(p, 0) for p in pkgs_list)
    cat_pkgs = len([p for p in pkgs_list if p in train_pkgs])
    pct = 100 * cat_funcs / total_funcs
    print(f"  {cat:25s}: {cat_funcs:>7,} funcs ({pct:4.1f}%) from {cat_pkgs} pkgs")

print(f"\n{'='*60}")
print("RECOMMENDATION")
print(f"{'='*60}")
print("""
  1. coreutils v8.32 — HIGHEST priority. Same gnulib foundation as
     recutils cross-project set. Fills missing gnulib function variants.
  2. Git — MEDIUM priority. Has gnulib overlap + diverse VCS internals.
  3. GDB — SIZE priority. Adds ~40K functions for volume, but low
     cross-project benefit (debugger-specific names).
  4. Vim — LOW priority. Editor-specific names, minimal xproj overlap.
  5. nmap — LOW priority. Network tool but different patterns from nginx.

  For cross-project benefit: coreutils3 + Git > GDB + Vim + nmap
  For dataset volume: GDB alone adds ~40K
  For diversity: GDB + Vim cover new domains (debugging, editing)
""")
