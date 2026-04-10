"""Show complete dataset distribution."""
import json, sys
from collections import Counter, defaultdict

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"

with open(f"{DATA}/match_index.json") as f:
    mi = json.load(f)

pkgs = Counter()
opts = Counter()
pkg_opts = defaultdict(Counter)
pkg_bins = defaultdict(set)

for v in mi.values():
    b = v["binary"]
    pkg = b.split("_")[0]
    pkgs[pkg] += 1
    pkg_bins[pkg].add(b)
    for p in b.split("_"):
        if p in ("O0", "O1", "O2", "O3"):
            opts[p] += 1
            pkg_opts[pkg][p] += 1
            break

categories = {
    "GNU Core Utils": ["coreutils", "coreutils2", "findutils", "findutils2", "sed", "sed2",
                        "tar", "tar2", "gzip", "gzip2", "patch", "patch2", "grep", "grep2",
                        "gawk", "which", "which2", "diffutils"],
    "Text Processing": ["m4", "make", "less", "bison", "flex", "enscript", "texinfo",
                         "spell", "indent", "wdiff", "gawk2"],
    "Dev Tools": ["binutils", "bash", "bc"],
    "Networking": ["wget", "inetutils"],
    "Crypto/Security": ["openssl", "nettle"],
    "Compression": ["lz4", "zlib", "xz"],
    "Data/Libraries": ["sqlite", "lua", "jq"],
    "System/Monitoring": ["strace", "screen", "nano", "htop", "time", "cpio",
                           "units", "units2"],
    "Embedded": ["busybox"],
    "Typography": ["groff"],
    "Other GNU": ["hello", "direvent", "cppi", "csplit2", "combinatorics", "rush",
                   "gcal", "sharutils", "gnuchess", "gperf", "datamash", "ed"],
}

pkg_cat = {}
for cat, cat_pkgs in categories.items():
    for p in cat_pkgs:
        if p in pkgs and p not in pkg_cat:
            pkg_cat[p] = cat
for p in pkgs:
    if p not in pkg_cat:
        pkg_cat[p] = "Other"

total_bins = sum(len(v) for v in pkg_bins.values())
print("=" * 80)
print("COMPLETE DATASET DISTRIBUTION")
print("=" * 80)
print("Total: {:,} functions | {} packages | {} binaries".format(len(mi), len(pkgs), total_bins))
print()

print("OPTIMIZATION LEVEL DISTRIBUTION")
print("-" * 40)
for o in sorted(opts):
    pct = 100.0 * opts[o] / len(mi)
    bar = "#" * int(pct / 2)
    print("  {}: {:>7,} ({:4.1f}%) {}".format(o, opts[o], pct, bar))

print()
print("CATEGORY DISTRIBUTION")
print("-" * 80)
cat_totals = defaultdict(int)
cat_pkg_count = defaultdict(int)
for p, c in pkg_cat.items():
    cat_totals[c] += pkgs[p]
    cat_pkg_count[c] += 1

print("  {:<20s} {:>10s} {:>6s} {:>5s}".format("Category", "Functions", "Pct", "Pkgs"))
print("  {} {} {} {}".format("-" * 20, "-" * 10, "-" * 6, "-" * 5))
for cat in sorted(cat_totals, key=lambda x: -cat_totals[x]):
    pct = 100.0 * cat_totals[cat] / len(mi)
    print("  {:<20s} {:>10,} {:>5.1f}% {:>5d}".format(cat, cat_totals[cat], pct, cat_pkg_count[cat]))

print()
print("PER-PACKAGE BREAKDOWN")
print("-" * 95)
print("  {:<16s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>5s} {:<15s}".format(
    "Package", "Total", "O0", "O1", "O2", "O3", "Bins", "Category"))
print("  {} {} {} {} {} {} {} {}".format(
    "-" * 16, "-" * 7, "-" * 7, "-" * 7, "-" * 7, "-" * 7, "-" * 5, "-" * 15))
for pkg in sorted(pkgs, key=lambda x: -pkgs[x]):
    po = pkg_opts[pkg]
    cat = pkg_cat.get(pkg, "Other")
    print("  {:<16s} {:>7,} {:>7,} {:>7,} {:>7,} {:>7,} {:>5d} {:<15s}".format(
        pkg, pkgs[pkg], po.get("O0", 0), po.get("O1", 0), po.get("O2", 0),
        po.get("O3", 0), len(pkg_bins[pkg]), cat))

print()
print("CROSS-PROJECT SET (excluded from training k-NN index)")
print("-" * 60)
xproj = ["tengine", "angie", "nginx118", "recutils", "curl"]
for pkg in xproj:
    status = "in match_index" if pkg in pkgs else "demo/labels only"
    count = pkgs.get(pkg, 0)
    print("  {}: {:,} functions ({})".format(pkg, count, status))
