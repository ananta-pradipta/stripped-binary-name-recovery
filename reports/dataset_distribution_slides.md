# Dataset Distribution & Package Coverage

Slide deck: package-level breakdown across train / val / test / cross-project splits, with opt-level coverage and function-type descriptions.

Format: Markdown with `---` slide separators. Paste into reveal.js, Marp, or manually port to Google Slides / PowerPoint.

---

## Slide 1 — Dataset Split Overview

| Split | Packages | Binaries | Functions | Opt Coverage | Name Overlap w/ Train |
|---|---|---|---|---|---|
| **Train** | 58 | 432 | 177,824 | O0/O1/O2/O3/default | 100% (self) |
| **Val** | 10 | 17 | 7,823 | O0, O2, default (no O1/O3) | **~65%** (bash_O0 drags) |
| **Test** | 9 | 18 | 13,559 | O0, O2, default (no O1/O3) | **~93%** |
| **Cross-Project** | 4 | 34 (33 active) | 9,492 | **O0/O1/O2/O3** (full coverage) | **~66%** weighted |

**Opt distribution (% of functions):**

| Split | O0 | O1 | O2 | O3 | default |
|---|---|---|---|---|---|
| Train | 51.4% | 12.9% | 15.8% | 11.7% | 8.2% |
| Val | 58.7% | 0% | 20.3% | 0% | 21.0% |
| Test | 65.3% | 0% | 16.7% | 0% | 18.0% |
| Cross-Project | ~43% | ~15% | ~27% | ~15% | 0% |

**Key observations:**
- Val/test have **no O1 or O3 functions at all** — cross-project is the only split providing full opt-level coverage.
- Val overlap is artificially low due to bash_bash_O0 (51% of val, only 32.3% overlap). All other val packages have 95-100% overlap.
- Test is clean: 93% weighted overlap.
- Cross-project overlap (~66%) is intentionally designed to include both high-overlap (nginx forks ~99%) and low-overlap (recutils ~19%) packages.

---

## Slide 2 — Training Set (Top 15 Packages by Function Count)

| Package | Opt Levels | Functions | Function Types |
|---|---|---|---|
| binutils | O0/O1/O2/O3/def | 51,415 | ELF/DWARF parsing (bfd_*), disassembly, symbol lookup |
| coreutils | O0/O1/O2/O3/def | 14,398 | File utilities + gnulib helpers |
| sqlite | O0/O2 | 10,527 | B-tree, VDBE, SQL parser (sqlite3_*) |
| groff | O0/O1/O2/O3 | 8,803 | Text formatting, troff/eqn/tbl/pic |
| coreutils2 | O0/O1/O2/O3/def | 8,760 | Extended GNU utilities (dd, df, du, ls, shred, sort, stty) |
| strace | O0/O2 | 4,873 | syscall tracing, ptrace decoders |
| curl | O0/O2 | 4,827 | HTTP/FTP/SMTP client (curl_*) |
| tar | O0/O1/O2/O3 | 4,699 | Archive handling + gnulib |
| lua | O0/O2 | 4,170 | Lua VM, stdlib (lua_*, luaL_*) |
| gawk | O0/O1/O2/O3 | 4,125 | AWK interpreter, regex, field parsing |
| libxml2 | O0/O2 | 3,794 | XML parser (xml*, html*), DOM tree |
| findutils | O0/O1/O2/O3/def | 3,256 | find/xargs + gnulib |
| inetutils | O0/O1/O2/O3/def | 3,248 | Network tools (ftp, ping, telnet, traceroute) |
| less | O0/O1/O2/O3/def | 3,164 | Pager, termcap |
| jq | O0/O2 | 3,085 | JSON query language, lexer/parser |

**58 total training packages** span GNU core, hub libs, system tools, interpreters, and archive/compression.

---

## Slide 3 — Training Set (Bottom Packages) & Hub Libraries

| Category | Packages (8) | Function Types |
|---|---|---|
| **Hub Libraries** | zlib, libxml2, libsodium, libarchive, libpng, libyaml, pcre2, expat, sqlite, lua, curl | Compression (deflate), XML/HTML parsing, crypto primitives, archive codecs, image codecs, YAML, regex, SQL engine, scripting |
| **GNU Core Tools** | coreutils, coreutils2, binutils, groff, findutils, texinfo, bison, m4, gawk, sed, grep, gzip, xz, tar, cpio, patch, diffutils, flex, make, bc, hello, less, nano, which, time, units, indent | gnulib helpers (xmalloc/xrealloc/error/quotearg/savedir), file ops, parsing, regex, text processing |
| **System / Daemon** | strace, htop, acct, inetutils, direvent, dos2unix, jq, screen, nginx, bash, rush, sqlite | Process monitoring, network tools, file watchers, shells, HTTP server |
| **Niche / Small** | spell, gperf, enscript, datamash, cppi, csplit2, combinatorics, lz4, bzip2, libyaml, expat, time | Spell check, hash generation, PostScript, stats, compression |

---

## Slide 4 — Validation Set (10 packages, 7,823 functions)

Val = held-out binaries from same packages as train, at different opt levels. **Name overlap** = fraction of val function names that also appear in training (at any opt level of any package).

| Package | Held-out Opts | Functions | Name Overlap with Train | Function Types |
|---|---|---|---|---|
| **bash** | **O0 only** | **4,010 (51%)** | **32.3%** ⚠️ | Shell parser, exec, expansion, builtins. **77.7% are O0-only helpers inlined at higher opts** |
| coreutils | O0/O2/def | 1,000 | 97.1% | File utilities + gnulib |
| tar | def | 746 | 100.0% | Archive + gnulib |
| gawk | def | 645 | 100.0% | AWK interpreter |
| screen | O2 | 520 | 100.0% | Terminal multiplexer |
| m4 | O2 | 455 | 100.0% | Macro processor + gnulib |
| findutils | O2 | 355 | 100.0% | find/xargs + gnulib |
| which | O2/def | 42 | 95.2% | Command lookup |
| time | O0 | 33 | 63.6% | Timing wrapper |
| less | O2 | 17 | 100.0% | lesskey utility |
| **Val total** | — | **7,823** | **~65%** (weighted) | Dominated by bash_O0 drag |

**Caveat:** bash_bash_O0 alone accounts for 51% of val functions, and **only 32.3% of its names are in training** (the rest are O0-only helpers inlined at higher opts). Every other val package has ≥95% name overlap. Val F1 = 0.4978 is dragged down almost entirely by bash_O0 — a composition artifact, not a model weakness.

---

## Slide 5 — Test Set (9 packages, 13,559 functions)

| Package | Held-out Opts | Functions | Name Overlap with Train | Function Types |
|---|---|---|---|---|
| binutils | O0/O2/def | 7,078 | **99.2%** ✅ | nm-new variants: ELF/DWARF parsing, symbol lookup |
| bison | O0/O2 | 3,908 | **76.4%** ⚠️ | LALR parser generator + obstack; some O0-only helpers |
| coreutils | O0/O2/def | 1,618 | 99.1% | Held-out utility binaries |
| patch | def | 518 | 100.0% | Diff application + gnulib |
| coreutils2 | def | 219 | 100.0% | du, mktemp |
| findutils | def | 102 | 100.0% | xargs |
| inetutils | def | 98 | 100.0% | ping |
| less | O0 | 13 | 53.8% | lessecho |
| time | def | 5 | 100.0% | Timing wrapper |
| **Test total** | — | **13,559** | **~93%** (weighted) | Binutils-dominated (52%), biased toward ELF/DWARF parsing |

Test has much cleaner overlap (~93% weighted) than val (~65%) — no dominant O0-only outlier like bash.

---

## Slide 6 — Cross-Project Set (4 packages, 34 binary-opt combos, 9,492 scored functions)

Packages **never seen** in train/val/test at any opt level. Selected to share code lineage with training.

| Package | Type | Binaries | Opts | Scored Fns | Name Overlap vs Training | Shared Function Types |
|---|---|---|---|---|---|---|
| **tengine** | nginx fork (Taobao) | 1 (nginx) | O0, (O2 bug) | 554 | **81.7%** | `ngx_http_*`, `ngx_event_*`, `ngx_conf_*`, `ngx_buf_*`, `ngx_string_*` — direct nginx lineage |
| **angie** | nginx fork (ex-nginx devs) | 1 (angie) | O0/O1/O2/O3 | 3,893 | **89.9%** | Full nginx `ngx_*` namespace + angie `angie_api_*` extensions |
| **nginx118** | nginx 1.18.0 (LTS, Apr 2020) | 1 (nginx118) | O0/O1/O2/O3 | 3,470 | **99.3%** | Complete `ngx_*` namespace, older version of the nginx code in training |
| **recutils** | GNU text DB tool | 6 (recfix, recdel, recfmt, recinf, recins, recsel) | O0/O1/O2/O3 | 1,575 | **19.5%** (gnulib only) | gnulib helpers (`xmalloc`, `error`, `quotearg`, `c_isalpha`, `_gl_alloc_nomem`, `hash_*`) — shared with every GNU training package |
| **Total** | | **9 unique binaries** | | **9,492** | ~66% weighted | |

Binary-opt combos: 2 (tengine) + 4 (angie) + 4 (nginx118) + 24 (recutils: 6 bins × 4 opts) = **34 total** (33 active; tengine_O2 has 0 matches due to BAP/debug address misalignment bug).

**Final measured F1 (P2 mode, contrastive_model.pt ep46, job 904339):**

| Package | N | F1 (P2) | EM (P2) | Comment |
|---|---|---|---|---|
| nginx118 | 3,470 | **0.8134** | 53.1% | Highest F1 — oldest nginx version, near-complete namespace overlap |
| tengine | 554 | **0.7613** | 62.3% | Highest EM — O0-only due to preprocessing bug |
| angie | 3,893 | **0.7392** | 45.1% | Largest scored count |
| recutils | 1,575 | **0.3572** | 28.6% | Only gnulib overlaps; application code (rec_*) unseen |
| **Overall** | **9,492** | **0.7043** | **46.3%** | **Paper headline number** |

**Non-tengine packages eval at all 4 optimization levels** — cross-project is the only split providing O1/O3 coverage.

---

## Slide 7 — Shared Function-Type Categories Across Splits

| Category | Examples | Share Pattern |
|---|---|---|
| **gnulib helpers** | `xmalloc`, `xrealloc`, `xstrdup`, `error`, `quotearg_*`, `hash_lookup`, `c_isalpha`, `getopt_long`, `savedir`, `rpmatch` | Vendored by every GNU package → appears in ~27 training packages AND cross-project recutils → strong transfer layer |
| **nginx HTTP/event** | `ngx_*` (ngx_http_*, ngx_event_*, ngx_conf_*, ngx_buf_*) | Training `nginx` → cross-project tengine / angie / nginx118 (all nginx forks) |
| **Hub library APIs** | `xml*`, `html*`, `png_*`, `crypto_*`, `archive_*`, `sqlite3_*`, `pcre2_*`, `lua_*`, `z_*` | Vendored in training hub packages; would transfer to any package statically linking these libs (none of our cross-project do) |
| **Package-specific app code** | `rec_*` (recutils), `ngx_api_angie_*` (angie), `bash_parse_*`, `nm_find_*` | Unique per package, NOT shared → hard to recover via k-NN retrieval |

**Retrieval works best** on functions whose names are in the first two categories. Package-specific application code is the limiting factor for cross-project F1.

---

## Slide 8 — Key Observations & Paper Positioning

1. **Optimization-level coverage gap:** Train has O0/O1/O2/O3/default; val/test have ONLY O0 and O2. Cross-project is the only split providing O1/O3 eval signal.
2. **O0 dominance:** Train is 51% O0, val 59% O0, test 65% O0 — distribution mismatch biases metrics toward O0 performance.
3. **Val composition artifact:** `bash_bash_O0` dominates val (51% of functions), and 78.6% of its names are O0-only helpers absent at higher opts. Val F1 = 0.4978 largely reflects this artifact, not a model capacity issue.
4. **Cross-project set is curated for measurable transfer:** 4 packages, 9 unique binaries, 34 binary-opt combos (33 active), 9,492 scored functions. 3 nginx-family packages (tengine/angie/nginx118) cover direct code lineage; recutils covers the gnulib-transfer story.
5. **Shared function types explain retrieval performance:** nginx lineage → 0.74–0.81 F1 (P2); gnulib layer → 0.36 F1; truly novel package-specific code → ~0 F1.
6. **Final paper numbers (contrastive_model.pt, P2 mode):**
   - **Val F1 = 0.498** (dragged by bash_O0, 7,823 functions)
   - **Test F1 = 0.770** (13,559 functions, clean in-distribution)
   - **Cross-project F1 = 0.704** (9,492 functions, 4 packages × 4 opts)

---

*File: `reports/dataset_distribution_slides.md`*
*Updated 2026-04-05. Reflects v3 hub-expanded dataset (203K functions), contrastive_model.pt (P3 ep46), final 4-pkg cross-project eval (job 904339). Binary count corrected from 26 → 34.*
