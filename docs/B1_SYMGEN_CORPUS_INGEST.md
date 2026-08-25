# B1 — SymGen corpus ingest (approved 2026-08-25 23:11 UTC, "based on current BAP-only result")

Source: Zenodo 15530083 ("Binary Files of SymGen", NDSS'25). 33 projects × {x86_64, x86_32, arm, mips} × {O0..O3}:
`binaries.tar.gz` (4.7 GB, unstripped ELFs), `unstripped_decompiled_data.tar.gz` (1.2 GB, Ghidra JSON per binary:
name → decomp_code / args_metadata / assembly / function_address), `stripped_decompiled_data.tar.gz` (1.1 GB),
`source_projects.tar.gz` (0.2 GB). Wulver: `/project/hz79/_shared/cs785/symgen_corpus/` (binaries tarball symlinked
from the April download in the course dir; x86_64 unstripped decomp already extracted there).

## Overlap audit (`scripts/symgen_overlap_audit.py`, name-normalized vs `split_v2.json` meta.roles)

| class | projects |
|---|---|
| **EXCLUDED — collide with our val/test tiers (9)** | coreutils (val_xproj + xproject_NCT), diffutils (FT + NCT), gettext (FT), gawk, grep, gzip, inetutils, tar, units (NCT) |
| already in our train pool (9) | bash, binutils, cflow, curl, datamash, mailutils, openssl, texinfo, wget2 |
| new (15) | adns, dico, freeipmi, gmp, gss, libiconv, libidn2, libmicrohttpd, libpng, libredwg, libtool, libunistring, ncurses, poke, readline |

Ingestible: **24 projects, x86_64, O0–O3**. Version differences (e.g. grep-3.8 vs our grep) do NOT make a project disjoint.

## Hygiene rules (unchanged from split policy v3)
1. Package-disjoint eval: the 9 excluded projects never enter any training set (either head).
2. Body-dedup: test/val functions whose token-hash occurs in the enlarged train are dropped from scoring (`eval_v2.py`
   already does this against whatever train set the checkpoint saw — re-run, don't reuse P2/A1a dropped-lists).
3. Train dedup: NOT required (user, 2026-08-24); ingest all builds.
4. A4 head trains on SymGen **stripped** decompilations (FUN_xxx bodies, like our own A4 input), never on the
   unstripped text (real callee names in bodies = train/test mismatch and mild leakage). Unstripped JSON is used only
   for labels (name ↔ function_address) — same role as our debug ELF `.symtab`.

## Pipeline
- **BAP heads**: extract x86_64 ELFs (job 1195865) → strip copies (`strip --strip-all`) → labels from unstripped
  `.symtab` (same `relift_v2.py` path as dataset v2; container recipe in [[project_wulver_bap_container]]) → BAP +
  eh_frame rooter lift on Wulver CPU nodes → `match_index` rows with `corpus: symgen_zenodo` → DatasetV2 with
  `corpora: [local_main, symgen_zenodo]`. Expect ~2,000 binaries (24 proj × 4 opt × ~20 bins) ≈ the size of relift v2
  (1,890 bins, which took ~1 day wall-clock on CPU nodes).
- **A4 head**: join stripped-decomp JSON (address-keyed) with unstripped names → `results/a4_ft/train_symgen.jsonl`
  (same schema as `a4_build_ft.py` output) → concatenate with our 190K rows; `--max-src 1024`.
- **External benchmark**: SymGen's paper split is by ??? (see §Open) — if it is project-level, score our heads on
  their held-out projects that are not in our train (candidates: the 15 new projects, held out entirely as a
  second FT-style test) → reported as "SymGen-corpus FT" next to our own test. Decision below.

## Open decisions
- Which of the 15 new projects to hold out as external benchmark vs ingest as train. Proposal: hold out 5
  (gmp, libpng, ncurses, libmicrohttpd, poke — distinct domains, none gnulib-heavy), ingest 10 + the 9 train-pool
  overlaps = 19 projects. Report SymGen-34B on the same 5 (its LoRA was trained on the *full* SymGen corpus, so its
  number on them is train-contaminated — must be stated).
- Enlarged-corpus retrains: A1a-config encoder (retrieval/decoder) and A4 — after the current A4 run finishes.

## Status log
- 2026-08-25 23:15 UTC: audit done; downloads of source/decomp tarballs running; job 1195865 extracting x86_64 ELFs
  + decomp function census.
