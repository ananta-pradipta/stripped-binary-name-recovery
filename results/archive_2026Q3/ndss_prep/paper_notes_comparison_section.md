# Paper-writing notes: baseline comparison section (collected 2026-08-05/06)
# User instruction: include a short explanation of these scoring conventions
# IN THE PAPER (not just as a repo caveat). No paper edits until the
# experiment phase closes — this file is the staging area.

## Must-explain in prose near the comparison table
0. **PRIMARY TABLE SCORES ALL SYSTEMS AGAINST RAW GROUND TRUTH** (user decision
   2026-08-07, on the principle that evaluation must be against the symbol-table
   name). BLens's own-space number becomes a SECONDARY column with one
   explanatory sentence. Rationale: our sub-token F1 already gives partial
   credit, so raw scoring does not punish BLens's normalization — expansions
   still score (quote_argument_free vs quotearg_free = 0.4), truncations still
   score (ngx_http_core_post_access vs ..._phase = 0.89) — it merely stops
   rewarding it, and truncation is a real capability limit (an incomplete name
   is an incomplete answer to the analyst). Under the single raw basis:
   ours 0.546-0.551 > BLens ~0.507 > SymGen 0.498; far transfer ours 0.151 >
   BLens 0.111. REFRESH the matched tables on this basis + the final no-rerank
   configuration before writing the section.

1. **BLens scoring convention (short paragraph, user-requested).** BLens
   canonicalizes ground-truth names before training (abbreviation expansion,
   token drops, truncation: quotearg_free -> quote_argument_free;
   ngx_http_core_post_access_phase -> ngx_http_core_post_access). We score its
   predictions against ITS OWN canonical targets — the convention its paper
   uses and the charitable choice. The choice is material: 0.507 (raw names)
   vs 0.641 (own space) on the same 6,381 matched functions. Ours and SymGen
   are scored against raw nm symbols. State: no neutral option exists when
   systems define different output vocabularies; we report the charitable one
   and note the aggregate ordering is unchanged (our lead widens under raw).
1b. **BLens FT numbers are basis-dependent — print BOTH (measured 2026-08-07).**
   Scoring identical COMBO predictions against BLens's normalized targets vs raw
   nm symbols: FT 0.338 vs 0.111 (inflation +0.227), NCT 0.498 vs 0.411 (+0.086).
   The 2.6x-larger FT inflation is mechanical: FT packages use compressed
   identifiers (dash averages 1.16 sub-tokens/name — popredir, evalvar), which
   raw scoring makes all-or-nothing while BLens's expansion creates partial-credit
   surface. Under the raw basis (the one ours and SymGen use) BLens FT = 0.111,
   BELOW our 0.151. Do NOT state "BLens beats us on far transfer" without the
   basis qualifier; print both columns and say cross-system FT comparison is only
   valid within one basis.

2. **Matched-subset protocol.** Every number in the comparison table is on
   three-way-shared functions (identical population, identical sub-token F1);
   own-population numbers are stated once and marked non-comparable.
3. **SymGen caveats.** (a) Its 4-pkg eval population is the .dynsym-leak
   subset (ghidra_decomp_v2 skips FUN_*); internal callee names reach its
   decompiled input on those packages. (b) 9.3% of its v2 inputs contain the
   GT name verbatim (replace-first-only masking; recutils 15.3%). (c) Its
   dash/gettext/psmisc numbers in our table come from OUR corrected FT3
   protocol (internal functions, address-matched GT) — the original 5newpkg
   eval was 100% import stubs and is reported nowhere except as a methodology
   warning. (d) recutils has no valid FT3 run; its SymGen cell carries the
   dynsym caveat.
4. **tengine matched rows are O0-only** (our graph coverage); BLens's own
   tengine number (0.532) is propped by statically-linked OpenSSL functions,
   runtime scaffolding, and O2 duplicates outside the shared population.
5. **Parameter/cost framing.** 25M (ours, BAP lift only) vs ~200M ensemble
   (BLens, PalmTree+CLAP embeddings) vs 34B (SymGen, per-function Ghidra
   decompilation; +31-83% wall time for the decompile pass — cite our
   same-hardware Table-5 measurements). SymGen long-input failure modes seen
   in FT3: prompt-echo degeneration, empty </s> outputs.
6. **The unified table layout the user approved (2026-08-06 Discord):** one
   table, NCT block / FT block with subtotals, three-way matched keys per
   row, ours-base + ours-clanginv + SymGen + BLens columns, ALL-7 row:
   ours 0.546/0.551 > BLens 0.513 > SymGen 0.498.
7. **2-D coverage framing** (evidence axis vs corpus-homolog axis): SymGen's
   valid FT profile tracks the evidence-yield diagnostic (dash 0.07->0.026,
   gettext 0.71->0.135, psmisc 0.53->0.342); dash is corpus-axis (busybox-ash
   homologs) where we beat the 34B model 5x. Everyone hits the FT wall
   (0.095/0.106/0.172) — an order of magnitude below NCT for all systems.


## EVAL-SET COMPOSITION — the CCS "experimental setting" criticism (assessed 2026-08-07)
The reviewer objection is substantially CORRECT and our own measurements prove it. It is a
composition problem, not dishonest selection, and the fix is reporting discipline:
1. **Three of seven eval packages are nginx forks** (nginx118/angie/tengine) = 7,917 of 13,581
   functions (58%). They are not independent samples. Effective package count is 5, not 7.
2. **That family has a training sibling** — the older `nginx` package remains in train under
   every split definition (leakage audit: nginx-family verbatim coverage flows through it).
   Our strongest numbers therefore come from the most favourable possible setting.
3. **n-weighting amplifies it**: headline 0.555 (n-wt) vs 0.412 (per-package mean) — same
   model, same data; the gap is pure composition.
4. Domain narrowness: overwhelmingly GNU/coreutils-style C.
**MANDATORY WRITING CHANGES (no new experiments needed):**
- Lead with three STRATA, not one aggregate: near-clone ~0.78 / far transfer ~0.15 /
  unseen domain ~0.02 (mbedtls+lmdb+jansson, 42,730 fns, EM 0.0%).
- Always print per-package mean beside n-weighted, and explain the difference.
- Collapse the nginx family to ONE row (or report family means) — three forks are one sample.
- State the nginx-sibling-in-training relationship in the main text, not an appendix.
- Do not lead with 0.555 unqualified: a reviewer computes 0.412 in ten minutes, and finding
  it themselves is far more damaging than us stating it first.
**Defensive evidence we DO have** (cite when answering this objection): self-published unseen-
domain floor (0.018-0.032, EM 0.0%), our own leakage audit (88%/74.5% verbatim), the
third-party Ubuntu build set, and the matched-subset + raw-label protocol discipline.
