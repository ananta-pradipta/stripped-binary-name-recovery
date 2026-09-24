# Dual-Head HyDRA — fresh experimental programme (plan v0, 2026-08-17)

Scope: answer the 8-point directive of 2026-08-17. Everything below is gate-disciplined: no phase starts until the previous phase's verification passes, and every fix is asserted on its *effect* (preflight discipline), not on configuration.

## 0. Ground truth I am planning against

**The CCS paper's contributions (paper_ccs.tex, HyDRA):**
1. Ext-call paradox + cascaded gated fusion w/ conditional bypass (ext → callee → caller).
2. Adaptive dual-path inference: k-NN retrieval head + GRU decoder (LM-pretrained sub-token prior), per-binary computable gate.
3. Efficiency: ~32M params, no decompilation (BAP-IR only), no LLM, 41–70% wall-clock saving; 7-pkg held-out eval vs SymGen+LoRA / BLens.

**Why it was rejected (meta-review + persona review):** generalization claims not supported (leakage: dash/gettext/psmisc silently in train; 88% verbatim name overlap; 0.770→0.738 gap "too tight"); dataset scale/coverage (300K vs 1M+; x86-64 GCC only); no SymLM/Epitome comparison; scale vs pretrain not disentangled (2×2 since done); gate is hand-set τ; recutils/OOV is the sub-area's core failure; ML contributions not intuitive.

**What the Aug-2026 research arc established (must respect, not repeat):**
- Composition of *novel* names is closed 4 ways (retmem, unified U1/U2/U3, P1 heads, RARC/RCEM/FEC/SECC). All heads ≈ 0.07–0.10 F1 on truly-novel names. Selection ≈ 14–19% of oracle. → The dual-head narrative CANNOT be "decoder composes novel names". It must be **"retrieval for in-coverage recognition + generation for graceful degradation, with a router that knows which regime it is in"** — measured with coverage-conditioned metrics and selective-prediction curves, not headline generalization.
- Every learned router failed for the SAME reason: the dev set on which it was tuned did not resemble the test regime (validation paradox / prior shift). That is a *dataset* defect (point 6), and it is the reason a learned gate can be re-tried honestly once the dev set is fixed.

## 1. Branch reset (point 1–2) — proposal, needs your OK before I cut

- **Base = `dev`** (CCS-era code, post-submission, non-anonymized; `ccs-anon` is byte-identical except anonymization). `dev`'s loader still has the silent "new binary → train" default that caused the leakage; the guarded loader lives on Wulver `ccs/src/preprocessing/build_dataset.py` (XPROJ guard) — I port that guard in as fix #1.
- **New branch: `dualhead-hydra`** cut from `dev`.
- **Nothing deleted.** `unified`, `openvocab`, `ndss27`, `dualspace` stay as archive branches; I add tags `archive/…` so they are findable. `results/experiment_log.md` (3,987 lines, append-only) and every `*_REPORT.md` are carried onto the new branch under `results/archive_2026Q3/` as the "lessons learnt" record. Untracked data dirs on disk stay untouched (I'll list what is safe to purge later; disk purge only on your explicit word).
- Uncommitted WIP on `unified` gets one archive commit so nothing is lost.

## 2. BAP-pipeline defect list (point 3) — from the logs, with the fix + the test for each

| # | Defect (evidence) | Effect on the paper | Fix | Verification (must pass before Phase 2) |
|---|---|---|---|---|
| B1 | **Collate edge misalignment** (`collate_fn` offsets `edge_index` by real block count, nodes padded to 30) → ~63/64 functions per batch trained/indexed with scrambled CFG edges; GAT became a set encoder (log 2026-08-05, E2 2026-08-13) | Every trained checkpoint saw wrong graphs; retrieval index vs query in different spaces | Offset by `max_blocks`; unit test on batch=1 vs batch=32 equivalence | cos(z_batch1, z_batch32) ≥ 0.999 on 1,024 fns; edge-permutation sensitivity test (GAT output must change when edges are shuffled) |
| B2 | **`CALL_<sym>` block-token channel missing** for dash/gettext/psmisc/grep/sed (0% of fns; `*_external.json` fine) — parse_bap.py batch artifact | 3 of 4 FT eval packages had a crippled input; grep/sed in train | Re-parse ALL graphs with one deterministic parse_bap run; add invariant "if external.json non-empty ⇒ ≥1 CALL_<sym> token" | Per-package channel-presence table, no package at 0%; graphs byte-reproducible across two runs |
| B3 | **Address matching**: BAP `sub_` addresses at variable offset from `nm` (tengine_nginx_O2 matched 0/569; only ENDBR64 −4 fallback exists) | Whole binaries silently dropped; ~12% "loader drop" unexplained | Range-based matching (nm addr+size containment), PIE rebase detection, per-binary match-rate report | Match rate ≥95% for every binary or the binary is listed as dropped with reason; zero silent drops |
| B4 | **Build mismatch**: 15 `data/stripped/*_stripped` are a different build than their `.bir` (curl_O0, xmllint, zlib×4, grep×4, sed×4) | String/rodata channels read from the wrong ELF; labels possibly misaligned | Re-lift from the matching build (or drop with record); `rcdg_stage0_elfcheck.py --all` becomes a pipeline gate | ELF build-id == the build-id recorded in the .bir manifest for 100% of retained binaries |
| B5 | **Duplicate builds mislabelled as opt levels** (groff O0/O2 95% identical, mailutils 92%, gnuchess 80%, texinfo 69%, zlib 58%) | Contrastive O0/O2 pairs partly trivial; "opt-mix" stats wrong | Detect via graph-hash identity; relabel opt as `default`; exclude near-identical pairs from contrastive pretraining | Pair-identity rate reported per package; no pretraining pair with token-Jaccard = 1.0 |
| B6 | **PLT-stub graphs** named after library symbols live in `data/graphs/` (`sed_sed_O2_strlen.json`) | Possible label leakage / noise if they enter training as named functions | Audit whether they enter the dataset; exclude non-`sub_` graphs at load | Count of non-sub graphs admitted = 0 |
| B7 | **V3 discards signals present in .bir**: immediates/magic constants, rodata string refs (uppercase hex — first pass silently matched nothing), global-data addresses, arg count/register identity, expression/def-use structure | Lexical evidence (strings) unavailable to either head; ~27KB semantics → ~600 tags | Adopt `parse_bap_v2` outputs as *additional* channels: (a) per-function string-ref set, (b) immediate buckets incl. magic constants, (c) arg-count/calling-convention features. Feed to BOTH heads (retrieval embedding + decoder), not generation-only | Channel coverage stats per package; ablation with/without each channel on the new dev set |
| B8 | **Silent-default split leakage** (dash/gettext/psmisc → train) | The rejected headline | Port XPROJ guard; split file becomes git-tracked and hashed into every checkpoint | Loader raises on any unassigned binary; checkpoint stores split-hash |
| B9 | **Function-level leakage across "package-level" splits** (linking mode / gnulib / version channel; 88% verbatim overlap) | Reviewer's "too tight" attack | Function-level dedup across splits (exact token-stream hash + name), SymGen-style; report both dedup'd and raw | Leakage table: verbatim-name %, token-stream-dup %, composability % per eval package |

Everything above is preprocessing/loader-side; the raw `.bir` (936 binaries, 4.5 GB local) is preserved so no re-lifting is needed except for B4.


## 2b. Phase-0 audit results (2026-08-17, local corpus, all 488K graphs; see phase0_audit/AUDIT.md) — plan deltas

| # | Measured | Delta to plan |
|---|---|---|
| B3 | Median per-binary match rate 0.72 (O0 0.998, O1 0.71, O2 0.64, O3 0.70); 99.7% of misses = BAP emitted no `sub_` at the label address; `main` missing in 831/942 binaries although a symbol-named `<bin>_main.json` graph exists in 813; +16,949 further (binary,label) pairs with a symbol-named graph but no match; 58,367 duplicate samples (19%) from thunk+body double-matching; 55 zero-match binaries (recutils×36, cflow, cvs, lighttpd, tinycc, tengine, wget_O0) most at Δ=0 (matcher never run) | **B3a** matcher v2: symbol-named graphs + PIE rebase + thunk/body dedup + per-binary report. **B3b (new)** eh_frame rooter: on 30 random O1–O3 stripped binaries, `.eh_frame` FDE starts cover median 99.3% of labelled functions vs BAP 71% (bash O2/O3 22%→100%). Root BAP on FDE starts (deployable on stripped ELFs). Test: discovery ≥0.95/binary. |
| B2 | 53 binaries with 0% `CALL_<sym>`: dash/gettext/psmisc/grep/sed + cvs/lighttpd/tinycc + libyaml; stale-parser root cause confirmed | as planned; cvs/lighttpd/tinycc "vocab-domain miss" conclusion is confounded — re-evaluate |
| B5 | 310 opt-pairs ≥0.5 identical; datamash O0=O3 1.0, hello O0=O1=O2 1.0, gperf O1=O2=O3 1.0, groff O0-O2 0.93–0.98, traceroute, ginstall-info, strace | opt truth from graph-hash identity in the manifest; drop identical pairs from contrastive; relabel |
| B6 | 42 label-noise samples (BAP mis-named subs); 100,843 non-sub graph files; thunk resolution can absorb PLT-stub graphs | exclude non-sub keys; thunk resolver must reject stub bodies |
| B8 | 414/881 binaries (32.6% of fns) unsplit → silently train under dev loader; split file names non-existent ids | as planned (guard + tracked split); corpus manifest first |
| B1 | failing unit test written | as planned |
| corpus | local `data/` (939 bir, 302K fns, 431/17/18/4 split) ≠ Wulver `ccs/data` (933 bins, 613K graphs, 821/17/18/77/208 split) | **P0.1 corpus manifest** (local+Wulver): per binary build-id, debug/stripped/bir/graphs presence, opt truth, label source; single source of truth for dataset v2 |

## 3. Dataset redesign (point 6)

Problems: val is by-binary *within training packages* → SEEN-dominated (validation paradox; every learned gate tuned on it failed on test); package-level split ≠ function-level leakage; coverage (77 GNU-heavy pkgs, x86-64 GCC) is the reviewers' #1 concern; scale 300K vs 1M+ (but blind expansion 397K→1.4M *regressed* xproj — curation > scale).

Design (subject to the Phase-0 literature check):
- **Three-tier protocol** (BLens ladder + SymGen dedup): (i) in-distribution test (binary-level, dedup'd), (ii) **package-disjoint dev** that mirrors the test regime mix (used for ALL threshold/gate/router selection), (iii) held-out cross-project test, stratified by *coverage regime* — near-clone (version/fork), far-transfer, and lexically-isolated — with a leakage table published per package.
- **Coverage expansion, curated:** add the domains already built (mbedTLS/lmdb/jansson, expat-class parser), Clang corpus (97,619 verified fns), O0–O3; add ≥2 non-GNU domains per regime; every candidate package passes B4/B5/B9 audits before admission. Target ~500K–1M functions only if per-package curation holds; scale is not the goal, regime coverage is.
- **Benchmark alignment (research task, Phase 0):** check availability/licensing of SymLM's, BLens's (XFL Debian), SymGen's released datasets; if one is usable in x86-64 stripped ELF form, we evaluate on it as an external benchmark so a SymLM/Epitome comparison becomes possible.

## 4. Metrics (point 7)
- Primary: sub-token **F1** (micro), plus **per-package macro-F1** (n-weighted aggregate is dominated by nginx-family — reviewers noticed).
- Secondary: EM, EdSim/NgSim per table (promised in rebuttal).
- New, narrative-bearing: **coverage-conditioned F1** (seen / novel-composable / OOV strata, with leakage %), **risk–coverage / selective prediction** (AURC, F1@k% coverage — a router that abstains is deployable value), **calibration** (ECE was ≥0.56 on all heads — fixing this is itself a contribution), efficiency (params, ms/function, end-to-end wall clock).

## 5. Experiment ladder (point 4–5, 8) — each rung has a gate

- **Phase 0 — Audit & research (no training).** Reproduce the CCS numbers on `dev` code from the shipped checkpoint (sanity anchor); run the B1–B9 measurement scripts to *quantify* each defect before fixing; literature/benchmark check (SymLM/BLens/SymGen/Epitome data availability, current SOTA table). Gate: every defect has a number and a test.
- **Phase 1 — Pipeline fixes B1–B9 + dataset v2.** Deterministic re-parse, new split file, leakage table. Gate: all verification cells in §2 pass; dataset card written.
- **Phase 2 — Retrain the CCS architecture unchanged on dataset v2** (encoder + fusion + decoder + k-NN, fixed collate). This is the honest new baseline and tells us how much of the old headline was leakage vs. real. Gate: reproducible ±0.005 across 2 seeds; leakage table clean.
- **Phase 3 — Dual-head v2.** (a) enriched channels B7 into both heads; (b) router re-tried on the *fixed* dev set: learned per-query gate over [retrieval margin, ext-Jaccard, coverage features] with calibrated confidence + abstention; (c) decoder as graceful-degradation head (evaluate by selective metrics, not novel-EM). Gate: dual > best single head by ≥+0.01 F1 AND better AURC on the package-disjoint dev; else the dual claim is reported as a coverage-boundary result, not oversold.
- **Phase 4 — Baselines clean.** Re-run BLens and SymGen+LoRA on dataset v2 under the identical protocol (their previous fine-tunes included dash/gettext/psmisc). Envs exist on Wulver.
- **Phase 5 — Ablations & robustness.** Fusion cascade, channel ablations, gate sensitivity, 2×2 scale/pretrain (already done — re-verify on v2), Clang/opt-level robustness, efficiency table.

## 6. What I need from you (blocking only for the branch cut)
1. Confirm base branch = `dev` (vs `ccs-anon`), new branch name `dualhead-hydra`.
2. Confirm "discard" = archive (tags + results copied), not delete. Disk purge of untracked experiment dirs only on your list.
3. Target venue / deadline, if any (affects how deep Phase 3 goes before we lock a dataset).

Non-blocking assumptions I'll proceed under: BAP-only, no decompiler, no LLM ≥1B; Wulver hz79 for GPU; local for BAP; every result to Discord + experiment_log.
