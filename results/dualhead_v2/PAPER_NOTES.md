# Paper notes (collected during experiment phase — writing itself is user-gated)

## Evaluation-protocol strictness (user directive 2026-09-01: "mention how all evaluation protocol is more strict than theirs")
Dedicated paragraph (likely in Evaluation Setup or a "Comparison fairness" box) arguing our protocol is
strictly harder than every baseline's own, point by point:

1. **Split unit — package family, not project/binary/function.** XFL & AsmDepictor split at random
   function level; SymLM & SymGen at binary level; BLens/Epitome at project level. We split by package
   *family* (name-overlap ≥0.35 chains packages), which additionally separates version pairs and forks
   (diffutils2/3, nginx→angie/tengine/openresty) that even project-level splits leak across.
2. **Test-side exact-body removal against train** (SymGen-style dedup, applied on top of the disjoint
   split): any test function whose V3 token sequence appears anywhere in train is excluded (−81,988
   rows). Cite SymGen's own demonstration: AsmDepictor 0.72 → 0.05 F1 after dedup (14×); their
   in-dataset dup ablation ~2.9×. SymLM confirmed no-dedup (GitHub issues); BLens excludes
   empty/overlapping/locally-bound only.
3. **Dynsym exclusion (unique to us):** names exported in .dynsym survive stripping, so "predicting"
   them is reading the input (−12,746 rows). No baseline drops these.
4. **Residual near-corpus mass is labeled, not blended:** near-duplicates that exact hashing cannot
   catch are quarantined by construction — family-linked or ≥60%-verbatim-name packages are tagged NCT
   and every table reports FT/NCT separately, plus seen/novel-name strata. Closest prior = BLens's
   third tier ("reduced shared components": 0.79→0.46→0.32) — cite as the overlap-controlled precedent
   our FT regime generalizes.
5. **Macro alongside micro always** (bdb+icu = 44% of raw test would otherwise dominate micro).
6. **Identical policy applied to every baseline we retrain** (SymGen-34B LoRA, BLens on our tier,
   same scored keys, same metric-v2 canonicalization) — strictness never differentially penalizes us.
7. Framing sentence: "our protocol combines SymGen's deduplication with BLens's overlap-controlled
   cross-project evaluation, and strengthens both (family-level disjointness, dynsym exclusion,
   labeled residual-overlap regimes)."

Sources: memory research_dataset_distributions.md (paper details verified 2026-08-03),
project_leakage_table_v2 (old 89.5% tok-identical test → why hygiene matters), split policy v3 in
data/split_v2.json + docs/DATASET_V2_CARD.md.

## Metric presentation (user directive 2026-09-01)
- **Macro (per-package mean F1) = the default metric in the paper body.** The terms "micro/macro"
  are considered confusing/non-principled — do not use them in the body; call the default simply
  "F1 (averaged per package)" or "per-package F1". Function-weighted (micro) numbers +
  the micro-vs-macro comparison go to the **appendix as an ablation**.
- Caveats raised at note time (user to decide during writing): (1) published baseline numbers
  (SymGen 0.38, BLens 0.46/0.32, SymLM 0.277) are function-level — any table citing them must
  use the matching aggregation or footnote the difference; (2) selective-prediction/router
  numbers are currently function-level — recompute per-package variants or scope the claim.
  All per-package numbers already exist in every results JSON (macro_pkg fields) — no reruns needed.

## Balanced 50/50 seen/novel view (user request 2026-09-01)
Derived (no resampling): each cell = mean(seen-name, novel-name score) from score_symgen_full_sem.json.
Union bal-macro 0.377 / bal-micro 0.458 / bal-EM 35.6%; SymGen 0.210/0.197/5.1%; retrieval 0.314/0.402.
Balanced lens rewards dual capability (union leads SymGen +0.17 macro) but reweights a 12/88 reality
50/50 — label loudly; suggested placement: appendix next to strata table; body keeps blended + strata.
Also: SymLM-style semantic F1 (CodeWordNet) computed for everything — uniform +0.01–0.02, ordering
unchanged → cite as evidence low scores are task difficulty, not metric harshness.

## Published baseline numbers — placement (discussed 2026-09-01)
Main comparison table = same-data same-metric retrained systems ONLY. Published numbers appear only in
related-work prose or a separated "original paper, different data — not comparable" block, for three
purposes: (1) preempt "SymGen says 0.38, you show 0.196" — the delta IS the protocol-strictness
evidence; (2) systems not retrained (SymLM 0.277 via BLens's measurement, XFL, AsmDepictor) are only
positionable via published numbers; (3) SymGen own-adapter vs our-tier adapter (0.120 vs 0.124 FT)
shows our retraining didn't cripple it.

## Novel-name evidence-stratified claim (2026-09-01, results/novel_head_analysis.json)
"The 34B LLM's novel-name advantage disappears once input evidence is controlled for": on rows where
GT tokens are present in the shared decomp input, A4-220m ≥ SymGen-34B (full-evidence 0.550 vs 0.464);
SymGen's aggregate edge lives entirely in weak/zero-evidence rows (65% of novel mass, everyone ~0.04)
and pretraining-familiar projects (angie ngx_*, fossil sqlite3 internals) → corpus prior / borderline
contamination, not composition. Heads complementary (oracle +34%). A4 defect: C++ targets not demangled
(icu mangled fragments) — fix candidate.

## Other queued paper points (from earlier sessions)
- Ghidra-vs-BAP ablation reading (FINAL_TABLES T3d): decompiled text worth +0.029 head / +0.017
  system micro; code-LM on raw BAP-IR ≈4× the GRU decoder on FT/novel → FT gap was LM pretraining +
  capacity, not the IR. Supports BAP-only variant as a legitimate reduced-dependency configuration.
- val-cannot-arbitrate finding (C1 design doc addendum): package-disjoint val ties (~0.1595) across
  C1 variants while test separates them; val NCT = 3 packages. Report as a protocol lesson.
- Dual-head narrative measured cross-system (P4 interim) and now within our system (T3d): retrieval
  owns seen/NCT, generation owns novel/FT, no system has both; router captures ~92% of 2-head oracle.
- "votes" tokenizer naming: consider renaming in paper (frequency-voting ≠ Epitome's multi-model
  voting) — old CLAUDE.md note.

## Why CodeT5p-220m as the generation head (2026-09-02, for Design/Discussion section)
Justification ladder (strongest first):
1. EMPIRICAL ON OUR PROTOCOL: SymGen (NDSS'25) = CodeLlama-34B + LoRA on decomp; our fully
   fine-tuned 220m + evidence digest beats it overall (0.212-0.217 vs 0.145 micro), on FT
   (0.144-0.151 vs 0.118) and on novel names (0.148-0.156 vs 0.128) with 150x fewer params.
   Evidence-cut analysis: with full input evidence 220m 0.550 > 34B 0.464; the 34B's edge is
   confined to zero-evidence rows + pretraining-familiar projects (sqlite3/bfd/ngx memorized).
2. FULL FINE-TUNE VS LORA: stripped-binary decomp is far OOD from LLM pretraining text; 220m
   permits full-weight adaptation on 190K rows, 1xA100-40G, 6.5h. LoRA on 34B adapts a
   rank-limited subspace (SymGen's own choice, forced by scale).
3. ARCHITECTURE FIT: encoder-decoder; bidirectional encoding of 1280-token input -> ~5-subtoken
   name. CodeT5+ pretrains with span denoising (mask-filling identifiers) — near-isomorphic to
   our [MASK]-the-name formulation. Decoder-only chat LLMs are tuned for long-form generation.
4. INFERENCE ECONOMICS: 268K-function eval sweeps in ~3h on one A100 enable daily ablation
   cycles (9 full-test sweeps this week). SymGen-34B full-test required 34 shards over days.
5. CONTAMINATION HYGIENE: big code LLMs saw our test packages' source on GitHub (SymGen novel-EM
   concentrated in nginx-fork/sqlite internals). REFORGE (arXiv 2607.07738) and REBench
   (arXiv 2604.27319) both name pretraining leakage as the central confound in LLM binary-naming
   evals. A small, older-corpus model keeps our strict-eval story coherent.
6. ATTRIBUTION: contribution = evidence construction + routing, not scale. Small LM keeps digest
   deltas measurable; the from-scratch ablation (running) quantifies the LM prior itself.
   Aside: the from-scratch 220m NaN'd under bf16 (T5 random-init instability) — pretrained
   weights are load-bearing even for numerical stability.
7. SCALE PATH: CodeT5p family 220m->16B; 770m sbatch prepared (dh2_sbatch/a4_train_770m.sbatch)
   as a controlled scale ablation on the winning recipe. GenNm (NDSS'25) fine-tunes
   CodeGemma-2B/CodeLlama-7B for the sibling task (variable names) — the "fine-tune small-ish,
   don't prompt huge" pattern is now the literature consensus.

## Ablation attribution matrix (2026-09-02, response to user's architecture-justification question)
User's observation is CORRECT: the scratch run measures the PRETRAINING PRIOR, not "the code LM
as an architecture choice". Full matrix:

TIER A — why dual-head at all (evidence largely COMPLETE):
 A1 heads-alone vs routed vs oracle: R 0.127 / A4 0.217 / GBT 0.2335 / oracle 0.2533 (test).
 A2 complementarity: retrieval owns NCT+seen (NCT 0.564, seen-EM 0.708) with novel-EM 0.0002;
    generation owns FT+novel (0.151/0.156) where retrieval collapses (FT 0.039). Cross-system:
    SymGen-34B shows the same FT-strong/NCT-weak profile => regime split is task structure,
    not our artifact. Instance-level: A4-only EMs 1,780 vs SG-only 4,049 vs shared 2,769.
 A3 single-model alternatives FAILED with gates (2026-08): retrieval-augmented decoder,
    retrieve-and-edit, learned fusion, composition-as-decoding — all closed negative.
    Dual-head was arrived at BY ELIMINATION, not assumed (honest narrative for paper).
 A4 router ablation: regime-rule 0.2132 < conf-threshold 0.2147-0.2228 < learned GBT 0.2335
    (router2_eval) + calibrated abstention (0.96@5%).

TIER B — generation head choices:
 B1 input representation: BAP-IR text 0.155 vs Ghidra decomp 0.184 (same LM) — DONE.
 B2 evidence digest: none 0.184 -> modctx 0.2095 -> poolctx 0.2169 — DONE (3 points).
 B3 pretraining prior: scratch (random-init, same arch/data/recipe) — RUNNING.
    Reading: if scratch ~= old GRU decoder (0.077), architecture w/o prior buys ~nothing at our
    budget and the prior is the payload; if scratch >> 0.077, transformer+text contributes too.
 B4 scale: 770m sbatch prepared, run on winning recipe — PENDING.
 B5 enc-dec vs decoder-only: NOT CLEANLY ABLATABLE — no public checkpoint pair shares a
    pretraining corpus across architectures; a dec-only run (e.g. small Qwen-Coder) would be a
    "design alternative" comparison, not an ablation. Optional; state the confound if run.
 NOTE architecture-vs-pretraining cannot be fully factorized with public checkpoints; B3's
    scratch + comparison against old custom decoder brackets it honestly.

TIER C — retrieval head choices:
 C1 MISSING (cheap, high-value): kNN over LM-text embeddings (pool the fine-tuned A4 encoder)
    vs our contrastive BAP-graph embeddings. Answers the reviewer question "why not one model
    for both heads?". Eval-only GPU job (embed 190K train + 268K test, no training).
 C2 our encoder's pretraining: July 2x2 factorial (scale +0.104 dominates, pretrain needs
    capacity) — DONE on older setup, cite with caveat.
