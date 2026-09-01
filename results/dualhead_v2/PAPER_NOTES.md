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
