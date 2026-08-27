# C1 — Name-aware contrastive training for the retrieval head (design, 2026-08-25)

Status: DESIGN. Not launched. Gate numbers below come from `eval_v2.py` only.

## 1. Problem this attacks

Diagnosis chain (P2-Baseline / A1a, metric v2, package-disjoint test):

| stratum | retrieval (A1a enc + string rerank) | vocab-oracle | gap |
|---|---|---|---|
| ALL micro / macro | 0.118 / 0.381 | — | — |
| seen-name (NCT) | 0.635 (decoder) / 0.792 (retrieval, P4 NCT sample) | ~1.0 | selection |
| novel-name (FT) | 0.024 (decoder) / 0.037 (retrieval, P4 FT sample) | ~0.33 (RAEC Stage A oracle) | selection ≫ vocabulary |

The retrieval head is the system. Its embedding space is trained **only through the decoder's
cross-entropy** (the encoder never sees a name-similarity signal), plus optional exact-name
NT-Xent (`compute_contrastive_loss` in `src/training/train.py`, positives = identical name
only, historically off: `--contrastive-weight 0.0`). Consequences:

1. **Selection gap.** A query's nearest train neighbours share *token* structure, not *name*
   structure. Among top-20 candidates the right name is present far more often than it is
   ranked first (RAEC/SECC: selection = 14–19 % of oracle everywhere). Nothing in training
   pushes `read_config_file` closer to `parse_config` than to `write_log_line`.
2. **Novel names are unreachable by construction.** With exact-name positives, a test
   function whose name never occurs in train has *no* positive geometry to fall into. A
   name-similarity-weighted objective gives it partial credit for landing near
   `*_config_*` functions, which is exactly what sub-token F1 rewards.

Prior negative result to state honestly: exact-name NT-Xent as a train-time loss was a WASH
in the CE/generator era ([[project_dualhead_track_20260814]], "contrastive loss (as
train-time loss)" in the DO-NOT-RETRY list). That verdict was measured on the **generator**
and with **exact-name** positives. C1 changes both the positive definition and the head
being optimised, so it is not a retry of the same experiment; it is still a risk.

## 2. Objective

For a batch of encoder embeddings `z_i` (post-fusion, the vector `dump_embeddings_v2.py`
indexes), names `n_i`, packages `p_i`:

```
s_ij   = SubtokenF1(n_i, n_j)                      # metric v2 split_name, i != j
w_ij   = s_ij * (1 + beta * [p_i != p_j])          # cross-package positives up-weighted
P_ij   = w_ij / sum_k w_ik                          # soft target distribution (rows with sum 0 are skipped)
Q_ij   = softmax_j( cos(z_i, z_j) / tau )           # j != i
L_C1   = mean_i  KL(P_i || Q_i)                     # soft-label InfoNCE (SupCon with graded labels)
L      = L_CE(decoder) + lambda * L_C1
```

Defaults: `tau = 0.1`, `beta = 1.0`, `lambda in {0.3, 1.0}`. Ablation `s_ij -> [n_i == n_j]`
reduces to the existing exact-name loss (control run).

Retrieval vector: try both (a) backbone `z` (what the current index uses) and (b) a 2-layer
projection head `g(z)` trained with `L_C1` while `z` stays decoder-owned. (b) protects the
decoder head from any contrastive damage; (a) is simpler. Both are dumped by
`dump_embeddings_v2.py` in one job.

## 3. Batch construction (the part that decides whether this works)

`ContrastiveBatchSampler` currently packs ~32 exact-name pairs per batch. Replace with a
**name-aware sampler**:

* Positive pool for anchor `i`: train functions in a *different package* with
  `s_ij >= 0.5` (near-name), falling back to exact-name pairs in other binaries.
  Availability is what the C1 census (job 1195584, `results/c1_census/report.json`)
  measures — see §6.
* **Hard negatives**: from the A1a kNN dump (`results/emb_p3a/train_knn.npz`), each train
  function's top-10 train neighbours with `s_ij < 0.2`. These are literally the
  candidates the current retrieval head confuses. One hard negative per anchor in-batch.
* Batch = 16 anchors × (1 positive + 1 hard negative) + 16 random = 64 (fits A100 at the
  A3 config; cut to 32 if memory-bound).

Dedup interaction: protocol train is one sample per (token-hash, name), so within-binary
O0/O2 pairs survive only when bodies differ — the existing sampler already lives with that.

## 4. What must NOT change

* Split policy v3, eval hygiene, `eval_v2.py` scoring — untouched.
* Retrieval pipeline code path (index build, kNN, string rerank, router) — untouched; C1
  only changes the checkpoint the pipeline consumes.
* Decoder head: must not regress > 0.005 micro on val_xproj, otherwise use variant (b).

## 5. Gates and cost

| gate | metric | must beat |
|---|---|---|
| G1 | val_xproj retrieval micro/macro (eval_v2, k=1) | best BAP-only encoder (A1a 0.1147/0.3797 or A3+ if higher) |
| G2 | val_xproj retrieval + string rerank | 0.118 / 0.381 |
| G3 | novel-name stratum retrieval F1 | A1a novel 0.024 (decoder) / FT-sample 0.037 — target ≥ 0.06 |
| G4 | selective F1 @10 % coverage (router) | 0.754 (must not drop) |
| G5 | decoder micro on val_xproj | ≥ A3+/A1a − 0.005 |

Cost: 2 retrains (λ = 0.3, 1.0) + control (exact-name positives) = 3 A100 jobs of the
A3+ length; embedding dump + rerank + router per checkpoint ≈ 2 h CPU each.

## 6. Census results (job 1195584, `results/c1_census/report.json`, metric-v2 split_name, Jaccard on sub-token sets; candidate generation skips sub-tokens with df>3000 = {ossl, get})

Cross-package name availability (positive = a function in a *different* package):

| tier | n | exact name in another pkg | near (J >= 0.5, non-identical) | weak (J > 0 only) | none |
|---|---|---|---|---|---|
| train (anchors) | 190,151 | 17.0 % | 23.3 % | 53.9 % | 5.7 % |
| val_xproj vs train | 10,617 | 19.4 % | 26.9 % | 51.5 % | 2.2 % |
| val FT / NCT | — | 10.4 % / 97.7 % | 29.9 % / 1.1 % | 57.2 % / 1.2 % | 2.5 % / 0 |
| test vs train | 268,178 | 12.9 % | 19.3 % | 63.6 % | 4.2 % |
| test FT / NCT | — | 2.4 % / 65.2 % | 20.1 % / 15.1 % | 72.6 % / 18.8 % | 4.9 % / 1.0 % |

Reading:
* **40.3 % of train anchors** have a cross-package positive at J >= 0.5 (17.0 exact + 23.3
  near) — clears the 30 % rule in the decision below, so the name-aware sampler is not
  degenerate. Near-positives look semantically right (`parse_entries`↔`rsrc_parse_entries`,
  `do_statistics`↔`do_disk_statistics`, `file_reader_init`↔`bzip2_reader_init`).
* **FT ceiling for name geometry**: on test-FT only 2.4 % of names exist verbatim in
  train, but 22.5 % have a J >= 0.5 train name and 95 % share at least one informative
  sub-token. Exact-name positives can therefore teach nothing about FT; graded positives
  are the only training signal that touches this stratum. The J >= 0.5 stratum bounds
  what a near-name retrieval can score: ~0.22 × (F1 of the matched name, ≈0.6) ≈ 0.13
  micro on FT if selection were perfect — vs 0.037 now. Weak overlaps (72.6 %) are the
  partial-credit tail sub-token F1 actually pays for.
* NCT is already saturated on exact names (val 97.7 %); C1 must not disturb it (gate G4/G5).

Decision: proceed with the full design (name-aware sampler + hard negatives + λ sweep).
The "hard-negative-only" fallback is not needed as the primary run; keep it as an ablation.

## 7. Relation to C2 (candidate scoring) and A4

C1 shapes the *code* embedding with name geometry. C2 adds a *name text* encoder and trains
code↔name alignment (CLIP-style); at inference it scores evidence-assembled candidate names
instead of generating them. C1 is a prerequisite: C2's negatives and positives use the same
`s_ij` and sampler. A4 (decompiled-text generation head) is orthogonal — it feeds the
generation head; C1/C2 feed retrieval/scoring. Router logic unchanged either way.

## Addendum (2026-08-27): val_xproj cannot arbitrate data-side changes — proposal for a non-GNU dev slice

Three experiments moved val_xproj and not the test tier: A3+ (val ≈, test −0.009), A4 run 2 (+SymGen rows: val +0.037,
test −0.008), BAP v3 retrain (+SymGen corpus: val +0.03–0.04 on decoder and retrieval, test ±0.00). Val's far-transfer
packages (direvent, rush, wdiff, spell, cppi, csplit2, iotop, tig, zstd) are 7/10 GNU/gnulib-flavoured; the additions
were gnulib-heavy. The test tier's mass is non-GNU (icu, libsodium, mbedtls, tinycc, tcsh, cvs, sysstat…).

Options (none free of cost):
1. **Non-GNU dev slice from the train pool** — move 3–4 non-GNU train packages (candidates: lmdb, jansson, lighttpd
   are already test; from train: e.g. `lz4`, `xz`, `libyaml`, `expat`… need the roles list) into a second dev tier.
   Cost: they leave training; every checkpoint would need retraining for strict comparability. Only worth it for the
   final frozen runs.
2. **Report per-regime val and gate on retrieval val** (adopted 2026-08-26) — catches representation regressions
   (A3+) but not GNU-domain overfitting of data additions.
3. **State it as a finding**: model selection on a domain-skewed dev set selects for the dev domain; report test for all
   variants with the dev-selected one marked. This is what the results ledger does now.
Recommendation for the paper: (3) now, (1) if a final re-run pass happens after the heads are frozen.
