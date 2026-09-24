# Encoder diagnostic probe — is the contrastive encoder taking an external-call shortcut?

**Date:** 2026-08-05 · **Checkpoint:** `ccs/checkpoints/best_model_paper_clean.pt`
(leakage-clean baseline, epoch 48, val F1 0.5033, 7,004-name vocab)
**Split:** `ccs/data/split_assignments_paper_clean_strict_idx.json` (passed explicitly;
no split file was modified)
**Scripts:** `scripts/probe_encoder.py`, `scripts/probe_pretrain_pairs.py`,
`scripts/probe_encoder.sbatch` · **Jobs:** 1160244 (main), 1160267 (M3 rerun)
**Raw results:** `results/ndss_prep/probe_encoder.json`,
`results/ndss_prep/probe_encoder_m3.json`, `results/ndss_prep/probe_pretrain_pairs.json`

Measurement only — nothing was retrained, no split/vocab file was touched.

Scale: 243,289 clean-train functions (k-NN index), 13,581 cross-project functions
loaded from graphs across all 7 evaluation packages, 2,000 functions per analysis
group, 400,000 sampled pairs per group, 6,000 sampled pretraining pairs.

---

## VERDICT

**The shortcut hypothesis is CONFIRMED at the embedding level, and PARTIALLY
confirmed at the objective level — external-call matching is a real shortcut, but
it is the *second* strongest one. The strongest is plain bag-of-token overlap.**

Three things are true simultaneously:

1. **Embedding similarity is governed by external-call overlap, not by semantics.**
   Two functions with the *same name* but *different* external calls sit at cosine
   0.13 (training pkgs) / 0.05 (dash-gettext-psmisc). Two functions with
   *different names* but the *same* external calls sit at 0.11 / 0.14. Controlling
   for ext-call overlap, the name-semantic signal in the embedding is
   approximately zero. This is the shortcut signature, and it is unambiguous.

2. **But the contrastive objective is not primarily solved by ext calls.** Ext-call
   Jaccard alone separates a true positive pair from an in-batch negative with
   **AUC 0.78**, and is unavailable for 42% of positives (neither view has any ext
   call). Raw token-multiset Jaccard alone reaches **AUC 0.88** and is available on
   100% of pairs. Since the block encoder mean-pools token embeddings, its
   representation is close to a bag of instruction types — so the dominant
   trivializing feature is the token bag, and ext calls ride along inside it.

3. **17% of the "positive" pairs are literally the same token stream**, and the
   mean token-multiset Jaccard across positives is 0.66. The pretraining task is
   mostly asking the encoder to match near-identical inputs.

**A second, separate result refutes the standing explanation for the FT/NCT gap.**
The O0↔O2 embedding cosine for the *same* function is ~0.3 on **training**
packages, ~0.3 on FT packages and ~0.26–0.43 on NCT packages (M3). There is no
regime gap in encoder invariance — the pretraining objective failed to deliver its
target (~1.0) everywhere, including on its own training distribution. The
prediction-consistency gap measured on 2026-08-05 is therefore a *decoder/retrieval*
effect (dense neighbourhoods absorb the drift), not evidence that the learned
invariance is package-specific. That framing in the experiment log needs correcting.

**Consequences for the two planned fixes:**

- **(i) Hard-negative mining — WORTH PURSUING, but redesign the mining criterion.**
  Mining negatives that share external calls attacks the AUC-0.78 feature and
  leaves the AUC-0.88 one intact. Mine on **token-bag similarity** (optionally
  jointly with ext-call similarity); that is the feature that currently makes the
  task trivial. Expected effect is on the FT regime, where the crosstab shows the
  semantic signal is being crowded out entirely.
- **(ii) Ext-call masking augmentation — LOW VALUE AS SPECIFIED.** It removes a
  feature that is absent on 42% of positives anyway and leaves the stronger
  shortcut untouched. If pursued at all, mask the *whole* call channel (ext-call
  encoder input **and** the `CALL_<sym>` block tokens — the `ext_all` condition
  below), not just the encoder input. Even then the residual token bag still
  separates positives from negatives.

**Two defects found along the way outrank both fixes in expected value** — see
Findings D1 and D2. Neither is an architecture problem.

---

## D1. The batched collate misaligns edge indices (k-NN index and queries are in different spaces)

`collate_fn` (in `src/training/train.py` and `scripts/eval_cross_project.py`)
offsets `edge_index` by each sample's **actual** `num_blocks`, while the node
tensor is padded to `max_blocks = 30` and the pooling vector is
`arange(B).repeat_interleave(30)`. Sample *b*'s nodes therefore live at rows
`b*30 + i`, but its edges were shifted by `sum(num_blocks[:b])`. Every sample
after the first in a batch receives edges pointing into an earlier sample's rows.

Measured on 1,024 clean-train functions:

| comparison | mean cos | median | 5th pct | frac < 0.99 |
|---|---|---|---|---|
| batch=1 vs batch=32, repo convention | **0.958** | 0.970 | 0.885 | **84.0%** |
| batch=1 vs batch=32, offset fixed to `max_blocks` | **0.997** | 1.000 | 0.987 | 10.4% |
| repo convention vs fixed, both batch=32 | 0.961 | 0.975 | 0.886 | 78.6% |

The near-1.0 second row proves the discrepancy is the offset convention and
nothing else. Two consequences:

- Cross-project **queries** are encoded at batch=1 (`predict_binary_with_embeddings`),
  the k-NN **index** is built batched. They are not in the same space.
- **Training** also ran with the misaligned convention, so the GAT learned on
  scrambled connectivity. Fixing the collate is therefore a **retrain-required**
  change, not a drop-in patch.

**Honest counterweight:** rebuilding the k-NN index with the fixed convention
changes retrieval quality only marginally (top-1 F1 nginx118 0.859→0.878,
angie 0.725→0.752, dash 0.203→0.219, gettext 0.030→0.030). So the bug is real
and worth fixing, but it is not the lever that explains far-transfer failure —
and it hints that the GAT contributes little to `z` in the first place.

---

## D2. Five packages have no `CALL_<sym>` token channel at all — three of them are FT evaluation packages

Across 79 packages (measured on `*_sub_*` function graphs only):

| package | regime | % functions with any named `CALL_<sym>` token |
|---|---|---|
| **dash, gettext, psmisc** | FT eval | **0.0** |
| **grep, sed** | training | **0.0** |
| xz / pcre2 / libxml2 (lowest nonzero) | training | 13–19 |
| recutils | FT eval | 70.7 |
| nginx118 / angie / tengine | NCT eval | 73–79 |
| coreutils / binutils / bash | training | 43–90 |

In those five packages every call token is `CALL_INTERNAL` or `CALL_INDIRECT`, at
every optimization level. Their `data/external_calls/*_external.json` files are
populated normally (dash: 81 distinct callees), so the ext-call *encoder* channel
works — only the block-token channel is missing. This looks like a `parse_bap.py`
batch/version difference, not a property of the binaries.

**Why it matters:** dash / gettext / psmisc are the three worst FT packages on
every metric we report (F1 0.036–0.152; cross-optimization prediction consistency
11.5–22.2%), while recutils — the one FT package that *has* the channel — is the
best (0.328 F1, 44.1% consistency). Three of four FT evaluation packages are in the
zero set versus two of the other 75 packages. **The FT-vs-NCT regime comparison in
the 2026-08-05 "ENCODER INSTABILITY" entry is confounded by a missing input
channel** and should be re-reported with dash/gettext/psmisc separated from
recutils until the graphs are rebuilt.

*Measurement trap for whoever re-checks this:* globbing
`data/graphs/<pkg>_*.json` gives a false nonzero, because that directory also
holds 2-token PLT-stub files named after the library symbol
(`sed_sed_O2_strlen.json`) which naturally contain a `CALL_<sym>` token. Restrict
to `*_sub_*.json`, equivalently to `match_index.json` keys.

---

## M1. What embedding cosine actually tracks

400,000 sampled pairs per group; standardized OLS betas of cosine on
[ext-call Jaccard, block-count ratio, name sub-token F1], restricted to pairs
where both functions have at least one external call.

| group | ext-Jaccard β | block-ratio β | **name-F1 β** | R² |
|---|---|---|---|---|
| train | 0.355 | 0.071 | 0.268 | 0.235 |
| xproj NCT (nginx118/angie/tengine) | **0.573** | 0.118 | **0.100** | 0.392 |
| xproj FT no-channel (dash/gettext/psmisc) | 0.433 | 0.133 | 0.270 | 0.345 |
| xproj FT recutils | 0.468 | 0.104 | 0.380 | 0.581 |

Restricted to **cross-package** pairs, the semantic signal collapses while the
ext-call signal does not:

| group (different-package pairs) | ext-Jaccard β | name-F1 β | ratio |
|---|---|---|---|
| train | 0.349 | 0.244 | 1.4× |
| xproj NCT | 0.571 | 0.098 | **5.8×** |
| xproj FT no-channel | 0.383 | 0.015 | **25×** |

The decisive cut is the 2×2 crosstab (high = ≥0.6, low = ≤0.1 for Jaccard, =0 for
name F1). Mean cosine per cell:

| group | same name, same ext | **same name, DIFF ext** | **DIFF name, same ext** | diff name, diff ext |
|---|---|---|---|---|
| train | 0.582 (n=62) | **0.131** (n=19) | **0.114** (n=1,718) | 0.030 |
| xproj FT no-channel | 0.562 (n=482) | **0.048** (n=108) | **0.137** (n=5,749) | 0.028 |
| xproj FT recutils | 0.696 (n=3,745) | **0.175** (n=1,214) | **0.226** (n=1,273) | 0.035 |

Read the two middle columns. Functions that mean the same thing but call different
libraries are *no closer* — and on cross-project data are **2.9× farther** — than
functions that mean different things but call the same libraries. (NCT cells are
degenerate: every nginx name shares the `ngx_` prefix, so name-F1 is never 0.)

Baseline for calibration: random pairs sit at cosine 0.02–0.08. The space is
near-orthogonal, so all of these are small absolute numbers; the *ratios* are the
result.

---

## M2. Ablation displacement — which input channel is the embedding built on

Mean cosine between the original embedding and the embedding after zeroing one
input, computed only over functions where that channel is actually present
(lower = larger displacement = more reliance).

| group | ext encoder | callee ctx | caller ctx | `CALL_<sym>` tokens | ext encoder + call tokens | all code tokens → `<UNK>` |
|---|---|---|---|---|---|---|
| train | **0.512** | 0.534 | 0.711 | 0.969 | 0.729 | 0.338 |
| xproj NCT | **0.469** | 0.515 | 0.681 | 0.927 | 0.689 | 0.219 |
| xproj FT no-channel | **0.512** | 0.524 | 0.724 | 1.000¹ | 0.710 | 0.350 |
| xproj FT recutils | **0.447** | 0.629 | 0.668 | 0.955 | 0.582 | 0.402 |

¹ exactly 1.0 because those packages have no named call tokens to mask (D2).

Channel presence rates: ext 40–70%, callee 34–76%, caller 28–66%.

**Reading:** among the three context channels the external-call encoder is the
single largest contributor everywhere, marginally ahead of callee context and
clearly ahead of caller context. Destroying the code tokens moves the embedding
much more than any single context ablation — but that is a far larger intervention
(every token, not one channel), so it does not license "code dominates". What it
does license: **the encoder is not exclusively ext-driven**. That is the part of
the hypothesis that does not survive.

The `CALL_<sym>` token channel by itself is nearly inert (0.93–0.97), i.e. the
ext-call signal reaches `z` through the ext-call *encoder*, not through the block
tokens.

---

## M3. Cross-optimization embedding drift

Same source function encoded at O0 and at O2; compared against a within-package
mismatched baseline. Reported per package with the fraction of pairs whose O0 and
O2 token streams are byte-identical, because several packages turned out to be
duplicates rather than genuine optimization pairs.

| package | regime | n pairs | **cos(O0, O2) same fn** | median | random baseline | z | identical inputs |
|---|---|---|---|---|---|---|---|
| busybox | TRAIN | 400 | **0.228** | 0.201 | 0.038 | 3.5 | 0% |
| sqlite | TRAIN | 400 | **0.268** | 0.244 | 0.040 | 3.9 | 0% |
| strace | TRAIN | 400 | **0.272** | 0.254 | 0.042 | 3.8 | 0% |
| libxml2 | TRAIN | 400 | **0.287** | 0.247 | 0.050 | 2.9 | 0% |
| curl | TRAIN | 400 | **0.313** | 0.294 | 0.028 | 5.3 | 0% |
| tmux | TRAIN | 400 | **0.337** | 0.294 | 0.029 | 4.5 | 0% |
| binutils | TRAIN | 400 | **0.357** | 0.339 | 0.034 | 5.8 | 0% |
| coreutils | TRAIN | 400 | **0.407** | 0.404 | 0.056 | 4.5 | 1% |
| coreutils4 | TRAIN | 400 | **0.410** | 0.403 | 0.072 | 4.1 | 3% |
| coreutils3 | TRAIN | 400 | **0.423** | 0.422 | 0.071 | 4.8 | 4% |
| dropbear | TRAIN | 400 | 0.546 | 0.470 | 0.047 | 7.4 | 16% |
| ~~groff~~ | TRAIN | 400 | ~~0.973~~ | ~~1.000~~ | 0.065 | 10.0 | **95% — DEGENERATE** |
| angie | NCT | 400 | **0.256** | 0.163 | 0.083 | 1.7 | 0% |
| nginx118 | NCT | 400 | **0.428** | 0.380 | 0.087 | 3.6 | 0% |
| gettext | FT | 213 | **0.278** | 0.258 | 0.034 | 3.8 | 1% |
| dash | FT | 218 | **0.295** | 0.281 | 0.032 | 4.3 | 0% |
| recutils | FT | 252 | **0.511** | 0.506 | 0.086 | 3.1 | 4% |

*(psmisc had only 29 usable O0/O2 pairs and tengine none — those binaries exist at a
single optimization level — so both are omitted. groff is excluded from any mean:
95% of its "O0/O2 pairs" are byte-identical token streams, i.e. the two binaries are
the same code. gnuchess, mailutils, texinfo and zlib showed the same duplication in
the first run (58–92% identical) — worth a separate corpus-hygiene check.)*

**This is the result that contradicts the standing interpretation.** Non-degenerate
training packages average **cos = 0.33** (range 0.23–0.42). FT packages: 0.28, 0.30,
0.51. NCT packages: 0.26, 0.43. **There is no regime gap in cross-optimization
embedding stability.** The encoder is equally far from optimization-invariant
everywhere — the pretraining objective's target was ~1.0 and it delivers ~0.3 on
its own training distribution.

So the prediction-consistency gap reported on 2026-08-05 (NCT 46–54% vs FT 12–22%)
is **not** produced by the encoder being more invariant on familiar packages. The
embeddings drift just as much on training packages; what differs is that in dense,
well-covered regions of the training manifold a drifting embedding still lands
among near-clone neighbours with the same name, so the *decoder/retrieval output*
is stable even though the *representation* is not. Downstream stability is masking
upstream instability.

Practical consequence: **cross-optimization prediction consistency is a decoder-side
metric, not the encoder-quality proxy the earlier entry proposed.** The encoder-side
version of that metric is this cosine, and by that measure the contrastive
pretraining has largely not achieved its stated objective anywhere.

---

## M4. k-NN neighbourhood quality vs the vocabulary oracle

Index: 243,289 clean-train embeddings, 42,692 distinct training names. Queries:
250 per package. Oracle = best sub-token F1 achievable by *any* name in the
training vocabulary.

| package | regime | oracle F1 | top-1 | top-5 best | top-50 best | **capture @top-1** | capture @top-50 |
|---|---|---|---|---|---|---|---|
| nginx118 | NCT | 0.998 | 0.859 | 0.909 | 0.954 | **86%** | 96% |
| tengine | NCT | 0.924 | 0.795 | 0.842 | 0.877 | **86%** | 95% |
| angie | NCT | 0.972 | 0.725 | 0.804 | 0.868 | **75%** | 89% |
| recutils | FT | 0.844 | 0.283 | 0.352 | 0.465 | **34%** | 55% |
| dash | FT | 0.867 | 0.203 | 0.279 | 0.379 | **23%** | 44% |
| psmisc | FT | 0.829 | 0.148 | 0.192 | 0.340 | **18%** | 41% |
| gettext | FT | 0.534 | 0.030 | 0.044 | 0.125 | **6%** | 24% |

This reproduces the oracle-ceiling result on a 250-function-per-package sample and
sharpens it: **the correct-enough name is not merely ranked badly, it is not in the
top 50 at all.** Going from top-1 to top-50 recovers only 44%→24% of the oracle for
FT packages. Retrieval depth is not the fix; the geometry is wrong.

Note the ordering within FT tracks D2 exactly: recutils (has the call channel)
0.283 > dash 0.203 > psmisc 0.148 > gettext 0.030.

---

## M5 / pretraining-pair audit (`probe_pretrain_pairs.py`)

411,313 positives in `data/pretrain_pairs.json`; 6,000 sampled; every pair crosses
optimization levels (0% same-level).

| property | value |
|---|---|
| pairs whose two views are **byte-identical token streams** | **17.3%** |
| mean token-multiset Jaccard between the two views | **0.657** |
| pairs with identical *non-empty* ext-call sets | 29.0% |
| pairs where **neither** view has any ext call | **42.4%** |

Discriminability against in-batch negatives (Mann–Whitney AUC; 0.5 = useless,
1.0 = solves the task):

| feature alone | mean on positives | mean on negatives | **AUC** | coverage |
|---|---|---|---|---|
| token-multiset Jaccard | 0.657 | 0.304 | **0.877** | 100% |
| external-call Jaccard | 0.552 | 0.030 | **0.780** | 58% pos / 71% neg |

Both are shortcuts. The token bag is the stronger one and it is always available.
A mean-pooling block encoder computes something very close to a token bag, so the
architecture and the objective are aligned on the *wrong* invariant.

---

## Recommended next steps, in expected-value order

1. **Rebuild `parse_bap.py` graphs for dash / gettext / psmisc / grep / sed** so
   they carry the `CALL_<sym>` channel, then re-run the FT evaluation. Until then,
   every FT-vs-NCT claim is confounded (D2). This is data-engineering work, not
   modelling work, and it is cheap.
2. **Correct the "ENCODER INSTABILITY" entry on two counts**: separate recutils from
   dash/gettext/psmisc (D2), and retract the claim that the learned invariance is
   package-specific — at the embedding level it is uniformly absent (M3).
   Cross-optimization *prediction* consistency should be relabelled a
   decoder/retrieval-robustness metric.
3. **Fix the collate edge offset and retrain** (D1). Expect little retrieval gain
   on its own, but it is a correctness bug that undermines the GAT and makes every
   graph-encoder ablation in the paper hard to defend if a reviewer finds it.
4. **Hard-negative mining keyed on token-bag similarity** (fix (i), redesigned).
   This is the one architectural/objective change the probe supports.
5. Ext-call masking augmentation (fix (ii)) — deprioritize, or fold into 4 as the
   `ext_all` variant.

## Limitations

- Cross-project functions come from raw graphs (the production query path);
  `match_index.json` has no entries for nginx118/angie/tengine/recutils, so the
  dataset path alone would have covered only 3 of 7 packages.
- The clean-train index here is 243,289 functions; the paper pipeline uses 241,174
  because it also drops curl. Retrieval numbers are therefore close to, but not
  identical with, the published k-NN rows.
- M2's "all code tokens → `<UNK>`" row is a much larger intervention than the
  single-channel ablations and is reported for scale only, not as a like-for-like
  comparison.
- One anomaly left unexplained: for recutils, ablating ext-encoder + call tokens
  together (0.582) displaces *less* than ablating the ext encoder alone (0.447).
  Gated re-balancing can produce this, but it was not investigated.
- M3 pairs a ground-truth name to at most one function per optimization level by
  taking the largest body (a name can match both an O0 thunk and its body, and the
  `-4` address fallback adds more). Requiring strict uniqueness instead retains only
  ~1% of cross-project pairs. Where both rules apply the answer is unchanged
  (busybox 0.228 vs 0.230, coreutils3 0.423 vs 0.423, tmux 0.337 vs 0.337).
- psmisc (29 usable pairs) and tengine (0 — single optimization level available)
  are missing from M3 entirely.
