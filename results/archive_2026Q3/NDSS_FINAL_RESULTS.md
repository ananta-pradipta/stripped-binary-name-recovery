# NDSS'27 Submission — Complete Verified Performance Results

**Compiled 2026-08-10.** Every number in this document is anchored to a job ID and a results JSON.
This supersedes all gate-era tables. Reporting rules: `docs/WRITING_NOTES.md`.

**Model of record (everywhere):** `best_model_cont_control.pt` — 32,208,484 parameters
(15.7M encoder+fusion, 16.5M GRU decoder; composition head +4.1M separate).
**Protocol of record:** retrieval = k-NN + BinFilter τ=0.5 over the clean 241,174-function GCC train
index, raw cosine, no re-rank. Decoder = GRU beam k=5 (autoregressive baseline). Composition =
Linear(1024→4000) on frozen z, BCE, threshold 0.30 (VAL-tuned), unseeded init (±0.01 across runs).

---

## 1. In-distribution test set (GCC, 13,301 functions, 18 held-out binaries of training packages)

| head | F1 | EM |
|---|---|---|
| **Retrieval (k-NN+BinFilter)** | **0.8053** | **74.6%** |
| k-NN no filter | 0.7855 | 73.2% |
| Decoder (beam 5) | 0.7792 | 71.8% |

Anchor: job 1168004 (`results/tau05_t0.5.json`). This 0.8053 is the anchor value every subsequent
eval run must reproduce (and did, in jobs 1169445/1169447).
⚠️ The older "0.770 / 70.8% (n=13,559)" test figure belongs to the pre-sprint Cycle-B model — the
draft should be updated to 0.805/74.6% for one-model consistency (pending author decision).

---

## 2. Cross-project, GCC clean-7 (13,581 functions, 7 fully held-out packages)

### 2.1 Per-package, retrieval + decoder (job 1168004, `tau05_t0.5.json`)

| package | n | J_max | retr F1 | retr EM | dec F1 | dec EM |
|---|---|---|---|---|---|---|
| nginx118 | 3,470 | 1.00 | **0.889** | 75.3% | 0.821 | 67.5% |
| angie | 3,893 | 0.97 | **0.839** | 67.6% | 0.754 | 59.3% |
| tengine | 554 | 0.70 | **0.756** | 69.7% | 0.704 | 59.0% |
| **NCT subtotal** | **7,917** | | **0.855** | **71.1%** | 0.780 | 62.9% |
| recutils | 2,550 | 0.35 | 0.332 | 23.4% | **0.346** | 25.5% |
| dash | 1,324 | 0.29 | **0.238** | 22.7% | 0.133 | 12.5% |
| gettext | 1,518 | 0.17 | **0.038** | 1.1% | 0.037 | 1.0% |
| psmisc | 272 | 0.39 | 0.157 | 9.6% | **0.167** | 12.1% |
| **FT subtotal** | **5,664** | | **0.223** | **16.6%** | 0.205 | 15.3% |
| **Aggregate** | **13,581** | | **0.591** | **48.4%** | 0.540 | 43.0% |

Headline chain: reported 0.5915 → reproduced 0.5913 (job 1168004). BinFilter worth +0.0396
(0.5517 → 0.5913). Decoder 0.5399.

### 2.2 Stratified by name coverage, all three heads (N1 run, job 1169047, `ndss_dual_head_eval.json`)

| stratum | n | retrieval | decoder | composition |
|---|---|---|---|---|
| seen | 10,151 | **0.757** | 0.690 | 0.380 |
| novel_comp | 1,509 | 0.081 | 0.065 | **0.103** |
| oov | 1,921 | 0.082 | 0.071 | **0.124** |
| novel+oov | 3,430 | 0.081 | 0.069 | **0.114** |
| overall | 13,581 | **0.586**† | 0.533 | 0.313 |

† Stratified harness reproduces the 0.591 headline at 0.586 (~1% index-construction difference;
both runs anchored). Composition beats the decoder +57% on novel+OOV (notes-approved figure:
0.1074 vs 0.0686). Decoder exact matches on novel+OOV: **0 of 3,430**.

### 2.3 Regime × stratum × head (computed from N1 per-function records, 2026-08-09)

| regime | stratum | n | retr F1 | dec F1 | comp F1 |
|---|---|---|---|---|---|
| NCT | seen | 7,435 | 0.892 | 0.815 | 0.433 |
| NCT | novel_comp | 320 | 0.297 | 0.196 | 0.247 |
| NCT | oov | 162 | 0.365 | 0.299 | **0.369** |
| NCT | ALL | 7,917 | 0.857 | 0.779 | 0.424 |
| FT | seen | 2,716 | 0.386 | 0.349 | 0.219 |
| FT | novel_comp | 1,189 | 0.022 | 0.030 | **0.056** |
| FT | oov | 1,759 | 0.056 | 0.050 | **0.095** |
| FT | ALL | 5,664 | 0.207 | 0.189 | 0.146 |

NCT is 94% seen names; FT is 48% — the regime gap is mostly stratum mix (coverage boundary).
Composition beats BOTH other heads on FT novel_comp and FT oov (52% of FT functions).

### 2.4 Per-package, all three heads (same records)

| package | n | retr | dec | comp |
|---|---|---|---|---|
| nginx118 | 3,470 | 0.892 | 0.820 | 0.440 |
| angie | 3,893 | 0.840 | 0.753 | 0.420 |
| tengine | 554 | 0.755 | 0.704 | 0.349 |
| recutils | 2,550 | 0.310 | 0.330 | 0.284 |
| dash | 1,324 | 0.227 | 0.121 | **0.006** |
| gettext | 1,518 | 0.022 | 0.018 | 0.033 |
| psmisc | 272 | 0.171 | 0.155 | 0.169 |
| ALL | 13,581 | 0.586 | 0.533 | 0.308 |

Composition profile: strong where names decompose into frequent atoms (recutils 0.284 ≈ retrieval),
dead where names are idiosyncratic single tokens (dash 0.006).

### 2.5 Composition head reliability (canonical atom splitter)

| scope | n | set precision | set recall |
|---|---|---|---|
| FT novel+OOV | 2,948 | **0.115** | 0.069 |
| ALL novel+OOV | 3,430 | 0.176 | 0.102 |

→ a low-precision hint channel: ~1 in 9 asserted components correct on the hard stratum. Framed in
the paper as prioritization evidence, not naming.

### 2.6 Complementarity and oracle ceilings (N2/N3)

- Composition contributes ≥1 correct sub-token that retrieval misses entirely on **10.8%** of
  novel+OOV functions (mean unique_C 0.115); retrieval's mirror figure on seen is 73.3%.
- Oracle(R+C) − R: **+0.0535** on novel+OOV, **+0.0312** overall. Oracle(R+C+GRU): 0.6366.
- Token-level fusion FAILED its pre-registered bar (+0.004 vs +0.010 required) → dual output
  presented, no fusion, no routing (three routing variants all negative; per-binary gate −0.037).

---

## 3. Baselines (matched-key protocols; ours = pure retrieval everywhere)

### 3.1 SymGen+LoRA (CodeLlama-34B fine-tuned on our corpus) — corrected leak-free protocol

**FT, stratified (ft3 protocol, matched n=3,830):**

| stratum | n | ours F1 / EM | SymGen F1 / EM |
|---|---|---|---|
| seen | 1,955 | **0.377 / 34.9%** | 0.156 / 8.5% |
| novel_comp | 698 | 0.020 / 0.0% | **0.156 / 3.3%** |
| oov | 1,177 | 0.048 / 0.0% | **0.273 / 3.7%** |
| **ALL FT** | **3,830** | **0.211 / 17.8%** | 0.192 / 6.1% |

**NCT (corrected run, job 1169309, scored `symgen_nct_matched.json`, matched n=5,778):**

| stratum | n | ours F1 / EM | SymGen F1 / EM |
|---|---|---|---|
| seen | 5,338 | **0.880 / 74.8%** | 0.324 / 6.6% |
| novel_comp | 235 | 0.267 / 0.0% | **0.360 / 6.4%** |
| oov | 103 | 0.329 / 0.0% | **0.428 / 4.9%** |
| **ALL NCT** | **5,778** | **0.845 / 70.2%** | 0.324 / 6.5% |

- SymGen inference: 39,421 s for 10,256 functions on A100-80GB (3.84 s/fn; internal timing datum,
  not for publication without three-boundary protocol).
- Leak validation: discarded v2 arm scored OOV EM 31.7% > seen (name readable in prompt);
  corrected run: flat ~5–6% EM everywhere. 🚫 v2 numbers (0.7352 OOV, 0.6437 pairwise aggregate,
  recutils 0.626) remain forbidden.
- **The finding that must be reported:** SymGen wins novel_comp (8× FT) and OOV (5.7× FT) in both
  regimes, with non-zero novel EM where we score exactly 0. Report both readings (representation
  ceiling vs pretraining contamination); do not use contamination to dismiss.

**FT per-package (3-way matched keys, n=3,258):**

| package | n | ours | SymGen | BLens |
|---|---|---|---|---|
| recutils | 1,027 | **0.336** | 0.308 | 0.117† |
| dash | 1,004 | **0.241** | 0.027 | 0.029 |
| gettext | 1,045 | 0.025 | **0.151** | 0.037 |
| psmisc | 182 | 0.187 | **0.378** | 0.168† |
| **ALL** | **3,258** | **0.199 / 16.1% EM** | 0.175 / 5.6% | 0.070 / 3.0% |

† BLens n differs slightly in its own pairwise table; three-way values shown.
Per-package honesty: SymGen wins gettext and psmisc outright; our FT aggregate win is carried by
recutils and dash.

### 3.2 BLens (retrained COMBO c+p + LoRD on our corpus) — pairwise matched n=9,710

| package | n | ours F1 / EM | BLens F1 / EM |
|---|---|---|---|
| nginx118 | 2,654 | **0.882 / 74.3%** | 0.615 / 19.1% |
| angie | 2,483 | **0.828 / 66.3%** | 0.592 / 18.6% |
| tengine | 554 | **0.756 / 69.7%** | 0.156 / 0.4% |
| recutils | 1,625 | **0.335 / 24.3%** | 0.117 / 4.4% |
| dash | 1,014 | **0.239 / 22.8%** | 0.029 / 1.3% |
| gettext | 1,183 | **0.042 / 1.3%** | 0.034 / 1.2% |
| psmisc | 197 | **0.190 / 13.2%** | 0.155 / 9.1% |

Regime view (notes-anchored, matched n=9,475): NCT ours 0.847/70.9% vs BLens 0.558/17.3%;
FT ours 0.202/17.1% vs 0.083/3.4%. BLens wins no package on matched keys. BLens is scored on raw
nm labels (same basis as everyone); its self-reported numbers use its own normalized name space.

### 3.3 Parameters (measured on artifacts)

| system | parameters |
|---|---|
| **Ours, full** | **32,208,484** (ckpt 386.8 MB) |
| — encoder+fusion (retrieval-sufficient) | 15,669,000 |
| — GRU decoder | 16,539,484 |
| — composition head | ~4.1M |
| BLens COMBO (measured) | 153,547,153 (LoRD stage additional, uncounted) |
| SymGen | CodeLlama-34B + LoRA r=8 (38 MB adapter) |

Claim of record: "~1,000× fewer parameters than the largest baseline" (34B/32.2M ≈ 1,056×).
vs BLens only ~4.8×. 🚫 No wall-clock claims (no three-boundary measurement exists).

---

## 4. Clang cross-compiler experiments (jobs 1169445 / 1169446 / 1169447, 2026-08-10)

**Eval set:** all 7 clean-7 packages recompiled Clang 18.1.8 at O0–O3, identical BAP pipeline,
23,336 functions. ⚠️ Clang gettext statically links libtextstyle/libxml2 → 12,415 fns incl.
library code (GCC: 1,518). **Never quote the n-weighted aggregate cross-compiler; per-package
mean is the metric of record.**

**Models:**
- *Zero-shot* = `best_model_cont_control.pt` unchanged, GCC-only index (241,174). Reproduced the
  0.8053 GCC test anchor exactly.
- *+Mixed training* = `best_model_gcc_clang_cont.pt` (job 1169446): identical recipe incl.
  continued-pretrain encoder, + Clang builds of 21 TRAINING packages (97,619 fns; preflight: 1,407
  train bins, 586 Clang, 0 leaks into eval buckets). Mixed index 337,038 fns. Val F1 0.5033.

### 4.1 Retrieval-head F1 per package

| package | GCC ref | Clang zero-shot | Clang +mixed training |
|---|---|---|---|
| nginx118 | 0.889 | 0.495 | **0.528** |
| angie | 0.839 | 0.464 | **0.489** |
| tengine | 0.756 | 0.465 | **0.488** |
| recutils | 0.332 | 0.159 | **0.232** |
| gettext | 0.038 | 0.096 | **0.136** |
| dash | 0.238 | 0.128 | **0.143** |
| psmisc | 0.157 | 0.178 | **0.188** |
| **pkg-mean** | **0.464** | **0.284** | **0.315** |

EM (zero-shot / matched): nginx118 34.3/40.2, angie 31.0/35.8, tengine 30.7/35.1, recutils
11.2/18.7, gettext 6.7/10.3, dash 12.1/13.4, psmisc 12.8/16.3 (%).

### 4.2 Findings

1. **Zero-shot cross-compiler costs 0.18 pkg-mean F1** (0.464 → 0.284), concentrated on NCT
   (index anchoring: Clang forks no longer near-match GCC index entries). gettext/psmisc score
   *higher* under Clang (population differs).
2. **Compiler-diverse training improves all 7 packages** (mean → 0.315, recutils +0.073) at **no
   GCC cost** — GCC test 0.8053 → 0.8073, clean split verified (no eval package in training under
   either compiler).
3. **BinFilter is inert on Clang** (±0.000 vs +0.04 on GCC) — fingerprint matching is
   compiler-sensitive.
4. **Retrieval's margin over the decoder vanishes cross-compiler** (0.231 vs 0.225 n-wt) — the
   retrieval advantage is index proximity, not the head.
5. Recipe confound measured: old mixed model (plain encoder, Aug 4) pkg-mean 0.292 vs matched
   0.315 → continued pretraining worth +0.023 on Clang; direction as predicted. Paper uses the
   MATCHED column only.
6. Honest scope: 0.315 ≪ 0.464 → "partial cross-compiler transfer with substantial degradation."
   Clang training corpus is ~1/3 the GCC corpus; extension is future work.
7. 🚫 The gate-era Clang table (dash 0.93, psmisc O2=1.00 — memorized leaked functions) remains
   forbidden.

---

## 5. Anchors index

| result | job(s) | artifact |
|---|---|---|
| GCC clean-7 headline + per-package | 1168004 | `results/tau05_t0.5.json` |
| Stratified 3-head + per-function records | 1169047 | `results/ndss_dual_head_eval.json` |
| Rank audit (frequency skew) | — | `results/COMPOSITION_RANK_AUDIT.md` |
| SymGen FT (ft3) + 3-way + BLens pairwise | 1161536/7 + | `results/ndss_prep_matched_subset_clean7.json` |
| SymGen NCT corrected | 1169309 | `results/symgen_nct_matched.json` |
| Clang zero-shot + old-mixed | 1169445 | `results/clang_clean_{control,gccclang}.json` |
| Clang matched retrain + eval | 1169446/7 | `results/clang_clean_gccclang_cont.json` |
| BLens COMBO param count | — | measured from checkpoint (611 MB) |
