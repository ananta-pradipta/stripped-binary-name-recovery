# HyDRA: Hybrid Retrieval and Generation with Adaptive Routing for Function Name Recovery in Stripped Binaries

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![Ghidra 11](https://img.shields.io/badge/Ghidra-11.3-green.svg)](https://ghidra-sre.org/)
[![CodeT5+ 220M](https://img.shields.io/badge/backbone-CodeT5%2B%20220M-orange.svg)](https://huggingface.co/Salesforce/codet5p-220m)

> NJIT: Ananta Dian Pradipta, Robert Blacha, Zhihao Lin, Haotian Zhang

---

## Abstract

HyDRA recovers function names from stripped x86-64 binaries with **one fine-tuned 220M-parameter code language model serving two heads**. Function name recovery faces two regimes at once: *reused code* (library routines, versions, forks) whose name already exists somewhere, and *genuinely new code* whose name must be composed from evidence. The model's decoder is the **generation head** (composes names), its mean-pooled encoder states drive the **retrieval head** (copies the name of the most similar training function), and a **learned per-function router** picks the head from four confidence features. Every prediction carries a **calibrated abstention score** so an analyst can apply only the names above a chosen confidence. Each function is presented as its masked Ghidra decompilation plus a **binary-neighbourhood context**: string literals and library-call identifiers of the functions adjacent to it in the address space.

Because published evaluations of this task leak (on a 300K-function corpus split at the binary level, 89.5% of test functions are token-identical to a training function and 97% of test names occur in training), the repository also releases **LineageBench**, a lineage-aware evaluation protocol that uses package families to build two explicit transfer settings (family-disjoint far transfer, and reuse-heavy transfer in which related software is intentionally kept in training), deduplicates test bodies against training, excludes linker-visible names, and reports by transfer regime and name category, and applies it to the public **Punstrip** benchmark.

**Key results (220M parameters):**
- **LineageBench test tier (268,178 functions, 50 held-out packages):** 0.472 package-level / 0.237 function-level sub-token F1, vs. SymGen-34B 0.196 / 0.145 and BLens 0.171 / 0.059 trained on the same data.
  - **Far transfer (FT, 27 packages):** 0.169 package-level / 0.145 function-level F1 (SymGen-34B 0.144 / 0.118, BLens 0.021 / 0.013)
  - **Reuse-heavy transfer (RHT, 23 packages):** 0.828 package-level / 0.693 function-level F1 (SymGen-34B 0.257 / 0.276, BLens 0.346 / 0.287)
- **Punstrip (public cross-project split, BLens's own evaluator):** 0.549 full / 0.467 strict, vs. published BLens 0.461 / 0.293 and SymGen-34B trained on Punstrip 0.435 / 0.395.
- **Selective prediction:** 0.95 F1 on the 5% of functions ranked most confident, 0.90 at 10%, 0.72 at 20% (ECE 0.036).
- **Efficiency:** 155x fewer parameters and ~95x faster per-function inference than the 34B baseline; 6.5 h fine-tuning on one A100-40GB.

---

## Architecture

```
Stripped binary
    |
    v
Input construction (Ghidra headless)
  • decompile function f, replace every FUN_xxxx placeholder with [MASK]
  • binary-neighbourhood context: up to 40 identifier tokens (strings + library calls) mined from the
    ±10 address-adjacent functions, ranked by how many neighbours contain them, prepended as a C comment
    x_f = Digest(neighbours) || Mask(Decompile(f))
    |
    v
One backbone: CodeT5+ 220M encoder-decoder, fine-tuned once on (x_f -> name)
  • Generation head (HyDRA-G):  decoder, greedy decoding
                                -> n_gen, confidence c_gen = length-normalised log-probability
  • Retrieval head  (HyDRA-R):  e_f = mean-pooled encoder states
                                cosine k-NN over 190K embedded training functions (FAISS)
                                -> n_ret, top-1 similarity s1, runner-up s2
    |
    v
Router rho(z), z = < s1, s1 - s2, c_gen, s1 - c_gen >
  • MLP (64, 32 hidden units), fit on the validation tier only,
    weighted binary cross-entropy with weight |F1_R - F1_G| per function
  • n_hat = n_ret if rho(z) >= 0.5 else n_gen   (12.5% of test functions go to retrieval)
    |
    v
Abstention alpha(z, rho): regressor -> expected F1 q_hat of the routed name
  • the analyst applies n_hat only if q_hat >= tau; otherwise the function stays FUN_xxxx
```

---

## Key Findings

| Finding | Evidence |
|---|---|
| Lookup and composition are different mechanisms | Retrieval is exact on seen names (81.8% EM) and structurally silent on novel ones (0.1%); the 34B generative model reaches 6.9% EM on the same seen names. Unified designs (retrieval-augmented decoding, retrieve-and-edit, logit fusion, re-ranking) all lost to routing. |
| One fine-tuned model serves both heads | The generation model's own encoder out-retrieves a separately trained contrastive graph encoder (0.138 vs. 0.127 F1; seen-category EM 0.816 vs. 0.708). |
| The name's evidence is often outside the function | 66% of novel-name functions carry none of their name's sub-tokens in their own body; 65% have one in the ±10 address neighbours, 42% the naming prefix. The digest lifts the head by +0.025 F1 and first puts the 220M head above the 34B model on novel names. |
| Routing needs weighting, not a particular learner | An unweighted router collapses to 92% retrieval; weighted by \|F1_R - F1_G\|, MLP, gradient-boosted trees, random forest, SVM and a depth-3 tree all land within 0.002. |
| Pretrained code prior is most of the payload | CodeT5+ from random initialisation reaches 0.069 F1 with collapsed outputs; the public checkpoint reaches 0.213. |
| Published evaluations measure recognition | A 32M graph-encoder system scores 0.738 F1 on a binary-level split and 0.086 on the family-disjoint, deduplicated re-split of the same data. |
| Baselines' F1 is precision bought with abstention | On Punstrip under BLens's evaluator: BLens P 0.656 / R 0.355 (abstains on 46%), HyDRA P 0.557 / R 0.542 (names every function). |

---

## Repository Structure

```
stripped-binary-name-recovery/
├── scripts/
│   ├── design_split_v2.py           # LineageBench: package families, tiers, regimes -> data/split_v2.json
│   ├── export_baseline_protocol.py  # Protocol tier records (binary, address, name, regime, category, flags)
│   ├── build_corpus_manifest.py     # Corpus manifest (binaries, packages, compilers, optimisation levels)
│   ├── a4_build_modctx_dm.py        # HyDRA inputs: masked Ghidra text + binary-neighbourhood context (canonical targets)
│   ├── a4_build_modctx.py           # Digest variants used in the ablations
│   ├── a4_build_poolctx.py          #   (wider three-tier digest)
│   ├── a4_build_baptext.py          #   (linearised BAP-IR instead of decompiled text)
│   ├── a4_train_codet5p.py          # Fine-tune CodeT5+ 220M (generation head)
│   ├── a4_predict.py                # Greedy decoding + confidence c_gen for the protocol tiers
│   ├── c_lmemb_knn.py               # Retrieval head: encoder embeddings, cosine k-NN, s1 and margin
│   ├── router3_eval.py              # Router (GBT / MLP), abstention regressor, per-package tables
│   ├── ablation_router.py           # Router feature ablations, selective prediction F1@c%
│   ├── score_pr.py                  # Precision / recall / F1 / EM for every system on identical keys
│   ├── punstrip/                    # Punstrip rebuild, SymGen/BLens glue, BLens-evaluator re-implementation, strata
│   ├── dh2_sbatch/                  # Slurm launchers (cluster paths are placeholders)
│   └── 0[1-4]_*.sh                  # Legacy corpus compilation and BAP lifting pipeline
│
├── src/                             # Legacy BAP-IR pipeline (builds the benchmark; decompiler-free ablation)
│   ├── preprocessing/parse_bap.py   # BAP-IR -> instruction-type token sequence (= body hash for dedup)
│   ├── models/                      # Graph-attention encoder + GRU decoder (CCS-era system, ablation only)
│   └── evaluation/metrics.py        # split_name, compute_subtoken_f1, compute_subtoken_precision_recall
│
├── baselines/
│   ├── symgen/                      # CodeLlama-34B + LoRA with the authors' pipeline; predictions and logs
│   ├── training_logs/
│
├── results/
│   ├── dualhead_v2/                 # LineageBench: per_pkg_single_backbone.json, score_symgen_full*.json,
│   │                                #   score_pr.json, case_study_*.json, FINAL_TABLES.md, PAPER_NOTES.md
│   ├── punstrip/                    # Punstrip: score_report_extra_final.json (both scorers), punstrip_strata_v2.json
│   └── experiment_log.md            # Every number in the paper with the job that produced it
│
├── configs/                         # dualhead_v2_*.yaml (current), optimized*.yaml / ablation_model*.yaml (legacy)
├── docs/                            # DATASET_V2_CARD.md, DUALHEAD_HYDRA_PLAN.md, protocol design notes, contributor guides
├── data/split_v2.json               # LineageBench split manifest
├── data/                            # Corpus, decompilations, embeddings: not tracked (see Dataset)
└── checkpoints/                     # Model weights: not tracked (see Dataset)
```

---

## Quick Start

```bash
git clone https://github.com/ananta-pradipta/stripped-binary-name-recovery.git
cd stripped-binary-name-recovery

# Environment: Python 3.10+, PyTorch 2.5, transformers, faiss, scikit-learn; Ghidra 11.3 headless for decompilation
bash scripts/01_setup_environment.sh
source activate.sh
pip install -r requirements.txt
```

The scripts refer to the workspace that holds the corpus, Ghidra decompilations and embeddings through
`$WORKSPACE` (cluster paths in the Slurm launchers); set it to your copy.

### Full pipeline (headline model)

```bash
# 1. LineageBench: families, tiers, deduplication, linker-visible filter, regimes and name categories
python3 scripts/design_split_v2.py
python3 scripts/export_baseline_protocol.py

# 2. Inputs: masked Ghidra decompilation + binary-neighbourhood context for train / val / test
python3 scripts/a4_build_modctx_dm.py --src $WORKSPACE/results/baseline_protocol_v2 --out $WORKSPACE/data/modctx_dm

# 3. Generation head: fine-tune CodeT5+ 220M (3 epochs, lr 5e-5, batch 32, bf16, ~6.5 h on one A100-40GB)
python3 scripts/a4_train_codet5p.py --model Salesforce/codet5p-220m --data-dir $WORKSPACE/data/modctx_dm --tag modctx_dm --bf16
python3 scripts/a4_predict.py $WORKSPACE/checkpoints/a4_modctx_dm/best --tiers val,test --tag modctx_dm --bf16

# 4. Retrieval head: embed train / val / test with the fine-tuned encoder; cosine k-NN; s1 and margin
python3 scripts/c_lmemb_knn.py

# 5. Router + abstention (fit on the validation tier only) and the per-package / regime / category tables
python3 scripts/router3_eval.py --a4 $WORKSPACE/results/a4_modctx_dm/val_test_preds.tsv --out results/dualhead_v2/router.json
python3 scripts/score_pr.py $WORKSPACE/blens/LORD-inference-logs-test.txt     # P / R / F1 / EM, all systems, identical keys
```

### Baselines

```bash
# SymGen-34B: authors' pipeline, one LoRA epoch on the LineageBench training tier (4 x A100-40GB)
see baselines/symgen/ and scripts/a4_build_symgen_ft.py

# BLens: authors' code, CLAP + PalmTree embeddings extracted by us, 80 + 80 epochs, selection on our validation tier
see scripts/blens_v2/ and results/dualhead_v2/PAPER_NOTES.md

# Punstrip: rebuild the 10,047 Debian binaries, train a second HyDRA on Punstrip-train only, score with BLens's evaluator
see results/punstrip/PLAN.md and scripts/punstrip/
```

### HPC (Slurm)

```bash
sbatch scripts/dh2_sbatch/a4_train.sbatch        # fine-tune the generation head
sbatch scripts/dh2_sbatch/a4_predict.sbatch      # decode the protocol tiers
sbatch scripts/punstrip/score_extra_final.sbatch # Punstrip: both scorers, every system
```

---

## Evaluation

Predictions and references are demangled, split on underscores, camel-case and digits, lower-cased and compared
as sub-token multisets: per function TP = |Y_hat ∩ Y|, P = TP / |Y_hat|, R = TP / |Y|, F1 = 2PR / (P + R); exact match (EM)
is case-sensitive canonical string equality. Function-level scores are means over functions; package-level scores are
means of the per-package means (two packages hold 44% of the test functions, so package-level F1 is the headline).

### Main results (LineageBench test tier, 268K functions, 50 packages; every system trained on the same tier)

| System | Params | P (fn) | R (fn) | F1 (fn) | F1 (pkg) | EM | FT F1 (fn / pkg) | RHT F1 (fn / pkg) |
|---|---|---|---|---|---|---|---|---|
| SymGen-34B (CodeLlama + LoRA, authors' pipeline) | 34B | 0.154 | 0.144 | 0.145 | 0.196 | 2.9% | 0.118 / 0.144 | 0.276 / 0.257 |
| BLens (authors' code, CLAP + PalmTree) | ~200M | 0.077 | 0.054 | 0.059 | 0.171 | 1.2% | 0.013 / 0.021 | 0.287 / 0.346 |
| HyDRA-G (generation head only) | 220M | 0.226 | 0.209 | 0.213 | 0.400 | 5.9% | 0.144 / 0.167 | 0.553 / 0.674 |
| HyDRA-R (retrieval head only) | 220M | 0.141 | 0.138 | 0.138 | 0.419 | 10.2% | 0.037 / 0.075 | 0.643 / 0.824 |
| **HyDRA (routed)** | 220M | **0.249** | **0.234** | **0.237** | **0.472** | 10.7% | **0.145** / **0.169** | **0.693** / **0.828** |
| HyDRA with oracle head choice (upper bound) | 220M | – | – | 0.255 | 0.505 | – | 0.160 / 0.189 | 0.730 / 0.876 |

### Context-source study (2026-09-25)

Controlled comparison of what feeds the generation head's 40-token context digest (same backbone, recipe and budget;
only the set of source functions changes), LineageBench test F1 (fn / pkg): none 0.184 / 0.360; 20 random functions
of the same binary 0.193 / 0.389; ±10 address neighbours (adopted) 0.213 / 0.400; callers + callees 0.214 / 0.403;
address ∪ callers/callees 0.217 / 0.405. Address vs callers/callees is a tie (package-level −0.003, 95% CI [−0.011, +0.005]);
the two heads agree on 12% of names, an oracle choice reaches 0.261, and picking the more confident head gives 0.222.
64% of the ±10 window lies in the target's source file (8% random; 59% on Debian builds). Full tables, locality, layout
deltas and raw outputs: `results/dualhead_v2/context_study/`; builders in `scripts/ctx_checks/`.

### By name category (F1 / EM)

| System | Seen (33.7K, 12.6%) | Novel-known (81.6K, 30.4%) | Novel-OOV (152.8K, 57.0%) |
|---|---|---|---|
| SymGen-34B | 0.262 / 6.9% | 0.168 / 2.3% | 0.106 / 2.3% |
| BLens | 0.382 / 10.3% | 0.019 / 0.0% | 0.011 / 0.1% |
| HyDRA-R | **0.866 / 81.4%** | 0.046 / 0.0% | 0.027 / 0.0% |
| HyDRA-G | 0.664 / 39.0% | **0.201** / 1.8% | **0.119** / 0.8% |
| **HyDRA (routed)** | 0.856 / 77.2% | 0.200 / 1.7% | **0.119** / 0.8% |

Seen: the name occurs in training. Novel-known: unseen name whose sub-tokens all occur in training names.
Novel-OOV: at least one sub-token never seen. Retrieval owns the first category, generation the other two; the
router keeps both.

### Ablations (test tier F1, fn / pkg)

| Variant | F1 |
|---|---|
| CodeT5+ 220M from random initialisation (same data, digest, schedule) | 0.069 (outputs collapse) |
| Public checkpoint, masked decompiled text only | 0.188 |
| + binary-neighbourhood context (±10 neighbours) = HyDRA-G | 0.213 / 0.400 |
| Linearised BAP-IR instead of decompiled text (decompiler-free) | 0.184 (head), 0.220 (system) |
| Wider three-tier digest | 0.217 on test, but lower on validation (not adopted) |
| Second pass feeding predictions back into digests | 0.201 (worse in every category) |
| Router: unweighted MLP | 0.164 (routes 92% to retrieval) |
| Router: weighted MLP / GBT / random forest / SVM / depth-3 tree | 0.236–0.237 |

### Selective prediction (abstention score, routed system)

| Coverage | 5% | 10% | 20% | 30% | 50% | 100% |
|---|---|---|---|---|---|---|
| F1 on covered functions | 0.948 | 0.904 | 0.722 | 0.561 | 0.389 | 0.237 |

Expected calibration error 0.036; correlation between predicted and actual F1 0.74.

### Punstrip (public cross-project split, 22,926 functions, identical keys for every system)

| System | P | R | F1 (full) | F1 (strict) | our F1 fn / pkg | seen | novel |
|---|---|---|---|---|---|---|---|
| **HyDRA (routed, trained on Punstrip-train only)** | 0.557 | **0.542** | **0.549** | **0.467** | **0.447 / 0.626** | 0.761 | **0.284** |
| HyDRA-G | 0.528 | 0.514 | 0.521 | 0.447 | 0.414 / 0.607 | 0.662 | 0.285 |
| HyDRA-R | 0.366 | 0.360 | 0.363 | 0.204 | 0.311 / 0.530 | 0.758 | 0.078 |
| SymGen-34B, LoRA on Punstrip-train | 0.439 | 0.430 | 0.435 | 0.395 | 0.316 / 0.434 | 0.416 | 0.264 |
| BLens, published model | **0.656** | 0.355 | 0.461 | 0.293 | 0.394 / 0.622 | **0.775** | 0.196 |
| XFL, published | 0.610 | 0.195 | 0.296 | 0.085 | 0.223 / 0.516 | 0.517 | 0.071 |
| SymLM, published | 0.542 | 0.175 | 0.265 | 0.133 | 0.112 / 0.083 | 0.234 | 0.049 |
| AsmDepictor, published | 0.243 | 0.167 | 0.198 | 0.076 | 0.153 / 0.348 | 0.374 | 0.039 |

P, R, F1 (full) and F1 (strict) under BLens's evaluator (label-set metric over its 1,024 canonical labels; strict removes
duplicate bodies, forbidden labels and crt stubs, 16,405 functions). Re-scoring the authors' csv reproduces their paper
to ±0.001 (BLens 0.461 / 0.293, XFL 0.296 / 0.085).

### Case study (whole binaries, routed system as configured above)

| Binary | Regime | Functions | F1 / EM | F1@10% / EM@10% | F1@50% / EM@50% |
|---|---|---|---|---|---|
| cvs (O0) | far transfer | 1,216 | 0.385 / 26.4% | 0.992 / 94.3% | 0.657 / 51.0% |
| lighttpd (O2) | far transfer | 385 | 0.169 / 0.0% | 0.259 / 0.0% | 0.218 / 0.0% |
| nginx 1.18 (O2) | reuse-heavy transfer | 397 | 0.950 / 85.9% | 1.000 / 100% | 1.000 / 100% |

Debian's shipped `tcpreplay` 4.2.6-1 (Punstrip test, 1,398 functions): the two CVE sites present as functions
(`dlt_en10mb_encode`, CVE-2018-17974; `get_l2len`, CVE-2018-20553) are named exactly with confidence >= 0.999 in every
executable that contains them; suite-wide F1 0.752 / EM 59.2%.

### Cost

| System | Params | Training | Inference / function |
|---|---|---|---|
| **HyDRA** | 220M | 6.5 h, one A100-40GB | ~40 ms |
| BLens (retrained) | ~200M | 53 h + one day of embedding extraction | ~20 ms |
| SymGen-34B + LoRA | 34B | ~200 GPU-h | ~3.8 s |

---

## Dataset

- **Corpus:** 77 open-source packages, GCC and Clang, O0–O3; 1,890 binaries, 874,803 labelled functions
  (138 duplicate builds and 37 binaries with fewer than 20 functions excluded). Functions rooted from `.eh_frame`
  (median coverage on optimised binaries 99%).
- **Per-function record:** BAP instruction-type token sequence (body hash), Ghidra decompilation with the placeholder
  masked, string literals and named library calls, debug name, linker-visible flag.
- **LineageBench tiers:** 16 package families (>= 35% overlap of names present in < 3 packages); train 59 packages /
  997 binaries, validation 10 / 104 (package-disjoint from both other tiers; coreutils3, csplit2 and cppi are
  reuse-heavy relative to training by design so the router sees both regimes), test 50 / 611. Train deduplicated to one pair per (body hash, name):
  434,651 -> 190,151. Validation and test drop bodies that occur in training (-81,988) and linker-visible names (-12,746):
  test 268,178 functions (27 FT packages, 223K functions; 23 RHT, 45K), validation 10,617.
- **Punstrip:** the public Debian corpus of XFL / BLens; 10,047 binaries rebuilt from snapshot.debian.org by exact
  symbol-table match; 395K train / 18K val / 23.9K test functions; a second HyDRA trained only on its training split.
- **Not tracked here (size):** decompilations, embeddings, the 190K-function retrieval index, checkpoints. They are
  distributed with the replication package referenced in the paper.

---

## Scoring a new system on LineageBench

Join your predictions on `(binary, address)` with the protocol tier records, canonicalise (demangle, split, lower-case)
and score with `src/evaluation/metrics.py` (`compute_subtoken_f1`, `compute_subtoken_precision_recall`); report
function-level and package-level means, per regime (FT / RHT) and per name category. Prediction dumps of every system
in the paper are under `results/`, so a new comparison needs no retraining of the baselines.

---

## References

- He, J., Ivanov, P., Tsankov, P., Raychev, V., & Vechev, M. (2018). **Debin: Predicting Debug Information in Stripped Binaries.** CCS'18.
- David, Y., Alon, U., & Yahav, E. (2020). **Neural Reverse Engineering of Stripped Binaries using Augmented Control Flow Graphs** (NERO). OOPSLA'20.
- Jin, X., Pei, K., Won, J. Y., & Lin, Z. (2022). **SymLM: Predicting Function Names in Stripped Binaries via Context-Sensitive Execution-Aware Code Embeddings.** CCS'22.
- Patrick-Evans, J., Dannehl, M., & Kinder, J. (2023). **XFL: Naming Functions in Binaries with Extreme Multi-Label Learning.** IEEE S&P'23.
- Benoit, T., Wang, Y., Dannehl, M., & Kinder, J. (2025). **BLens: Contrastive Captioning of Binary Functions using Ensemble Embedding.** USENIX Security'25.
- Jiang, L., Jin, X., & Lin, Z. (2025). **Beyond Classification: Inferring Function Names in Stripped Binaries via Domain Adapted LLMs** (SymGen). NDSS'25.
- Ye, G., et al. (2024). **Epitome: Improving Function Name Recovery in Stripped Binaries** (project-split evaluation). FSE'24.
- Wang, Y., Le, H., Gotmare, A., Bui, N., Li, J., & Hoi, S. (2023). **CodeT5+: Open Code Large Language Models for Code Understanding and Generation.** EMNLP'23.
- Allamanis, M. (2019). **The Adverse Effects of Code Duplication in Machine Learning Models of Code.** Onward!'19.
- Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., & Lewis, M. (2020). **Generalization through Memorization: Nearest Neighbor Language Models.** ICLR'20.

---

## License

MIT License (see `LICENSE`).
