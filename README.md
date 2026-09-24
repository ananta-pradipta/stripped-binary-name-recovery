# HyDRA: Hybrid Decoder-Retrieval with Adaptive Routing for Dual-Regime Function Name Recovery in Stripped Binaries

> NJIT: Ananta Dian Pradipta, Robert Blacha, Zhihao Lin, Haotian Zhang

This repository holds the code, benchmark protocol, prediction dumps and result tables behind the FSE submission
*HyDRA: Hybrid Decoder-Retrieval with Adaptive Routing for Dual-Regime Function Name Recovery in Stripped Binaries*.
The earlier CCS-era system (graph-attention encoder over BAP-IR with cascaded gated fusion) is retained under `src/`
because it builds the benchmark and serves as the decompiler-free ablation; it is no longer the system under test.

---

## What HyDRA does

Function name recovery faces two regimes at once: **near-clone code** (library routines, versions, forks) whose name
already exists somewhere, and **genuinely new code** whose name must be composed from evidence. HyDRA keeps one head
per regime and pays for one model:

```
stripped binary --Ghidra--> masked decompiled function  +  module-context digest
                             (FUN_xxxx -> [MASK])          (strings + library calls of the +-10 address neighbours)
        |
        v
CodeT5+ 220M encoder-decoder, fine-tuned once on (input -> name)
        |-- decoder            -> generated name  n_gen, confidence c_gen        (generation head, HyDRA-G)
        |-- mean-pooled encoder -> cosine k-NN over 190K embedded training fns
                               -> retrieved name n_ret, similarity s1, margin s1-s2   (retrieval head, HyDRA-R)
        |
        v
router rho(z), z = <s1, s1-s2, c_gen, s1-c_gen>, MLP fit on the validation tier (weighted by |F1_R - F1_G|)
abstention alpha(z, rho) -> expected F1 q_hat; the analyst applies the name only if q_hat >= tau
```

## LineageBench: the evaluation protocol

Published function-naming evaluations leak: on a 300K-function corpus split at the binary level, 89.5% of test
functions are token-identical to a training function and 97% of test names occur in training. LineageBench is a
protocol plus a controlled corpus:

1. **Package families** (>=35% overlap of rare names) so that versions and forks never straddle the split.
2. **Whole-family tiers**: train 59 packages / 997 binaries, validation 10 / 104, test 50 / 611 (GCC and Clang, O0-O3).
3. **Deduplication and filtering**: one training pair per (body hash, name); test/validation functions whose body
   occurs in training are dropped, as are linker-visible names (exported for dynamic linking, still readable after strip).
4. **Reporting** per transfer regime (far transfer FT vs near-clone transfer NCT) and per name category
   (seen / novel-known / novel-OOV).

Test tier: 268,178 functions, 50 packages (27 FT, 23 NCT). Every baseline is trained, selected and scored under the
same tiers and canonicalisation. The same protocol is applied to the public Punstrip benchmark (XFL, BLens).

## Headline results

Package-level sub-token F1 is the headline (two packages hold 44% of the test functions); function-level F1 is in
every table as well. Precision, recall, F1 and exact match are defined in the paper (Eq. 5).

| System (same training data) | Params | F1 fn-level | F1 pkg-level | FT | NCT |
|---|---|---|---|---|---|
| SymGen-34B (CodeLlama + LoRA, authors' pipeline) | 34B | 0.145 | 0.196 | 0.118 | 0.276 |
| BLens (authors' code, CLAP + PalmTree) | ~200M | 0.059 | 0.171 | 0.013 | 0.287 |
| HyDRA-G (generation head only) | 220M | 0.213 | 0.400 | 0.144 | 0.553 |
| HyDRA-R (retrieval head only) | 220M | 0.138 | 0.419 | 0.037 | 0.643 |
| **HyDRA (routed)** | 220M | **0.237** | **0.472** | **0.145** | **0.693** |

By name category (F1 / exact match): retrieval owns seen names (0.870 / 81.8%), generation owns novel names
(0.201 novel-known, 0.119 novel-OOV); the routed system keeps both (0.859 / 77.5%, 0.201, 0.120).

Punstrip (public cross-project split, 22,926 functions, BLens's own evaluator, full / strict): HyDRA 0.549 / 0.467,
SymGen-34B trained on Punstrip-train 0.435 / 0.395, BLens published 0.461 / 0.293. Precision / recall under that
evaluator: HyDRA 0.557 / 0.542, BLens 0.656 / 0.355 (abstains on 46%).

Selective prediction: 0.95 F1 on the 5% of functions the abstention score ranks highest, 0.90 at 10%, 0.72 at 20%
(ECE 0.036). Cost: 6.5 h fine-tuning on one A100-40GB, ~40 ms per function at inference; 155x fewer parameters and
95x faster per function than the 34B baseline.

## Repository map

```
scripts/
  a4_build_modctx_dm.py      build HyDRA inputs: masked Ghidra text + module-context digest (demangled targets)
  a4_build_modctx.py, a4_build_poolctx.py, a4_build_baptext.py   digest variants / BAP-text ablation inputs
  a4_train_codet5p.py        fine-tune CodeT5+ 220M (generation head)
  a4_predict.py              greedy decoding + confidence c_gen for the protocol tiers
  c_lmemb_knn.py             retrieval head: mean-pooled encoder embeddings, FAISS/torch cosine k-NN, s1 and margin
  router3_eval.py, ablation_router.py   routers (GBT / MLP), feature ablations, selective prediction
  design_split_v2.py, export_baseline_protocol.py, build_corpus_manifest.py   LineageBench construction
  punstrip/                  Punstrip rebuild, SymGen/BLens retrain glue, BLens-evaluator re-implementation, strata
  dh2_sbatch/                Slurm launchers used on the cluster (paths are cluster-specific)
src/                         legacy BAP-IR pipeline: parse_bap.py (instruction-type tokens = body hash), GNN models,
                             metrics.py (sub-token P/R/F1, exact match, split_name)
baselines/                   SymGen (LoRA) and BLens retrain recipes and outputs
results/dualhead_v2/         LineageBench results: per_pkg_single_backbone.json (per-package / regime / category),
                             score_symgen_full*.json (baselines on identical keys), case_study_*.json, FINAL_TABLES.md
results/punstrip/            Punstrip reports: score_report_extra_final.json (both scorers), punstrip_strata_v2.json
results/experiment_log.md    every number in the paper, with the job that produced it
docs/                        dataset card, protocol design notes, contributor guides
data/split_v2.json           LineageBench split manifest (families, tiers, regimes)
```

## Reproducing the main table

Cluster paths in the scripts (`/project/.../dh2`) point at the workspace where the corpus, Ghidra decompilations and
embeddings live; set them to your own copy. The order is:

```bash
# 1. LineageBench: families, tiers, dedup, linker-visible filter -> data/split_v2.json, protocol tier files
python3 scripts/design_split_v2.py && python3 scripts/export_baseline_protocol.py

# 2. Inputs: masked Ghidra decompilation + module-context digest for train/val/test
python3 scripts/a4_build_modctx_dm.py --src <protocol dir> --out <data dir>

# 3. Generation head: fine-tune CodeT5+ 220M (3 epochs, lr 5e-5, batch 32, bf16) and decode the tiers
python3 scripts/a4_train_codet5p.py --model Salesforce/codet5p-220m --data-dir <data dir> --tag modctx_dm --bf16
python3 scripts/a4_predict.py <checkpoint> --tiers val,test --tag modctx_dm --bf16

# 4. Retrieval head: embed train/val/test with the fine-tuned encoder, cosine k-NN, s1 and margin
python3 scripts/c_lmemb_knn.py

# 5. Router + abstention (fit on validation only) and every table of the paper
python3 scripts/router3_eval.py --a4 <a4 preds tsv> --out results/dualhead_v2/router.json
```

Baselines: `baselines/symgen/` (authors' pipeline, one LoRA epoch on the LineageBench training tier) and
`baselines/blens*/` (authors' code, 80 + 80 epochs, CLAP and PalmTree embeddings extracted by us). Punstrip:
`scripts/punstrip/` and `results/punstrip/PLAN.md`.

## Scoring a new system on LineageBench

Join your predictions on `(binary, address)` with the protocol tier records, canonicalise (demangle, split on
underscores / camel-case / digits, lower-case) and score with `src/evaluation/metrics.py`
(`compute_subtoken_f1`, `compute_subtoken_precision_recall`); report function-level and package-level means, and
per regime and name category. Prediction dumps of every system in the paper are under `results/`, so a new
comparison needs no retraining of the baselines.

## References

- He et al. **Debin**, CCS 2018. - David et al. **NERO**, OOPSLA 2020. - Jin et al. **SymLM**, CCS 2022.
- Patrick-Evans et al. **XFL**, IEEE S&P 2023 (Punstrip). - Benoit et al. **BLens**, USENIX Security 2025.
- Jiang et al. **SymGen** ("Beyond Classification"), NDSS 2025. - Wang et al. **CodeT5+**, EMNLP 2023.
- Ye et al. **Epitome**, FSE 2024. - Allamanis. **The adverse effects of code duplication**, Onward! 2019.

## License

MIT License (see `LICENSE`).
