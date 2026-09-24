# Production Retrieval-Index Reconciliation (spec §2, Priority Zero)

**Date:** 2026-08-13 · **Artifact:** `production_train_function_ids.txt` (package, binary,
address, name, in_paper_clean flag; one row per index member)

## Set arithmetic

```
historical (paper protocol of record)  = 241,174   (NDSS_FINAL_RESULTS.md: "clean 241,174-function
                                                    GCC train index", anchor 0.5913, job 1168004)
current recovered (frozen-dump index)  = 243,289
intersection                           = 241,174
only_in_243289                         =   2,115
only_in_241174                         =       0
```

## The 2,115-function difference, exactly

The 2,115 functions are **all and only the `grep` (1,267) and `sed` (848) package train
functions**: 243,289 − 2,115 = 241,174 with zero remainder. Per the leakage audit
(`results/ndss_prep/leakage_audit.md`), the paper-clean train definition ("Train B",
`best_model_paper_clean.pt`, logged 241,174) excludes all 7 cross-project packages **plus
grep and sed** as leakage-suspect; the Aug-9 dual-head dump instead built its index from
`ds.get_splits()` train membership, which retains grep+sed. No other category (duplicates,
extraction version, address-matching, added packages) contributes: the two sets are otherwise
identical, proven by exact element-wise reproduction of the dump's `random.Random(1234)`
80,000-function probe sample from the recovered 243,289-member list.

(A note in `results/ndss_prep/probe_encoder.md` attributes the difference to "dropping curl" —
that note is wrong; curl's train functions are present in both sets. The grep+sed arithmetic
is exact.)

## Which set backs which frozen artifact

| Artifact | Index |
|---|---|
| Frozen clean-7 retrieval dump (`ndss_dual_head_eval.json`, Aug-9; its own ALL macro 0.5861) | 243,289 (grep+sed IN) |
| Paper anchor 0.5913 (job 1168004, `tau05_t0.5.json`) | 241,174 (grep+sed OUT) |

## Decision: corpus of record

**`D_train` := the 241,174-function paper-clean index** (drop grep+sed from the recovered set).
Rationale: (a) it is the published protocol of record and the definition of the 0.5913 anchor
this experiment must reproduce; (b) grep/sed were excluded for leakage hygiene and re-admitting
them would be a protocol regression; (c) U0 is being rebuilt from scratch under one pipeline
anyway, so consistency with the paper matters more than consistency with the Aug-9 dump, which
is superseded by the rebuilt U0. The Aug-9 dump's inclusion of grep+sed is recorded as a
(small) protocol deviation of that dump.

All of: U0 rebuilt index, U1 training population, token vocabulary, SEEN/NOVEL/OOV strata, and
frequency statistics now derive from exactly `D_train` (rows flagged `in_paper_clean_241174=1`).
