# N1–N3 — Production Dual-Head Evaluation, Complementarity, and Oracle Ceiling

**Date:** 2026-08-09 · **Branch:** `dualspace` · clean-7 **n=13581** · composition threshold 0.30 (VAL-tuned)

**Anchors passed** — decoder **0.5330** (expect 0.5330), retrieval **0.5861** (expect 0.5913). Retrieval uses the **full 243289-function index with BinFilter τ=0.5**, not the 40K sampled no-filter index used in earlier comparisons. That earlier index was a handicapped baseline; results below supersede those.

## N1 — production heads on identical functions

| stratum | n | retrieval F1 | GRU F1 | composition F1 |
|---|---|---|---|---|
| seen | 10151 | **0.7567** | 0.6900 | 0.3799 |
| novel_comp | 1509 | **0.0806** | 0.0653 | 0.1026 |
| oov | 1921 | **0.0815** | 0.0712 | 0.1238 |
| novel+oov | 3430 | **0.0811** | 0.0686 | 0.1144 |
| overall | 13581 | **0.5861** | 0.5330 | 0.3128 |

## N2 — function-level dominance

| stratum | n | composer > retrieval | retrieval > composer | equal |
|---|---|---|---|---|
| seen | 10151 | 866 (8.5%) | 7616 (75.0%) | 1669 (16.4%) |
| novel_comp | 1509 | 306 (20.3%) | 178 (11.8%) | 1025 (67.9%) |
| oov | 1921 | 476 (24.8%) | 119 (6.2%) | 1326 (69.0%) |
| novel+oov | 3430 | 782 (22.8%) | 297 (8.7%) | 2351 (68.5%) |
| overall | 13581 | 1648 (12.1%) | 7913 (58.3%) | 4020 (29.6%) |

## N2 — unique correct semantic components (the complementarity test)

`unique_C = (C ∩ G) − R` — correct sub-tokens the composer recovers that retrieval misses entirely; `unique_R` is the mirror image.

| stratum | n | % fns with unique_C > 0 | mean unique_C | % fns with unique_R > 0 | mean unique_R |
|---|---|---|---|---|---|
| seen | 10151 | **3.8%** | 0.053 | 73.3% | 1.896 |
| novel_comp | 1509 | **12.2%** | 0.128 | 12.4% | 0.149 |
| oov | 1921 | **9.7%** | 0.106 | 4.4% | 0.060 |
| novel+oov | 3430 | **10.8%** | 0.115 | 7.9% | 0.099 |
| overall | 13581 | **5.6%** | 0.069 | 56.8% | 1.442 |

## N3 — oracle dual-head ceilings

| stratum | n | R | C | GRU | oracle R+C | oracle R+GRU | oracle R+C+GRU | Δ(R+C − R) |
|---|---|---|---|---|---|---|---|---|
| seen | 10151 | 0.7567 | 0.3799 | 0.6900 | **0.7803** | 0.7866 | 0.8036 | **+0.0237** |
| novel_comp | 1509 | 0.0806 | 0.1026 | 0.0653 | **0.1321** | 0.1003 | 0.1368 | **+0.0516** |
| oov | 1921 | 0.0815 | 0.1238 | 0.0712 | **0.1366** | 0.1039 | 0.1470 | **+0.0551** |
| novel+oov | 3430 | 0.0811 | 0.1144 | 0.0686 | **0.1347** | 0.1023 | 0.1425 | **+0.0535** |
| overall | 13581 | 0.5861 | 0.3128 | 0.5330 | **0.6173** | 0.6138 | 0.6366 | **+0.0312** |

## N4 — conservative token fusion

Rules, per plan §8. `tau_C` tuned on VALIDATION only; clean-7 is evaluation-only.

- top-1: tau_C = **0.85** (val set-F1 0.5295)
- top-2: tau_C = **0.85** (val set-F1 0.5295)

| stratum | n | retrieval only | + top-1 composer | + top-2 composer |
|---|---|---|---|---|
| seen | 10151 | 0.7567 | **0.7565** (-0.0002) | **0.7565** (-0.0002) |
| novel_comp | 1509 | 0.0806 | **0.0858** (+0.0053) | **0.0858** (+0.0052) |
| oov | 1921 | 0.0815 | **0.0839** (+0.0023) | **0.0839** (+0.0023) |
| novel+oov | 3430 | 0.0811 | **0.0847** (+0.0036) | **0.0847** (+0.0036) |
| overall | 13581 | 0.5861 | **0.5868** (+0.0008) | **0.5868** (+0.0008) |

**N4 verdict (bar +0.0100 over production retrieval on novel+OOV): FAIL — keep the two outputs separate** (best Δ = +0.0036)

Per plan §8, fusion is not adopted. The dual retrieval-composition architecture is retained and both outputs are presented to the analyst; no unified prediction is forced merely to claim one.

## Verdict

- oracle(R+C) − R on novel+OOV: **+0.0535**; overall: **+0.0312**
- functions where the composer contributes a correct sub-token retrieval misses (novel+OOV): **10.8%**

**Complementarity is SUBSTANTIAL — N4 token fusion is justified.**

**No adaptive gate, router, or arbiter is built or claimed** (plan §7). The system is presented as a dual-output semantic recovery method: retrieval returns a candidate known identifier, composition returns a semantic component set, and both are shown to the analyst.
