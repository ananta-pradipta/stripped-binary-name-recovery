# Final Summary Tables — Dual Retrieval–Composition System (Professor Update)

**Compiled 2026-08-10, branch `openvocab`.** Architecture frozen. Every number
traces to a job ID; commit IDs in the experiment log. Population: GCC clean-7
cross-project, n = 13,581 unless stated. Splitter `v1-canonical-2026-08-10`.

---

## Table A — Complete-name prediction (existing verified results; unchanged)

| system | population | name F1 | EM |
|---|---|---|---|
| **Retrieval (k-NN + BinFilter)** | clean-7 full | **0.591** | **48.4%** |
| GRU decoder | clean-7 full | 0.540 | 43.0% |
| SymGen+LoRA (34B) | matched FT n=3,830 | 0.192 (ours 0.211) | 6.1% (ours 17.8%) |
| SymGen+LoRA (34B) | matched NCT n=5,778 | 0.324 (ours 0.845) | 6.5% (ours 70.2%) |
| BLens (retrained) | matched n=9,710 | 0.358 (ours 0.586) | 11.2% (ours 48.1%) |

---

## Table B — Semantic composition (atom sets, same canonical splitter/evaluator)

Overall semantic P/R/F1 on clean-7 (learned composers at matched operating
points; name-token rows = tokenized complete-name predictions, P8 re-scoring):

| system | sem P | sem R | sem F1 | notes |
|---|---|---|---|---|
| A0 fixed-4K | 0.429 | 0.251 | 0.299 | |
| H1 hybrid | 0.430 | 0.259 | 0.303 | |
| **H2 set-structured hybrid** | 0.431 | 0.258 | 0.303 | set-level ≈ H1 (by design; value is Table C) |
| Retrieval-name tokens | 0.591 | 0.587 | 0.587 | dominated by seen names |
| GRU-name tokens | 0.537 | 0.536 | 0.535 | |
| SymGen-FT tokens (n=4,937) | 0.203 | 0.193 | 0.185 | |
| SymGen-NCT tokens (n=7,445) | 0.329 | 0.444 | 0.334 | |

Key band/subset facts (3-seed means):
- rare / very-rare recall: A0 0.081 / 0.005 → H1 & H2 0.081 / **0.062–0.071** (12×)
- tail added precision at 2% coverage: H1 ~0.45 → **H2 0.67–0.80**
- retrieval-failure subset (retr F1 = 0, n=4,271): our composers ≈ 0.053 sem F1;
  **SymGen-FT tokens 0.178, SymGen-NCT tokens 0.401** — the LLM's name tokens
  remain the strongest semantic evidence exactly where our retrieval fails
  (contamination caveats apply; report, do not hide).

---

## Table C — Selective semantic evidence (H2's headline; controlled 3-seed protocol)

`results/phase2_frontier_final.tsv` is the result of record. Discrepancy note:
the earlier "50% → 1.6% coverage" H1 figure was seed 42 alone; H1 is
seed-unstable (seed 42 ≈ 1.5%, seeds 123/7 ≈ 0.07%).

| metric | operating point | H1 (3-seed) | **H2 (3-seed)** |
|---|---|---|---|
| precision @ coverage | 0.5% | 0.49 ± 0.05 | **0.98 ± 0.01** |
| | 1% | 0.48 ± 0.06 | **0.84 ± 0.14** |
| | 2% | 0.41 ± 0.01 | **0.74 ± 0.06** |
| | 5% | 0.18 ± 0.02 | **0.33 ± 0.02** |
| coverage @ precision | 50% | 0.5% ± 0.8% | **3.3% ± 0.2%** |
| | 70% | 0.07% | **2.1% ± 0.4%** |
| | 90% | 0.07% | **0.9% ± 0.2%** |
| | 95% | 0.07% | **0.9% ± 0.2%** |

Causal note: K=2 (from tail-cardinality census, P99.9 = 2); the no-Hungarian
control performs comparably, so the gain is attributed to K-queries +
existence gating + validation calibration, not the matching algorithm.

---

## Phase-3 copyability census (evidence-grounded OOV potential; no model)

Evidence = external calls, string constants, NEEDED libraries, dynamic-symbol
**imports only** (defined `.dynsym` excluded — documented internal-name leak).

Atom population (50,297 GT atom slots): HEAD_KNOWN 91.4% · TAIL_KNOWN 4.5% ·
OOV_COPYABLE 1.1% · OOV_NONCOPYABLE 3.0%.

| subset | n fns | % GT atoms in evidence | % fns ≥1 copyable | % fns ≥2 |
|---|---|---|---|---|
| ALL | 13,581 | 16.9% | 39.2% | 15.0% |
| **retrieval F1 = 0** | **4,271** | **47.1%** | **55.5%** | **35.3%** |

**The directive's key number: 55.5% of retrieval-failure functions carry ≥1
correct semantic atom in observable lexical evidence.** Source decomposition:
external calls + dynsym imports dominate (≈6,200 correct atoms), strings add
≈4,200 (2,300 alone), libraries are negligible (37 functions).

**Zero-training IDF baseline: fails** (retr-F1=0 subset: sem F1 0.004–0.007,
3–5 false hints/function). The evidence signal exists but undirected
extraction drowns in pool size — per the directive, the census stands as the
result and motivates a *learned, z-conditioned* copy mechanism as the next
extension; no trained copy head is claimed or built.

---

## Final architecture (frozen, of record)

```
                         Binary function
                               |
                         Shared encoder (frozen; 0.8053 test anchor)
                               |
                               z
                  _____________|_____________
                 |                           |
            Retrieval                  Composition
        (k-NN + BinFilter)                   |
                 |                    ________|________
        complete-name                |                 |
         recognition            fixed 4K head     H2 K-query tail
        0.591 F1 / 48.4% EM     common concepts   decoder (K=2, calibrated)
                                     |            rare concepts
                                     +-----+------+
                                           |
                              calibrated semantic evidence
                                           |
                    [next extension: evidence-grounded copy branch —
                     motivated by the 55.5% census number, not yet built]
```

## Scientific-story assessment (per the directive)

- **Strong complete-name prediction** ✓ Table A.
- **Clean semantic evidence** ✓ Table C: H2 turns the tail channel into a
  calibrated, seed-stable hint source (up to 98% precision).
- **Useful information beyond retrieval** ✓ partially: composition rescues
  components (examples B/C), and the census shows the larger untapped source
  is observable lexical evidence (55.5% of retrieval failures).
  ✗ not yet: set-level composition on retrieval failures is still weak
  (~0.05 sem F1), and SymGen's tokens remain stronger there — the honest gap.
- **Controlled false-hint risk** ✓ Table C + example F (the ~5% confident-wrong
  residual is quantified and must be stated).
