# Baseline Feasibility Assessment — 9 Papers for Reproduction

Deep research compiled 2026-04-06. Each paper assessed for end-to-end reproducibility on our hardware (Wulver: A100 10GB MIG, course_gpu QoS max 1 GPU) and our dataset.

## Summary Table (ranked by feasibility)

| # | Paper | Venue | Code | Checkpoint | VRAM (train) | VRAM (inf) | Disassembler | Feasibility | Key blocker |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **llasm** | TOSEM 2025 | [GitHub](https://github.com/Sandspeare/llasm) | [HF encoder](https://huggingface.co/sandspeare/llasm-encoder) + [HF decoder](https://huggingface.co/sandspeare/llasm-decoder) | 4×A100 80GB | ~26-40GB (Vicuna-13B) | Assembly (IDA/BinaryCorp) | **MEDIUM** | 13B inference needs 26GB+ VRAM → won't fit on 10GB MIG. 4-bit quant → ~8GB feasible |
| 2 | **SymLM** | CCS 2022 | [GitHub](https://github.com/OSUSecLab/SymLM) | Trex encoder only (incompatible) | 1×GPU 10GB | 1×GPU 10GB | Ghidra (free) | **MEDIUM** | Trex 2021 checkpoint deleted from GDrive. We trained from scratch: F1=0.36 (vs paper 0.53). Email sent to authors. |
| 3 | **AsmDepictor** | AsiaCCS 2023 | [GitHub](https://github.com/agwaBom/AsmDepictor) | [Zenodo](https://zenodo.org/record/7978756) (160MB) | 64GB | <10GB | IDA Pro | **MEDIUM-LOW** | Pretrained ckpt released BUT vocab file NOT released (33,546-entry vocab unreconstructable from released data). Blocked. |
| 4 | **XFL** | IEEE S&P 2023 | [GitHub](https://github.com/lmu-plai/xfl) (xfl-r refactor) | Not released | Unknown | Unknown | DEXTER static features | **MEDIUM-LOW** | Output is multi-label TAGS not name strings → metric mismatch. DEXTER build needed. No trained model released. |
| 5 | **SymGen** | NDSS 2025 | [GitHub](https://github.com/OSUSecLab/SymGen) | LoRA for 34B only | 1×A100 80GB | 1×A100 80GB (34B) | Ghidra (free) | **LOW** | Only 34B LoRA released. Our max GPU is 10GB MIG. 4-bit quant needs ~22GB → blocked. |
| 6 | **BLens** | USENIX Sec 2025 | [GitHub](https://github.com/lmu-plai/blens) | [Zenodo](https://zenodo.org/) | 80GB rec. | 80GB rec. | Ghidra (multi-stage) | **LOW** | 4 upstream embedding models (PalmTree+CLAP+DEXTER+VarCLR). 80GB VRAM. 200+200 epochs. 50GB disk/experiment. |
| 7 | **Epitome** | FSE 2024 | **NOT released** | **NOT released** | Unknown | Unknown | Assembly + PLM + GNN | **INFEASIBLE** | No code, no checkpoint, no data released. Cite-only. |
| 8 | **NFRE** | ISSTA 2021 | [GitHub](https://github.com/USTC-TTCN/NFRE) (dataset toolkit only) | **NOT released** | Unknown | Unknown | BAP | **INFEASIBLE** | Repo is dataset-building toolkit. No model code, no checkpoint. Would need full reimplementation. |
| 9 | **GenNm** | NDSS 2025 | arXiv page | Unknown | Unknown | Unknown | Decompiled C | **N/A — WRONG TASK** | Recovers **variable** names, NOT function names. Out of scope for our comparison. |

---

## Per-Paper Deep Assessment

### 1. llasm (TOSEM 2025) — MEDIUM feasibility ⭐ NEW FINDING

**Full title:** "llasm: Naming Functions in Binaries by Fusing Encoder-only and Decoder-only LLMs"
**Authors:** Zihan Sha, Hao Wang, Zeyu Gao et al. (Tsinghua University / NISL)
**Venue:** ACM TOSEM Vol 34, Issue 4 (2025) — top-tier SE journal
**DOI:** 10.1145/3702988

**Architecture:** LLaVA-style fusion — custom assembly encoder (0.1B params) + Vicuna-13B decoder via projection layer. Two-stage training: (1) pretrain projection, (2) instruction-tune full model.

**Artifacts:**
- Code: [github.com/Sandspeare/llasm](https://github.com/Sandspeare/llasm) ✅
- Encoder checkpoint: [HuggingFace sandspeare/llasm-encoder](https://huggingface.co/sandspeare/llasm-encoder) (0.1B, MIT) ✅
- Decoder LoRA: [HuggingFace sandspeare/llasm-decoder](https://huggingface.co/sandspeare/llasm-decoder) (LoRA for Vicuna-13B, MIT) ✅
- Dataset: BinaryCorp (69K binaries, 1.58M functions). "Will release after publication" — may or may not be available yet.

**Hardware:**
- Training: 4×A100 80GB, 24h pretrain + 24h instruction-tune = 48h total. **NOT feasible on our hardware.**
- Inference: Vicuna-13B base → ~26GB fp16, ~8GB 4-bit quantized. **4-bit on our 10GB MIG is tight but possibly feasible.**

**Reported metrics:** P=0.633, R=0.581, **F1=0.606** on unseen binaries (8,000 test binaries from distinct projects).

**Baselines they compared against:** XFL, SymLM, NERO, DEBIN, AsmDepictor + encoder ablations (PalmTree, jTrans, Trex). All prior baselines scored <0.12 F1 on their eval.

**Reproduction path for us:**
- Download encoder (HF, small) + decoder LoRA (HF, small) + Vicuna-13B base (~26GB fp16 or ~7GB 4-bit)
- Load with `BitsAndBytesConfig(load_in_4bit=True)` + LoRA adapter — should fit on 10GB MIG
- Need to tokenize our binaries into their expected assembly format (Capstone-based, similar to AsmDepictor)
- Run inference on our cross-project binaries
- **Estimated effort: 1-2 days** (download + format conversion + inference)
- **Risk: Vicuna-13B 4-bit on 10GB may be tight with KV cache. Need ~8-9GB.**

**Why important:** llasm reports F1=0.606 on UNSEEN projects — this is the most directly comparable number to our cross-project F1=0.704. If we can run llasm on our data, the comparison is very strong.

### 2. SymLM (CCS 2022) — MEDIUM feasibility (ALREADY ATTEMPTED)

**Status:** We already trained SymLM from scratch (no Trex) on their sample data → F1=0.359 on valid. Currently processing our cross-project binaries through their Ghidra+prepare_dataset pipeline (job 904689 running overnight). Email sent to Prof Xin Jin for Trex 2021 checkpoint.

**Key issue:** Trex pretrained encoder checkpoint incompatible (2024 version replaces 2021 architecture). Without Trex, SymLM is handicapped by ~15-20pp F1.

**Our reproduction result:** F1=0.359 (no Trex, their sample data). Paper: F1=0.53 (with Trex).

### 3. AsmDepictor (AsiaCCS 2023) — MEDIUM-LOW feasibility (BLOCKED)

**Status:** Pretrained checkpoint (160MB) downloaded from Zenodo. BUT the checkpoint expects a 33,546-entry shared vocabulary that was NOT released. Reconstructing vocab from released train_source.txt gives 33,446 (100 short). Adding test data gives 34,402 (856 over). Without exact vocab, embedding weights index into wrong tokens → inference gives garbage.

**Workaround attempted:** Train+test concatenation, min_freq sweep. None hit exactly 33,546.

**Resolution needed:** Email AsmDepictor authors (agwaBom) for the original `1_train.json` or vocab file. Alternatively, retrain from scratch (needs 64GB VRAM → blocked).

### 4. XFL (IEEE S&P 2023) — MEDIUM-LOW feasibility

**Status:** Code available via xfl-r refactored repo. Uses DEXTER static feature extractor.

**Key issues:**
- Output is multi-label **tags** (sub-tokens as independent labels), NOT a generated name string. Direct F1 comparison requires recomposing names from tags — metric alignment is tricky.
- DEXTER needs to be built and configured separately.
- No pretrained model released — must train from scratch on our data.
- Dataset is 10K+ Debian binaries (different from ours).

**Estimated effort:** 2-3 days (build DEXTER + adapt to our data + train + eval).

### 5. SymGen (NDSS 2025) — LOW feasibility

**Status:** Repo cloned, CodeLlama-34B downloaded (63GB), Ghidra installed on Wulver. BUT:
- Only 34B LoRA adapter released (no 7B/13B variant)
- 34B fp16 needs ~68GB VRAM → won't fit
- 34B 4-bit needs ~22GB → won't fit on 10GB MIG
- course_gpu QoS limits us to 1×a100_10g (10GB)
- Tried debug_gpu for a100_40g → billing-expired account

**Resolution needed:** Full A100 40GB access (HPC admin request) or 7B variant (not released).

### 6. BLens (USENIX Sec 2025) — LOW feasibility

**Status:** Repo available. Requires ensemble of 4 pretrained embedding models (PalmTree + CLAP + DEXTER + VarCLR) fused with a "LORD" decoder. 

**Key blockers:**
- 80GB VRAM recommended for training
- 50GB disk per experiment
- 200+200 epoch training schedule
- Must wire up 4 separate upstream embedding stacks
- Our max GPU is 10GB → completely infeasible

**Resolution:** Cite-only. Their published cross-project F1=0.46.

### 7. Epitome (FSE 2024) — INFEASIBLE for reproduction

**Status:** No code, no checkpoint, no dataset released anywhere. Paper only cites SymLM and NFRE repos as baselines.

**Published metrics:** F1=71.96% weighted macro (P=73.13, R=70.84) on 2.6M functions.

**Important note:** Their "votes-based name tokenization" is a 3-model voting system (N-gram TF + unigram LM + rule-based), NOT the same as our frequency-based sub-token tokenizer despite naming similarity. See `project_paper_positioning.md` for differentiation guidance.

**Resolution:** Cite-only with F1=0.72.

### 8. NFRE (ISSTA 2021) — INFEASIBLE for reproduction

**Status:** GitHub repo is a dataset-building toolkit ONLY. No model implementation, no checkpoint, no training code.

**Resolution:** Cite-only. Would need full reimplementation from paper to reproduce.

### 9. GenNm (NDSS 2025) — OUT OF SCOPE

**Task mismatch:** GenNm recovers **variable names** inside decompiled functions, NOT function names. Different task. Do not compare head-to-head on our metric.

**Resolution:** Mention in related work for completeness, do NOT include in comparison table.

---

## Recommended Reproduction Priority

### Tier 1 — Directly reproducible with our hardware (10GB MIG)

| Paper | Action | Effort | Expected result |
|---|---|---|---|
| **llasm** ⭐ | Download HF checkpoints, load Vicuna-13B in 4-bit + LoRA, Capstone tokenize our binaries, run inference | **1-2 days** | Their F1=0.606 on unseen projects, directly comparable to our 0.704. **Strongest possible comparison.** |
| **SymLM** | Already trained from scratch (F1=0.36). Cross-project eval running overnight (job 904689). If Trex arrives from email, retrain. | **Done + 1 day if Trex arrives** | F1=0.36 (no Trex) or ~0.50 (with Trex) on our xproj |

### Tier 2 — Blocked but solvable with author help

| Paper | Action | Effort if unblocked |
|---|---|---|
| **AsmDepictor** | Email authors for vocab file (1_train.json). If received, inference is 1 hour on our data. | 1 day after response |
| **SymGen** | Need A100 40GB access OR 7B LoRA (not released). HPC admin request pending. | 1-2 days if unblocked |

### Tier 3 — Cite-only (no reproduction feasible)

| Paper | Published number to cite |
|---|---|
| **BLens** | Cross-project F1 = 0.46, cross-binary F1 = 0.77 |
| **Epitome** | F1 = 0.72 weighted macro |
| **XFL** | Precision = 0.825 (tags, not names — metric mismatch) |
| **NFRE** | Cite paper, no public numbers easily comparable |
| **GenNm** | Out of scope (variable names) |

---

## Recommended Paper Comparison Table

```
System              Venue        Approach                Input       Our Reproduction   Paper-Reported
─────────────────────────────────────────────────────────────────────────────────────────────────────
Ours (FuncR)        —            GAT+GNN+k-NN+contrast  BAP-IR      F1 = 0.704 (xproj) —
llasm               TOSEM 2025   LLaVA-style encoder+    Assembly    TBD (inference)    F1 = 0.606
                                 Vicuna-13B decoder
SymLM               CCS 2022     Trex encoder + naming   Ghidra ICFG F1 = 0.359*        F1 = 0.53
                                 head                                (*no Trex)
AsmDepictor         AsiaCCS 2023 Transformer enc-dec     Assembly    Blocked (vocab)    F1 = 0.715†
SymGen              NDSS 2025    CodeLlama-34B LoRA      Ghidra C    Blocked (VRAM)     F1 ≫ prior‡
BLens               USec 2025    4-model ensemble        Ghidra      Blocked (VRAM)     F1 = 0.46 (xproj)
Epitome             FSE 2024     Votes tok + multi-task  Assembly    No code            F1 = 0.72
XFL                 S&P 2023     Extreme multi-label     DEXTER      Not attempted      P = 0.825§
NFRE                ISSTA 2021   Bi-GRU + GNN            BAP         No model code      —

† AsmDepictor's 0.715 is on THEIR test set, not ours
‡ SymGen reports relative improvements, not absolute F1 in comparable terms
§ XFL outputs tags, not name strings — metric differs
```

---

## Sources
- [llasm GitHub](https://github.com/Sandspeare/llasm)
- [llasm HF encoder](https://huggingface.co/sandspeare/llasm-encoder)
- [llasm HF decoder](https://huggingface.co/sandspeare/llasm-decoder)
- [llasm ACM TOSEM](https://dl.acm.org/doi/10.1145/3702988)
- [SymGen GitHub](https://github.com/OSUSecLab/SymGen)
- [BLens GitHub](https://github.com/lmu-plai/blens)
- [BLens arXiv](https://arxiv.org/abs/2409.07889)
- [SymLM GitHub](https://github.com/OSUSecLab/SymLM)
- [AsmDepictor GitHub](https://github.com/agwaBom/AsmDepictor)
- [AsmDepictor Zenodo](https://zenodo.org/record/7978756)
- [XFL GitHub](https://github.com/lmu-plai/xfl)
- [Epitome arXiv](https://arxiv.org/abs/2405.09112)
- [NFRE GitHub](https://github.com/USTC-TTCN/NFRE)
- [GenNm NDSS](https://www.ndss-symposium.org/ndss-paper/unleashing-the-power-of-generative-model-in-recovering-variable-names-from-stripped-binary/)
