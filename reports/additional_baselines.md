# Additional Baseline Papers for Binary Function Name Recovery

Supplement to `related_work_comparison.md`. Papers found via deep search on 2026-04-05.

## New findings (not previously covered)

### 1. Epitome — PACMSE / FSE 2024 ⭐ HIGHEST RELEVANCE
- **Title:** "Enhancing Function Name Prediction using Votes-Based Name Tokenization and Multi-Task Learning"
- **Authors:** Zhang, Xu et al.
- **Venue:** PACMSE Vol 1, Issue FSE (July 2024)
- **arXiv:** https://arxiv.org/abs/2405.09112
- **Approach:** Pre-trained assembly language model + GNN + **votes-based name tokenization** + multi-task learning (adds function semantics similarity prediction task across opt levels)
- **Input:** Disassembled x86/ARM/MIPS assembly
- **Dataset:** **2,597,346 functions** across 5 opts (O0-Os) × 4 archs (x64, x86, ARM, MIPS)
- **Reported improvement:** +44.34% precision, +64.16% recall, +54.44% F1 over SOTA (baseline unspecified in abstract)
- **Code availability:** Not clearly indicated from abstract; need to check full paper
- **Why critical for our paper:** Uses **votes-based name tokenization**, the SAME technique as our V3 Votes tokenizer. Multi-task learning with cross-opt similarity is architecturally similar to our contrastive cross-opt pairs. **This is probably the closest methodological relative of our work that we were missing.**

### 2. HexT5 — ASE 2023
- **Title:** "HexT5: Unified Pre-Training for Stripped Binary Code Information Inference"
- **Authors:** (undisclosed in summary, from Kehuan Zhang's group HKU)
- **Venue:** ASE 2023 (38th IEEE/ACM International Conference on Automated Software Engineering)
- **DOI:** 10.1109/ASE56229.2023.00099
- **Approach:** CodeT5-based unified encoder-decoder, pretrained on pseudo-code with multiple objectives
- **Tasks covered:** function name recovery, variable name recovery, code summarization, similarity detection (all in one model)
- **Input:** Ghidra/IDA decompiled pseudo-code
- **Why relevant:** Directly comparable on function name task. One of the most recent T5-based approaches.

### 3. Bin2Summary — PACMSE 2024
- **Title:** "Bin2Summary: Beyond Function Name Prediction in Stripped Binaries with Functionality-Specific Code Embeddings"
- **Venue:** PACMSE Vol 1, Issue FSE 2024
- **DOI:** 10.1145/3643729
- **Approach:** Functionality-specific code embeddings + attention-based seq2seq for natural language summaries
- **Reported:** 0.728 precision, 0.729 recall on 38,167 functions from 16 projects (e.g., coreutils)
- **Task mismatch:** Their output is natural-language summaries, not function names. Adjacent task — could cite for context but NOT a direct baseline.

### 4. BinT5
- Mentioned as a baseline in HexT5 and "How Far Have We Gone..." paper
- CodeT5 fine-tuned on binary code for summarization (primarily)
- May have a function-name variant — need to dig deeper
- Predecessor of HexT5

### 5. "How Far Have We Gone in Stripped Binary Code Understanding Using LLMs" — arXiv 2024
- **arXiv:** 2404.09836
- **Approach:** Benchmark study, not a new model. Evaluates 20 LLMs + 4 deep learning baselines (SymLM, NER, BinT5, HexT5) on 2 tasks: function naming + summarization
- **Models benchmarked:** CodeGen, WizardCoder, CodeLlama (multiple sizes), DeepSeek-Coder, ChatGLM, Vicuna, Llama-2, Mistral, Mixtral, ChatGPT + the 4 DL models
- **Dataset:** 2,000 curated functions from 12 real-world C projects (FFmpeg, Redis, OpenSSL, ...) across 8 domains, **public release**
- **Why extremely valuable:** (a) it's a READY-MADE public benchmark we can run our model on, (b) gives us LLM baselines we wouldn't otherwise be able to reproduce (ChatGPT, Mixtral), (c) includes code snippets of 4 DL baselines' evaluation format
- **Action item:** get their dataset and run our model on it → additional data point for the paper

### 6. In Nomine Function — arXiv 2019
- **Title:** "In Nomine Function: Naming Functions in Stripped Binaries with Neural Networks"
- **arXiv:** 1912.07946
- **Approach:** seq2seq encoder-decoder, bidirectional LSTM, assembly input
- **Venue:** arXiv only (no published venue found), 2019
- **Status:** Pre-DEBIN-era, likely superseded but might be cited for historical coverage
- **Feasibility:** Low priority — older than DEBIN, probably not worth reproducing

### 7. VarBERT — Oakland (IEEE S&P) 2024
- **Title:** "Len or index or count, anything but v1: Predicting Variable Names in Decompiled Code"
- **Venue:** IEEE S&P 2024
- **Task:** **Variable name recovery**, NOT function name recovery
- **Out of scope** for our paper's main comparison, but could be cited in related work for context (same authors / adjacent research line)

### 8. EKLAVYA — USENIX Security 2017
- **Title:** "Neural Nets Can Learn Function Type Signatures from Binaries"
- **Task:** Function TYPE signature recovery (argc, argv types), not names
- **Out of scope** but often cited in related-work sections for historical coverage

## Revised baseline ranking (merged with original list)

**Tier 1 — Must include in comparison (HIGH relevance, feasible to reproduce or cite):**
1. **SymLM** (CCS 2022) — currently training from scratch on Wulver
2. **AsmDepictor** (AsiaCCS 2023) — public pretrained checkpoint on Zenodo
3. **BLens** (USENIX Sec 2025) — cite published numbers (80GB VRAM out of reach)
4. **SymGen** (NDSS 2025) — cite published numbers (80GB VRAM out of reach)
5. **XFL** (IEEE S&P 2023) — xfl-r refactor available, multi-label tags
6. **⭐ Epitome** (FSE 2024) — NEWLY FOUND, most methodologically similar (votes tokenization + multi-task). **MUST check their paper + code if available.**

**Tier 2 — Strong supporting citations:**
7. **HexT5** (ASE 2023) — decompiled pseudo-code baseline
8. **NERO** (OOPSLA 2020) — IDA-dependent, cite paper
9. **DEBIN** (CCS 2018) — cite paper, stack too old
10. **NFRE** (ISSTA 2021) — cite paper, no model code
11. **DIRTY** (USENIX Sec 2022) — variable names, task partial overlap

**Tier 3 — Related work context only:**
12. **"How Far..."** (arXiv 2024) — benchmark study, useful for citing LLM numbers + get their public dataset
13. **Bin2Summary** (FSE 2024) — summarization, adjacent task
14. **BinT5** — CodeT5 precursor to HexT5
15. **In Nomine Function** (arXiv 2019) — historical, pre-DEBIN
16. **VarBERT** (S&P 2024) — variable names, different task
17. **EKLAVYA** (USENIX Sec 2017) — type signatures, different task
18. **GenNm** (NDSS 2025) — variable names, different task

## Actionable next steps

1. **Epitome investigation (DONE, 2026-04-05):**
   - Code availability: **NO public code release.** Paper only cites SymLM and NFRE (the latter contacted authors) as baselines.
   - Their votes tokenizer ≠ ours (see below), no prior-art issue.
   - Cite Epitome in related work with F1 71.96% (weighted macro, P=73.13, R=70.84 on 2.6M fns).
   - Differentiate: we use BAP-IR (decompiler-free), simpler rule-based sub-token splitter, k-NN retrieval with binary fingerprint filter, cross-opt NT-Xent contrastive.

2. **Clarification on "votes" tokenization terminology (IMPORTANT):**
   - **Our `build_votes.py` is a simple rule-based identifier splitter** (underscore + camelCase + lowercase). NO voting, NO multiple models.
   - **Epitome's votes tokenization is genuinely different** — 3-model ensemble (N-gram transition frequency + unigram LM + rule-based longest English/programming word), ≥2-of-3 agreement.
   - These are completely different methods that happen to share the word "votes."
   - **ACTION: rename our tokenizer in the paper** to avoid reviewer confusion. Candidates: "Sub-token identifier splitter", "Rule-based name splitter", "V3 name tokenizer."
   - **ACTION: fix CLAUDE.md** which currently misdescribes our tokenizer as "3 models vote on sub-token boundaries, 95% less OOV than BPE" — the "3 models vote" part is factually incorrect for our implementation.

3. **"How Far Have We Gone..." dataset (2024-09836, ICSME 2024):**
   - Paper claims to release a 2,000-function benchmark but **no public URL exists** in the arXiv text or ACM/IEEE versions. Need to email corresponding author: `sycheng@ustc.edu.cn` (Shaoyin Cheng, USTC).
   - Would be highly valuable if obtained: gives us LLM baselines (ChatGPT, Mixtral, CodeLlama, DeepSeek-Coder) for free.

4. **BinMetric (IJCAI 2025):** 6-task binary analysis benchmark. Tasks: decompilation, summarization, assembly instruction gen, call-site reconstruction, signature recovery, algorithm classification. **Function name recovery NOT among the 6 tasks.** Not a direct baseline for us.

## Sources
- [Epitome arXiv 2405.09112](https://arxiv.org/abs/2405.09112)
- [Epitome ACM DL](https://dl.acm.org/doi/10.1145/3660782)
- [HexT5 ASE 2023](https://conf.researchr.org/details/ase-2023/ase-2023-papers/96/HexT5-Unified-Pre-training-for-Stripped-Binary-Code-Information-Inference)
- [Bin2Summary FSE 2024](https://dl.acm.org/doi/10.1145/3643729)
- [How Far Have We Gone arXiv 2404.09836](https://arxiv.org/html/2404.09836v1)
- [In Nomine Function arXiv 1912.07946](https://arxiv.org/abs/1912.07946)
- [VarBERT Oakland 2024](https://sefcom.asu.edu/publications/varbert-oakland24.pdf)
- [GenNm NDSS 2025](https://www.ndss-symposium.org/wp-content/uploads/2025-276-paper.pdf)
