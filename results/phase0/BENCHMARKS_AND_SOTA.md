# Benchmarks and SOTA for stripped-binary function-name recovery (verified 2026-08-17)

Scope: what external datasets exist, whether a BAP-based pipeline can consume them, what the
2025–2026 SOTA looks like, and how the field handles leakage and metrics. Every row below was
checked against a primary source (paper PDF via pdftotext, GitHub README, or Zenodo API) on
2026-08-17 unless marked **UNVERIFIED**. "BAP-usable" = ELF binaries (unstripped or debug+stripped)
are downloadable so we can lift them ourselves.

---------------------------------------------------------------------------------------------------
## (i) Dataset table

| Dataset (venue) | Size | Arch / compiler / opts | Split protocol | Raw ELF? | License | Download | BAP-usable? |
|---|---|---|---|---|---|---|---|
| **SymLM** (CCS'22, Jin et al.) | 1,431,169 fns, 16,027 binaries (incl. Hikari-obfuscated), 27 projects [paper §5.1]. Zenodo x64.zip: 4,618 files in 36 project dirs × O0–O3 (I range-read the zip central directory) | x64/x86/ARM/MIPS; **gcc-7.5**; O0–O3 (+4 obfuscations) | 8:1:1 **binary-level**, random; **no dedup**; no split file published (dataset_generation README: user assembles train/valid/test) | **Yes** — Zenodo ships raw unstripped project builds (`x64/O2/coreutils/ls` etc.; incl. .so, scripts). Ghidra-based extraction in repo | CC-BY-4.0 (Zenodo) | https://zenodo.org/record/8306055 (9.4 GB: x64 4.4 GB, x86 3.8, arm 0.6, mips 0.55; 2023-08-31); code https://github.com/OSUSecLab/SymLM | **YES** (strip ourselves; GT from symtab/DWARF). Caveat: Trex Dec-2021 checkpoint gone → SymLM model itself hard to rerun |
| **Punstrip/XFL Debian** (ACSAC'20 / S&P'23) | 741,724 fns, 10,047 C binaries, >3,000 Debian packages | mixed compilers/versions/opts as shipped by Debian maintainers (x86-64) | XFL: 90:5:5 random over binaries (503 test binaries); "function bodies are never shared between training and testing sets"; unseen-name subset reported (Tab. V) | **No.** Only scripts (`punstrip/debian-unstripped`: "unable to host our full dataset online due to space limitations"); XFL Zenodo = models + PostgreSQL feature tables (46.9 GB) | GPL-3.0 | scripts https://github.com/punstrip/debian-unstripped ; XFL https://github.com/lmu-plai/xfl ; https://zenodo.org/records/10733597 (2024-03-18) | **NO as-is**; would need re-download of Debian pkgs (versions drift → not bit-identical to XFL/BLens) |
| **BLens** (USENIX Sec'25, Benoit et al.) | same Punstrip corpus; splits ≈80/10/10 in two settings; cross-project test = 23,875 fns → strict 17,384 | as above | (1) **cross-binary** (binary-level), (2) **cross-project** (package-level), (3) **strict** = cross-project minus hash dups, 20 "free" functions, and functions whose name is in train AND contains one of 40 excluded prefix/co-occurring words (6,491 fns removed) | **No** — Zenodo `data.tar.gz` (8.4 GB) = pre-computed DEXTER/CLAP/PalmTree/VarCLR embeddings, tokenizers, train/val/test split files, logs of BLens/XFL/SymLM/AsmDepictor/HexT5 | GPL-3.0+ | https://zenodo.org/records/14713022 (v5, 2025-04-01); https://github.com/lmu-plai/blens | **NO** (no ELF). Split lists are published, so a rebuilt Debian corpus could be aligned by binary name — best-effort only |
| **SymGen "Beyond Classification"** (NDSS'25, Jiang/Jin/Lin) | 2,237,915 fns, 9,842 binaries, 33 GNU/C projects | x86-64/x86-32/ARM/MIPS; **GCC-9.4.0**; O0–O3 (+ Hikari obfuscated subset) | 8:1:1 **binary-level**; dedup by (name OR identical body) + callee-normalised body; `divide_dataset.py` uses unseeded `random.shuffle`, split-dump line commented out → **no canonical split file** (only sample train/test JSON) | **Yes** — `binaries.tar.gz` 4.88 GB by arch/opt/project (paper: builds with and without debug info); also stripped/unstripped Ghidra decompiled JSON, source tarballs, obfuscated binaries | CC-BY-4.0 (Zenodo); code "Other"/Apache-2.0 per README | https://zenodo.org/records/15694344 (latest, 2025-06-18; earlier 15530083/14252147); https://github.com/OSUSecLab/SymGen | **YES** — best candidate; LoRA weights released → direct SymGen rerun on any split |
| **Epitome** (FSE'24, Zhang et al.) | 2,597,346 fns, 36,275 binaries, 174 projects (Buildroot) | x86/x64/ARM/MIPS; O0–O3+Os; compiler = Buildroot GCC (version not stated) | 5-fold, 8:1:1, **source-level** (all opt levels of one source in one set); IDA Pro 7.3 + LLVM-IR plugin | **No** — README: "training data is too large … will be made public later" (repo last push 2025-06-22, still not); only a processed test set on Google Drive | none stated | https://github.com/Xiaolinger-Z/Epitome | **NO** |
| **NERO** (OOPSLA'20, David et al.) | 67,246 procedures; 60,403/2,034/4,809 | x86-64 GNU packages; filenames encode `<compiler>-<version>__O<level>__<pkg>__<exe>` | 8:1:1 **package-level**; dedup: dropped multi-version pkgs, C++, tests, static linking | **Yes** — `nero_dataset_binaries.tar.gz` (40 MB) TRAIN/VALIDATE/TEST dirs, executables contain debug info | CC-BY-4.0 | https://zenodo.org/record/4099685 ; https://github.com/tech-srl/Nero | **YES** (small; legacy cross-project sanity set; used by XFL & SymLM as external test) |
| **AsmDepictor** (AsiaCCS'23) | ~ (train_source.txt 0.66 GB, test 0.16 GB) | x86-64 assembly text (IDA) | random function split; exact-input dedup only (SymGen shows leaked pairs) | **No** — assembly text pairs + model | CC-BY-4.0 | https://zenodo.org/record/7978756 ; https://github.com/agwaBom/AsmDepictor | **NO** |
| **HexT5** (ASE'23) | – | pseudo-code (IDA) | project-level (per Shang et al. 2025) | **No** — only model weights | – | https://zenodo.org/records/11393904 ; https://github.com/USTC-TTCN/hext5 | **NO** |
| **GenNm** (NDSS'25) | variable-name task (not function names) | decompiled code | – | artifact repo only | – | https://github.com/XZ-X/gennm-ndss-ae | n/a (wrong task) |
| **REBench** (AIWare'26 + arXiv 2604.27319, Won/Jin/Ma/Lin, 2026-04-30) | superset of SymLM+Trex+[Xu et al.] corpora; after filtering x64: 2.87M/2.45M/2.51M/2.88M fns for O0–O3 (Table 1) | x64/x86/ARM32/MIPS32; O0–O3; C/C++ | LLM benchmark: random **function-level** samples (2,500 test + 10,000 train per arch×opt); dedup by code similarity over asm+decompiled; PLT/trivial (<10 lines) removed; **no project split** | **Yes** — `REBench_binaries.zip` 7.23 GB (symbol binaries; you strip), `REBench_decomp.zip` 3.46 GB (Ghidra 11.4.2 / IDA 7.6) | CC-BY-4.0 | https://zenodo.org/records/19899116 ; https://github.com/OSUSecLab/REBench | **YES** for binaries, but protocol is LLM-oriented and function-level |
| **BinaryLLMs-Eval** (Shang et al., EMSE'25 / arXiv 2504.21803) | 14,000 items (12 projects × arch/opt), fine-tune set 124,819 fns from 51 GNU pkgs (BinKit gcc-11.2.0, x64 O0) | x64/x86/ARM/MIPS; O0–O3 | project-disjoint benchmark vs GNU fine-tune set; expert-reviewed | pseudo-code (IDA) | – | https://github.com/Sxxxw/BinaryLLMs-Eval | **UNVERIFIED** whether ELF shipped |
| **BinaryCorp** (jTrans, ISSTA'22; used by llasm TOSEM'25) | 26M fns (BinaryCorp-26M) / 3M | x64 ArchLinux pkgs, gcc+clang, O0–O3/Os | similarity task; project split | raw binaries on private cloud (cloud.vul337.team) | MIT (code) | https://github.com/vul337/jTrans | **MAYBE** (huge; availability of private host not guaranteed) |
| **ALLSTAR** (JHU/APL, 2019; used by AGNOMIN ACSAC'25 as DAB-9k) | >30,000 Debian Jessie pkgs; unstripped ELF + source + .o | amd64/i386/armel/mips/ppc/s390x; Debian-maintainer flags | none (raw corpus) | **Yes** (rsync/HTTP) | not stated | https://allstar.jhuapl.edu/ | **YES** as a source for a Debian cross-project set with published manifest |
| Debin (CCS'18) | "thousands" of Debian binaries | x86/x64/ARM | – | **No** (models only) | Apache-2.0 | https://github.com/eth-sri/debin | NO |

Key takeaway: only **SymGen, SymLM, NERO, REBench, ALLSTAR** ship ELF files today. The XFL/BLens
Debian corpus — the one with the richest cross-paper comparison table — cannot be downloaded.

---------------------------------------------------------------------------------------------------
## (ii) SOTA table (function-name recovery; sub-token/word-level F1 unless noted)

| Paper (venue, year) | Benchmark / split | Metric | Value | LLM? | Decompiler? |
|---|---|---|---|---|---|
| NERO (OOPSLA'20) | own GNU, package split | F1 | 0.455 | no | IDA for boundaries |
| XFL (S&P'23) | Punstrip Debian, random 90:5:5, L=1024 | micro P/R/F1 | 0.835/0.575/**0.681**; Nero set F1 0.582 | no | Ghidra/radare boundaries |
| SymLM (CCS'22) | own, binary split, x64 avg | P/R/F1 | 0.726/0.764/**0.751** (unknown-binary eval "<0.10") | no (Trex enc.) | Ghidra |
| Epitome (FSE'24) | own 2.6M, source-level 5-fold | P/R/F1 | 0.731/0.708/**0.720**; on SymLM-x64: Epitome 0.887 vs SymLM 0.773 | no | IDA Pro |
| AsmDepictor (AsiaCCS'23) | own, function split | F1 / Jaccard* | 0.715 / 0.754 (SymGen re-eval after dedup: 0.05) | no | IDA |
| SymGen (NDSS'25) | own dedup, binary split | P/R/F1 | all-arch **0.277**; x86-64 0.373/0.388/**0.380**; SymLM 0.10, XFL 0.16, AsmDep 0.05 on same set | yes (CodeLlama-34B LoRA) | Ghidra |
| BLens (USENIX Sec'25) | Punstrip cross-binary / cross-project / strict | F1 (RougeL, Bleu, VarCLR) | **0.772 / 0.460 / 0.294**; SymLM 0.704/0.277/0.195; XFL 0.625/0.295/0.085; AsmDep 0.407/0.200/0.090 | no (CLAP/PalmTree/DEXTER ensemble) | IDA Pro 7.6 |
| llasm (TOSEM'25) | BinaryCorp + Debin sets | P/R/F1 | "up to +19.9/40.7/36.5% over SOTA" — absolute values **UNVERIFIED** (paywalled) | yes (asm-encoder + decoder LLM) | no |
| Shang et al. BinaryLLMs-Eval (EMSE'25, arXiv 2504.21803) | 12 unseen projects | token P/R/F1 (×100) | zero-shot CodeLlama-34B **26.75** (x64); SymLM 6.66, NER 11.17, HexT5 3.00; fine-tune +6.3 F1 avg | yes | IDA |
| REBench (AIWare'26, 2026-04) | own, function-level, LoRA-tuned 13–15B LLMs | P/R/F1 | x64 O0 best **0.156** (Phi4), Qwen2.5 0.139, CodeLlama 0.065; assembly-input lower | yes | Ghidra/IDA |
| SemFlow (Cybersecurity, 2026-05-19) | SymGen dataset (per abstract), 4 arch × O0–O3 | F1 rel. to SymGen | up to +13.4% (x86-64), +20.3% (x86-32), +34.3% (ARM), +158.5% (MIPS); absolute **UNVERIFIED** (only abstract accessible) | yes (call-graph propagation over LLM) | yes |
| SYMSEM (OpenReview E14Vr16HwD, under review) | – | – | CodeLlama-34B "self-transformative" augmentation; numbers **UNVERIFIED** (Cloudflare wall) | yes | yes |
| BinLLM (IPCCC'25, 2025-11) | own | – | **UNVERIFIED** (paywalled) | yes | – |
| AGNOMIN (ACSAC'25, arXiv 2509.25514) | ALLSTAR DAB-9k, package split 9:1 | P@k/R@k (no F1) | +27% P@5, +56% R@5 vs XFL/SymLM/CFG2VEC | no | Ghidra PCode |
| ReCopilot (arXiv 2505.16366, industry) | own file-level bench | Rouge | not comparable | yes (Qwen) | IDA 9.0 |
| SoK AI-Augmented Binary Reversing (arXiv 2606.17398, 2026-06-16) | survey of 144 papers | – | lists only XFL/SymGen/SymLM/BLens under "function symbol name recovery"; flags train–test leakage as "persistent threat to validity" | – | – |

Cross-paper comparability that actually exists:
* **SymGen dataset**: SymGen, SymLM, XFL, AsmDepictor (all in SymGen Tab. I) + SemFlow'26 (relative). No canonical split → "same dataset, own split".
* **SymLM x64 dataset**: SymLM (own), Epitome (5-fold rerun of both).
* **Punstrip Debian**: XFL, BLens, SymLM, AsmDepictor, HexT5 (BLens Tabs 1/3/5, logs on Zenodo) — but binaries unavailable.
* Same model, different papers → wildly different SymLM F1: 0.751 (own) / 0.773 (Epitome) / 0.704→0.277→0.195 (BLens tiers) / 0.10 (SymGen dedup) / 0.067 (Shang). Any "SOTA table" must be per-benchmark, never pooled.

---------------------------------------------------------------------------------------------------
## (iii) Leakage-control practice

| Paper | Split unit | Dedup | Reports seen/unseen-name or overlap breakdown? | Cross-project definition |
|---|---|---|---|---|
| NERO'20 | package | removed multi-version pkgs, C++, test binaries, static linking (cites Allamanis'18) | no | train/val/test "from completely separate projects and packages" |
| XFL'23 | random binaries 90:5:5 | body-level: "function bodies are never shared" between train/test | **yes** — Tab. V restricts test to names absent from training; states names like hash/get_line/usage recur across sets | none (Debian is inherently multi-project, but split is not project-aware) |
| SymLM'22 | binary 8:1:1 | none (author-confirmed in issues; SymGen: many dups) | partial: "unknown binaries" eval on NERO non-leaked functions; notes 11.8% eval vocab OOV | uses NERO's test set as unknown binaries |
| AsmDepictor'23 | function | exact-input dedup | no | none |
| Epitome'24 | source (all opts of one source together), 5-fold | implied by source split | "domain-knowledge shared/deviated" unseen-project sets (accuracy) | new projects akin/dissimilar to training domains |
| SymGen'25 | binary 8:1:1 | (i) same name OR identical body → 1 copy; (ii) callee-normalised body (Alg. 1); ablation with/without in-dataset dups (2.9× effect); membership-inference probe of Code Llama (EM 0, CodeBLEU 0.098) | no explicit seen-name split, but name-dedup makes test names largely train-disjoint | not defined |
| BLens'25 | binary vs **project** vs strict | strict: hash dups + 20 free functions + prefix/word exclusion list; explicitly says removing all name-overlapping test functions would be wrong (main, print_help) | **yes** — 3-tier ladder 0.772/0.460/0.294 + per-word tables | "an entire project is allocated to one of the sets" (Debian package = project) |
| Shang'25 (BinaryLLMs-Eval) | project-disjoint benchmark | check no benchmark fn in fine-tune set | no | 12 GitHub projects vs 51 GNU fine-tune projects |
| REBench'26 | random function | code-similarity dedup on asm+decompiled; PLT/trivial removed | no | none; argues decompilation transforms mitigate LLM contamination |
| AGNOMIN'25 | package 9:1 | "randomly splitting … led to data leakage issues" | no | package-level |
| SoK'26 | – | recommends dedup + split construction; leakage = persistent threat | – | – |

Nobody publishes a full "verbatim-name overlap %" statistic for their test set; the closest are XFL Tab. V
(unseen-name subset) and BLens strict (name-in-train ∧ excluded-word). Our W1 audit (88% verbatim overlap)
would be the first quantified report — a strength if framed as a dataset-card field.

---------------------------------------------------------------------------------------------------
## (iv) Metrics practice

* **Sub-token/word P/R/F1** — universal (NERO: case/order/duplicate-insensitive, alphabetic only; SymLM: word-set with
  CodeWordNet synonym/morphology matching counted as correct; SymGen/Epitome/Shang: token-set P/R/F1; REBench: sub-token
  decomposition + WordNet synonyms + exact-match tiers).
* **Ranking metrics** — XFL/AGNOMIN: P@k, PSP@k, nDCG@k, CG@k (multi-label framing).
* **Order/semantic metrics** — BLens: RougeL, smoothed BLEU-4, VarCLR embedding similarity (BERTScore discussed);
  ReCopilot: Rouge; AsmDepictor: Jaccard*.
* **Exact match** — rarely a headline; SymGen uses EM only for the contamination probe; edit-distance similarity is
  used by us but not by these papers.
* **Selective prediction / calibration** — BLens tunes a confidence threshold on validation (0.194 cross-project, 0.398
  cross-binary; empty predictions allowed) and reports precision separately; XFL tunes threshold p_t for F1. No paper
  reports ECE/reliability diagrams or risk–coverage curves. "Coverage"/abstention rates appear only implicitly
  (BLens case study "___ empty prediction"; XFL 60.7% abstention in BLens reproduction).
* **Free functions** — BLens/XFL count 20 statically inferable functions (csu_init …) as correct; disclose if adopted.

---------------------------------------------------------------------------------------------------
## (v) Recommendation

**Adopt two external benchmarks:**

1. **SymGen dataset (Zenodo 15694344), x86-64 subset, our own published split** — primary. Reasons: only large,
   modern (GCC 9.4, O0–O3, 33 projects, ~560K x86-64 fns est. = 2.24M/4, verify) corpus with raw ELF + stripped/debug
   pairs, CC-BY-4.0, BAP-consumable; comes with SymGen LoRA weights (we already have the FT env), and SymGen's Table I
   gives SymLM/XFL/AsmDepictor numbers on it; SemFlow'26 also reports on it. Because SymGen's split is unseeded, we
   must (a) publish `split.json` (binary-level 8:1:1 replicating their protocol AND a project-level variant), (b) rerun
   SymGen LoRA on our split, (c) apply their name-or-body + callee-normalised dedup and report both with/without.
   Caveat to state: heavy GNU overlap with our own corpus → this is an in-distribution benchmark; cross-project claims
   come from a project-held-out split, not from SymGen's protocol.
2. **SymLM dataset (Zenodo 8306055), x64 subset** — secondary, specifically to answer the reviewer's "compare with
   SymLM and Epitome": Epitome ran 5-fold on exactly this subset (0.887 vs SymLM 0.773), so reporting our 5-fold F1
   there is the only route to Epitome comparability (Epitome's own data is unreleased). Same caveats: no split file,
   no dedup in the original; publish ours. (SymLM's model is bit-rotted — cite BLens/SymGen/Epitome reproductions.)
   Cheap add-on: **NERO** (40 MB, package split, CC-BY-4.0) as the legacy unseen-binary set that SymLM/XFL both used.

**Do not** try to reproduce the Punstrip/BLens Debian corpus (binaries not hosted; Debian versions drift); cite BLens'
three-tier ladder as protocol precedent and, if a Debian-style cross-project set is wanted, build it from **ALLSTAR**
with a published package manifest and sha256s.

**Dataset card must contain (to survive review):**
* Package list with exact versions/commit hashes; compiler versions (GCC/Clang) and full flags per opt level; arch.
* Per-binary sha256 for stripped and debug builds; ELF-vs-BIR consistency check (we already found 15 mismatched
  `_stripped` files — the card must certify zero).
* Function counts per split and per opt level; split unit (project vs binary) with `split.json`; the "GNU/gnulib shared
  component" policy (BLens-style exclusion list) with counts removed.
* Dedup policy and effect: exact-body hash, name+body, callee-normalised (SymGen Alg. 1), and results with/without.
* **Verbatim-name overlap statistics**: % test functions whose name appears in train; % with byte-identical body in
  train; F1 broken down seen-name / unseen-name / novel-vocabulary (XFL Tab. V + BLens strict as precedent).
* Label extraction (nm/DWARF, symbol binding filters: drop size-0, overlapping, local symbols as XFL/BLens do); name
  tokenizer + vocab; free-function list if any.
* Tool provenance (BAP version, Ghidra version for baselines), metric definitions (P/R/F1 variant, EM, EdSim, NgSim),
  selective-prediction curve (risk–coverage) since no prior paper reports calibration.
* For LLM baselines: contamination probe (SymGen-style membership inference) and note that CodeLlama likely saw the
  GNU sources.
* Licenses: source packages (GPL etc.), released artifacts (CC-BY-4.0 like SymGen/SymLM/REBench), and the redistribution
  basis for compiled binaries.

Files fetched for verification live in this directory: xfl.txt, blens.txt, symgen.txt, symlm.txt, epitome.txt,
nero.txt, rebench.txt, emp.txt, agnomin.txt, sok.txt (pdftotext dumps).

## Correction 2026-09-03 — Punstrip/XFL/BLens Debian corpus IS obtainable by rebuild
The August entry marked the XFL/Punstrip corpus "not obtainable". Verified today: (1) `lmu-plai/xfl`
ships `xfl/evaluation/dataset.txt` (manifest of all 10,047 binaries as Debian package + install
path, 3,465 packages) and `dataset_eval_split.dill` (9,042/502/503 binary split); (2) the BLens
Zenodo artifact (10.5281/zenodo.15119877, already at
`baselines/blens_user_env/zenodo/data.tar.gz` on HPC) contains the function-level ground truth
for the cross-project setting (train 394,985 fns/9,042 bins/3,112 pkgs; val 18,081/367/173; test
23,875/451/174; records = bin path, address, name, tokenised name), the cross-binary and strict
splits, embeddings, and raw logs of BLens/XFL/SymLM/AsmDepictor/HexT5; (3) `punstrip/debian-unstripped`
has the build scripts (apt download pkg + pkg-dbgsym from Debian Sid, eu-unstrip). Only the ELF
files are withheld (BLens: GPL-3 redistribution; Punstrip: hosting space). Package versions are not
pinned (Sid, ~2019-20) → rebuild from snapshot.debian.org and verify per binary against the
(address, name) manifest. Estimated 1–2 weeks incl. Ghidra decompile of ~10K binaries and a
CodeT5p fine-tune on the Punstrip train split. User-gated; not started.
