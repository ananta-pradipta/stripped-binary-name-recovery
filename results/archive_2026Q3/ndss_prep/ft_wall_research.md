# FT-wall second research pass (2026-08-06): mechanism families beyond composition/OOV

Full agent deliverable preserved below-in-summary; ranked against the PROVEN washout mechanism,
the 7-null list, and the churn/capture measurements. See Discord relay + experiment_log.

## Ranked
1. **Two-encoder inference** (zero training, testable tonight): frozen PRETRAINED encoder for the
   k-NN index/queries, FT model for decoding. Washout sidestepped by construction. Lit: RETRO
   (frozen retriever, ICML'22), LP-FT (ICLR'22), frozen-PLM retrieval (AAAI'23 wksp), the whole
   BCSD field (SAFE/Asm2Vec/jTrans; Marcelli USENIX'22) = existence proof for never-CE-finetuned
   retrieval spaces. Decisive: capture@50 on dash/gettext under {FT, pretrained, RRF fusion}
   indices, nginx guard. Risk: pretrained f lacks fusion context; O0/O2 objective never pulled
   cross-package homologs together — raw capture may be worse; fusion/LP-head mitigations.
2. **WiSE-FT α-interpolation** of shared encoder weights (pretrained↔FT), retrieval-only
   (CVPR'22; model soups ICML'22 — strongest OOD evidence in the robust-FT family; L2-SP/R3F/
   ULMFiT have weak/no OOD evidence). Same harness as #1: capture-vs-α curve; mid-α hump =
   deployable immediately. Risk: linear connectivity may fail (check norms at α=0.5 first).
3. **Whole-binary call-graph alignment name transfer** (2-4 days): seeded graph matching
   (PLT anchors) query↔top-k training binaries; names transferred through confident alignments.
   Lit: DeepBinDiff NDSS'20, SigmaDiff NDSS'24, QBinDiff belief-prop ASE'21, Punstrip ACSAC'20
   (CRF over call graph > per-function). The only family injecting global structural evidence
   no per-function system (ours/SymGen/BLens) uses — consistent with all three hitting the wall.
   Decisive: dash↔busybox single-pair alignment, name-F1 on aligned subset.
4. **Test-time contrastive adaptation** on the target binary (TTT++/TENT-parameterized,
   LN-only + moment matching; positives = the target's own O0-O3 quadruplets): the one
   intervention point CE can never wash out (runs after CE). Risk: query-index calibration
   drift (queries move, index doesn't); collapse monitors required.
5. **Package-episodic FT with homolog retrieval supervision** (the one retraining bet; re-ID
   episodic-DG literature is the credible precedent; DomainBed kills generic DG regularizers).
   5-epoch learnability gate on non-gnulib held-out name-mates before any full run.
6. **DFR-style frozen-encoder head retrain** ({frozen-pretrained, frozen-FT} × balanced
   name-mate positives): doubles as THE diagnostic for whether washout is head-deep or
   encoder-deep (hours; paper-grade either way).
7. Late-interaction (ColBERT MaxSim) rerank over block/context vectors of the top-1000
   (BEIR OOD-robustness precedent); needs block-IDF to avoid boilerplate matching.
8. LP-FT/anchored refit — only if #2 shows a mid-α hump; risk = slow-motion washout (8th null).

Filtered out with reasons: IRM/GroupDRO/DANN-on-package (DomainBed: nothing beats tuned ERM by
>1pt; no retrieval-task wins outside re-ID), TENT entropy objective (documented collapse under
skewed shift), anything pretraining-only (washout makes it moot).

Tonight sequencing: #1 + #2 share one re-embedding harness → one script, six index variants,
one capture@50 table. This week: #3's single-pair probe + #6's 2x1 diagnostic before any retrain.
