# HANDOFF (2026-08-27)
State: experiment ladder complete except user-deferred baseline retrains; paper user-gated.
- Numbers: results/dualhead_v2/FINAL_TABLES.md (audited); job ids: results/dualhead_v2/RESULTS_LEDGER.md; narrative log: results/experiment_log.md (top state pointer + sprint summary at bottom).
- Final heads: BAP retrieval encoder = C1 soft λ1.0 (Wulver dh2/checkpoints/c1_soft_l10_seed42.pt); generation head = A4 run 1 (dh2/checkpoints/a4_codet5p220m_v1/best; run 2 = a4_codet5p220m_symgen_v2); router = GBT on 11 features (scripts/router2_eval.py, fit on val).
- Headline: test 0.2052/0.4387 (micro/macro), oracle 0.2241/0.4713, selective F1 0.880@10% / 0.690@20%.
- Wulver workspace: /project/hz79/_shared/cs785/dh2 (scripts mirrored here under scripts/, sbatch templates under scripts/dh2_sbatch/). Data v3: /project/hz79/_shared/cs785/relift_ws/data (match_index_v3, split_v3, votes_v3); cache dh2/data_cache/corpus_v3.pkl.
- Memory (auto): ~/.claude/projects/-home-apradipta-cs785-project/memory/project_fundamental_redesign_20260824.md (READ FIRST), feedback_audit_completed_jobs.md, feedback_wulver_job_config.md (node/memory traps).
- Only job running: C1 λ0.3 training (1198921). Its best epoch moved to 9 → when it ends, run the final chain (eval/embdump/a1zt/router_c1soft_l03.sbatch + union_c1soft_l03.sbatch; exact commands in memory project_fundamental_redesign_20260824.md).
