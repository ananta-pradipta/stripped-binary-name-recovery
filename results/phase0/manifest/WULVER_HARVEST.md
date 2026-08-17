# Wulver debug-ELF harvest (2026-08-17)

Goal: bring every unstripped x86-64 ELF (has `.symtab`) that exists under `/project/hz79/_shared/cs785` on Wulver but has no local debug ELF into `data/raw_wulver/<id>` for `scripts/relift_v2.py`. Wulver was read-only throughout; nothing else in the repo was modified.

## Method

1. Walked the Wulver tree with a pure-python ELF header parser (magic, class, e_machine=62, section names, `.note.gnu.build-id`); skipped python envs, HF caches, JDK/Ghidra, tarball stages, graphs/labels/bir dirs. Directories scanned: `data/{debug,raw,cross_project,stripped,string_lexicon,external_calls}`, `ccs`, `clang_leaky_ws`, `clang_o1o3`, `data_clang`, `data_clang_train`, `data_hybrid`, `strlex_ws`, `demo`, `build_tmp`, `results`, `scripts`, `src`, `configs`, `tools`, `home_archive_adp232/{icfg,slurm_logs}`, `baselines/{SymLM,SymGen,blens,AsmDepictor,xfl,llasm,prorec,blens_user_env/{bin_chunks,bin_chunks_xproj,blens_data}}`. Also checked `~adp232` on Wulver (unrelated MASTER/StockMixer only) and `/project/hz79/_shared` (only `cs785`).
2. Kept only x86-64 ELF64 with `.symtab` (`has_symtab=1`); dropped build-tree duplicates (`clang_o1o3/build/**`, `demo/build/**`, `build_tmp/**` — the latter are byte-identical to the `*_O3` files in `data/raw`, e.g. `build_tmp/gawk-5.2.2/gawk` == `data/raw/gawk2_gawk_O3`; kept in the TSV as `build-tree-dup` for provenance) plus openssl test/fuzz binaries, gnulib-tests, ghidra, sample binaries.
3. Local inventory = `corpus_manifest.tsv` `has_debug_elf` paths + `data/raw/<id>[_sym]` + `data/cross_project/{debug,candidates}` + `ftdomains*/bins/*.debug` + `clang_*/bins/*.debug` + `demo/raw/*_sym` (1,617 ids).
4. Full listing: `results/phase0/manifest/wulver_debug_elfs.tsv` (380 rows; columns id, path, size, has_symtab, has_debug_sections, build_id, elf_type, source_group, local_path, local_build_id, status).

## Result summary

- Wulver debug ELFs found (with .symtab, after de-dup of build trees): 353 files, 353 distinct ids.
- **Copied (no local debug ELF): 283 files, 1338.0 MB (1337990744 bytes)** into `data/raw_wulver/`. All 283 verified after copy (size + build-id + `.symtab` present). No `_sym` renaming was needed (all Wulver basenames already equal the id).
- Local debug ELF exists and matches Wulver: 8 (cflow x4 same build-id; nginx118/tengine/... clang bins identical size, no build-id).
- Local debug ELF exists but Wulver copy is a DIFFERENT build (report only, NOT copied): see table below.

## Copied files per package

| package | files | MB | Wulver source | build-id check vs local stripped ELF |
|---|---|---|---|---|
| angie | 4 | 16.5 | SymLM/xproj_debug | bid-verified |
| atop | 8 | 4.9 | data/debug | no-buildid-on-wulver |
| bdb | 40 | 617.2 | data/debug | no-buildid-on-wulver |
| cvs | 4 | 11.0 | cross_project/debug_unseen | no-local-stripped-bid |
| diffutils | 16 | 4.9 | data/raw | no-buildid-on-wulver |
| findutils2 | 8 | 5.8 | data/raw | no-local-stripped-bid |
| fossil | 4 | 58.6 | cross_project/debug_unseen | BID-MISMATCH-vs-local-stripped, no-local-stripped-bid |
| gawk2 | 4 | 9.5 | data/raw | no-buildid-on-wulver |
| gdbm | 12 | 2.2 | data/debug | no-buildid-on-wulver |
| grep2 | 4 | 2.6 | data/raw | no-local-stripped-bid |
| gzip2 | 4 | 1.1 | data/raw | no-local-stripped-bid |
| icu | 44 | 410.4 | data/debug | no-buildid-on-wulver |
| iotop | 4 | 1.6 | data/debug | no-buildid-on-wulver |
| libxml2 | 6 | 24.7 | data/debug | no-buildid-on-wulver |
| lighttpd | 4 | 6.0 | cross_project/debug_unseen | no-local-stripped-bid |
| lsof | 4 | 2.7 | data/debug | no-buildid-on-wulver |
| mksh | 4 | 4.1 | data/debug | no-buildid-on-wulver |
| mutt | 8 | 10.8 | data/debug | no-buildid-on-wulver |
| nginx114 | 2 | 6.9 | SymLM/xproj_debug | no-local-stripped-bid |
| nginx118 | 4 | 14.8 | SymLM/xproj_debug | bid-verified |
| nginx126 | 2 | 7.3 | SymLM/xproj_debug | no-local-stripped-bid |
| openresty | 4 | 40.0 | SymLM/xproj_debug | no-local-stripped-bid |
| patch2 | 4 | 2.0 | data/raw | no-local-stripped-bid |
| procps | 8 | 0.5 | data/debug | no-buildid-on-wulver |
| sed2 | 4 | 2.9 | data/raw | no-local-stripped-bid |
| sysstat | 28 | 10.6 | data/debug | no-buildid-on-wulver |
| tar2 | 4 | 7.7 | data/raw | no-local-stripped-bid |
| tcsh | 3 | 4.0 | data/debug | no-buildid-on-wulver |
| tdb | 16 | 0.6 | data/debug | no-buildid-on-wulver |
| tengine | 2 | 8.9 | SymLM/xproj_debug | bid-verified |
| tig | 4 | 6.1 | data/debug | no-buildid-on-wulver |
| tinycc | 4 | 3.9 | cross_project/debug_unseen | no-local-stripped-bid |
| units2 | 4 | 1.0 | data/raw | no-local-stripped-bid |
| which2 | 4 | 0.3 | data/raw | no-local-stripped-bid |
| zstd | 4 | 25.7 | data/debug | no-buildid-on-wulver |
| **total** | **283** | **1338.0** | | |

Notes on the build-id column: Wulver-compiled binaries in `data/debug` and `data/raw` were linked without `--build-id` (annobin `.gnu.build.attributes` only), so they cannot be build-id-verified against local stripped ELFs; `relift_v2.py`/`rcdg_stage0_elfcheck.py` should verify by section layout / FDE count. Where both sides had a build-id: angie x4, nginx118 x4, tengine x2 match the local stripped ELFs exactly (`SymLM/xproj_debug` copies are the true debug twins of the local BIRs). **fossil_fossil_O0/O2** (`cross_project/debug_unseen`) have build-ids bf02306b../d77be070.. that do NOT match the local stripped build-ids in the manifest (4364276a../f9c514d1..) — copied anyway (no local debug exists) but treat as a different build.

## Ids where a local debug ELF exists but the Wulver copy differs (NOT copied)

| id | Wulver path | Wulver bid / size | local path | local bid / size |
|---|---|---|---|---|
| libxml2_xmllint_O1 | data/debug/libxml2_xmllint_O1 | - / 4140240 | data/raw/libxml2_xmllint_O1_sym | ab28b5e9fb7b / 221128 |
| libxml2_xmllint_O3 | data/debug/libxml2_xmllint_O3 | - / 5821120 | data/raw/libxml2_xmllint_O3_sym | d2f99a5d53cb / 241072 |
| openssl_openssl_O0 | data/raw/openssl_openssl_O0 | - / 16225296 | data/raw/openssl_openssl_O0 | 4d4eab666a6f / 16998184 |
| openssl_openssl_O1 | data/raw/openssl_openssl_O1 | - / 21181976 | data/raw/openssl_openssl_O1 | 64f342953f21 / 22104080 |
| openssl_openssl_O2 | data/raw/openssl_openssl_O2 | - / 22100136 | data/raw/openssl_openssl_O2 | 50c9e61d1ad6 / 23351088 |
| openssl_openssl_O3 | data/raw/openssl_openssl_O3 | - / 23685664 | data/raw/openssl_openssl_O3 | adbfe350e86a / 24902744 |
| patch_patch | demo/raw/patch_patch_sym | 8af836f2a68e / 741416 | data/raw/patch_patch_sym | fd6fc0ea4eda / 741416 |
| recutils_recdel_O0 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recdel_O0 | 72211e387bd0 / 148608 | data/raw/recutils_recdel_O0_sym | 42ccc1cb38ee / 149320 |
| recutils_recdel_O1 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recdel_O1 | 7961f8771cab / 243832 | data/raw/recutils_recdel_O1_sym | 0adbb23398d0 / 250104 |
| recutils_recdel_O2 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recdel_O2 | 02a9747a7ab3 / 240408 | data/raw/recutils_recdel_O2_sym | 78e77d437da3 / 247584 |
| recutils_recdel_O3 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recdel_O3 | e2ea8571808f / 249944 | data/raw/recutils_recdel_O3_sym | be6e621e75e6 / 257016 |
| recutils_recfix_O0 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recfix_O0 | b84975d76c8f / 148984 | data/raw/recutils_recfix_O0_sym | 303571904144 / 149680 |
| recutils_recfix_O1 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recfix_O1 | 2fa728be8850 / 240256 | data/raw/recutils_recfix_O1_sym | 38e91816159b / 242240 |
| recutils_recfix_O2 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recfix_O2 | c722bb73594f / 232752 | data/raw/recutils_recfix_O2_sym | f6e7ab9ab41d / 239776 |
| recutils_recfix_O3 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recfix_O3 | 7258d8beacaa / 246392 | data/raw/recutils_recfix_O3_sym | 142af2922458 / 253304 |
| recutils_recfmt_O0 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recfmt_O0 | 5239c387cea5 / 149232 | data/raw/recutils_recfmt_O0_sym | 9c0d43456a53 / 149840 |
| recutils_recfmt_O1 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recfmt_O1 | eae5f1074f94 / 241016 | data/raw/recutils_recfmt_O1_sym | e4ffcd621e31 / 247088 |
| recutils_recfmt_O2 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recfmt_O2 | 051180a04ff8 / 233648 | data/raw/recutils_recfmt_O2_sym | a9898cfea846 / 244736 |
| recutils_recfmt_O3 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recfmt_O3 | b641776eeb91 / 248256 | data/raw/recutils_recfmt_O3_sym | 791ba25ddb38 / 254896 |
| recutils_recinf_O0 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recinf_O0 | 9c031cd0d18f / 144424 | data/raw/recutils_recinf_O0_sym | 1b5a4545c96a / 149120 |
| recutils_recinf_O1 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recinf_O1 | 6b76ae109f48 / 243512 | data/raw/recutils_recinf_O1_sym | 819c3b46432a / 245304 |
| recutils_recinf_O2 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recinf_O2 | bac8558b99d4 / 235920 | data/raw/recutils_recinf_O2_sym | 18c73152f506 / 242752 |
| recutils_recinf_O3 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recinf_O3 | d2c389df7455 / 249552 | data/raw/recutils_recinf_O3_sym | cdb423b5707a / 252184 |
| recutils_recins_O0 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recins_O0 | e1aabc9f83ef / 154392 | data/raw/recutils_recins_O0_sym | affeb5dd7df7 / 150992 |
| recutils_recins_O1 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recins_O1 | 74d0589ccc26 / 250160 | data/raw/recutils_recins_O1_sym | 8d99aa82e84b / 256408 |
| recutils_recins_O2 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recins_O2 | dafb060e0865 / 246832 | data/raw/recutils_recins_O2_sym | c4f2c1a003db / 249824 |
| recutils_recins_O3 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recins_O3 | d4b3c8d3f689 / 256560 | data/raw/recutils_recins_O3_sym | 71cfbb304470 / 263544 |
| recutils_recsel_O0 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recsel_O0 | c20db9ea535b / 155400 | data/raw/recutils_recsel_O0_sym | 800de9f55e31 / 156088 |
| recutils_recsel_O1 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recsel_O1 | c4bcc7076dbf / 251616 | data/raw/recutils_recsel_O1_sym | 90384542be40 / 258328 |
| recutils_recsel_O2 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recsel_O2 | 5cccaa240008 / 248424 | data/raw/recutils_recsel_O2_sym | 2d2c40c6a724 / 251800 |
| recutils_recsel_O3 | baselines/SymLM/dataset_generation/xproj_debug/recutils_recsel_O3 | ca8f4ac7e4e7 / 257968 | data/raw/recutils_recsel_O3_sym | beef241e71e7 / 265328 |
| tcsh_tcsh_O2 | data/debug/tcsh_tcsh_O2 | - / 1462560 | data/cross_project/candidates/tcsh_tcsh_O2 | 78051804e6d9 / 1559336 |

- recutils x24 (`baselines/SymLM/dataset_generation/xproj_debug/`): a different recutils build than local `data/raw/recutils_*_sym` (and than the local stripped ELFs / BIRs); ignore for re-lift.
- libxml2_xmllint_O1/O3: Wulver `data/debug` copies are 4.1/5.8 MB (static libxml2 link) vs local 221/241 KB `_sym` (dynamic link) -- different builds; local O0/O2 debug do not exist locally at all and were copied (`libxml2_xmllint_O0`, `libxml2_xmllint_O2`, plus xmlcatalog x4). Local `data/stripped/libxml2_xmllint_O0/O2` are known build-mismatched (see project_rcdg.md), so verify before use.
- tcsh_tcsh_O2: local `data/cross_project/candidates/tcsh_tcsh_O2` (1,559,336 B) vs Wulver `data/debug/tcsh_tcsh_O2` (1,462,560 B), neither has a build-id -- different builds; O0/O1/O3 were copied.
- openssl_openssl_O0..O3: Wulver `data/raw` copies (no build-id, 16.2/21.2/22.1/23.7 MB) vs local (with build-id, 17.0/22.1/23.4/24.9 MB) -- different builds.
- patch_patch (demo, no opt tag): Wulver `demo/raw/patch_patch_sym` bid 8af836f2.. vs local `data/raw/patch_patch_sym` bid fd6fc0ea.. -- different builds. `demo/raw/diffutils_{cmp,diff,diff3,sdiff}_sym` are already present locally in `demo/raw/`.

## Debug ELFs that do NOT exist anywhere on Wulver (searched all locations above)

These ids have graphs/labels/stripped ELFs on Wulver (from `ccs/data` / `data/stripped`, compiled ~Mar 20 or Apr 26 2026) but the unstripped binaries were only ever written to `$DEBUG=/project/hz79/_shared/cs785/data/debug` or a `/tmp/p0_0_6_*_$SLURM_JOB_ID` build dir, and are gone:
- **coreutils2_{dd,df,du,ls,mktemp,realpath,shred,sort,stty}_{O0,O1,O2,O3,default}** -- 45 ids, only `data/stripped/*_stripped` (Mar 20) + labels (200 fns each) remain; no sbatch in `ccs/` references coreutils2 (built by an older `scripts/expand_bap_only.sh` flow); `build_tmp/coreutils-8.32` contains only `src/make-prime-list`.
- **diffutils2_{cmp,diff,diff3,sdiff}_O0..O3** -- 16 ids, stripped + labels only.
- **inetutils2_{dnsdomainname,hostname,logger}_O0..O3** -- 12 ids, stripped + labels only.
- **zsh_zsh_O0..O3** -- `data/debug/zsh_zsh_O*` exist but are STRIPPED (0 `.symtab`, no `.debug_*`; zsh links with `-s`), and `data/labels/zsh_zsh_O*_labels.json` have 0 functions -> zsh debug never existed.
- **nginx114_nginx114_O1/O3, nginx126_nginx126_O1/O3** -- never built (only O0/O2 exist anywhere: labels, stripped, SymLM xproj_debug).
- **dash_dash_clang_O0/O2** -- only labels/graphs in `data_clang/`; the clang bins on Wulver (`clang_o1o3/bins`) are O1/O3 only and are already local. `angie/nginx118/tengine/gettext/psmisc/recutils _clang_O1/O3` likewise already local (identical sizes).
- Sibling `*2_` packages that WERE recovered: gawk2, grep2, gzip2, sed2, tar2, units2, which2, patch2, findutils2 (find, xargs), diffutils_(cmp|diff|diff3|sdiff)_O0..O3 -- all from Wulver `data/raw/` (56 files).

## Exact commands

```bash
# 1. scan (pure-python ELF parser streamed to Wulver, output captured locally; run per top-level dir)
ssh wulver 'python3 - /project/hz79/_shared/cs785/<dir>' < scratchpad/scan.py > scratchpad/scan_<dir>.tsv
# 2. compare with local inventory -> results/phase0/manifest/wulver_debug_elfs.tsv + files_from.txt (283 relative paths)
# 3. copy (flattened basenames == ids; no rename needed)
rsync -av --no-relative --files-from=scratchpad/files_from.txt wulver:/project/hz79/_shared/cs785/ /home/apradipta/cs785-project/data/raw_wulver/
#   -> sent 14,062 bytes  received 1,338,333,910 bytes; total size 1,337,990,744
# 4. verify: size, .symtab, build-id re-parsed for all 283 -> 283 OK / 0 bad
```

## Source manifest of the copied files (relative to `/project/hz79/_shared/cs785/`)

```
angie_angie_O0	baselines/SymLM/dataset_generation/xproj_debug/angie_angie_O0	3701000	329e88015892d70c80b0802a55f5311e1b03a995
angie_angie_O1	baselines/SymLM/dataset_generation/xproj_debug/angie_angie_O1	4137928	1f107d08f865cb7e91f5d1123579c877f6ace061
angie_angie_O2	baselines/SymLM/dataset_generation/xproj_debug/angie_angie_O2	4261352	eda729b59488fd39755d47b5b34b9135fa3fee5e
angie_angie_O3	baselines/SymLM/dataset_generation/xproj_debug/angie_angie_O3	4386656	452cab28566fcdd4da40cb68298b92eac83d48d8
atop_atop_O0	data/debug/atop_atop_O0	1156928	-
atop_atop_O1	data/debug/atop_atop_O1	1156928	-
atop_atop_O2	data/debug/atop_atop_O2	1156928	-
atop_atop_O3	data/debug/atop_atop_O3	1156928	-
atop_atopacctd_O0	data/debug/atop_atopacctd_O0	61728	-
atop_atopacctd_O1	data/debug/atop_atopacctd_O1	61728	-
atop_atopacctd_O2	data/debug/atop_atopacctd_O2	61728	-
atop_atopacctd_O3	data/debug/atop_atopacctd_O3	61728	-
bdb_db_archive_O0	data/debug/bdb_db_archive_O0	13904832	-
bdb_db_archive_O1	data/debug/bdb_db_archive_O1	15379904	-
bdb_db_archive_O2	data/debug/bdb_db_archive_O2	15818840	-
bdb_db_archive_O3	data/debug/bdb_db_archive_O3	16373568	-
bdb_db_checkpoint_O0	data/debug/bdb_db_checkpoint_O0	13905456	-
bdb_db_checkpoint_O1	data/debug/bdb_db_checkpoint_O1	15381664	-
bdb_db_checkpoint_O2	data/debug/bdb_db_checkpoint_O2	15820184	-
bdb_db_checkpoint_O3	data/debug/bdb_db_checkpoint_O3	16374928	-
bdb_db_dump_O0	data/debug/bdb_db_dump_O0	13912048	-
bdb_db_dump_O1	data/debug/bdb_db_dump_O1	15392384	-
bdb_db_dump_O2	data/debug/bdb_db_dump_O2	15830648	-
bdb_db_dump_O3	data/debug/bdb_db_dump_O3	16385560	-
bdb_db_hotbackup_O0	data/debug/bdb_db_hotbackup_O0	13912368	-
bdb_db_hotbackup_O1	data/debug/bdb_db_hotbackup_O1	15396552	-
bdb_db_hotbackup_O2	data/debug/bdb_db_hotbackup_O2	15831184	-
bdb_db_hotbackup_O3	data/debug/bdb_db_hotbackup_O3	16386224	-
bdb_db_load_O0	data/debug/bdb_db_load_O0	13937584	-
bdb_db_load_O1	data/debug/bdb_db_load_O1	15429112	-
bdb_db_load_O2	data/debug/bdb_db_load_O2	15866896	-
bdb_db_load_O3	data/debug/bdb_db_load_O3	16432224	-
bdb_db_printlog_O0	data/debug/bdb_db_printlog_O0	14402072	-
bdb_db_printlog_O1	data/debug/bdb_db_printlog_O1	15904336	-
bdb_db_printlog_O2	data/debug/bdb_db_printlog_O2	16346336	-
bdb_db_printlog_O3	data/debug/bdb_db_printlog_O3	16901224	-
bdb_db_recover_O0	data/debug/bdb_db_recover_O0	13910456	-
bdb_db_recover_O1	data/debug/bdb_db_recover_O1	15383016	-
bdb_db_recover_O2	data/debug/bdb_db_recover_O2	15822184	-
bdb_db_recover_O3	data/debug/bdb_db_recover_O3	16376888	-
bdb_db_replicate_O0	data/debug/bdb_db_replicate_O0	13910832	-
bdb_db_replicate_O1	data/debug/bdb_db_replicate_O1	15389368	-
bdb_db_replicate_O2	data/debug/bdb_db_replicate_O2	15824392	-
bdb_db_replicate_O3	data/debug/bdb_db_replicate_O3	16379000	-
bdb_db_stat_O0	data/debug/bdb_db_stat_O0	13910656	-
bdb_db_stat_O1	data/debug/bdb_db_stat_O1	15388400	-
bdb_db_stat_O2	data/debug/bdb_db_stat_O2	15823424	-
bdb_db_stat_O3	data/debug/bdb_db_stat_O3	16382128	-
bdb_db_verify_O0	data/debug/bdb_db_verify_O0	13905288	-
bdb_db_verify_O1	data/debug/bdb_db_verify_O1	15381680	-
bdb_db_verify_O2	data/debug/bdb_db_verify_O2	15820392	-
bdb_db_verify_O3	data/debug/bdb_db_verify_O3	16375120	-
cvs_cvs_O0	data/cross_project/debug_unseen/cvs_cvs_O0	1946408	1f54a0ba4b2aa4ffc29e725d2e7bce2e1e87ce42
cvs_cvs_O1	data/cross_project/debug_unseen/cvs_cvs_O1	2716248	7c74d9c284ed9ff72cebd2e263831caff134768d
cvs_cvs_O2	data/cross_project/debug_unseen/cvs_cvs_O2	2989376	2a8e1d013d445de6393451261a61bb2ea7f4b75f
cvs_cvs_O3	data/cross_project/debug_unseen/cvs_cvs_O3	3354288	83c153427ba21fbc70aebd5adef06bffededae6b
diffutils_cmp_O0	data/raw/diffutils_cmp_O0	155848	-
diffutils_cmp_O1	data/raw/diffutils_cmp_O1	190832	-
diffutils_cmp_O2	data/raw/diffutils_cmp_O2	205488	-
diffutils_cmp_O3	data/raw/diffutils_cmp_O3	224472	-
diffutils_diff3_O0	data/raw/diffutils_diff3_O0	189224	-
diffutils_diff3_O1	data/raw/diffutils_diff3_O1	245168	-
diffutils_diff3_O2	data/raw/diffutils_diff3_O2	278648	-
diffutils_diff3_O3	data/raw/diffutils_diff3_O3	335992	-
diffutils_diff_O0	data/raw/diffutils_diff_O0	417200	-
diffutils_diff_O1	data/raw/diffutils_diff_O1	542336	-
diffutils_diff_O2	data/raw/diffutils_diff_O2	607720	-
diffutils_diff_O3	data/raw/diffutils_diff_O3	701888	-
diffutils_sdiff_O0	data/raw/diffutils_sdiff_O0	167672	-
diffutils_sdiff_O1	data/raw/diffutils_sdiff_O1	197664	-
diffutils_sdiff_O2	data/raw/diffutils_sdiff_O2	225824	-
diffutils_sdiff_O3	data/raw/diffutils_sdiff_O3	245672	-
findutils2_find_O0	data/raw/findutils2_find_O0	839192	ccab27de9c94ad3cda27cb4611c97bf2db27c5c7
findutils2_find_O1	data/raw/findutils2_find_O1	1094296	b9647ae627c8da782851cbbfcf78ee1fa91f8513
findutils2_find_O2	data/raw/findutils2_find_O2	1240232	d4469e2067ed301eae32d8bf1abce82d2c5e7477
findutils2_find_O3	data/raw/findutils2_find_O3	1521344	bb4a8395885c2fee31c6a9f49cb7dfdae0444dda
findutils2_xargs_O0	data/raw/findutils2_xargs_O0	198736	634dde95a56473227ac2d0ec931e4ddf459b45e1
findutils2_xargs_O1	data/raw/findutils2_xargs_O1	248680	733d35399f86cdcebaf4b6109f6a4b3385a8d870
findutils2_xargs_O2	data/raw/findutils2_xargs_O2	283112	bac2bf5f89a3ed02ddd83f93fe6cea377a5931a2
findutils2_xargs_O3	data/raw/findutils2_xargs_O3	329392	18eb3dba1a3ff512a90e63d17d167899eeb7649b
fossil_fossil_O0	data/cross_project/debug_unseen/fossil_fossil_O0	9155616	bf02306b3669797c0e5fdcd8f811996d762c118a
fossil_fossil_O1	data/cross_project/debug_unseen/fossil_fossil_O1	13037072	3ab983a5c79c76d31ccd3bbfbe08462ee2256797
fossil_fossil_O2	data/cross_project/debug_unseen/fossil_fossil_O2	16654520	d77be070bffb0b37470383bb9b4eaa24b5567c82
fossil_fossil_O3	data/cross_project/debug_unseen/fossil_fossil_O3	19796032	73b0a5f17afd6fc5cbfbf4068905e522db0f770d
gawk2_gawk_O0	data/raw/gawk2_gawk_O0	1468760	-
gawk2_gawk_O1	data/raw/gawk2_gawk_O1	2332904	-
gawk2_gawk_O2	data/raw/gawk2_gawk_O2	2685464	-
gawk2_gawk_O3	data/raw/gawk2_gawk_O3	3000208	-
gdbm_gdbm_dump_O0	data/debug/gdbm_gdbm_dump_O0	72176	-
gdbm_gdbm_dump_O1	data/debug/gdbm_gdbm_dump_O1	85080	-
gdbm_gdbm_dump_O2	data/debug/gdbm_gdbm_dump_O2	88992	-
gdbm_gdbm_dump_O3	data/debug/gdbm_gdbm_dump_O3	92120	-
gdbm_gdbm_load_O0	data/debug/gdbm_gdbm_load_O0	75256	-
gdbm_gdbm_load_O1	data/debug/gdbm_gdbm_load_O1	90224	-
gdbm_gdbm_load_O2	data/debug/gdbm_gdbm_load_O2	98464	-
gdbm_gdbm_load_O3	data/debug/gdbm_gdbm_load_O3	101696	-
gdbm_gdbmtool_O0	data/debug/gdbm_gdbmtool_O0	299536	-
gdbm_gdbmtool_O1	data/debug/gdbm_gdbmtool_O1	371080	-
gdbm_gdbmtool_O2	data/debug/gdbm_gdbmtool_O2	418160	-
gdbm_gdbmtool_O3	data/debug/gdbm_gdbmtool_O3	447976	-
grep2_grep_O0	data/raw/grep2_grep_O0	458056	35b17764437dcef6df66a878975fdea3b0348f5c
grep2_grep_O1	data/raw/grep2_grep_O1	573544	de3c1fb3c7babe6b033cbb994f3bce0c5041fd85
grep2_grep_O2	data/raw/grep2_grep_O2	726104	7d3138675454d8cd6f0cbf86e87a585a71f5d7ea
grep2_grep_O3	data/raw/grep2_grep_O3	868920	b116ec75daa9c4048e8d90d973639f60adca04ba
gzip2_gzip_O0	data/raw/gzip2_gzip_O0	195288	d3cfd54828e4252684b61475e2f3249ff78ca0c0
gzip2_gzip_O1	data/raw/gzip2_gzip_O1	259960	f87f6919a3641c1b57c76cbcc85c36378ed23a07
gzip2_gzip_O2	data/raw/gzip2_gzip_O2	285792	5d291f199ee46135c14ea15bfffb7f794400f65f
gzip2_gzip_O3	data/raw/gzip2_gzip_O3	343528	a3dc397bda5eb0ff332619b1508983ea31be6247
icu_derb_O0	data/debug/icu_derb_O0	18116392	-
icu_derb_O1	data/debug/icu_derb_O1	24191192	-
icu_derb_O2	data/debug/icu_derb_O2	27076832	-
icu_derb_O3	data/debug/icu_derb_O3	30498904	-
icu_genbrk_O0	data/debug/icu_genbrk_O0	6235688	-
icu_genbrk_O1	data/debug/icu_genbrk_O1	7798640	-
icu_genbrk_O2	data/debug/icu_genbrk_O2	8652288	-
icu_genbrk_O3	data/debug/icu_genbrk_O3	9878648	-
icu_gencfu_O0	data/debug/icu_gencfu_O0	6425944	-
icu_gencfu_O1	data/debug/icu_gencfu_O1	8220440	-
icu_gencfu_O2	data/debug/icu_gencfu_O2	9197960	-
icu_gencfu_O3	data/debug/icu_gencfu_O3	10399464	-
icu_gencnval_O0	data/debug/icu_gencnval_O0	644216	-
icu_gencnval_O1	data/debug/icu_gencnval_O1	788344	-
icu_gencnval_O2	data/debug/icu_gencnval_O2	843856	-
icu_gencnval_O3	data/debug/icu_gencnval_O3	951416	-
icu_gennorm2_O0	data/debug/icu_gennorm2_O0	1355280	-
icu_gennorm2_O1	data/debug/icu_gennorm2_O1	1830360	-
icu_gennorm2_O2	data/debug/icu_gennorm2_O2	2008496	-
icu_gennorm2_O3	data/debug/icu_gennorm2_O3	2435320	-
icu_genrb_O0	data/debug/icu_genrb_O0	17997912	-
icu_genrb_O1	data/debug/icu_genrb_O1	24900080	-
icu_genrb_O2	data/debug/icu_genrb_O2	27894312	-
icu_genrb_O3	data/debug/icu_genrb_O3	31368232	-
icu_gensprep_O0	data/debug/icu_gensprep_O0	546656	-
icu_gensprep_O1	data/debug/icu_gensprep_O1	637880	-
icu_gensprep_O2	data/debug/icu_gensprep_O2	677584	-
icu_gensprep_O3	data/debug/icu_gensprep_O3	776584	-
icu_icuinfo_O0	data/debug/icu_icuinfo_O0	13528488	-
icu_icuinfo_O1	data/debug/icu_icuinfo_O1	19093584	-
icu_icuinfo_O2	data/debug/icu_icuinfo_O2	21469040	-
icu_icuinfo_O3	data/debug/icu_icuinfo_O3	24272696	-
icu_icupkg_O0	data/debug/icu_icupkg_O0	3846160	-
icu_icupkg_O1	data/debug/icu_icupkg_O1	5453624	-
icu_icupkg_O2	data/debug/icu_icupkg_O2	6061512	-
icu_icupkg_O3	data/debug/icu_icupkg_O3	7045880	-
icu_makeconv_O0	data/debug/icu_makeconv_O0	764680	-
icu_makeconv_O1	data/debug/icu_makeconv_O1	921200	-
icu_makeconv_O2	data/debug/icu_makeconv_O2	970832	-
icu_makeconv_O3	data/debug/icu_makeconv_O3	1190048	-
icu_pkgdata_O0	data/debug/icu_pkgdata_O0	4059776	-
icu_pkgdata_O1	data/debug/icu_pkgdata_O1	5708584	-
icu_pkgdata_O2	data/debug/icu_pkgdata_O2	6328040	-
icu_pkgdata_O3	data/debug/icu_pkgdata_O3	7313016	-
iotop_iotop_O0	data/debug/iotop_iotop_O0	268680	-
iotop_iotop_O1	data/debug/iotop_iotop_O1	410416	-
iotop_iotop_O2	data/debug/iotop_iotop_O2	423984	-
iotop_iotop_O3	data/debug/iotop_iotop_O3	530176	-
libxml2_xmlcatalog_O0	data/debug/libxml2_xmlcatalog_O0	2980128	-
libxml2_xmlcatalog_O1	data/debug/libxml2_xmlcatalog_O1	3919104	-
libxml2_xmlcatalog_O2	data/debug/libxml2_xmlcatalog_O2	4444824	-
libxml2_xmlcatalog_O3	data/debug/libxml2_xmlcatalog_O3	5546384	-
libxml2_xmllint_O0	data/debug/libxml2_xmllint_O0	3144184	-
libxml2_xmllint_O2	data/debug/libxml2_xmllint_O2	4687896	-
lighttpd_lighttpd_O0	data/cross_project/debug_unseen/lighttpd_lighttpd_O0	1187752	a906d4c4bcabd2d4e09e3adcde6e3ab7a04da01d
lighttpd_lighttpd_O1	data/cross_project/debug_unseen/lighttpd_lighttpd_O1	1472680	7ff0f85adefacab16743ed0d9fa4baf0c94305e6
lighttpd_lighttpd_O2	data/cross_project/debug_unseen/lighttpd_lighttpd_O2	1599144	b6e8d8b3203dfe1249ca8479816eb439ce3761ca
lighttpd_lighttpd_O3	data/cross_project/debug_unseen/lighttpd_lighttpd_O3	1788464	66a2b41321f123120b3be9e4a8231b9c4b54006a
lsof_lsof_O0	data/debug/lsof_lsof_O0	537928	-
lsof_lsof_O1	data/debug/lsof_lsof_O1	673392	-
lsof_lsof_O2	data/debug/lsof_lsof_O2	725040	-
lsof_lsof_O3	data/debug/lsof_lsof_O3	766576	-
mksh_mksh_O0	data/debug/mksh_mksh_O0	717328	-
mksh_mksh_O1	data/debug/mksh_mksh_O1	960928	-
mksh_mksh_O2	data/debug/mksh_mksh_O2	1115432	-
mksh_mksh_O3	data/debug/mksh_mksh_O3	1298592	-
mutt_mutt_O0	data/debug/mutt_mutt_O0	2074488	-
mutt_mutt_O1	data/debug/mutt_mutt_O1	2622360	-
mutt_mutt_O2	data/debug/mutt_mutt_O2	2784400	-
mutt_mutt_O3	data/debug/mutt_mutt_O3	3166264	-
mutt_mutt_dotlock_O0	data/debug/mutt_mutt_dotlock_O0	27608	-
mutt_mutt_dotlock_O1	data/debug/mutt_mutt_dotlock_O1	32544	-
mutt_mutt_dotlock_O2	data/debug/mutt_mutt_dotlock_O2	32896	-
mutt_mutt_dotlock_O3	data/debug/mutt_mutt_dotlock_O3	32896	-
nginx114_nginx114_O0	baselines/SymLM/dataset_generation/xproj_debug/nginx114_nginx114_O0	3193264	207ba4de0728e05df95a9b87fc43b1dbd082c9ea
nginx114_nginx114_O2	baselines/SymLM/dataset_generation/xproj_debug/nginx114_nginx114_O2	3737792	86f4b07b984cc2e21d28956180c6af0895bbb101
nginx118_nginx118_O0	baselines/SymLM/dataset_generation/xproj_debug/nginx118_nginx118_O0	3285776	3694319c9e8096569c0cb4812b3b4c09f2ec017d
nginx118_nginx118_O1	baselines/SymLM/dataset_generation/xproj_debug/nginx118_nginx118_O1	3732784	ba9fb2a6428b05f9ee1269ca2bcbafa39ebdab29
nginx118_nginx118_O2	baselines/SymLM/dataset_generation/xproj_debug/nginx118_nginx118_O2	3833992	19be64d8318f45b381a479861f29c5f529892d9f
nginx118_nginx118_O3	baselines/SymLM/dataset_generation/xproj_debug/nginx118_nginx118_O3	3962472	7a7f147324d225da91479a5717c4cbca8b981c23
nginx126_nginx126_O0	baselines/SymLM/dataset_generation/xproj_debug/nginx126_nginx126_O0	3371920	922ad728fe9587e7a2488a0b5eda7aac0bcd581b
nginx126_nginx126_O2	baselines/SymLM/dataset_generation/xproj_debug/nginx126_nginx126_O2	3934488	472303ea95e80fc107aa5a07547bc9301cb4a553
openresty_nginx_O0	baselines/SymLM/dataset_generation/xproj_debug/openresty_nginx_O0	8901520	ac10e9b1247b961425c29454b4cd388688254abc
openresty_nginx_O1	baselines/SymLM/dataset_generation/xproj_debug/openresty_nginx_O1	10125456	94179428b2708cc6e37f35a834274a19ef94eb1b
openresty_nginx_O2	baselines/SymLM/dataset_generation/xproj_debug/openresty_nginx_O2	10358136	ffd0cc0c2877705618259a2a13e647b8c75f28e4
openresty_nginx_O3	baselines/SymLM/dataset_generation/xproj_debug/openresty_nginx_O3	10637864	1df0a64411630a33459ffadb324b6beab65d7a7f
patch2_patch_O0	data/raw/patch2_patch_O0	387128	7cf4e580c6caafd2ecd6b0fa5e83f886fbf2f373
patch2_patch_O1	data/raw/patch2_patch_O1	487808	fcab3d19d423bbbea5ba8e2075f30bad934ac0dd
patch2_patch_O2	data/raw/patch2_patch_O2	547168	707fbac4e57d12805f6f35537d4ff5ebdb77c0d1
patch2_patch_O3	data/raw/patch2_patch_O3	620448	0ca7faad5d1ec53d8ead257fff458f16fb780cc6
procps_kill_O0	data/debug/procps_kill_O0	36416	-
procps_kill_O1	data/debug/procps_kill_O1	44992	-
procps_kill_O2	data/debug/procps_kill_O2	45728	-
procps_kill_O3	data/debug/procps_kill_O3	47824	-
procps_sysctl_O0	data/debug/procps_sysctl_O0	64224	-
procps_sysctl_O1	data/debug/procps_sysctl_O1	81608	-
procps_sysctl_O2	data/debug/procps_sysctl_O2	83952	-
procps_sysctl_O3	data/debug/procps_sysctl_O3	85368	-
sed2_sed_O0	data/raw/sed2_sed_O0	501912	566a460f2c1b33e57193289d98699792ae57f8f6
sed2_sed_O1	data/raw/sed2_sed_O1	685456	fa7f3183d8fbb139a4a03b70011f69be558d301d
sed2_sed_O2	data/raw/sed2_sed_O2	776864	31c54376d764deb406a7275178379706460e3687
sed2_sed_O3	data/raw/sed2_sed_O3	980648	48016998eaf713fbacdd2f53cd66ee0411e125f5
sysstat_cifsiostat_O0	data/debug/sysstat_cifsiostat_O0	134528	-
sysstat_cifsiostat_O1	data/debug/sysstat_cifsiostat_O1	134528	-
sysstat_cifsiostat_O2	data/debug/sysstat_cifsiostat_O2	134528	-
sysstat_cifsiostat_O3	data/debug/sysstat_cifsiostat_O3	134528	-
sysstat_iostat_O0	data/debug/sysstat_iostat_O0	189496	-
sysstat_iostat_O1	data/debug/sysstat_iostat_O1	189496	-
sysstat_iostat_O2	data/debug/sysstat_iostat_O2	189496	-
sysstat_iostat_O3	data/debug/sysstat_iostat_O3	189496	-
sysstat_mpstat_O0	data/debug/sysstat_mpstat_O0	188984	-
sysstat_mpstat_O1	data/debug/sysstat_mpstat_O1	188984	-
sysstat_mpstat_O2	data/debug/sysstat_mpstat_O2	188984	-
sysstat_mpstat_O3	data/debug/sysstat_mpstat_O3	188984	-
sysstat_pidstat_O0	data/debug/sysstat_pidstat_O0	201176	-
sysstat_pidstat_O1	data/debug/sysstat_pidstat_O1	201176	-
sysstat_pidstat_O2	data/debug/sysstat_pidstat_O2	201176	-
sysstat_pidstat_O3	data/debug/sysstat_pidstat_O3	201176	-
sysstat_sadf_O0	data/debug/sysstat_sadf_O0	1281056	-
sysstat_sadf_O1	data/debug/sysstat_sadf_O1	1281056	-
sysstat_sadf_O2	data/debug/sysstat_sadf_O2	1281056	-
sysstat_sadf_O3	data/debug/sysstat_sadf_O3	1281056	-
sysstat_sar_O0	data/debug/sysstat_sar_O0	526952	-
sysstat_sar_O1	data/debug/sysstat_sar_O1	526952	-
sysstat_sar_O2	data/debug/sysstat_sar_O2	526952	-
sysstat_sar_O3	data/debug/sysstat_sar_O3	526952	-
sysstat_tapestat_O0	data/debug/sysstat_tapestat_O0	131880	-
sysstat_tapestat_O1	data/debug/sysstat_tapestat_O1	131880	-
sysstat_tapestat_O2	data/debug/sysstat_tapestat_O2	131880	-
sysstat_tapestat_O3	data/debug/sysstat_tapestat_O3	131880	-
tar2_tar_O0	data/raw/tar2_tar_O0	1324152	a0e6eb398e1dcd4d75ba54c54be92caf4840735a
tar2_tar_O1	data/raw/tar2_tar_O1	1860760	5cfbee6b268b3467bdd5574e1919172d79fe312f
tar2_tar_O2	data/raw/tar2_tar_O2	2088792	6bef907128b2eea8cff51036c0a4e281b4d2e9fb
tar2_tar_O3	data/raw/tar2_tar_O3	2423424	dc9abddb9409dfeeee52ec76cdc27b2579649529
tcsh_tcsh_O0	data/debug/tcsh_tcsh_O0	1003856	-
tcsh_tcsh_O1	data/debug/tcsh_tcsh_O1	1324552	-
tcsh_tcsh_O3	data/debug/tcsh_tcsh_O3	1707528	-
tdb_tdbbackup_O0	data/debug/tdb_tdbbackup_O0	26056	-
tdb_tdbbackup_O1	data/debug/tdb_tdbbackup_O1	31920	-
tdb_tdbbackup_O2	data/debug/tdb_tdbbackup_O2	32200	-
tdb_tdbbackup_O3	data/debug/tdb_tdbbackup_O3	32976	-
tdb_tdbdump_O0	data/debug/tdb_tdbdump_O0	23592	-
tdb_tdbdump_O1	data/debug/tdb_tdbdump_O1	26464	-
tdb_tdbdump_O2	data/debug/tdb_tdbdump_O2	27176	-
tdb_tdbdump_O3	data/debug/tdb_tdbdump_O3	28704	-
tdb_tdbrestore_O0	data/debug/tdb_tdbrestore_O0	23112	-
tdb_tdbrestore_O1	data/debug/tdb_tdbrestore_O1	26792	-
tdb_tdbrestore_O2	data/debug/tdb_tdbrestore_O2	27744	-
tdb_tdbrestore_O3	data/debug/tdb_tdbrestore_O3	27904	-
tdb_tdbtool_O0	data/debug/tdb_tdbtool_O0	51072	-
tdb_tdbtool_O1	data/debug/tdb_tdbtool_O1	60208	-
tdb_tdbtool_O2	data/debug/tdb_tdbtool_O2	58704	-
tdb_tdbtool_O3	data/debug/tdb_tdbtool_O3	62768	-
tengine_nginx_O0	baselines/SymLM/dataset_generation/xproj_debug/tengine_nginx_O0	4076960	5cc9387de097f71c989b52c3957264f9fceae482
tengine_nginx_O2	baselines/SymLM/dataset_generation/xproj_debug/tengine_nginx_O2	4776024	732474ce4fc54fcf511796e3a43382d8af6f292b
tig_tig_O0	data/debug/tig_tig_O0	1244264	-
tig_tig_O1	data/debug/tig_tig_O1	1465304	-
tig_tig_O2	data/debug/tig_tig_O2	1573896	-
tig_tig_O3	data/debug/tig_tig_O3	1804584	-
tinycc_tcc_O0	data/cross_project/debug_unseen/tinycc_tcc_O0	586984	2acaefcff5e43c4f573af9c5815c20709990202e
tinycc_tcc_O1	data/cross_project/debug_unseen/tinycc_tcc_O1	818600	cc5832167f7a26de9ba099a9d8d1698bbc96312f
tinycc_tcc_O2	data/cross_project/debug_unseen/tinycc_tcc_O2	1029736	8605ec31cf44ca3e31de291672c1a8e344c43c64
tinycc_tcc_O3	data/cross_project/debug_unseen/tinycc_tcc_O3	1464816	62fb26e8f5e23ae5f4a11113ad6b1bf78ec422a5
units2_units_O0	data/raw/units2_units_O0	196448	95444ab822c245360a698612091d1ebcb64afb14
units2_units_O1	data/raw/units2_units_O1	273752	de841e4272f765c59e01d8b839278d05aa4470d1
units2_units_O2	data/raw/units2_units_O2	276744	d124f0eb4c4f8bc59d5c43e310bb3f42ca74689f
units2_units_O3	data/raw/units2_units_O3	281616	7160459ede0ad52144d4aa4318480f4335968ed1
which2_which_O0	data/raw/which2_which_O0	58384	c95881c26b57cf6d88c5372814666d7b6c217ca9
which2_which_O1	data/raw/which2_which_O1	72952	cf4bc94f0b5f40353dfdf0be1890fe36f8abf33a
which2_which_O2	data/raw/which2_which_O2	77928	f5d5e2129640715f85dbb4819718043510f8f955
which2_which_O3	data/raw/which2_which_O3	80032	a95a1cbfa08a95a910d53b0621f2beb2044d5379
zstd_zstd_O0	data/debug/zstd_zstd_O0	3590000	-
zstd_zstd_O1	data/debug/zstd_zstd_O1	6165472	-
zstd_zstd_O2	data/debug/zstd_zstd_O2	6958520	-
zstd_zstd_O3	data/debug/zstd_zstd_O3	8953016	-
```
