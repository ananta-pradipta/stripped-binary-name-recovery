# Non-ML Baseline: Ghidra Function ID (FID) Signature Matching

Built in response to CCS Reviewer C ("the paper compares HyDRA with other
ML-based approaches, but it does not compare it with non-ML approaches").
Ghidra's Function ID (FID) analyzer is the open-source equivalent of IDA
FLIRT: it hashes function bodies (normalized, relocation-agnostic) into a
signature database and matches unknown functions against it.

All numbers below use `src/evaluation/metrics.py` (the same subtoken-F1 /
exact-match / n-gram-sim / edit-sim implementation as the ML evaluation), so
they are directly comparable to the paper's other rows.

**Local only, not committed to git.** Ghidra 11.0.3 PUBLIC, `~/ghidra_11.0.3_PUBLIC`.

---

## 1. Two findings before the numbers (read these first)

### 1.1 Ghidra's shipped FID libraries are irrelevant to our corpus (confirmed empirically)

Ghidra 11.0.3 ships 10 pre-built `.fidbf` signature libraries under
`Ghidra/Features/FunctionID/data/`: `vs2012`, `vs2015`, `vs2017`, `vs2019`,
`vsOlder`, each in x86/x64 — **every one of them is a Visual Studio / MSVC
runtime signature set, built for Windows PE binaries.** They are active by
default. On our Linux/GCC/ELF x86-64 corpus they contribute **zero** matches
(confirmed by running with them active vs. explicitly deactivated —
identical results either way).

This is exactly the fallback condition anticipated in the task spec ("if
FidDb turns out to be unavailable/empty"). We therefore built our **own**
custom FID database from debug (unstripped) builds of our training-side
packages — the methodologically correct FLIRT-equivalent experiment, and a
direct structural analogue of what our ML model does (learn from a training
split, evaluate on held-out packages).

### 1.2 Four of the seven xproj packages leak names via `.dynsym` (data artifact, not a matching result)

`strip` removes `.symtab` but **must** leave `.dynsym` intact for dynamic
linking to work. `.dynsym` normally only lists *imported* (UND) symbols. But
if a binary is built with exported visibility (`-rdynamic`, or simply no
`-fvisibility=hidden`), its own internally-defined functions also appear in
`.dynsym` as **DEFINED** entries — with their real names, at their real
addresses. Reading that is not signature matching, not recognition, not
ML-equivalent effort of any kind; it is parsing a standard ELF section.

We checked `readelf --dyn-syms <binary> | grep -v UND | grep FUNC` for every
package's O2 stripped binary:

| package | defined FUNC entries leaked via `.dynsym` |
|---|---|
| dash | 0 |
| gettext | 0 |
| psmisc | 0 |
| grep | 6 (negligible) |
| sed | 6 (negligible) |
| recutils | 84–86 per binary (≈55–65% of that binary's GT) |
| nginx118 | 457 |
| angie | 509 |
| **tengine** | **570 (GT has 569 — literally 100% of ground truth is exported)** |

**tengine's "default Ghidra, zero custom signatures, zero ML" result is a
vacuous 100% EM.** This is a build-configuration artifact of how these
particular nginx-family/recutils binaries were compiled for this dataset
(plausibly nginx's dynamic-module-loading architecture requires exported
symbols), not a property of "stripped binaries" in general, and not
something either Ghidra or our ML model is doing anything clever to recover.
**We are flagging this explicitly rather than quietly reporting the inflated
number** — dash/gettext/psmisc/grep/sed are the only packages in the 7-pkg
xproj set where a coverage/EM number reflects genuine difficulty.

---

## 2. Baseline 2 (headline): custom-FID, leave-one-package-out signature matching

**Scope:** the 5 packages with local debug (unstripped, real `.symtab`) AND
stripped ELF pairs at O2: dash, gettext, grep, psmisc, sed — 1,323 ground
truth functions total. For each held-out package, a custom FID database was
built by hashing every function in the debug builds of the *other 4*
packages (FID population is literally "read the correct name straight off
`.symtab`, then hash the function body" — no leakage across the fold
boundary; population and eval binaries never overlap by package).

Two conditions per fold, same binaries, only the active FID database differs:
- **control** — default Ghidra, zero active FID databases (isolates what
  Ghidra recovers "for free" from ELF structure alone — PLT/import thunks,
  `_init`/`_fini`/`frame_dummy`/`register_tm_clones` convention labels, etc.)
- **treatment** — control + our custom LOPO FID database active

| fold (held out) | n | control EM | treatment coverage | treatment F1 | treatment EM | precision on named (EM) | n newly named by FID |
|---|---|---|---|---|---|---|---|
| dash | 250 | 0.000 | 1.6% | 0.008 | 0.8% | 100.0% (2/2) | 2 |
| gettext | 469 | 0.000 | 5.8% | 0.006 | 0.6% | 100.0% (3/3) | 3 |
| grep | 338 | 0.000 | 30.8% | 0.267 | 26.0% | 84.6% (88/104) | 94 |
| sed | 215 | 0.000 | 46.5% | 0.417 | 40.9% | 88.0% (88/100) | 94 |
| psmisc | 51 | 0.000 | 25.5% | 0.098 | 9.8% | 38.5% (5/13) | 5 |
| **aggregate** | **1,323** | **0.000** | **18.7%** | **0.144** | **14.1%** | — | **198** |

**Control aggregate is exactly 0.000 EM / F1 ≈ 0.001** — every function
"named" by plain Ghidra without our custom library is wrong (mostly
`_INIT_0`/`_FINI_0`-style convention labels that don't match GT text at
all). The entire treatment number is attributable to the custom FID
database.

**Reading this table is the coverage-boundary story from the opposite
direction of our paper's thesis:**
- Where near-duplicate, statically-linked shared code exists across packages
  (grep/sed both pull in large overlapping gnulib helper sets — quotearg,
  xmalloc family, `full_write`, etc.), FID gets both **decent coverage
  (31–47%) and near-ML-grade precision on what it names (85–88% EM)**.
- Where it doesn't (dash is a POSIX shell with almost no gnulib overlap with
  grep/sed/gettext/psmisc; gettext's own helper functions are largely
  gettext-specific), **FID coverage collapses to 1.6–5.8%** — it has nothing
  to match against, so it correctly stays silent rather than guessing.
- This is a clean instance of "regime (a)": signature matching gets
  near-perfect precision on exact/near-clone code and total collapse
  elsewhere, with no middle ground — exactly the boundary our ML model
  (which *can* interpolate/generalize past exact duplication, at lower but
  non-zero accuracy) is positioned against.

Per-fold and per-function detail: `results/ndss_prep/ghidra_fid_lopo_{dash,gettext,grep,sed,psmisc}.json`

---

## 3. Baseline 1: weak default (no custom FID at all), extended to all 7 xproj packages

For recutils/nginx118/angie/tengine we have no local debug binaries, so no
custom-FID LOPO experiment is possible; only the "default Ghidra, no FID"
condition was run, for the fuller 7-package picture the reviewer likely
expects. **Given the §1.2 caveat, read these as upper bounds contaminated by
`.dynsym` export leakage, not as a real matching result:**

| package | n | coverage | F1 | EM | caveat |
|---|---|---|---|---|---|
| dash/gettext/grep/psmisc/sed (control, from §2) | 1,323 | 3.8% | 0.001 | 0.0% | clean |
| recutils (9 binaries) | 1,224 | ≈68% | ≈0.63 | ≈0.62 | `.dynsym` leaks ≈55–65% of GT per binary |
| nginx118 | 1,160 | 39.9% | 0.396 | 39.4% | `.dynsym` leaks 457 defined FUNC entries |
| angie | 1,297 | 39.8% | 0.394 | 39.2% | `.dynsym` leaks 509 defined FUNC entries |
| tengine | 569 | 100.0% | 1.000 | 100.0% | **vacuous — `.dynsym` leaks 570/569 = all of GT** |
| recutils+nginx118+angie (excl. tengine) | 3,701 | 48.9% | 0.470 | 46.6% | still leak-inflated, do not present as "Ghidra performance" |

Full detail: `results/ndss_prep/ghidra_fid_control_extra_final.json`

---

## 4. Timing (Table 5 input — replaces the literature citation with our own measurement)

**Machine:** AMD Ryzen 5 7535HS (12 threads), 7.4 GB RAM, WSL2
(`Linux 6.6.87.2-microsoft-standard-WSL2`) — laptop-class, not a dedicated
server. Ghidra 11.0.3 PUBLIC, OpenJDK 17.0.19. All times wall-clock via
`/usr/bin/time -v`, single binary per `analyzeHeadless` invocation (includes
JVM startup + project creation + ELF load + auto-analysis + script export +
project save).

### 4.1 Analysis-only (no decompilation, no custom FID) — the fair comparison point vs. our BAP-IR lift

This is the **same class of work** our BAP-IR pipeline does (recover
function boundaries / a typed IR), so it's the correct number to set next to
our own BAP lift time — NOT the number to compare against BLens/SymGen (see
§4.2).

| binary | size | wall time | internal auto-analysis time |
|---|---|---|---|
| psmisc_peekfd_O2 | 14 KB | 6.55 s | 1 s |
| psmisc_killall_O2 | 32 KB | 7.75 s | 2 s |
| dash_dash_O2 | 123 KB | 13.43 s | 8 s |
| grep_grep_O2 | 166 KB | 13.68 s | 9 s |
| nginx118_nginx118_O2 | 706 KB | 27.67 s | 21 s |
| angie_angie_O2 | 777 KB | 28.60 s | 22 s |
| tengine_nginx_O2 | 962 KB | 32.03 s | 26 s |

**Mean 18.53 s, range 6.55–32.03 s, total 129.71 s across 7 binaries.**
Fixed per-invocation overhead (JVM startup + project I/O, roughly
size-independent) ≈ 4.7–6.7 s (mean 5.82 s); the remainder scales
with binary size/function count — time roughly **5× from smallest to
largest binary while size grows ≈68×**, i.e. sublinear in size but
monotonically increasing.

### 4.2 Decompile-add-on: the extra cost BLens/SymGen-style pipelines pay that we don't

We separately measured calling `DecompInterface.decompileFunction()` on
every discovered function (full pseudocode-C generation, 30 s/function
timeout) — the step required to hand an LLM readable code, which our
BAP-IR-token pipeline never performs:

| binary | analysis-only wall | + full decompile wall | decompile-phase-only (internal) | n functions |
|---|---|---|---|---|
| psmisc_killall_O2 | 7.75 s | 10.20 s | 1.16 s | 143 |
| dash_dash_O2 | 13.43 s | 18.50 s | 4.97 s | 422 |
| grep_grep_O2 | 13.68 s | 21.87 s | 7.94 s | 585 (1 decompile failure/timeout) |
| nginx118_nginx118_O2 | 27.67 s | 50.76 s | 23.83 s | 1,433 |

Decompilation adds **31–83%** on top of plain analysis for these binaries,
growing with function count — it nearly **doubles** total wall time for the
largest (nginx118: 27.67 s → 50.76 s). **We did not run full decompilation
in the accuracy experiments above (§2–3)** — FID matching itself doesn't
need it — so §4.1's numbers are a fair floor for "our BAP-lift-equivalent
cost," and §4.2 quantifies the extra cost a decompiler-pseudocode-dependent
pipeline (BLens/SymGen-class) pays on top, **measured on our own hardware
and our own binaries**, not cited from the literature.

Raw JSON: `results/ndss_prep/ghidra_timing.json`

---

## 5. Deliverables

- `scripts/ghidra_fid_scripts/ExportFunctions.java` — per-function
  (address, name, symbol-source, is-thunk) dump post-script
- `scripts/ghidra_fid_scripts/FidPopulateAll.java` — builds/populates a
  custom FID database from every program in a project, with optional
  package-prefix exclusion for LOPO folds
- `scripts/ghidra_fid_scripts/FidSetOnlyActive.java` /
  `FidSetAllInactive.java` — FID-database activation control (needed because
  `FidFileManager` keeps a flat, Preferences-persisted list — every populate
  run leaves its file active until explicitly told otherwise, which WILL
  leak across LOPO folds if not reset; see §6 below)
- `scripts/ghidra_fid_scripts/DecompileAllFunctions.java` — timing-only
  probe for §4.2
- `scripts/ghidra_fid_baseline.py` — scores control vs. treatment exports
  against `data/labels/*_labels.json` (or `cross_project/ground_truth.json`
  fallback), address-space auto-detected (PIE `rel_addr` vs non-PIE `addr`)
- `scripts/ghidra_fid_control_only.py` — same scoring, control-condition
  only, for packages with no local debug binaries
- `results/ndss_prep/ghidra_fid_baseline_FINAL.json` — consolidated
  accuracy + timing
- `results/ndss_prep/ghidra_fid_lopo_{dash,gettext,grep,sed,psmisc}.json` —
  per-fold, per-function detail
- `results/ndss_prep/ghidra_fid_control_extra_final.json` — the 4
  dynsym-contaminated packages, control-only
- `results/ndss_prep/ghidra_timing.json` — raw timing data

## 6. Pitfall for anyone re-running this: FID activation is a leaky global, not a per-project setting

`FidFileManager` persists the active/inactive FID file list to
`~/.ghidra/.ghidra_11.0.3_PUBLIC/preferences` (`FID.USER.ADDED` /
`FID.INACTIVE` keys), **shared across every headless invocation by the same
user, independent of which Ghidra project is open.** Populating fold B's
database does NOT deactivate fold A's — we hit this directly (an earlier
`control_extra` pass for recutils/nginx118/angie/tengine accidentally ran
with the psmisc-holdout LOPO database still active from the prior fold,
inflating recutils' numbers; caught by comparing against a from-scratch
rerun after explicit `FidSetAllInactive`, and the two extra-package control
passes in §3 are the corrected, clean ones). Always call
`FidSetOnlyActive.java <path>` (or `FidSetAllInactive.java` for a true
zero-FID control) immediately before any pass whose FID state matters, never
assume the state left by a previous invocation.
