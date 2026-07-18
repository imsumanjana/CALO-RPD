# CALO-RPD Studio v3.3.0 — Deep Repository Audit

**Audited artifact:** `CALO-RPD-main(3).zip`  
**Audit date:** 17 July 2026  
**Audit mode:** source review, scientific-formulation review, automated tests, coverage, lint/security scans, packaging, manifest/freeze verification, and artifact/provenance inspection  
**Repository size:** 386 archive entries; 2,130,306 uncompressed bytes; no unsafe archive paths  
**Python codebase:** 266 Python files; 30,787 lines; 1,235 functions; 204 classes

## 1. Executive verdict

**Release decision: NO-GO for publication-grade benchmark claims and NO-GO for a v3.3 production release until the P0/P1 items are corrected.**

The repository has a strong modular foundation, a coherent 16-workspace PyQt6 workflow, a common ORPD evaluation path, centralized seed/evaluation-budget machinery, accelerator parity infrastructure, reproducible manifests, and a verified frozen CALO bundle. It compiles and builds clean source/wheel artifacts, and 106 non-GUI unit/scientific/integration tests pass.

However, the audit found **4 critical, 9 high, 10 medium, and 1 low findings**. The most serious are:

1. weighted CVaR is mathematically wrong for unequal/discrete weights;
2. generic shunt-control bounds overwrite the actual IEEE case shunts/reactors;
3. the persistent XPU sidecar contains a guaranteed undefined-name failure in progress reporting;
4. publication statistics can treat verified-but-infeasible solutions as valid objective evidence;
5. article-ready output can include unverified runs;
6. requested 31–50 repetitions are silently reduced to 30.

These are not cosmetic defects. They can change the optimization problem, robust objective, run count, accelerator success rate, rankings, plots and manuscript conclusions.

### Overall scorecard

| Area | Score | Audit judgement |
|---|---:|---|
| Architecture and separation | 8/10 | Strong package decomposition and shared services, but several oversized modules remain. |
| Core numerical correctness | 4/10 | Base AC power flow is reasonably tested; CVaR, shunt formulation and Q-limit validation contain major defects. |
| Benchmark fairness | 6/10 | Common evaluator/budget/seed/freeze are good; formulation and run-count defects undermine fairness. |
| Accelerator/runtime reliability | 5/10 | FP64/device-resident design is credible, but XPU sidecar is broken and physical GPU/XPU was not testable here. |
| Reproducibility/provenance | 6/10 | Hash manifests and frozen files are strong; version drift, stale artifacts, paths and no lock/CI weaken it. |
| Automated testing | 4/10 | Core tests pass, but full suite fails and coverage is only 37%. |
| GUI/workflow | 7/10 | Current source is internally coherent; tests lag the 16-workspace/Portfolio Manager design. |
| Security/trust boundaries | 6/10 | No high Bandit finding; unsafe checkpoint/pickle loading must be constrained. |
| Publication readiness | 3/10 | Publication filtering and evidence-basis defects are blockers. |
| **Overall** | **5.4/10** | **Promising research platform, not yet defensible as a frozen publication benchmark release.** |

## 2. Scope and limitations

### Included

- every Python module under `calo_rpd_studio/`, `calo_bootstrap/`, and `tests/`;
- project metadata, scripts, requirements, documentation, patch notes, release record, manifests and frozen files;
- algorithm registry and common optimizer/evaluator interfaces;
- AC power flow, ORPD decoder, constraints/objectives, robust scenarios/CVaR and independent validation;
- experiment scheduling, 80/10/10 device allocation plumbing, persistent workers, resume and result database;
- GUI construction and workflow in headless Qt mode;
- publication export, campaign packaging, plots and shipped example/stale evidence;
- wheel/sdist construction and hash/freeze verification.

### Not included

- a full 20-algorithm × 30–50-run × multi-case campaign;
- physical NVIDIA CUDA or Intel XPU throughput/utilization validation, because this audit environment did not expose those devices;
- visual inspection on every Windows DPI/font/display configuration;
- experimental validation against a laboratory or commercial ORPD reference model.

Therefore, the audit can reject release readiness based on demonstrated defects, but it does not certify physical GPU speed or end-to-end scientific accuracy after fixes.

## 3. Automated evidence

| Check | Result |
|---|---|
| Archive safety | 386 entries, no absolute/traversal paths |
| Python compilation | Passed (`compileall`) |
| Core test suites | **106 passed** (`tests/unit`, `tests/scientific`, `tests/integration`) |
| Full test suite with PyQt6/PYPOWER | **123 passed, 7 failed** |
| Coverage, core suites | **37%** (16,155 statements; 10,193 missed) |
| Ruff | **26 errors**, including F821 undefined `iteration` |
| Bandit | 0 high; 23 medium; 49 low (72 total findings) |
| Frozen CALO manifest | Passed across 50 files; SHA-256 `0d22395fbedd4834ab23aea6fea90f3544b3e7639fc5dc449d54b5a1cfd39fcd` |
| Repository file manifest | `sha256sum -c MANIFEST.sha256` passed |
| Wheel build | Passed; 252 entries |
| Source distribution build | Passed; 296 entries |
| Package version | 3.3.0 |

The release record reported 104 passed and 25 skipped because PyQt6 and PYPOWER were missing. Installing the declared dependencies exposed seven GUI/workflow test failures, so the published validation record is incomplete rather than a true full release gate.

## 4. Positive architecture findings

- `main.py`/bootstrap are thin entry points; domain logic is packaged rather than concentrated in one script.
- The GUI currently defines 16 workspaces, including Portfolio Manager, Validation & Audit, Publication Export, Resume Center and Benchmark & Evidence.
- All 20 primary algorithms are registered in one place. Non-CALO algorithms are routed to the tensor baseline suite only when the torch backend is requested; AI inference fields are injected only for CALO.
- A common normalized decoder, ORPD evaluator, constraint policy and central evaluation budget reduce algorithm-specific physics drift.
- Seed generation is centralized by run index, supporting paired stochastic comparisons.
- The v3.3 freeze manifest includes code, policy checkpoint and configuration material and verifies successfully.
- CPU/reference and accelerator parity tests exist, use float64, and are a useful foundation.
- The wheel and source distribution build without errors; console scripts are declared for GUI, training, validation, parity, benchmark and export.
- No archive traversal issue, `shell=True`, or Bandit high-severity vulnerability was detected.

These strengths should be retained while repairing the scientific and release-gate defects.

## 5. Detailed findings

### F-001 — CRITICAL: Weighted empirical CVaR is mathematically incorrect when alpha cuts through a probability mass

**Category:** Scientific correctness  
**Location:** `calo_rpd_studio/robustness/cvar.py:4-5; accelerated/torch_orpd.py:203-211; accelerated/device_resident_orpd.py:575-584`

**Evidence.** For values [0,100], weights [0.96,0.04], alpha=0.95, repository returns 4.0; exact upper-tail CVaR is 80.0. All CPU and accelerator implementations include the entire VaR atom rather than the fractional mass needed to make a 1-alpha tail.

**Impact.** Robust objective values, rankings, and conclusions can be materially wrong for unequal scenario weights or scenario counts whose cumulative mass does not align with alpha. CPU/accelerator parity cannot detect it because the same defect is duplicated.

**Required correction.** Implement fractional-tail weighted CVaR from the quantile integral; validate alpha in (0,1), nonnegative finite weights, and positive total weight. Add exact edge-case tests and cross-backend tests with unequal weights.

### F-002 — CRITICAL: Generic 0-5 MVAr shunt controls overwrite case-specific fixed reactors/capacitors

**Category:** Scientific formulation  
**Location:** `calo_rpd_studio/orpd/variable_decoder.py:18-24,45-50,62`

**Evidence.** The same 0..5 MVAr range is assigned to every default shunt bus. In PYPOWER case57 the selected buses start at 10.0, 5.9 and 6.3 MVAr; in case118 selected buses include -40, -25, 6..20 MVAr. Decoding writes the decision directly to BUS.BS, replacing the original value.

**Impact.** The optimized network is not the intended IEEE case and may remove reactors, collapse capacitors, or change base-case feasibility. Cross-case objective values and literature comparisons are not scientifically defensible until each case formulation is explicitly defined.

**Required correction.** Create versioned, case-specific ORPD control definitions with absolute versus incremental semantics, engineering bounds, discrete steps, and citations/provenance. Preserve fixed shunts outside the declared controls. Add base/control-vector regression tests per case.

### F-003 — HIGH: Benchmark decision dimensions are implicit and inconsistent across cases

**Category:** Benchmark formulation  
**Location:** `calo_rpd_studio/power_system/case_loader.py:8-17; calo_rpd_studio/orpd/variable_decoder.py:18-24,38-50`

**Evidence.** Default decoded dimensions are case30=14 (6 Vg, 0 taps, 8 shunts), case57=27, case118=77, case300=176 (69 Vg, 107 taps, 0 shunts). Stock PYPOWER data and hard-coded control lists determine the formulation silently.

**Impact.** A paper can label a study IEEE 30/57/118/300 while actually solving a different control-variable problem than the cited literature or another implementation. Case300 is not treated with the same shunt-control policy.

**Required correction.** Export a formulation manifest for every task: case checksum, control buses/branches, bounds, step sizes, initial values, dimension and fixed-device treatment. Require explicit case formulation profiles rather than hidden defaults.

### F-004 — CRITICAL: Persistent XPU sidecar progress callback raises NameError

**Category:** Runtime reliability  
**Location:** `calo_rpd_studio/compute/persistent_accelerator_sidecar.py:121-143`

**Evidence.** Line 131 assigns last_iteration = iteration, but iteration is undefined and last_iteration is not declared nonlocal. Ruff reports F821. The callback is passed into run_single/run_ablation and the outer handler converts the exception to a failed run.

**Impact.** The claimed persistent XPU execution path can fail on its first progress emission, directly breaking the 80/10/10 throughput engine on systems that use the sidecar.

**Required correction.** Read iteration from payload, declare nonlocal only if state is needed, and add an end-to-end sidecar protocol test that exercises at least two progress frames and job completion.

### F-005 — CRITICAL: Publication statistics include independently verified but infeasible solutions

**Category:** Publication integrity  
**Location:** `calo_rpd_studio/results/publication_export.py:16-30; publication_export/verified_runs.csv; publication_export/descriptive_statistics.csv`

**Evidence.** Exporter filters verified_only but groups every row by algorithm for objective statistics without filtering feasible=True. The shipped export contains one infeasible CALO-without-AI run (violation 0.057979...) and reports objective 2.3665 as publication descriptive evidence.

**Impact.** Infeasible objective values can be reported as best/mean performance, invalidating rankings and manuscript tables under a feasibility-first ORPD formulation.

**Required correction.** Separate verification status from feasibility. Compute objective statistics only on feasible verified runs; export feasibility rate, violation statistics, and no-feasible-run status separately. Refuse best-objective claims when no verified feasible run exists.

### F-006 — HIGH: Article-ready package still derives narrative, tables and figures from all runs when validation is incomplete

**Category:** Publication integrity  
**Location:** `calo_rpd_studio/benchmarking/package.py:71-95,113-150,266; visualization/publication_evidence.py:21-25,264-343`

**Evidence.** Verified evidence is generated, but article_lines uses evidence.interpretations from all runs; downstream tables iterate all-run evidence; figure generation loads database.list_runs without verified_only. A warning file is emitted but artifacts are still produced.

**Impact.** Unverified runs can enter article text, rankings, boxplots and critical-difference graphics while the package is branded article-ready. A warning is insufficient to prevent accidental publication.

**Required correction.** Make verified-only the default publication basis. Either abort article-ready export until the intended validation set is complete or label and segregate diagnostic all-run artifacts outside the publication package.

### F-007 — HIGH: Requested 31-50 benchmark repetitions are silently reset to 30

**Category:** Reproducibility  
**Location:** `calo_rpd_studio/benchmarking/campaign.py:78-109; experiments/experiment_config.py:122-125; portfolio/models.py:84-105`

**Evidence.** build_campaign writes config.runs from campaign.runs, then config.validate overwrites it with the default JOURNAL portfolio requirement. Demonstration: requested 35 and 50 both create tasks with 30 runs and 600 jobs instead of 700/1000 for 20 algorithms.

**Impact.** The CLI accepts 30-50 runs but does not execute the requested evidence volume, causing incorrect campaign manifests, statistical power, elapsed-time estimates and publication claims.

**Required correction.** Synchronize campaign runs into a CUSTOM/TRANSACTIONS portfolio before validation, or make campaign tasks bypass/override portfolio normalization explicitly. Add tests for 30, 31, 35 and 50.

### F-008 — HIGH: Reactive-power-limit cross-validation is not independent

**Category:** Independent validation  
**Location:** `calo_rpd_studio/power_system/independent_validator.py:49-64`

**Evidence.** When the internal solver performs PV-to-PQ switching, PYPOWER is run on internal.case, which already contains transformed bus types and adjusted generator Q values, with ENFORCE_Q_LIMS=0.

**Impact.** The cross-check can confirm only the final transformed power flow, not whether the internal Q-limit detection, switching sequence and final state were correct from the original case.

**Required correction.** Run the independent solver from the original controlled case with its own Q-limit enforcement, then compare convergence, final bus types, clamped generators, voltages, Q outputs and losses.

### F-009 — HIGH: Invalid robust-scenario settings are accepted and can silently become deterministic

**Category:** Configuration validation  
**Location:** `calo_rpd_studio/experiments/experiment_config.py:19-29,88-151; experiments/experiment_runner.py:22-32; orpd/problem.py:21-23; accelerated/torch_orpd.py:68-71`

**Evidence.** ExperimentConfig.validate does not validate scenario mode, count, standard deviations, outage indices, renewable bus or capacity-factor parameters. count=0 is accepted; generators return an empty list; `scenarios or [Scenario(base)]` silently substitutes the deterministic base case.

**Impact.** A user can request robust uncertainty and unknowingly execute a deterministic study, corrupting labels, risk metrics and publication evidence.

**Required correction.** Validate scenario settings centrally and reject empty scenario sets. Use `if scenarios is None` rather than truthiness fallback. Persist the generated scenario count, weights and checksum in every run.

### F-010 — HIGH: The complete declared-dependency test suite is not green

**Category:** Release quality  
**Location:** `tests/gui/test_gui_startup.py; tests/gui/test_guided_workflow.py; RELEASE_VALIDATION.md:16-33`

**Evidence.** With PyQt6 and PYPOWER installed, full offscreen pytest reports 123 passed and 7 failed. Failures include stale 14-vs-16 workspace expectations, old page indices, old scroll count and old 50/30/20 defaults. The release record reported 104 passed/25 skipped because key dependencies were absent.

**Impact.** Release validation masked feature-drift regressions and cannot serve as a reliable gate. GUI navigation and workflow changes lack synchronized tests.

**Required correction.** Update tests to the 16-workspace/Portfolio Manager workflow and 80/10/10 defaults, then require zero failures with all mandatory dependencies installed.

### F-011 — HIGH: Batch/CLI problem construction bypasses structural case validation and zero impedance becomes an open circuit

**Category:** Input safety  
**Location:** `calo_rpd_studio/experiments/experiment_runner.py:33-45; power_system/case_validation.py:9-22; power_system/ybus.py:16-25`

**Evidence.** build_problem loads a case and immediately constructs the ORPD problem without validate_case. validate_case only warns for zero reactance. build_ybus sets y=0 for exactly zero complex impedance, electrically deleting an in-service zero-impedance branch.

**Impact.** Malformed custom cases can run silently with changed topology and produce plausible but invalid optimization results.

**Required correction.** Make validation mandatory in CaseLoader/build_problem; reject or explicitly regularize zero-impedance in-service branches; validate finite arrays, status values, limits, connectivity and branch ratings.

### F-012 — HIGH: Nineteen baseline implementations lack formulation-level regression evidence

**Category:** Algorithm validation  
**Location:** `tests/unit/test_algorithms.py:7-20; tests/unit/test_v3_accelerated_backend.py:31-62; docs/algorithm_sources.md:30-44`

**Evidence.** Tests run each method for only 40 evaluations on a sphere function and assert finite output/metadata. There are no operator-equation tests, known seeded trajectories, published reference values, or legacy-vs-tensor equivalence tests per algorithm.

**Impact.** The label canonical is not independently substantiated. A shared dispatcher or operator defect can change a baseline and alter CALO comparisons without test detection.

**Required correction.** For every baseline, add deterministic one-step/operator tests, seeded short trajectory snapshots, boundary/repair tests, budget accounting tests and CPU-vs-tensor parity against the declared source formulation.

### F-013 — HIGH: Scientific tests do not cover publication benchmark conditions

**Category:** Scientific testing  
**Location:** `tests/scientific/test_ieee_cases.py:1-10; tests/unit/test_robustness.py:1-14`

**Evidence.** Power-flow cross-check covers only base case30/57/118. It omits case300, optimized mixed-variable control vectors, Q-limit switching parity, contingencies, renewable/load uncertainty, robust aggregation edge cases and post-optimization independent validation. The CVaR assertion is only `>=2`.

**Impact.** Core publication pathways can be wrong while the scientific suite remains green, as demonstrated by the CVaR defect.

**Required correction.** Add a compact but authoritative scientific regression matrix across all cases, formulation profiles, robust modes, discrete controls, Q-limit events and final independent validation.

### F-014 — MEDIUM: Coverage is 37% and critical orchestration/export paths are untested

**Category:** Test engineering  
**Location:** `Repository-wide`

**Evidence.** Unit+scientific+integration coverage: 16,155 statements, 10,193 missed, 37%. publication_export is 0%, final benchmark runner 0%, experiment_manager 0%, and publication_evidence 8%.

**Impact.** High-risk changes in scheduling, persistence, campaign packaging and publication evidence can regress without detection.

**Required correction.** Set staged coverage gates (for example 60% overall first, then 75%+) and require high branch coverage on publication, scheduler, campaign and validation modules.

### F-015 — MEDIUM: Bootstrap environment-state version is stale at 3.2.0

**Category:** Versioning  
**Location:** `calo_bootstrap/prerequisites.py:22,865-899; pyproject.toml:5-8`

**Evidence.** Package version is 3.3.0 but APP_VERSION remains 3.2.0 and controls first_launch_or_version_changed/cpu_fallback acceptance.

**Impact.** A 3.2 environment state can be treated as current by 3.3, skipping intended prerequisite re-checks after the CUDA-resident release.

**Required correction.** Read the installed package version from importlib.metadata with a safe fallback, and test version-transition behavior.

### F-016 — MEDIUM: v3.3 benchmark code is still labeled v2/v3.2 and writes benchmark_v2

**Category:** Provenance  
**Location:** `benchmarking/campaign.py:19,32,45; benchmarking/package.py:114; scripts/run_final_benchmark.py:1,25,27; gui/panels/benchmark_campaign_panel.py:121,171,209`

**Evidence.** Campaign names, errors, CLI help, GUI defaults and article summary contain stale release identifiers.

**Impact.** Output directories and manuscript evidence can be misattributed to the wrong software/freeze version.

**Required correction.** Derive release and freeze identifiers centrally, include manifest hash in paths/reports, and remove hard-coded historical labels.

### F-017 — MEDIUM: Repository claims no pre-populated results but ships stale v1.0.6 evidence

**Category:** Repository hygiene  
**Location:** `publication_export/README.md and all files under publication_export/`

**Evidence.** README says no pre-populated experimental results. The directory contains CSV/TeX/ZIP/metadata from software 1.0.6, one infeasible run, budget 100 and an old policy path.

**Impact.** Users may mistake stale diagnostic output for v3.3 validated evidence; repository statements are internally contradictory.

**Required correction.** Remove generated results from source control or place a clearly labeled synthetic example with schema/version checks and no publication claim.

### F-018 — MEDIUM: Frozen/provenance JSON embeds absolute developer paths and an empty historical-learning snapshot

**Category:** Portability/privacy  
**Location:** `historical_experience_v1.3.json; data/frozen/historical_training_snapshot_v2.json; data/trained_models/*.json`

**Evidence.** Several JSON files contain `C:\Users\User\Downloads...`; both historical snapshots contain zero policy trajectories, zero cross-algorithm solutions and zero parameter priors, yet the frozen manifest includes the snapshot.

**Impact.** Leaks local path information, reduces portability and can imply historical learning evidence that is absent.

**Required correction.** Sanitize paths to repository-relative provenance; clearly record zero-data snapshots as disabled/not applicable; freeze only meaningful, documented training inputs.

### F-019 — MEDIUM: Checkpoints and worker payloads use unsafe deserialization

**Category:** Security/trust boundary  
**Location:** `ai/model_io.py:4; algorithms/calo/ai_controller.py:195; algorithms/calo/training.py:736; algorithms/calo/heterogeneous_training.py:609; compute persistent-worker modules`

**Evidence.** Bandit reports unsafe torch.load(weights_only=False) and pickle deserialization. No high-severity issue was found, but these formats can execute code if files/frames are attacker-controlled.

**Impact.** Opening an untrusted checkpoint, resume file or manipulated local worker stream can execute arbitrary code.

**Required correction.** Treat all such inputs as trusted-only in UI/docs, verify hashes before load, use weights_only=True for model-only checkpoints, validate schemas and replace pickle protocols where practical.

### F-020 — MEDIUM: No exact dependency lock or CI workflow

**Category:** Reproducibility  
**Location:** `pyproject.toml:13-33; repository root`

**Evidence.** Dependencies are broad version ranges; there is no lock/conda environment and no .github/workflows directory. Per-run provenance helps but does not recreate the environment automatically.

**Impact.** Numerical behavior, GUI compatibility and packaging results can drift across NumPy/SciPy/PyTorch/PYPOWER versions and platforms.

**Required correction.** Publish hashed lock files for supported CPU/CUDA/XPU environments and run a CI matrix for Python 3.11-3.13, Linux/Windows, headless GUI and CPU scientific tests.

### F-021 — MEDIUM: Several modules are too large and combine many responsibilities

**Category:** Maintainability  
**Location:** `app/experiment_manager.py; algorithms/calo/heterogeneous_training.py; gui/plotting/plot_format_toolbar.py; results/database.py`

**Evidence.** Largest modules are 1,662, 1,246, 1,106 and 879 lines. The experiment manager mixes planning, scheduling, workers, persistence, progress and failure handling.

**Impact.** Reviewability, unit isolation and safe modification decline; critical paths have low coverage.

**Required correction.** Split orchestration into planner/scheduler/executor/persistence services, extract plotting control models, and isolate database repositories/migrations.

### F-022 — MEDIUM: Ruff reports 26 errors and lint is not a release gate

**Category:** Static quality  
**Location:** `Repository-wide; persistent_accelerator_sidecar.py:131`

**Evidence.** Findings include 12 unused imports, unused state/locals and the production F821 undefined-name defect. RELEASE_VALIDATION says Ruff was not installed.

**Impact.** Simple defects survive release packaging and dead code obscures numerical paths.

**Required correction.** Add `ruff check`, formatting, compileall and import checks to CI and release scripts; do not publish when F/E errors remain.

### F-023 — MEDIUM: Times New Roman output is not portable or preflighted

**Category:** Publication rendering  
**Location:** `GUI/plotting and visualization modules`

**Evidence.** Headless GUI tests emitted 219 `Font family Times New Roman not found` warnings. Matplotlib silently substitutes a font.

**Impact.** Exported 600-DPI figures may not meet declared journal typography requirements and can vary by workstation.

**Required correction.** Add a font availability preflight and explicit fallback disclosure; record the resolved font in export metadata. Do not bundle proprietary font files.

### F-024 — LOW: Archive contains no Git history or commit identity

**Category:** Source provenance  
**Location:** `Uploaded archive root`

**Evidence.** The zip has no .git directory. The freeze and file manifest are useful, but there is no auditable commit graph or signed tag in the delivered archive.

**Impact.** Change provenance and review history cannot be reconstructed from this artifact alone.

**Required correction.** Release from a tagged commit and include commit SHA, tag, source archive hash and signed release metadata.

## 6. Scientific formulation assessment

### 6.1 AC power flow

The internal Newton–Raphson path has comparatively strong unit/scientific coverage, and base cases 30, 57 and 118 are cross-checked against PYPOWER. This is a sound base. The remaining scientific risks are at the boundaries around the solver: case construction, controllable-device semantics, Q-limit independence, and robust aggregation.

### 6.2 ORPD controls

The current default formulation must not be described merely by an IEEE case name. At minimum, every published task needs an automatically generated control table showing:

- unique generator-voltage control buses and bounds;
- transformer branches, initial ratios, min/max/step and absolute/incremental interpretation;
- shunt buses, original BS, controlled value definition, min/max/step and sign convention;
- fixed shunts/reactors and all non-controlled devices;
- final dimension and decision-variable order;
- case checksum and source package/version.

Until the generic shunt definitions are replaced, results for case57 and case118 are especially unsafe because the decoder changes existing fixed shunts/reactors to a generic positive 0–5 MVAr range.

### 6.3 Robust studies

The expected-value, mean-risk and worst-case branches are structurally reasonable. CVaR must be replaced. The exact discrete weighted upper-tail calculation must include only the fraction of the VaR atom required to fill the tail probability. Tests should include:

- equal weights aligned with alpha;
- unequal weights where alpha lies inside an atom;
- repeated values/ties at VaR;
- alpha near 0 and 1;
- invalid/zero/negative/NaN weights;
- identical CPU, torch and device-resident outputs.

### 6.4 Independent validation

Validation should reconstruct and solve from the original controlled case, not the internally switched final case. A final validation record should include case/control hashes, solver options, Q-limit enforcement, final PV/PQ types, voltage/loss deltas, constraint decomposition and independent feasibility.

### 6.5 Algorithms and fairness

The repository does several fairness-critical things correctly: same normalized space, evaluator, budget and seed plan. But fairness is not complete until each baseline is formulation-tested. A single 679-line tensor dispatcher for 19 algorithms concentrates risk. One-step equations and seeded trajectory snapshots are essential before claiming canonical implementations.

## 7. GUI and workflow assessment

The current source is more advanced than its GUI tests. `MainWindow` defines 16 workspaces and correctly inserts Portfolio Manager between Algorithms and CALO Intelligence. The validation-open path targets `ValidationAuditPanel` at index 11. The source sequence is coherent, but test fixtures still assume 14 pages, old indices and 50/30/20 scheduling.

All mandatory GUI tests should run in offscreen mode on CI. At least one test must execute the complete guided path:

Power System → ORPD → Algorithms → Portfolio → CALO → Scenarios → Experiment → Statistics → Results Review → Validation → Publication Export.

The test should assert that stale upstream changes invalidate all dependent stages.

## 8. Accelerator and throughput assessment

The intended design—80% CUDA, 10% XPU, 10% CPU, with optional 100% CUDA—is consistently represented in current configuration and release notes. It refers to numerical job assignment, not guaranteed operating-system utilization, which is correctly stated.

Release cannot claim the persistent XPU path until F-004 is fixed and tested. After correction, target-hardware validation should measure:

- actual job counts by device and fallback/work-stealing reasons;
- population evaluations/s and power-flow solves/s;
- host-device transfer bytes and synchronization count;
- device memory high-water mark and OOM recovery;
- CPU/reference parity on deterministic and robust mixed-variable cases;
- reproducibility across repeated seeds and process restarts;
- graceful pause/resume and sidecar crash recovery.

## 9. Publication and evidence assessment

The publication pipeline currently confuses three concepts:

1. **stored** — a run exists;
2. **independently verified** — a cross-check was executed and passed;
3. **feasible** — the solution satisfies the ORPD constraint policy.

Only verified **and feasible** objective values should enter objective-performance summaries. Verified infeasible runs are still useful, but only for feasibility rates and violation distributions. Unverified runs may appear in diagnostic dashboards, never in article-ready tables/figures unless explicitly labeled exploratory.

Recommended publication gate:

- intended campaign plan hash matches executed jobs;
- every intended run is complete or explicitly accounted for;
- frozen CALO manifest passes;
- all published runs are independently verified;
- objective tables use verified feasible rows only;
- feasibility/violation statistics include every verified row;
- no algorithm with zero verified feasible runs receives an objective rank;
- figures and text derive from the same filtered evidence dataset;
- package records code, checkpoint, case, formulation, scenario, dependency and raw-array hashes.

## 10. Prioritized remediation plan

### P0 — stop-release corrections

1. Replace all three CVaR implementations and add exact cross-backend tests.
2. Replace generic shunt defaults with versioned case-specific formulation profiles; regenerate all prior benchmark evidence.
3. Fix the persistent XPU sidecar callback and exercise it end to end.
4. Make publication objective statistics verified-and-feasible only.
5. Make article-ready narratives/tables/figures verified-only or refuse incomplete export.
6. Fix campaign repetition synchronization so 30–50 means exactly the requested number.

### P1 — scientific/release gate

1. Redesign independent Q-limit cross-validation from the original case.
2. Reject invalid/empty scenario configurations and record scenario manifests.
3. Enforce case validation in every load/build path; reject zero-impedance active branches.
4. Update all seven failing GUI/workflow tests and run with full dependencies.
5. Add case/control-vector regressions, case300 coverage, robust/contingency tests and baseline formulation tests.
6. Remove stale v1.0.6 publication outputs and stale v2/v3.2 labels.

### P2 — hardening

1. Introduce CI, exact environment locks and coverage/lint gates.
2. Sanitize absolute paths and historical snapshots.
3. Harden checkpoint/worker deserialization.
4. Split oversized orchestration/database/plotting modules.
5. Add font preflight and resolved-font provenance.

## 11. Acceptance gates for the corrected release

The corrected repository should not be called publication-ready until all of the following pass:

- `python -m pytest -q` → zero failures with all mandatory dependencies;
- coverage ≥60% initially and ≥85% for CVaR, campaign, publication, scheduler and validator modules;
- `ruff check .` → zero errors;
- Bandit reviewed with no unresolved medium finding on user-controlled inputs;
- exact CVaR oracle tests pass on CPU/torch/device-resident backends;
- case formulation manifests and base/control regression values pass for 30/57/118/300;
- campaign requested runs equal stored/planned runs for 30, 31, 35 and 50;
- independent validation begins from the original case and confirms Q-limit outcomes;
- article-ready package contains no unverified objective evidence and no infeasible objective rankings;
- persistent CUDA/XPU/CPU tests pass on target hardware, including sidecar progress and crash recovery;
- wheel/sdist, source manifest and frozen manifest all pass from a clean tagged commit.

## 12. Final conclusion

CALO-RPD Studio v3.3.0 is a substantial and thoughtfully structured research application, not a superficial prototype. Its shared evaluator, workflow, freeze mechanism and accelerator architecture are valuable. Nevertheless, the demonstrated CVaR, shunt-formulation, publication-filtering, campaign-run and XPU-sidecar defects are sufficient to invalidate robust benchmark and publication claims in the current archive. Correcting those defects and rebuilding evidence from scratch is mandatory; patching only the GUI tests or release notes would not be adequate.
