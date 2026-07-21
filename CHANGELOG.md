# Changelog

## 4.1.0

- Added a database-backed CALO Policy Center with policy discovery/import, immutable SHA-256 provenance, scientific status/grades, active-policy selection, archive, and reference-safe deletion.
- Added paired Policy Qualification on development cases: Candidate vs selected reference vs No-AI CALO under identical seeds and FE budgets; IEEE 118/300 remain protected holdouts by default. A+ qualification requires statistically supported paired superiority rather than training loss/return alone.
- Added the native CALO v4.1 policy schema: 32 features = historical 24-D cognitive state plus HPEM occupancy/consensus/readiness, success-memory density, learning-lane fraction, precision status/radius, and variable-group concentration. Legacy 24-D checkpoints remain explicitly classified rather than silently treated as native.
- Added strict experiment-policy bindings containing policy ID/name/path/SHA, architecture, state/action/training schema, qualification status/grade, and deterministic setting. Checksum/schema mismatches block strict numerical execution.
- Added an explicit No-AI CALO mode. Unqualified policies require explicit research-only permission and cannot become the default active policy through normal activation.
- Updated policy-training checkpoint metadata/state to v4.1 semantics while retaining exact epoch-level training resume. Real-optimizer Policy Qualification remains mandatory because the lightweight PPO rollout environment is not claimed to be bit-identical to the complete runtime transition loop.
- Added centralized ExperimentWorkspaceRestorer and persistent workflow/UI snapshots. Existing experiments now restore scientific configuration, CALO intelligence settings, workflow access, stored Live Optimization histories, selected run/plot view, and result/statistics/validation context.
- Wired historical experiment opening/resume to `LiveOptimizationPanel.load_experiment()` so old convergence plots are reconstructed from stored numerical evidence.
- Added end-to-end CALO runtime timing metadata separating policy inference, candidate generation, evaluator, and learning update. Evaluator warm-up/calibration is explicitly labelled evaluator-only.
- Preserved the v4.0 seeded CALO search trajectory for No-AI regression cases; host-side overhead reductions do not alter frozen stochastic peer-sampling semantics.
- Added adaptive persistent exact-evaluation caching: exact within-batch deduplication always remains, while low-yield cross-batch storage can disable itself without changing requested FE accounting.
- Added an in-GUI v4.1 dispute register so unresolved/partial performance and policy-science issues cannot be mistaken for closed findings. Full Torch/CUDA-native CALO control and indefinite policy/experiment continuation are intentionally deferred.
- Removed obsolete root-level per-version `PATCH_NOTES_v*.md`; consolidated release history remains in this changelog.

## 4.0.0

- Introduced CALO v4, a tensor-native constraint-cognitive architecture implementing the repository upgrade plan with a maximum persistent tensor dimensionality of 3D.
- Added persistent personal/lineage memory so successful individual learning is not destroyed after environmental selection.
- Added Hierarchical Prefix Elite Memory (HPEM): one canonical `[7,D]` quality-diversity elite store deriving Best-1/3/5/7 `[4,D]` multi-resolution knowledge without duplicate storage.
- Added mixed-variable-aware HPEM distance semantics so discrete tap/shunt values that decode to the same lattice state are treated consistently.
- Added batched contextual operator and memory-level credit `[regime, operator, context]` / `[regime, memory, context]` with per-individual regime attribution and order-independent updates.
- Reworked bounded success-direction memory into compact 3D `[operator, history, variable]` storage with finite context-aware sampling.
- Added regime-aware variable-group intelligence for generator voltages, transformer taps, and shunts.
- Made `recovery_fraction` operational through bounded partial recovery without whole-population forgetting.
- Added behavior-driven epsilon control, memory readiness/consensus, and single-budget discovery/learning dual-lane search.
- Added counted cognitive precision refinement with adaptive trust-radius behavior and legal mixed-variable moves; no hidden extra optimizer or uncounted evaluations are used.
- Added exact decoded-control deduplication/cache while preserving requested function-evaluation accounting.
- Added reusable scratch buffers and temporary 3D broadcast calculations to reduce duplicate state and allocation overhead.
- Added strict benchmark guards that block historical priors/warm starts unless an explicit transfer-learning study disables strict mode.
- Expanded CALO ablation coverage and added v4 regression tests for HPEM, persistent pbest, credit stability, budget accounting, cache accounting, reproducibility, and benchmark independence.
- No claim of universal CALO superiority is made in this release; final ranking requires frozen 30–50 run paired feasible comparisons and independent validation.

## 3.4.3

- Fixed Portfolio Export appearing permanently stuck at 94% on the final `reproducibility_bundle` artifact. The final archive now reports bounded 94–99% sub-progress and reaches 100% only after the archive is closed and atomically committed.
- Replaced unbounded recursive output-directory zipping with a portfolio-scoped reproducibility bundle so unrelated/older exports are not accidentally re-compressed. Already-compressed PNG/PDF/ZIP/NPZ-type artifacts are stored without redundant deflate work and text artifacts use fast level-1 compression.
- Added safe cancellation inside reproducibility-bundle creation. A pause request removes the temporary archive, preserves the 16 completed artifacts, and resumes only the missing final bundle.
- Moved standard verified-publication export off the Qt GUI thread and added visible progress/cancellation so large publication packages no longer freeze the interface.
- Eliminated `RuntimeWarning: All-NaN slice encountered` in statistical, live-progressive, portfolio, and publication-evidence median/IQR convergence calculations by removing all-NaN grid columns before reduction.
- Added atomic temporary ZIP creation and a manifest snapshot inside the reproducibility bundle.
- Added regression tests for 100% completion, bundle scoping, safe pause/resume during artifact 17, and NaN-safe convergence export.

## 3.4.2

- Fixed Live Optimization being permanently locked to repeated run 1. Telemetry is now stored independently for every repeated run, with Auto-follow and manual run selection; no parallel-run telemetry is discarded.
- Added Portfolio-aware live/progressive previewing. Selected convergence/CALO diagnostics update live; single-run power-system plots become available immediately after a run is committed; repeated-run summaries update provisionally as evidence accumulates; final inference outputs remain explicitly final-stage.
- Replaced direct PYPOWER `ENFORCE_Q_LIMS=1` dependence with an independent PV-only aggregate Q-limit switching loop around PYPOWER's own Newton solver, matching CALO-RPD's retained REF/slack semantics and fixing IEEE-300 cross-validation.
- Corrected reference/slack reactive-power reporting so the stored `QG` reflects the actual solved network requirement even when it exceeds declared limits; violations are no longer silently clipped in result state.
- Added IEEE-300 to the command-line case validation matrix and mandatory internal-vs-PYPOWER scientific regression suite.
- Improved fairness-audit responsiveness with staged progress reporting, accelerator-first parity selection, and CPU-thread throttling during CPU fallback parity checks.
- Retained GPU-preferred numerical scheduling: 100% of tensor-compatible numerical work is assigned to CUDA when available, with XPU then CPU fallback; Qt, persistence, and independent PYPOWER validation remain host-side by necessity.
- Removed obsolete root-level `PATCH_NOTES_v*.md` files; release history is consolidated in `CHANGELOG.md`.

## 3.4.1

- Fixed `UnboundLocalError: combined` in `CrossRunBatchBroker` when concatenation fails before the merged matrix is created.
- Preserved and propagated the original batching error instead of masking it during performance-ledger accounting.
- Added representation/device/dtype/decision-width compatibility keys so NumPy and Torch populations, or tensors from different devices, are never merged accidentally.
- Added a broker-liveness watchdog and an outer worker guard so a daemon failure cannot leave experiment or GUI threads blocked indefinitely.
- Added validation for malformed/empty candidate matrices and regression tests for failed concatenation, recovery, mixed representations, and broker liveness.
- Scientific equations, CALO operators, seeds, evaluation budgets, and the v3.4 ORPD formulation remain unchanged.

## 3.3.0

- Added the CUDA-Resident Execution Engine for comparative evaluation and ORPD policy-training rollouts.
- Changed the recommended fixed device plan to 80% CUDA, 10% XPU and 10% CPU; added a 100% CUDA mode.
- Added a device-resident FP64 ORPD evaluator with tensor mixed-variable decoding, grouped PV/PQ switching, batched power flow, branch flows, objectives, constraints, L-index and robust aggregation.
- Removed population-level CUDA-to-NumPy-to-CUDA round trips from tensor-native optimizer evaluation; one packed host materialisation remains for the public result/provenance contract.
- Made all twenty primary algorithms and CALO ablation variants eligible for accelerator allocation under the torch FP64 backend.
- Added CUDA-priority work stealing for unstarted jobs and retained persistent cross-run batching.
- Updated policy-training rollout defaults and GUI presets to 80/10/10 and optional 100% CUDA.
- Added v3.3 parity, allocation and device-residency regression tests.

## 3.2.1

- Fixed a startup crash in Resume Center caused by connecting experiment-manager signals through an undefined local name.
- Resume Center now consistently uses the stored `self.manager` reference for completed, cancelled, and failed signals.
- Added a regression check so this constructor-scope error cannot silently return in future releases.

## 3.2.0

- Added Portfolio Manager immediately after Algorithms, with single-run and overall-experiment portfolio modes.
- Added evidence-strength profiles and article presets that derive the minimum paired repetitions from requested plots, tables, statistics, validation, robust scenarios, and benchmark-block requirements.
- Added portfolio-aware result storage and selective multi-format result-portfolio generation with line, scatter, bar, box, violin, and heatmap outputs.
- Added exact scientific fingerprints and read-only run reuse across compatible portfolios while excluding operational and portfolio-only settings from the hash.
- Added persistent campaigns, per-job journals, atomic result commit, duplicate protection, and exact remaining-job calculation.
- Added system-wide Resume Center and startup unfinished-work detection after unclean shutdowns.
- Added safe experiment pause at independent-run boundaries, emergency-stop recovery, policy-training epoch resume, bulk-validation resume, and artifact-level portfolio-export resume.
- Added shared-trace reference protection so deleting a reused run cannot remove an array file still referenced by another experiment.
- Retained the v3.1 persistent CUDA/XPU/CPU batched-throughput engine for comparative evaluation and CALO policy training.

## 3.1.0

- Added the **Batched Throughput Engine** with one persistent worker/runtime per CUDA, Intel XPU, and CPU lane.
- Added real cross-run batching that merges compatible population-evaluation requests while preserving per-run ordering, seeds, equations, and evaluation budgets.
- Added automatic FP64 evaluator microbatch calibration and reusable JSON throughput profiles.
- Added measured candidate-evaluations-per-second job allocation and concurrency planning; manual weighted scheduling remains available.
- Reworked batched AC power flow to build candidate admittance matrices, solve Newton–Raphson systems, calculate generation, and calculate branch flows through vectorized tensor batches, with isolated candidate-specific PV-to-PQ fallback.
- Added in-memory stage profiling for decoding, scenario preparation, power flow, objective/constraint evaluation, robust finalization, and cross-run batching.
- Added lower-frequency live telemetry and end-of-run buffered persistence to reduce synchronization and I/O pressure.
- Added persistent policy-training actor interpreters, persistent CPU rollout processes, discarded actor-throughput calibration, and measured-throughput episode allocation.
- Added cross-episode FP64 ORPD population batching during policy-training development stages while preserving one-policy-snapshot on-policy PPO collection.
- Added CALO Intelligence controls for adaptive training allocation, persistent actors, accelerator-native ORPD rollouts, batch size, batch window, and maximum merged candidates.
- Added `calo_v31_freeze.json` to freeze the v3.1 scientific backend and throughput orchestration before final TEST campaigns.
- Added v3.1 throughput-engine regression tests for cross-run merging, measured allocation, policy-lane tuning, and configuration round trips.

## 3.0.0

- Added a common PyTorch FP64 scientific backend for CPU, NVIDIA CUDA, and supported Intel XPU execution.
- Added batched AC Newton–Raphson power flow, branch-flow evaluation, constraint decomposition, robust aggregation, and Kessel–Glavitsch L-index on the selected torch device.
- Added exact torch mixed-variable decoding for continuous generator voltages and discrete transformer/shunt controls.
- Added canonical tensor kernels for all nineteen non-CALO primary algorithms; CALO uses the same physical evaluator plus its cognitive policy.
- Made the complete primary comparison and ablation plan accelerator-compatible under the torch backend.
- Added reproducible CPU/accelerator parity auditing and an optional mandatory parity gate before experiments.
- Added v3 backend, dtype, device, batch-size, and parity information to GUI controls, fairness reporting, and run provenance.
- Added a new v3 freeze manifest covering the accelerator evaluator and torch baseline suite.
- Retained the legacy CPU reference backend for independent scientific auditing.

## 2.0.3

- Added automatic visible-data axis fitting for Live Optimization so stale or excessively broad limits cannot hide convergence differences.
- Added zero-aware scaling for non-negative metrics such as constraint violation, feasibility ratios, diversity, and operator-success rates while best-feasible objective plots retain tight data-driven scaling.
- Added Plot Tools controls for Auto-fit visible data, zero-baseline inclusion, and auto-fit padding; manual axis bounds remain available when auto-fit is disabled.
- Selective preview filtering now participates in scaling, so the displayed axis range is recalculated from the series that are actually visible.
- Added reusable bulk-validation helpers for current-experiment, repository-wide, and campaign validation workflows.
- Added background bulk validation to Validation & Audit with current-experiment and all-not-yet-verified scopes, live progress, passed/failed/error counts, cancellation between runs, and a JSON summary.
- Existing verified runs are skipped by default; failed or unverified runs can be independently checked again after corrections.
- Added regression tests for tight convergence scaling, visible-series-only scaling, bulk run selection, and continuation after an individual validation error.

## 2.0.2

- Added synchronous weighted heterogeneous policy-rollout actors with configurable CUDA/XPU/CPU shares (default 50/30/20).
- All actor lanes receive the same policy snapshot per PPO epoch; stale or mismatched actor payloads are rejected before the on-policy buffer is built.
- Added batched policy inference on CUDA and Intel XPU actor lanes while CPU actor processes collect their assigned share concurrently.
- Added secondary-XPU-runtime actor support without replacing the primary CUDA-enabled PyTorch installation.
- Kept one centralized PPO learner device and delayed every update until all current-policy actor lanes finish, preserving PPO's on-policy semantics.
- Added deterministic largest-remainder allocation, effective-share reporting, actor-lane provenance, and candidate-checkpoint metadata.
- Added GUI controls for rollout mode and CUDA/XPU/CPU transition shares, with live attainable-allocation reporting.
- Protected the bundled frozen CALO v2 checkpoint from being overwritten by weighted candidate training.
- Preserved the original 23-file frozen CALO benchmark manifest; a newly trained candidate must be validated and explicitly re-frozen before TEST use.

## 2.0.1

- Added a deterministic weighted CUDA/XPU/CPU experiment scheduler with default 50/30/20 shares for accelerator-compatible CALO jobs.
- Added pre-run attainable-allocation reporting, planned device lanes in the run queue, and stored lane provenance.
- Kept conventional optimizers and AC power flow on CPU unless a genuine accelerator-native implementation exists; no false GPU assignment is reported.
- Added static lane concurrency caps while retaining utilization and memory thresholds as safety gates.

## 2.0.0

- Added a cryptographic **Frozen CALO** gate covering CALO mathematical implementation files, operator definitions, cognitive state, archive rules, PPO architecture, policy checkpoint, frozen training-data snapshot, default hyperparameters, mixed-variable decoding, and feasibility rules.
- Added bundled `calo_v2_freeze.json` verification and a final-benchmark rule that blocks TEST execution when any frozen file changes.
- Added the **Benchmark & Evidence** workspace for final 20-algorithm TEST campaigns.
- Added the standard v2 benchmark suite for IEEE 30-, 57-, 118-, and 300-bus systems.
- Added deterministic, mixed discrete-continuous, load-uncertainty mean-risk/CVaR, renewable-uncertainty mean-risk/CVaR, and N-1 branch/generator worst-case study profiles.
- Enforced exactly all 20 primary algorithms, equal objective-function evaluation budgets, shared run-level seed tuples within each task, and 30–50 independent runs per algorithm/task.
- Added campaign manifests with per-task configuration, planned job counts, experiment IDs, and completion state.
- Final campaign experiments are automatically classified and locked as **TEST**, preventing historical-learning leakage.
- Added bulk independent validation for completed campaign runs.
- Added campaign-level descriptive statistics, feasible-run rate, evaluations to first feasibility, runtime summaries, global Friedman testing, average ranks, CALO-vs-baseline Wilcoxon signed-rank tests with Holm correction, rank-based Cliff's delta effect sizes, and Nemenyi critical-difference information.
- Added evidence-based interpretation that reports observed winners, statistical support, non-significant outcomes, and cases where a baseline outperforms CALO without generating a universal superiority claim.
- Added advanced publication figures: median convergence with IQR bands, feasible-run probability, evaluations-to-feasibility distributions, constraint decomposition, CALO operator utilization, CALO operator success, cognitive-regime timelines, boxplots, violin plots, global ranking, critical-difference diagram, and robustness map.
- Added a Transactions research-package builder containing verified/all-run tables, raw JSON run records, compressed convergence/population arrays, seeds, controls, full reconstructed power-flow states, validation records, experiment configurations, frozen CALO source/checkpoint artifacts, statistical reports, article-ready evidence summary, figures, and a reproducibility ZIP archive.
- Added the `calo-rpd-final-benchmark` command-line entry point.
- Added IEEE 300-bus loading support to the standard case loader and benchmark CLI.
- Made the navigation list vertically scrollable so the expanded v2 workspace set does not compress on smaller displays.
- Added v2 freeze/campaign/unit tests and updated GUI regression expectations for the new Benchmark & Evidence workspace.

## 1.3.0

- Moved Live Optimization series visibility from the permanently expanded preview checkbox section into a context-sensitive **Preview series** icon inside Plot Tools.
- Added dynamic Preview series checkboxes with **Select all**, **Clear all**, and **Restore default**; preview visibility remains independent from export-series selection and never deletes raw data.
- Added a leakage-aware **Historical Experience Learning** subsystem under CALO Intelligence.
- Added explicit experiment roles **TRAIN**, **VALIDATION**, **TEST**, and **EXCLUDED**; existing and new experiments remain excluded from learning unless explicitly classified, and only eligible TRAIN experiments may enter a learning repository.
- Added database migration fields for experiment learning role, eligibility, and classification lock while preserving compatibility with existing v1.2 databases.
- Added checksum-verified historical experience repositories containing eligible CALO policy trajectories, cross-algorithm solution exemplars, and CALO parameter priors.
- Added offline historical CALO policy pretraining before fresh on-policy PPO; historical trajectories are never inserted into PPO's on-policy rollout buffer.
- Added conservative reconstruction of usable legacy v1.2 CALO diagnostic histories as lower-weight partial regime/operator examples when enough telemetry is available; reconstructed samples do not supervise the continuous parameter head.
- Added optional problem-compatible cross-algorithm population warm starting and historical CALO parameter-prior blending for practical warm-start operation.
- Added **Cold Start**, **Historical Warm Start**, and **Continual Learning** modes; continual mode rebuilds the eligible repository but never silently retrains or promotes the deployed policy.
- Added GUI controls to classify and lock historical experiments, build/preview repositories, configure historical policy pretraining, enable parameter priors, and enable optional population warm starts.
- Added CLI options for historical repository selection and historical pretraining epochs.
- Added regression, integration, and unit coverage for preview-series placement/filtering, train/test leakage protection, repository generation, legacy-trajectory reconstruction, and historical policy pretraining.

## 1.2.4

- Replaced the simple CUDA/CPU scheduler with an accelerator-first heterogeneous scheduler using the fixed default priority **NVIDIA CUDA → Intel XPU → CPU**.
- Added dynamic discovery and live telemetry for all verified CUDA devices and available Intel XPU devices; PyTorch backend IDs are shown separately from Windows Task Manager GPU numbering.
- Added separate CUDA compute/VRAM, XPU compute/device-memory, CPU utilization, and system-RAM admission thresholds plus explicit per-accelerator job caps.
- Added strict admission ordering: compatible CALO AI jobs receive first refusal on CUDA, then XPU, before CPU fallback is considered; conventional baseline algorithms remain CPU workloads.
- Added an isolated Intel-XPU sidecar runtime so mixed NVIDIA/Intel systems can retain a CUDA-enabled primary PyTorch build while using an independent XPU-enabled PyTorch interpreter for Intel-GPU jobs and policy training.
- Extended the first-launch prerequisite wizard to detect Intel graphics, provision/repair the secondary XPU runtime, verify real XPU computation, and retain full download bytes/speed/ETA installation telemetry.
- Extended CALO policy-training device selection to CUDA, direct XPU, secondary XPU runtime, and CPU with automatic CUDA → XPU → CPU preference.
- Added resource-inventory status to Experiment Manager and persisted CUDA/XPU/CPU device assignments and expanded accelerator provenance.
- Added XPU-aware experiment configuration fields, JSON schema entries, configuration round-trip coverage, resource-priority tests, XPU no-utilization fallback tests, and system-RAM safety tests.
- Preserved the scientific rule that running optimizer jobs are never migrated mid-run and that strict runtime comparisons should use CPU-only single-worker execution.

## 1.2.3

- Added structured prerequisite-installation progress telemetry to the first-launch wizard.
- Added coarse overall setup stage progress and explicit current-stage labels.
- Added exact per-artifact pip download byte counters and percentages using pip raw progress output.
- Added measured download speed, ETA, and total elapsed installation time.
- Suppressed repetitive raw progress records from the text log while retaining normal pip diagnostic output.
- Kept download totals scientifically honest: the GUI reports exact current-artifact bytes, while the overall bar reports installation stages because pip resolves the full dependency graph dynamically.

## 1.2.2

- Added a first-launch prerequisite setup wizard that runs before the PyQt6 application and can install/repair the scientific environment.
- Added automatic NVIDIA detection and CUDA-enabled PyTorch backend selection with verified GPU tensor execution and CPU fallback.
- Added adaptive heterogeneous CPU/GPU experiment scheduling for both primary comparisons and CALO ablation studies.
- Added configurable GPU utilization, GPU memory, CPU utilization, and GPU-job admission targets.
- Enabled CUDA CALO policy inference during experimental evaluation while retaining CPU AC power flow and baseline optimizers.
- Recorded per-run compute-device assignments and accelerator provenance.
- Added fairness warnings for heterogeneous runtime comparisons.
- Added regression tests for prerequisite backend selection, resource admission, scheduler GUI controls, and compute-policy configuration round-trips.

## 1.2.1

- Added resource-aware CALO policy training with spawn-safe parallel CPU rollout workers.
- Added automatic CUDA detection and selectable CPU/CUDA/automatic device routing for centralized PPO updates.
- Kept rollout workers on CPU and the accelerator in the parent training process to avoid unsafe CUDA tensor sharing across Windows worker processes.
- Added deterministic per-episode seeds so rollout results do not depend on worker scheduling order.
- Added GUI controls for PPO compute device and parallel rollout worker count, including a recommended-worker action and detected accelerator status.
- Added training metadata recording rollout worker count, selected PPO device, CUDA availability, GPU name, and transition count.
- Added CLI options for training device and rollout-worker count.
- Preserved scientific reproducibility and benchmark-leakage safeguards.

## 1.2.0

- Rebuilt the proposed optimizer as CALO Core v2.
- Added adaptive epsilon-feasibility, feasible and constraint-boundary archives, per-individual operators, mixed-variable neighbourhood learning, environmental selection, success-distribution memory, online operator credit, and temporary diversity recovery.
- Added separate objective and constraint stagnation state.
- Added component-wise constraint diagnostics, exact/epsilon feasible ratios, adaptive epsilon, diversity, operator success, archive size, regime, and first-feasibility telemetry.
- Rebuilt the CALO policy as a hierarchical regime/operator/Beta-parameter actor-critic.
- Replaced the earlier policy-gradient-style training with actual PPO mechanics including clipped ratios and GAE.
- Unified training and runtime operator/selection modules.
- Expanded the CALO ablation suite to nine fixed variants.
- Added checkbox-based selective live preview generated from current plot series names.
- Preserved square live preview, selective export, and 600–2400 DPI PNG output.
- Kept the repository free of GitHub Actions workflows.

## 1.1.0

- Added complete local experiment-history management for removing obsolete completed runs or entire experiments.
- Deleting a run also removes its validation records and the referenced compressed convergence/final-population trace array.
- Deleting an experiment removes its completed runs, failed-run records, validation records, and all referenced local trace-array files.
- Added guarded **Delete all experiment history** with an explicit `DELETE ALL` confirmation phrase.
- Added automatic SQLite WAL checkpointing and database compaction after destructive history operations.
- Added storage summaries showing experiment, run, validation, failure, trace-file, and referenced trace-storage counts.
- Added **Manage history** to Results Explorer and a globally accessible **Experiment history** section in Application Settings.
- External publication-export folders are deliberately left untouched because they are independent user-managed copies.
- Added integration and GUI regression coverage for history deletion and trace cleanup.

## 1.0.10

- Reorganized the Experiment Manager into an explicit three-step workflow: configuration, fairness audit, then study execution.
- Moved the fairness audit ahead of all comparison and ablation controls so the required sequence is visually unambiguous.
- Added a page-level vertical scroll area only to the genuinely long Experiment Manager workspace, preventing Qt from compressing spin boxes, combo boxes, labels, and buttons on laptop-height displays.
- Preserved a fixed workspace header, disabled horizontal scrolling, and retained full-width responsive controls.
- Added minimum control heights and a dedicated run-queue section for consistent readability.

## 1.0.9

- Connected the Experiment Manager's Parallel workers setting to the GUI execution backend using spawn-safe CPU process parallelism across independent optimizer/run jobs.
- Added throttled inter-process progress telemetry, safe cancellation, parent-only SQLite persistence, and exact run/algorithm queue updates.
- Added canonical execution planning so the GUI and backend agree on the exact number and identity of jobs.
- Clarified the difference between the primary 20-algorithm comparison and the seven-variant CALO ablation study.
- Added explicit job-count summaries and a confirmation dialog before CALO ablation execution.
- Added a recommended CPU worker selector and guidance that GPU/disk utilization is expected to remain low for this CPU-bound workload.
- Added a fairness notice that parallel throughput mode is not suitable for strict wall-clock runtime ranking because jobs contend for CPU resources.
- Prevented interleaved process-parallel telemetry from mixing repeated runs on the live convergence canvas.


## 1.0.8 — 2026-07-12

- Fixed an empty Live Optimization plot when no feasible incumbent had yet been found.
- Added Automatic convergence mode, which displays constraint-violation progress until feasibility is available for every monitored optimizer, then switches to best-feasible objective.
- Added explicit informative empty-state messages instead of blank Matplotlib canvases.
- Live Optimization now reloads stored convergence histories after experiment completion or cancellation, so the plot remains available when the page is opened after the run.
- Synchronized application version labels and provenance with the release version.

## 1.0.7

- Corrected Live Optimization convergence semantics: objective convergence now uses best feasible objective and constraint convergence is shown separately.
- Changed convergence x-axis from iteration count to objective-function evaluations for fair cross-algorithm comparison.
- Prevented repeated runs from being concatenated into one false convergence curve; the live view now resets per repeated run.
- Added convergence metric selector for best feasible objective and best normalized constraint violation.
- Fixed Results Explorer row selection so selected-run details and the review-to-validation transition always work.
- Made review-to-validation navigation atomic and independent of signal ordering.
- Updated statistical convergence to evaluation-aligned median best-feasible histories when v1.0.7 telemetry is available.

## 1.0.6 — 2026-07-12

- Fixed the Results Explorer review action so confirming a selected run now unlocks and immediately opens Validation & Audit on that exact run.
- Added explicit reviewed-run handoff from Results Explorer to Validation & Audit.
- Added export-time series selection generated from the legend-capable series currently available in the plot preview.
- Added Select all and Clear all actions for export series.
- Exported figures now include only checked data series and only the corresponding legend entries, while the live preview is restored unchanged after saving.
- Preserved square live preview, exact square export, organized popup plot tools, and 600–2400 DPI PNG selection.

## 1.0.5 — 2026-07-12

- Replaced the dense always-visible plot-formatting control area with a compact four-icon tool strip.
- Added focused popup editors for Text & labels, Plot appearance, Export figure, and Style profiles.
- Kept independent typography controls for titles, axis labels, tick labels, legends, and annotations.
- Preserved square live preview, exact square export, and 600–2400 DPI PNG selection.
- Added theme styling and GUI regression coverage for the popup-based plot tools.

## 1.0.4 — 2026-07-12

- Live Optimization now uses an exact 1:1 square Matplotlib preview.
- Live Optimization content is vertically scrollable so the square plot is never compressed on shorter displays.
- Live-plot exports are forced to an exact square page/canvas for PNG, SVG, and PDF.
- PNG export now provides a selectable 600–2400 DPI range with a 600 DPI default.
- Square exports lock width and height together and disable tight cropping to preserve exact 1:1 output dimensions.
- GitHub Actions workflow files were removed; the guided scientific workflow remains entirely inside the desktop software.

## 1.0.3 — 2026-07-11

- Reworked the desktop shell and visual system for a sharper modern interface.
- Removed unnecessary page-level scroll areas from compact and data-centric workspaces.
- Rebuilt the Dashboard and Experiment Manager layouts.
- Prevented duplicate experiment-start requests from raising an uncaught runtime error.
- Added explicit experiment busy-state handling and safer QThread lifecycle cleanup.

## 1.0.1 — 2026-07-11

- Restored PYPOWER cross-validation compatibility with NumPy 2.4 environments by importing only the required PYPOWER modules and adding graceful third-party failure handling.
- Constrained the supported NumPy range to versions below 2.4 for complete PYPOWER compatibility.
- Added deterministic Qt palettes and expanded light/dark theme rules so labels, controls, menus, tables, tabs, and toolbars remain readable regardless of the Windows system palette.

## 1.0.0 — 2026-07-11

- Complete CALO-RPD Studio desktop application.
- Twenty primary optimization algorithms through a common evaluation interface.
- AI-assisted Cognitive Adaptive Learning Optimizer (CALO).
- AC Newton-Raphson power flow, mixed-variable ORPD, robust scenarios, statistics,
  independent validation, reproducibility records, and publication export.
- Modern PyQt6 interface with thirteen scientific workspaces.
- Global plot-formatting toolbar with editable typography, labels, legends, axes,
  curves, markers, and vector/raster export.