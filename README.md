# CALO-RPD Studio

**CALO-RPD Studio 3.4.3** is a Python/PyQt6 research platform for deterministic and robust optimal reactive power dispatch (ORPD), reproducible comparison of twenty optimizers, CALO policy development, independent validation, statistical analysis, and publication evidence generation.

Version 3.4.3 is the **Publication Export Completion and Responsiveness Release**. It fixes the final portfolio-export artifact that could appear stuck at 94%, bounds reproducibility packaging to the current portfolio evidence, adds sub-progress and safe cancellation during archive creation, moves standard publication export off the GUI thread, and removes all-NaN convergence warnings. All v3.4.2 live-evidence, IEEE-300 validation, and GPU-preferred numerical scheduling behavior is retained.

## v3.4.3 publication export completion and responsiveness

- Portfolio artifact 17 (`reproducibility_bundle`) now reports 94–99% packing progress and reaches 100% only after an atomic ZIP commit.
- Reproducibility ZIP creation is scoped to current portfolio evidence rather than recursively compressing arbitrary files below the selected output directory.
- PNG, PDF, ZIP, NPZ, and other already-compressed artifacts are stored without redundant recompression; compressible text artifacts use fast compression.
- Safe Pause works during final ZIP creation and resumes without regenerating the first 16 completed artifacts.
- Standard verified-publication export runs in a background `QThread`, with progress and cancellation, preventing Qt UI stalls during report packaging.
- Median/IQR convergence calculations remove all-NaN leading columns before reduction, eliminating the `All-NaN slice encountered` warning without fabricating values.
- Reproducibility archives include a portfolio-manifest snapshot and are written through a temporary file before atomic replacement.

## v3.4.2 live evidence, IEEE-300, and responsiveness corrections

- Exact fractional-tail weighted CVaR on CPU, CUDA, and XPU-compatible PyTorch paths.
- Versioned case-specific IEEE 30/57/118/300 control formulations with explicit absolute/delta shunt semantics, preserved fixed reactors, exact bounds, dimensions, checksums, and per-task formulation manifests.
- Central case validation rejects malformed, disconnected, non-finite, or active zero-impedance networks.
- Robust scenarios are validated centrally; empty or invalid robust requests can no longer silently become deterministic, and every run stores normalized weights and transformed-case checksums.
- Independent PYPOWER validation starts from the original controlled case with independent reactive-limit enforcement and compares final bus types, generator Q, voltages, angles, and losses.
- Publication statistics and article packages use independently verified, feasible, finite runs only; article-ready packaging stops when validation is incomplete.
- Requested benchmark repetitions from 30 through 50 are preserved exactly.
- The persistent Intel-XPU progress callback defect is fixed and local worker frames now have bounded lengths and schema checks.
- Fairness and CPU/accelerator parity audits run in a background QThread, keep the GUI responsive, and use explicit FP64 tolerances.
- Model-only checkpoints use restricted `weights_only=True` loading. Application-created training resumes require a matching SHA-256 sidecar.
- Font preflight records whether Times New Roman is available and discloses a DejaVu Serif fallback without bundling proprietary fonts.

The scientific aim is formulation consistency, reproducibility, and higher throughput—not fabricated superiority. A faithful CPU and accelerator implementation must agree within declared tolerances, and publication claims remain gated by feasibility and independent validation.

## v3.2 Portfolio Planning and Universal Resume

The guided workflow is now **Power System → ORPD Formulation → Algorithms → Portfolio Manager → CALO Intelligence → Scenarios → Experiment Manager**. Portfolio Manager provides:

- **Single-run diagnostic portfolios**, restricted to scientifically valid one-run plots;
- **Overall experiment portfolios** with exploratory, journal, Transactions, or custom evidence strength;
- article presets for TLBO/MTLBO, deterministic CALO, robust CALO, and transfer/accelerator studies;
- dependency-aware output checks, including minimum algorithms, repetitions, benchmark blocks, robust scenarios, validation, CALO diagnostics, and accelerator records;
- portfolio-aware trace storage so unnecessary final populations or scenario arrays are not retained;
- exact result reuse based on case, formulation, evaluator, algorithm, policy, budget, seed, and scientific settings—not GUI or storage settings.

Every numerical campaign is journaled in SQLite with planned, active, completed, failed, reused, paused, and interrupted tasks. **Pause safely** stops new admissions and waits for active optimizer jobs to commit; an emergency stop retains completed jobs and restarts only interrupted jobs from their original seeds. On the next launch, stale running records are marked interrupted and the Startup Resume prompt opens Resume Center.

Resume coverage includes:

- comparison and ablation campaigns at committed independent-run boundaries;
- policy training from the last completed PPO epoch, including optimizer and random states;
- bulk validation from the remaining unverified run queue;
- article-portfolio export artifact-by-artifact using an atomic manifest.

The Result Portfolio exporter creates only selected and available evidence, including convergence plots, constraint decomposition, voltage/branch heatmaps, control plots, box/violin/scatter plots, feasibility evidence, CALO cognition/operator plots, statistical tables, captions, and a reproducibility archive. Unsupported evidence is skipped with an explicit scientific reason rather than producing an empty or misleading figure.

## v3.1 Batched Throughput Engine

The recommended **Auto-tuned Batched Throughput Engine** replaces per-run accelerator process creation with:

- one persistent NVIDIA CUDA worker;
- one persistent Intel XPU worker or isolated XPU sidecar;
- one controlled persistent CPU worker pool;
- cross-run batching of compatible populations;
- device-resident case, policy, decoder, and invariant tensors;
- automatic candidate-microbatch calibration;
- measured-throughput CUDA/XPU/CPU job allocation;
- reduced live-telemetry frequency and end-of-run buffered trace persistence;
- cumulative stage profiling stored with result metadata.

Calibration evaluates valid normalized candidates outside every optimizer budget, records candidate evaluations per second, and selects the best stable microbatch for each device. Allocation follows measured capacity rather than an arbitrary utilization percentage. A manual weighted scheduler remains available.

Cross-run batching does not combine incompatible cases, scenarios, objectives, constraint formulations, devices, or precisions. Every synchronous optimizer receives exactly its own ordered evaluations, and its random seed, equations, evaluation count, and stopping rule remain unchanged.

### Policy-training throughput

Weighted CALO policy training can now keep CUDA/XPU actors and the CPU rollout pool resident for the complete session. Short calibration episodes are discarded, then fresh episodes are allocated by measured complete actor transitions per second. During ORPD development stages, compatible episode populations use the same FP64 batch broker as comparative evaluation. All accepted actor trajectories still use one policy snapshot per PPO epoch; PPO begins only after matching current-policy trajectories return, preserving on-policy semantics.

The earlier v3.1 release used an 80/10/10 deterministic fallback. Version 3.4 supersedes that default with GPU-maximum 100/0/0 rollouts: CUDA first, then Intel XPU only when CUDA is unavailable, then CPU only when neither accelerator is usable. Measured utilization still depends on kernel size, synchronization, memory limits, and host-side orchestration.

## v3 scientific backend

The common accelerator evaluator supports:

- double-precision (`float64`) PyTorch execution;
- normalized continuous controls and exact discrete tap/shunt lattice decoding;
- batched candidate and scenario evaluation;
- AC Newton–Raphson power flow with candidate-specific convergence masks;
- aggregate generator reactive-limit enforcement and PV-to-PQ switching;
- complex branch power flows and active-loss calculation;
- bus-voltage, generator-P, generator-Q, branch-thermal, and power-flow constraints;
- Kessel–Glavitsch L-index;
- deterministic, mean-risk, worst-case, and CVaR aggregation;
- isolated fallback for candidates whose Q-limit switching produces different bus sets;
- CPU-reference reconstruction of final publication states.

A failed candidate is marked infeasible without terminating the entire batch.

## All twenty accelerator-compatible optimizers

The v3 torch-native suite contains canonical kernels for:

CALO, TLBO, PSO, CLPSO, MTLA-DE, QODE, Dragonfly, Simulated Annealing, Salp Swarm, continuous ACO/ACOR, Bat, Crow Search, Firefly, Flower Pollination, Grasshopper, Grey Wolf, Moth-Flame, Multi-Verse, Whale Optimization, and Imperialist Competitive Algorithm.

Every method uses the same:

- physical ORPD evaluator;
- mixed-variable decoder;
- constraint definitions and normalization;
- Deb feasibility-first comparison;
- objective-function evaluation accounting;
- robust scenario set;
- seed protocol;
- boundary policy;
- validation rules.

The legacy NumPy/PYPOWER-style CPU reference remains available as a trusted audit backend.

## CPU/accelerator parity gate

Before a final comparison, **Experiment Manager → Run CPU/accelerator parity audit** evaluates a reproducible candidate set on the CPU reference and selected accelerator backend. The fairness gate can require parity before execution.

The audit reports:

- maximum objective error;
- maximum normalized-violation error;
- maximum bus-voltage error;
- feasibility mismatches;
- case, device, dtype, and scenario count.

Final tolerances must be reported with the study. FP64 parity protects against a fast but scientifically different accelerator implementation.

## Throughput-aware device scheduling

With **PyTorch FP64 batched AC Newton–Raphson** selected, the task-share controls apply to the complete job plan. For example, an eight-algorithm × fifty-run plan contains 400 accelerator-compatible jobs. A requested 100% CUDA plan can therefore assign all 400 optimizer jobs to CUDA when a verified CUDA runtime and sufficient lane capacity are available.

The scheduler supports:

- automatic calibration and measured-throughput allocation;
- persistent per-device runtimes and cross-run batching;
- manual weighted CUDA/XPU/CPU assignment;
- accelerator-first admission;
- separate compute and memory safety thresholds;
- device-specific concurrency caps;
- fixed device assignment per run;
- recorded requested lane and actual device in provenance.

“100% CUDA jobs” does not mean Windows CPU utilization will be zero. Python orchestration, process management, case/scenario preparation, SQLite persistence, GUI rendering, file I/O, and independent CPU validation still use host CPU and RAM. It means the optimizer kernel and v3 physical evaluator for each assigned run execute through CUDA.

## Installation and first launch

Create and activate a Python 3.11+ virtual environment, then launch the prerequisite wizard:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python bootstrap.py
```

On Windows, double-click `start_calo.bat`. The first-launch wizard:

- detects NVIDIA CUDA and supported Intel XPU hardware;
- installs or repairs the correct PyTorch backend;
- provisions an isolated XPU sidecar on mixed-GPU systems when needed;
- displays download amount, speed, ETA, and installation stage;
- performs a real accelerator tensor test;
- verifies the remaining scientific dependencies before opening the GUI.

A manual editable install is also possible after prerequisites are ready:

```bash
python -m pip install -e . --no-deps
python main.py
```

## Recommended v3 experiment setup

For a solution-quality comparison:

1. Load and validate the power-system case.
2. Apply one fixed ORPD formulation.
3. Select all desired algorithms.
4. Choose **PyTorch FP64 batched AC Newton–Raphson**.
5. Choose **Auto-tuned Batched Throughput Engine** and calibrate the available devices.
6. Run and pass the CPU/accelerator parity audit.
7. Run the fairness audit.
8. Use equal objective-function evaluation budgets and common run-level seeds.
9. Independently validate stored final solutions in bulk.
10. Generate statistics and the evidence package from verified results.

For strict wall-clock comparisons, report complete hardware/software details and use the same backend/device policy for all compared methods. Solution-quality evidence should remain based on equal evaluation budgets.

## Frozen final benchmark workflow

The **Benchmark & Evidence** workspace uses `calo_v32_freeze.json`, which hashes the CALO method, accelerator power-flow/evaluator, tensor decoder, baseline torch kernels, policy checkpoint, historical training snapshot, and shared feasibility logic. TEST execution is blocked when frozen files differ.

The final campaign supports IEEE 30-, 57-, 118-, and 300-bus systems, 30–50 repeated runs, deterministic and mixed-variable studies, load/renewable uncertainty, contingencies, expected/mean-risk/worst-case/CVaR objectives, independent validation, and reproducibility export.

The Transactions package includes:

- raw convergence arrays and seeds;
- final normalized vectors and decoded controls;
- reconstructed power-flow states;
- objective and constraint components;
- validation records;
- experiment and environment configuration;
- freeze manifest and policy checkpoint metadata;
- best/mean/median/worst, standard deviation, IQR, confidence intervals;
- feasible-run rate and evaluations to first feasibility;
- Friedman ranking, Wilcoxon tests, Holm correction, effect sizes, and critical-difference information;
- publication plots and evidence-based interpretation.

No universal CALO-superiority claim is generated automatically.

## CALO Intelligence and historical learning

CALO supports cold start, historical warm start, and controlled continual-learning workflows. Experiments are classified as TRAIN, VALIDATION, TEST, or EXCLUDED. Only explicitly eligible TRAIN data can enter the experience repository. Historical CALO trajectories are used for offline pretraining before fresh on-policy PPO rollout generation; other algorithms may contribute validated solution knowledge but not CALO action imitation.

Policy training supports weighted CUDA/XPU/CPU actor lanes and centralized PPO learning. Candidate policies never silently overwrite the frozen publication checkpoint.

## Plotting, validation, and history management

- Live plots auto-fit visible selected series and retain zero for non-negative feasibility metrics.
- **Plot Tools → Preview series** provides selective preview checkboxes.
- Figure formatting includes fonts, labels, ticks, legends, lines, markers, grids, limits, and 600–2400 DPI PNG/SVG/PDF export.
- Validation & Audit supports current-experiment and repository-wide bulk validation with progress and cancellation.
- Results Explorer and Application Settings can delete selected runs, complete experiments, or all experiment history and referenced trace arrays.

## Command-line tools

```bash
calo-rpd-benchmark --case case30 --algorithms CALO,TLBO,PSO --runs 5 --budget 5000
calo-rpd-final-benchmark --runs 30 --budget 10000 --cases case30,case57,case118,case300
calo-rpd-parity --case case30 --device auto --candidates 8
calo-rpd-train --epochs 24 --episodes 12 --horizon 28 --seed 2026
calo-rpd-validate --case case118
calo-rpd-export --database calo_rpd_results.sqlite --experiment <EXPERIMENT_ID>
```

## Important scientific interpretation

GPU conversion primarily improves throughput, batching, and scale. It does not inherently improve convergence, feasibility rate, or final objective quality. Those properties must be demonstrated through repeated runs, ablation, validation, and statistics. Differences between CPU and accelerator results beyond the declared parity tolerance must be treated as implementation defects or explicitly investigated numerical effects—not as algorithmic gains.

## Documentation

- `docs/architecture.md`
- `docs/throughput_engine.md`
- `docs/mathematical_formulation.md`
- `docs/calo_methodology.md`
- `docs/algorithm_sources.md`
- `docs/reproducibility.md`
- `docs/validation.md`
- `docs/user_guide.md`
- `RELEASE_VALIDATION.md`

The repository contains no GitHub Actions workflows. MIT License. Scientific claims remain the responsibility of the experimenter and should be based on the predefined protocol and independently verified raw results.
