# CALO-RPD Studio

**CALO-RPD Studio 2.0.3** is a scientific desktop platform for deterministic and robust optimal reactive power dispatch (ORPD), reproducible comparison of twenty optimizers, and final evidence generation for the **Cognitive Adaptive Learning Optimizer (CALO)**.

The software uses one common physical evaluator for all algorithms, normalized mixed-variable encoding, AC Newton-Raphson power flow, explicit constraint audits, seeded experiment records, independent result validation, nonparametric statistics, and publication-grade export. CALO alone contains the AI-assisted adaptive controller; the remaining nineteen primary algorithms remain conventional comparison baselines.

Version 2.0.3 adds **intelligent visible-data plot scaling** and **bulk independent validation**. Live Optimization now auto-fits the currently visible preview series after every redraw, prevents stale axis limits from obscuring small convergence differences, and keeps zero visible for non-negative feasibility metrics such as normalized constraint violation. Plot Tools exposes Auto-fit visible data, zero-baseline behavior, and padding controls. Validation & Audit can now independently validate every not-yet-verified run in the current experiment or across the whole repository in one background operation with progress, cancellation, per-run status, and a final summary.

Version 2.0.2 extends weighted heterogeneous execution to **CALO policy training**. Each PPO epoch assigns fresh rollout episodes to synchronized CUDA, Intel XPU, and CPU actor lanes using configurable default shares of 50/30/20. Every lane receives the same policy snapshot, all returned trajectories are checked for that snapshot, and the centralized PPO learner updates only after the complete current-policy buffer is available. Version 2.0.1 introduced the corresponding weighted scheduler for experimental evaluation.

Version 2.0.0 introduces the **Frozen CALO + Benchmark & Evidence** workflow. A cryptographic freeze manifest locks CALO's mathematical implementation, operator definitions, cognitive state, archive rules, PPO architecture, policy checkpoint, training-data snapshot, default hyperparameters, mixed-variable decoding, and feasibility rules before final TEST execution. The new Benchmark & Evidence workspace plans and runs all twenty primary algorithms across IEEE 30-, 57-, 118-, and 300-bus systems under deterministic, mixed discrete-continuous, load-uncertainty, renewable-uncertainty, and contingency studies with expected, mean-risk, worst-case, and CVaR aggregation profiles. Final TEST experiments are automatically locked out of historical learning.

The Transactions research-package builder exports raw convergence arrays, seeds, final controls, reconstructed power-flow states, constraint data, validation status, experiment configurations, frozen CALO source/checkpoint artifacts, descriptive statistics, feasible-run rates, evaluations to first feasibility, Friedman ranking, Holm-corrected Wilcoxon tests, effect sizes, Nemenyi critical-difference information, publication figures, automatic evidence-based interpretation, and a reproducibility archive. No universal superiority statement is generated automatically.

## Installation and first launch

Create and activate a Python 3.11+ virtual environment, then launch the bootstrap. The first window is the **Prerequisite Setup Wizard**; it installs/repairs the scientific dependencies, detects NVIDIA and Intel graphics hardware, selects the primary PyTorch backend, provisions an isolated Intel-XPU runtime when needed on mixed-GPU systems, and verifies real accelerator computations before the PyQt6 application starts.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python bootstrap.py
```

On Windows, `start_calo.bat` or `run_windows.bat` provides the same first-launch flow. After a verified environment is recorded, later launches perform a quick check and open the main application directly unless repair is required.

A manual editable install remains possible:

```bash
python -m pip install -e . --no-deps
python main.py
```

The bootstrap deliberately owns PyTorch backend selection so a generic dependency resolver does not replace a verified CUDA wheel with another build.

## Command-line tools

```bash
calo-rpd-benchmark --case case30 --algorithms CALO,TLBO,PSO --runs 5 --budget 5000
calo-rpd-final-benchmark --runs 30 --budget 5000 --cases case30,case57,case118,case300
calo-rpd-train --epochs 24 --episodes 12 --horizon 28 --seed 2026
calo-rpd-validate --case case118
calo-rpd-export --database calo_rpd_results.sqlite --experiment <EXPERIMENT_ID>
```

## Frozen final benchmark workflow

The **Benchmark & Evidence** workspace is separate from exploratory Experiment Manager studies. It enforces the final protocol:

1. Verify the bundled `calo_v2_freeze.json` manifest.
2. Select benchmark systems and predefined deterministic/robust study profiles.
3. Use exactly all 20 primary algorithms, an equal objective-function evaluation budget, and 30–50 independent runs per algorithm and task.
4. Build a campaign manifest before execution.
5. Automatically classify every final campaign experiment as locked **TEST** data so it cannot enter historical learning.
6. Re-check the CALO freeze before final execution.
7. Independently validate completed stored solutions.
8. Generate the Transactions research package only when every planned task completed without recorded optimizer-job failures.

The standard suite includes IEEE 30-, 57-, 118-, and **300-bus** systems. The study library includes deterministic ORPD, mixed discrete-continuous ORPD, load uncertainty with mean-risk and CVaR, renewable uncertainty with mean-risk and CVaR, and selected N-1 branch/generator contingency studies with worst-case aggregation.

The campaign manifest records each task configuration and resulting experiment ID. The same run-level seed tuple is shared across algorithms within a task, while scenario generation remains reproducible through the stored scenario seed.

## Primary algorithms

CALO, TLBO, PSO, CLPSO, MTLA-DE, QODE, Dragonfly, Simulated Annealing, Salp Swarm, continuous-domain Ant Colony Optimization, Bat, Crow Search, Firefly, Flower Pollination, Grasshopper, Grey Wolf, Moth-Flame, Multi-Verse, Whale Optimization, and Imperialist Competitive Algorithm.

## Resource-aware CALO policy training

The default policy-training mode is a synchronous actor–learner architecture. At the start of every PPO epoch, one immutable policy snapshot is distributed to three rollout lanes:

- NVIDIA CUDA actors — requested 50% of episodes/transitions;
- Intel XPU actors — requested 30%;
- CPU actors — requested 20%.

The integer allocation uses the largest-remainder method. With the default 12 episodes per epoch, the plan is 6 CUDA, 4 XPU, and 2 CPU episodes. Unavailable accelerator shares are redistributed across available lanes and the GUI displays the effective allocation before training.

CUDA/XPU actor processes batch policy-network inference on their assigned device. The CPU lane uses spawn-safe parallel processes. All actors must return the same policy-snapshot hash; stale or mismatched trajectories are rejected. Only after every current-policy lane has finished are the trajectories merged into one on-policy buffer and used for a centralized PPO update on the selected learner device. The next epoch then receives the updated snapshot.

On a mixed NVIDIA/Intel host, the XPU actor can run through the isolated secondary runtime without replacing the primary CUDA-enabled PyTorch installation. The configured 50/30/20 values are transition shares, not guarantees of identical Task Manager utilization. Environment evolution, PYPOWER, AC power flow, and most physical constraint calculations remain host-CPU work, so CPU utilization can still be substantial and accelerator activity can be bursty.

Weighted training writes a **candidate** checkpoint and does not overwrite the bundled frozen policy. The candidate must be independently validated and explicitly re-frozen before it is used in a final TEST campaign.

## Accelerator-first heterogeneous experiment scheduler

For primary comparisons and CALO ablation studies, the default scheduler performs compatibility-aware admission in this order:

1. NVIDIA CUDA devices, while their configured compute and VRAM admission thresholds allow more compatible CALO jobs.
2. Intel XPU devices, while their configured memory threshold and utilization threshold (when the runtime exposes utilization telemetry) allow more jobs.
3. CPU processes, while CPU utilization and the system-RAM safety limit permit additional work.

The default targets are 70% CUDA utilization, 70% XPU utilization, and 50% CPU utilization, with separate device-memory and system-RAM safety limits. These are **soft admission thresholds**, not promises of exact operating-system utilization. The explicit maximum jobs per accelerator prevents runaway process creation while telemetry catches up.

A running optimization job is never migrated between CUDA, XPU, and CPU. Device assignment happens before the independent run starts and is stored in the run metadata. PyTorch backend identifiers such as `cuda:0` and `xpu:0` are backend-specific and do not necessarily equal Windows Task Manager's `GPU 0` / `GPU 1` labels.

Only accelerator-compatible CALO neural-policy work is routed to CUDA/XPU. AC Newton-Raphson power flow, physical constraint evaluation, and the conventional baseline optimizers remain CPU workloads. For strict publication-quality wall-clock comparisons, use CPU-only, single-worker execution.

## CALO Core v2

CALO Core v2 does not force the entire population into one search equation. It uses:

- adaptive epsilon-feasibility that decays to exact feasibility;
- a **Feasible Elite Archive** for objective-quality feasible solutions;
- a **Constraint Boundary Archive** for diverse low-violation routes toward feasibility;
- per-individual operator allocation rather than one operator for the whole generation;
- six operators: feasible-elite learning, constraint-boundary differential learning, cognitive teacher learning, success-distribution memory, mixed-variable neighbourhood learning, and diversity recovery;
- environmental selection from parents and offspring;
- separate objective and constraint stagnation tracking;
- online operator credit blended with the learned policy;
- physically aware local moves for discrete transformer taps and shunt steps;
- a hierarchical policy that controls search regime, operator distribution, and bounded continuous parameters.

The final solution remains subject to the common strict physical feasibility test. Epsilon-feasibility is a search mechanism only and does not relax the validity of reported solutions.

## CALO diagnostics

Live and stored CALO telemetry includes:

- best feasible objective;
- best total normalized constraint violation;
- bus-voltage, generator-Q, generator-P, branch-thermal, and power-flow violation components;
- exact feasible population ratio;
- epsilon-feasible population ratio;
- adaptive epsilon;
- population diversity;
- elite diversity;
- CALO search regime;
- active/dominant operator;
- per-operator online success rate;
- evaluations to first exact feasible solution;
- feasible and constraint-boundary archive sizes.

The Live Optimization page provides dedicated diagnostic plot modes for constraint decomposition, feasibility evolution, diversity, and operator success.

## Historical Experience Learning

CALO Intelligence now contains a dedicated **Historical experience learning** workspace. Existing experiments are excluded from learning by default and can be explicitly classified as:

- **TRAIN** — may be marked learning-eligible and admitted to the historical repository;
- **VALIDATION** — may be used to assess candidate policies but is never admitted to training;
- **TEST** — locked out of learning to protect final benchmark independence;
- **EXCLUDED** — ignored by the learning pipeline.

Three operating modes are available:

- **Cold Start** — no historical experience is used; recommended for strict independent final benchmarking.
- **Historical Warm Start** — a previously built, checksum-verified repository may provide historical policy pretraining, CALO parameter priors, cross-algorithm solution knowledge, and optional population warm starts.
- **Continual Learning** — the repository is rebuilt from currently eligible TRAIN experiments after relevant experiment/classification changes. A deployed policy is not silently retrained or promoted.

For policy learning, exact v1.3 CALO trajectories store the cognitive state, regime decision, operator decision, bounded parameter action, reward, and evaluation index. Historical trajectories are used only for **offline supervised/value pretraining**; PPO then generates fresh trajectories with the current policy and remains on-policy. Legacy v1.2 CALO diagnostic histories may be reconstructed conservatively as partial, lower-weight regime/operator training examples when sufficient telemetry exists; they do not supervise the continuous parameter head.

Cross-algorithm runs can contribute validated solution exemplars and problem-specific knowledge but cannot imitate CALO actions. Optional historical population warm starting and parameter priors are therefore separate from neural-policy pretraining. These options should be disabled when a study requires strict cold-start fairness.

## Selective live preview and export

The Live Optimization workspace uses a square 1:1 preview. Series visibility is no longer shown as a permanently expanded checkbox bar. Open **Plot Tools → Preview series** to display a compact dynamic checklist for the series currently available in the active diagnostic plot. **Select all**, **Clear all**, and **Restore default** change only the live preview and never delete raw data.

The separate **Export figure** popup independently controls which series are saved. Exported figures and legends contain only the selected curves. PNG export supports **600–2400 DPI**, and the live plot exports in exact square aspect ratio. SVG and PDF remain vector formats.

## Organized plot editing tools

Each scientific figure provides four core plot tools:

- **Text & labels** — font family, independent font sizes, bold/italic, title, axis labels, tick labels, legend text, and annotations.
- **Plot appearance** — axes, limits, scales, grids, line styles, widths, markers, and visibility.
- **Export figure** — PNG/SVG/PDF, selected series, square export where required, and 600–2400 DPI PNG resolution.
- **Style profiles** — save, load, reset, and apply-to-all-compatible-plots.

Live Optimization additionally exposes the context-sensitive **Preview series** tool inside the same Plot Tools strip.

## Guided scientific workflow

The GUI enforces the research sequence instead of exposing every workspace at once:

1. **Power System** — load a case, run the base AC power flow, and pass the independent PYPOWER cross-check.
2. **ORPD Formulation** — apply objectives, control variables, discrete device behavior, and constraints.
3. **Algorithms** — select the comparison methods and apply declared parameters.
4. **CALO Intelligence** — when CALO is selected, inspect/train/validate the CALO Core v2 policy and optionally classify historical experiments, build a leakage-aware experience repository, and configure cold-start or historical learning.
5. **Robust Scenarios** — apply deterministic or robust scenario configuration.
6. **Experiment Manager** — configure runs, pass the fairness audit, then execute the primary comparison or CALO Core v2 ablation study.
7. **Live Optimization** — becomes available when an experiment starts.
8. **Statistical Analysis** — becomes available after the experiment completes.
9. **Results Explorer** — review stored runs and send the selected run to validation.
10. **Validation & Audit** — independently re-evaluate the reviewed stored decision.
11. **Publication Export** — unlocks only after at least one result from the current experiment is verified.
12. **Benchmark & Evidence** — always available as the dedicated frozen final-campaign workspace for v2.0.0 TEST execution and Transactions-level package generation.

Dashboard and Application Settings remain available throughout. Changing an upstream scientific stage invalidates dependent downstream workflow state.

The bottom status bar reports **Ready**, **Busy**, **Completed**, **Failed**, or **Cancelled**, together with the active operation, progress, elapsed time, and safe cancellation for supported long-running tasks.

## Experiment execution modes

- **Primary Algorithm Comparison** runs exactly the algorithms selected on the Algorithms page. Twenty algorithms and five repeated runs create 100 independent jobs.
- **CALO Ablation Study** runs nine fixed CALO/TLBO variants: classical TLBO, legacy Gaussian MTLBO, CALO Core v2 without AI, CALO without epsilon-feasibility, CALO without dual archives, CALO without mixed-variable learning, CALO without success memory, CALO without diversity recovery, and complete CALO. Five repeated runs create 45 jobs.

The Experiment Manager provides **Adaptive hybrid CPU + GPU**, **GPU preferred with CPU fallback**, and **CPU only** scheduling. CUDA-capable CALO variants may run neural-policy inference on the GPU, while AC power flow, constraints, and the nineteen non-AI baselines remain CPU workloads. Soft admission targets control GPU utilization, GPU memory, CPU utilization, maximum GPU CALO jobs, and total concurrent jobs. Running jobs are never migrated between devices. Use one worker and CPU-only mode when comparing wall-clock runtime as a scientific metric.

## AI training and benchmark separation

The CALO Core v2 policy network is trained with actual PPO mechanics: clipped policy updates, generalized advantage estimation (GAE), value loss, entropy regularization, minibatches, multiple PPO epochs, and gradient clipping. The same runtime CALO Core v2 operator implementations, epsilon selection, archives, state builder, and mixed-variable operators are reused inside the training environment.

The built-in curriculum covers continuous unconstrained, constrained continuous, mixed discrete-continuous, and narrow-feasible-region tasks. An optional final curriculum stage can use explicitly supplied ORPD development-system case paths through the GUI or repeated `--development-case` command-line options. Final publication benchmark cases are not silently used for policy training; development systems must be declared and are stored in checkpoint metadata.

The packaged `calo_policy_v2.pt` is a reproducible reference checkpoint with metadata and checksum. Publication-scale studies should freeze the selected checkpoint before final benchmarking and retain its checksum in experiment provenance.

## Reproducibility

Each run records its seed tuple, algorithm parameters, case checksum, objective, scenario configuration, evaluation budget, convergence histories, CALO diagnostics, final physical state, software environment, and validation state. Important numerical arrays remain available as raw data rather than only as figures.

## Experiment history and trace cleanup

Open **Results Explorer → Manage history** or **Application Settings → Experiment history** to remove obsolete stored runs and their referenced trace arrays. The history manager can delete a selected run, a complete experiment, or all stored experiment history. Full-history deletion requires typing `DELETE ALL`.

External publication-export directories are intentionally not deleted automatically because they are independent user-managed copies. Destructive history actions are disabled while a scientific task is active.

## Documentation

- `docs/architecture.md`
- `docs/mathematical_formulation.md`
- `docs/calo_methodology.md`
- `docs/algorithm_sources.md`
- `docs/reproducibility.md`
- `docs/validation.md`
- `docs/user_guide.md`
- `RELEASE_VALIDATION.md`

## Repository automation

This repository does **not** include GitHub Actions workflows. The guided workflow described above is application behavior inside CALO-RPD Studio, not a GitHub CI/CD workflow.

## License

MIT License. Scientific results remain the responsibility of the experimenter. Comparative claims should be based on the complete predefined protocol, repeated runs, statistical analysis, and independently verified raw results.

## Weighted experiment scheduling (v2.0.1+)

The default experiment scheduler uses requested shares of 50% CUDA, 30% Intel XPU, and 20% CPU **for accelerator-compatible CALO jobs**. It does not relabel CPU code as GPU work. In a 20-algorithm × 5-run comparison there are 100 jobs, but only the five CALO jobs currently contain accelerator-compatible policy inference. With both CUDA and XPU available, those five CALO jobs are distributed approximately as 3 CUDA, 1 XPU, and 1 CPU; the other 95 baseline jobs remain CPU jobs. The Experiment Manager shows this attainable allocation before execution.

The AC Newton-Raphson power-flow calculation, physical constraint evaluation, and the 19 conventional baseline implementations are NumPy/SciPy CPU workloads. Even a CALO run assigned to CUDA or XPU still performs its power-flow evaluations on CPU, so high CPU utilization and low accelerator utilization can remain normal. Achieving a true 50/30/20 split across all benchmark work requires a separately validated GPU/XPU-native batched AC power-flow evaluator and accelerator-native implementations of the baseline algorithms.
