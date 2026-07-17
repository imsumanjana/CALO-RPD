# CALO-RPD Studio Architecture

CALO-RPD Studio separates the user interface, physical network model, ORPD formulation, optimization algorithms, CALO intelligence, robust scenarios, experiment execution, statistics, persistence, validation, visualization, and reporting.

The dependency direction is intentionally one-way:

`GUI → experiment services → optimizer interface → ORPD evaluator → AC power flow`

All twenty primary algorithms call the same `ORPDProblem.evaluate()` method. No optimizer contains a private power-flow implementation, private objective definition, or private constraint model. Decision variables are searched in a normalized `[0,1]^D` space and are converted to physical generator voltages, transformer taps, and shunt settings by one common mixed-variable decoder.

## Major packages

- `power_system`: MATPOWER-compatible case model, Y-bus, Newton-Raphson power flow, reactive-limit switching, branch flows, L-index, and independent cross-validation.
- `orpd`: decision variables, mixed-variable decoding, objectives, physical constraints, feasibility rules, and scenario-aware evaluation.
- `algorithms`: one shared optimizer contract, nineteen conventional baselines, CALO Core v2, and the legacy Gaussian MTLBO ablation implementation.
- `algorithms/calo`: constraint diagnostics, dual archives, epsilon environmental selection, six learning operators, success-distribution memory, online operator credit, hierarchical policy, PPO training, and CALO runtime.
- `robustness`: load and renewable uncertainty, contingencies, Monte Carlo scenarios, expected/mean-risk/worst-case/CVaR aggregation.
- `experiments`: immutable seed derivation, fairness audit, evaluation budgets, adaptive process-based CPU/GPU job admission, failure isolation, and provenance.
- `compute`: CUDA/CPU resource monitoring, CALO GPU-job eligibility, and soft utilization-based admission control.
- `calo_bootstrap`: standard-library-only first-launch prerequisite detection, hardware-aware PyTorch backend installation, verification, and setup wizard.
- `results`: SQLite metadata, compressed numeric arrays, integrity checks, independent solution validation, comparisons, rankings, history deletion, and verified-only export.
- `statistics`: descriptive measures, confidence intervals, Wilcoxon, Friedman, Holm correction, Cliff's delta, and average ranks.
- `visualization` and `gui.plotting`: raw-data-preserving plots, square preview/export, selective preview and export series, and popup-based publication formatting tools.
- `app` and `gui`: PyQt6 desktop application with guided scientific workflow, global task status, and thirteen workspaces.

## CALO Core v2 data flow

`Population → physical evaluation → constraint decomposition → cognitive state → regime policy/prior → operator distribution → online-credit fusion → per-individual operators → offspring → epsilon environmental selection → archives/memory/credit → diagnostics`

Strict physical feasibility remains defined by the common ORPD evaluator. Adaptive epsilon-feasibility is used only during CALO search and decays to zero.

## Long-running work and heterogeneous scheduling

The desktop experiment manager keeps the Qt event loop responsive while independent optimizer/run jobs are dispatched to spawn-safe worker processes. In v3, the PyTorch FP64 scientific backend provides accelerator-native mixed-variable decoding, batched AC Newton–Raphson power flow, branch-flow and constraint evaluation, robust aggregation, L-index calculation, and canonical tensor kernels for all nineteen baseline optimizers; CALO uses the same evaluator plus its neural controller. CUDA and XPU assignments are made before a run begins and are never migrated mid-run. SQLite persistence remains coordinated by the parent process to avoid concurrent write corruption. Inter-process progress telemetry is throttled to reduce overhead.

GPU utilization, GPU memory, and CPU utilization targets are **soft admission controls**. They decide whether another independent job may start; an active optimizer is never migrated between devices. This preserves run-level seed semantics and avoids invalid mid-run backend changes.

Use one worker and CPU-only execution for strict wall-clock runtime comparisons because concurrent or heterogeneous jobs compete for different resources. Use adaptive scheduling for faster solution-quality throughput under equal objective-function evaluation budgets.

## Persistence

SQLite uses WAL mode for transaction safety. Large arrays are stored in compressed NPZ files. Experiment configurations use YAML or JSON. A final result stores the normalized decision vector, decoded physical controls, scenario-wise bus voltages and angles, generator outputs, branch flows, branch loading, objective components, constraint components, convergence histories, CALO diagnostics where applicable, and case checksum.

## Plot architecture

`ScientificPlotWidget` stores raw series separately from style settings. The live plot uses a square display surface and exact square export. A Qt **Preview legend** dynamically creates checkboxes from the active series names and filters only the displayed preview. The export popup independently selects saved series and rebuilds the exported legend using only those selections.

## Repository automation

No `.github/workflows` directory is included. The application's guided scientific workflow is local desktop software behavior, not GitHub CI/CD automation.
