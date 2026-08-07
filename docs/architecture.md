# CALO-RPD Studio Architecture

**v12 development boundary:** this document describes implemented package and data-flow structure;
the presence of policy-training or policy-runtime components is not authority to execute them during
Phase 4. Phase 4 supports empty-policy operation and performs no policy training, evaluation,
qualification, registration, activation, protected campaign, or deletion. Executable compute modes
are CUDA-preferred and CPU-only. See `DOCUMENTATION_STATUS.md` for document precedence.

CALO-RPD Studio separates the user interface, physical network model, ORPD formulation, optimization algorithms, CALO intelligence, robust scenarios, experiment execution, statistics, persistence, validation, visualization, and reporting.

The dependency direction is intentionally one-way:

`GUI → experiment services → optimizer interface → ORPD evaluator → AC power flow`

All registered primary algorithms call the same `ORPDProblem.evaluate()` method. No optimizer contains a private power-flow implementation, private objective definition, or private constraint model. Decision variables are searched in a normalized `[0,1]^D` space and are converted to physical generator voltages, transformer taps, and shunt settings by one common mixed-variable decoder.

## Major packages

- `power_system`: MATPOWER-compatible case model, Y-bus, Newton-Raphson power flow, reactive-limit switching, branch flows, L-index, and independent cross-validation.
- `orpd`: decision variables, mixed-variable decoding, objectives, physical constraints, feasibility rules, and scenario-aware evaluation.
- `algorithms`: one shared optimizer contract, a runtime-enumerated comparator registry, CALO Core v2, and the legacy Gaussian MTLBO ablation implementation.
- `algorithms/calo`: constraint diagnostics, dual archives, epsilon environmental selection, six learning operators, success-distribution memory, online operator credit, hierarchical policy, PPO training, and CALO runtime.
- `robustness`: load and renewable uncertainty, contingencies, Monte Carlo scenarios, expected/mean-risk/worst-case/CVaR aggregation.
- `experiments`: immutable seed derivation, fairness audit, evaluation budgets, adaptive process-based CPU/GPU job admission, failure isolation, and provenance.
- `compute`: CUDA/CPU discovery, immutable 80%-of-currently-available VRAM/RAM admission contracts, device leases, and protection telemetry. Utilization is diagnostic only and never controls scheduling.
- `calo_bootstrap`: standard-library-only first-launch prerequisite detection, hardware-aware PyTorch backend installation, verification, and setup wizard.
- `results`: SQLite metadata, compressed numeric arrays, integrity checks, independent solution validation, comparisons, rankings, history deletion, and verified-only export.
- `statistics`: descriptive measures, confidence intervals, Wilcoxon, Friedman, Holm correction, Cliff's delta, and average ranks.
- `visualization` and `gui.plotting`: raw-data-preserving plots, square preview/export, selective preview and export series, and popup-based publication formatting tools.
- `app` and `gui`: PyQt6 desktop application with guided scientific workflow, global task status, and thirteen workspaces.

## CALO Core v2 data flow

`Population → physical evaluation → constraint decomposition → cognitive state → regime policy/prior → operator distribution → online-credit fusion → per-individual operators → offspring → epsilon environmental selection → archives/memory/credit → diagnostics`

Strict physical feasibility remains defined by the common ORPD evaluator. Adaptive epsilon-feasibility is used only during CALO search and decays to zero.

## Long-running work and heterogeneous scheduling

The desktop experiment manager keeps the Qt event loop responsive while independent optimizer/run jobs are dispatched to spawn-safe worker processes. The PyTorch FP64 scientific implementation provides CUDA-resident mixed-variable decoding, batched AC Newton–Raphson power flow, branch-flow and constraint evaluation, robust aggregation, L-index calculation, and tensor kernels for registered comparators; CALO uses the same evaluator plus its neural controller. A run uses CUDA when admitted, stages through available CPU RAM only when necessary, or uses CPU-only mode. It is never migrated silently during active execution. SQLite persistence remains coordinated by the parent process to avoid concurrent write corruption. Inter-process progress telemetry is throttled to reduce overhead.

Memory admission is capped at 80% of memory available at admission time, not 80% of installed capacity. Shared leases prevent concurrent jobs from collectively exceeding that ceiling. Utilization is observed for diagnostics only and is not a scheduling target. This preserves run-level seed semantics and avoids invalid mid-run compute changes.

Use one worker and CPU-only execution for strict wall-clock runtime comparisons because concurrent or heterogeneous jobs compete for different resources. Use adaptive scheduling for faster solution-quality throughput under equal objective-function evaluation budgets.

## Persistence

SQLite uses WAL mode for transaction safety and a durable `PRAGMA user_version`. Before upgrading any populated version-0 database, the application creates an integrity-checked SQLite online backup, records its path and SHA-256 in `schema_migrations`, performs all DDL in one transaction, and advances the version only after success. Newer unknown schemas fail closed. Large arrays are stored in compressed NPZ files. Experiment configurations use YAML or JSON. A final result stores the normalized decision vector, decoded physical controls, scenario-wise bus voltages and angles, generator outputs, branch flows, branch loading, objective components, constraint components, convergence histories, CALO diagnostics where applicable, and case checksum.

## Plot architecture

`ScientificPlotWidget` stores raw series separately from style settings. The live plot uses a square display surface and exact square export. A Qt **Preview legend** dynamically creates checkboxes from the active series names and filters only the displayed preview. The export popup independently selects saved series and rebuilds the exported legend using only those selections.

## Repository automation

`.github/workflows/ci.yml` defines separate source/scientific, Python/platform compatibility,
headless scientist-GUI, staged Python artifact, CPU-container, CUDA build-only, and manually gated
physical-CUDA qualification lanes. Third-party actions are pinned to immutable commit SHAs. The
release Python 3.11 lane installs a hash-complete CPU validation lock, verifies exact pins, approved
indexes and complete SHA-256 coverage without re-resolving, checks the generated experiment schema,
runs formatting/lint and a bounded mypy safety target, then executes the active development test suite. Historical release-freeze tests remain
separate because they describe immutable prior artifacts rather than the changing development tree.
