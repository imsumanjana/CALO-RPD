# CALO-RPD Studio Architecture

CALO-RPD Studio separates the user interface, physical network model, ORPD formulation,
optimization algorithms, CALO intelligence, robust scenarios, experiment execution,
statistics, persistence, validation, visualization, and reporting.

The dependency direction is intentionally one-way:

`GUI → experiment services → optimizer interface → ORPD evaluator → AC power flow`

All twenty primary algorithms call the same `ORPDProblem.evaluate()` method. No optimizer
contains a private power-flow implementation, private objective definition, or private
constraint model. Decision variables are searched in a normalized `[0,1]^D` space and are
converted to physical generator voltages, transformer taps, and shunt settings by one common
mixed-variable decoder.

## Major packages

- `power_system`: MATPOWER-compatible case model, Y-bus, Newton-Raphson power flow,
  aggregate reactive-limit switching, branch flows, L-index, and independent cross-validation.
- `orpd`: decision variables, mixed-variable decoding, objectives, constraints,
  feasibility-first comparison, and scenario-aware evaluation.
- `algorithms`: one shared optimizer contract, nineteen conventional baselines, the CALO
  architecture, and the legacy Gaussian MTLBO ablation implementation.
- `robustness`: load and renewable uncertainty, contingencies, Monte Carlo scenarios,
  expected/mean-risk/worst-case/CVaR aggregation.
- `experiments`: immutable seed derivation, fairness audit, evaluation budgets, sequential and
  process-based parallel execution, failure isolation, and provenance.
- `results`: SQLite metadata, compressed numeric arrays, integrity checks, independent solution
  validation, comparisons, rankings, and verified-only export.
- `statistics`: descriptive measures, confidence intervals, Wilcoxon, Friedman, Holm correction,
  Cliff's delta, and average ranks.
- `visualization` and `gui.plotting`: raw-data-preserving plots and a live formatting toolbar.
- `app` and `gui`: PyQt6 desktop application with thirteen scrollable workspaces.

## Long-running work

The desktop experiment manager executes numerical runs in a worker thread so Qt event handling
remains responsive. The command-line runner supports process-based parallel execution for
independent runs. Every failure is isolated to its run and persisted with the algorithm, seed,
exception type, traceback, evaluation count, and numerical state when available.

## Persistence

SQLite uses WAL mode for transaction safety. Large arrays are stored in compressed NPZ files.
Experiment configurations use YAML or JSON. A final result stores the normalized decision vector,
decoded physical controls, scenario-wise bus voltages and angles, generator outputs, branch flows,
branch loading, objectives, constraint components, and case checksum.
