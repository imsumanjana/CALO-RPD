# User Guide

## 1. Power System

Load IEEE 30-, 57-, or 118-bus data, inspect bus/generator/branch matrices, run the base AC power flow, and cross-check the state with PYPOWER when available.

## 2. ORPD Formulation

Select active-power loss, voltage deviation, L-index, or multi-objective optimization. Enable generator-voltage, transformer-tap, and shunt controls. Configure discrete device behavior and transformer steps.

## 3. Algorithms

Select any subset of the registered primary methods. Method settings are saved with the experiment. All optimizers use the same normalized decision space, decoder, physical evaluator, constraints, scenarios, and budget policy.

## 4. CALO Intelligence

During the v12 development line, CALO Intelligence is a policy-lifecycle and readiness surface. The
supported default is an empty policy store with explicit safe fallback. Existing policy files, if
present, are development-only, unqualified, inactive, non-final, and must not be used as final
candidates or as initial weights for a later policy.

Phase 4 is development completion only. The page may expose configuration and lifecycle metadata
needed to complete and inspect the implementation, but Phase 4 does not run policy training,
evaluation, qualification, registration, activation, or protected-case campaigns. Policy-dependent
actions must remain visibly unavailable when no separately qualified and explicitly activated policy
exists. Non-policy CALO development and baseline workflows must remain usable through the declared
safe fallback.

Executable compute modes are **CUDA-preferred** and **CPU-only**. CUDA admission is bounded by at
most 80% of currently free VRAM, while CPU admission is bounded by at most 80% of currently available
RAM. Intel XPU may appear only as historical or diagnostic metadata; it is not an executable mode.

After the Phase 4 development freeze, removal of old policies is a separate, explicitly authorized
inventory-first operation. Only after the policy store is verified empty may a completely new
A-E/F-off policy be trained. That new artifact remains an unqualified candidate until it passes the
separate qualification, registration, activation, and immutable experiment-binding gates. F remains
experimental, independently feature-flagged, and disabled by default. A policy-free Phase 5 route is
also valid.

The CALO ablation study remains separate from the primary benchmark. Its scientific execution is not
part of Phase 4 development.

## 5. Robust Scenarios

Choose deterministic, load uncertainty, Monte Carlo, renewable uncertainty, branch contingency, or generator contingency mode. Select expected, mean-risk, worst-case, or CVaR aggregation.

## 6. Experiment Manager

The workspace is organized in the required scientific order:

1. **Experiment configuration**
2. **Fairness audit**
3. **Run study**
4. **Run queue**

Choose run count, population size, budget policy, evaluation budget, worker preference, master seed, and result directory first. Run the fairness audit next. Primary comparison and CALO ablation controls remain locked until the current configuration passes the audit. Any later configuration change invalidates the previous audit and locks execution again.

The workspace body scrolls vertically on shorter displays so controls retain their normal height rather than being compressed.

## 7. Live Optimization

The page shows a square convergence preview and live telemetry. Automatic mode displays best normalized constraint violation until every represented optimizer has produced a feasible incumbent, then switches to best feasible objective.

Additional CALO Core v2 diagnostic plot modes are available:

- constraint decomposition;
- exact and epsilon-feasible population ratios;
- population and elite diversity;
- operator success rates.

The telemetry panel also shows:

- CALO regime;
- dominant operator;
- adaptive epsilon;
- bus-voltage, generator-Q, generator-P, and branch-thermal violation components;
- evaluations to first exact feasibility;
- reward and diversity information.

### Selective preview series and automatic scaling

Open **Plot Tools → Preview series** to see a dynamic checkbox list for the series available in the active plot. Unchecking a series removes it from the preview immediately while keeping its raw data. Use **Select all**, **Clear all**, or **Restore default** for bulk changes.

Live Optimization automatically fits the axes to the currently visible series. Non-negative feasibility metrics keep zero visible by default so the target of exact feasibility remains interpretable. Open **Plot Tools → Plot appearance** to adjust Auto-fit visible data, zero-baseline behavior, and padding, or disable auto-fit to enter fixed manual limits.

Preview selection and export selection are separate. A curve can be hidden in the live preview without changing stored data or future export choices.

## 8. Plot formatting and export

Every embedded scientific plot exposes four focused popup tools:

- **Text & labels**
- **Plot appearance**
- **Export figure**
- **Style profiles**

The user can edit installed font family, independent font sizes, bold/italic state, displayed title, X/Y labels, tick labels, legend labels and placement, axis scales and limits, grids, line width/style, markers, and visibility.

The export popup lists available series as checkboxes. Saved figures and legends include only the selected curves. The live plot exports in exact square aspect ratio. PNG resolution is selectable from **600 to 2400 DPI**; SVG and PDF are vector exports.

## 9. Statistical Analysis and Results Explorer

Use Statistical Analysis for repeated-run summaries and editable figures. Use Results Explorer to inspect controls, objectives, feasibility, convergence, metadata, and final scenario-wise physical state.

Select a run, review its details, then confirm the result review. The application opens Validation & Audit on that exact run.

## 10. Validation and Publication Export

Validation & Audit supports both single-run and bulk independent validation. Use **Validate current experiment** to process every not-yet-verified run in the selected experiment, or **Validate all not-yet-verified runs** to process the complete local repository. Bulk validation runs in the background, reports live progress and passed/failed/error counts, and can be cancelled between runs. Existing verified runs are skipped by default.

Publication Export includes verified runs only and produces CSV, LaTeX-compatible tables, experiment metadata, and a reproducibility archive.

## Guided workspace sequence

CALO-RPD Studio uses prerequisite locking so that a new user follows the scientific workflow in the correct order. Locked sidebar entries explain their prerequisite in the tooltip. The workflow banner identifies the next required action and provides direct navigation.

### Setup sequence

1. Load and independently validate the power-system case.
2. Apply the ORPD formulation.
3. Apply the algorithm selection.
4. Validate and apply CALO Intelligence when CALO is selected.
5. Apply robust scenario configuration.
6. Open Experiment Manager, run the fairness audit, and start execution.

### Post-experiment sequence

1. Complete Statistical Analysis.
2. Inspect a stored run in Results Explorer and confirm the review.
3. Independently validate one or more stored runs.
4. Export verified publication results.

### Bottom task bar

The application-wide bottom bar changes from **Ready** to **Busy** whenever a tracked task starts. When a percentage is available it displays determinate progress; otherwise it shows an indeterminate progress indicator. CALO training and optimization experiments expose safe cancellation through the same bar.

## Managing stored experiment history

Old local experiments can be removed from either **Results Explorer → Manage history** or **Application Settings → Experiment history**.

- **Delete selected run** removes one completed run, its validation records, and its referenced compressed trace array.
- **Delete selected experiment** removes the complete experiment, completed and failed-run records, validation records, and referenced trace arrays.
- **Delete all experiment history** removes all local experiment records and referenced trace arrays after the user types `DELETE ALL`.

External publication exports are independent files and are not removed automatically. History deletion is disabled while a scientific task is active.
