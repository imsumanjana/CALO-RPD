# User Guide

## 1. Power System

Load IEEE 30-, 57-, or 118-bus data, inspect bus/generator/branch matrices, run the base AC power flow,
and cross-check the state with PYPOWER when available.

## 2. ORPD Formulation

Select active-power loss, voltage deviation, L-index, or multi-objective optimization. Enable generator
voltage, transformer tap, and shunt controls. Configure discrete device behavior and transformer steps.

## 3. Algorithms

Select any subset of the twenty primary methods. Algorithm parameter dictionaries are editable as JSON
and saved with the experiment.

## 4. CALO Intelligence

Inspect the active policy checksum and metadata. Train a new policy with explicit epoch, episode, horizon,
seed, and learning-rate settings. Run the seven-case CALO ablation suite separately from the primary
benchmark.

## 5. Robust Scenarios

Choose deterministic, load uncertainty, Monte Carlo, renewable uncertainty, branch contingency, or
generator contingency mode. Select expected, mean-risk, worst-case, or CVaR aggregation.

## 6. Experiment Manager

The workspace is organized in the required scientific order: **1. Experiment configuration**, **2. Fairness audit**, **3. Run study**, then **Run queue**. Choose run count, population size, budget policy, evaluation budget, worker preference, master seed, and result directory first. Run the fairness audit next. Primary comparison and CALO ablation controls remain locked until the current configuration passes the audit. Any later configuration change invalidates the previous audit and locks execution again. The workspace body scrolls vertically on shorter displays so controls retain their normal height rather than being compressed. The run queue records completion, failure, or cancellation.

## 7. Live Optimization

View current objective, feasibility, evaluation count, CALO operator, diversity, feasible ratio, reward,
and live convergence.

## 8. Plot formatting

Every embedded scientific plot exposes the formatting toolbar. Choose the text target and edit installed
font family, font size, bold/italic state, displayed title, X/Y labels, legend labels, legend placement,
axis scales and limits, major/minor grids, axis width, line width/style, marker and size, and visibility.
Save/load style profiles, apply one style to all compatible plots, and export PNG, SVG, or PDF with
configurable dimensions, DPI, transparency, and tight bounds. Plot style never changes raw numerical data.

## 9. Statistics and Results

Use Statistical Analysis for repeated-run summaries and editable figures. Use Results Explorer to inspect
run controls, objectives, convergence, metadata, and final scenario-wise system state.

## 10. Validation and Publication Export

Independently validate selected runs. Publication Export includes verified runs only and produces CSV,
LaTeX-compatible tables, experiment metadata, and a reproducibility archive.

## Guided workspace sequence

CALO-RPD Studio uses prerequisite locking so that a new user follows the scientific workflow in the correct order. Locked sidebar entries show an explanation in their tooltip. The workflow banner above the workspace always identifies the next required action and provides a direct navigation button.

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

The bottom bar is application-wide. It changes from Ready to Busy whenever a tracked scientific task starts. When a percentage is available it displays determinate progress; otherwise it shows an indeterminate progress indicator. CALO training and optimization experiments expose safe cancellation through the same bar.

## Managing stored experiment history

Old local experiments can be removed from either **Results Explorer → Manage history** or **Application Settings → Experiment history**.

The history manager provides three levels of cleanup:

- **Delete selected run** — removes one completed run, its validation records, and its referenced compressed trace array.
- **Delete selected experiment** — removes the complete experiment, all completed and failed-run records, validation records, and referenced trace arrays.
- **Delete all experiment history** — removes all local experiment records and referenced trace arrays after the user explicitly types `DELETE ALL`.

The application shows record counts and referenced trace storage before deletion. External publication exports are independent files and are not removed automatically. History deletion is disabled while a scientific task is active.
