# User Guide

## 1. Power System

Load IEEE 30-, 57-, or 118-bus data, inspect bus/generator/branch matrices, run the base AC power flow, and cross-check the state with PYPOWER when available.

## 2. ORPD Formulation

Select active-power loss, voltage deviation, L-index, or multi-objective optimization. Enable generator-voltage, transformer-tap, and shunt controls. Configure discrete device behavior and transformer steps.

## 3. Algorithms

Select any subset of the registered primary methods. Method settings are saved with the experiment.
All optimizers use the same normalized decision space, decoder, physical evaluator, constraints,
scenarios, and budget policy. `TSH-CALO` is shown as a separate policy-gated row and remains disabled
until a future completely new post-development ensemble satisfies every immutable activation gate.

## 4. CALO Intelligence

CALO Intelligence is the policy lifecycle and readiness surface. The safe default is an empty policy
store with rule-only CALO available. Completed training outputs appear in Policy Library and require
an explicit import. Import never qualifies, activates, or binds a model.

Use **Policies → Train policy** for a new TSH-CALO campaign. **New training** is the ordinary
path: choose the eligible cases and visible campaign, ensemble, budget, compute, and PPO inputs,
then select a new training directory. **Check readiness** constructs the complete internal training
plan from those inputs plus application-owned safe defaults and current source identity; no external
settings-template file is required. The managed model-library location is intentionally not a
scientist input. **Refresh** updates its retained campaigns. Selecting a saved campaign loads that
campaign's exact retained plan internally for authenticated resume or compatible finite extension;
it never silently copies those settings into an unrelated new campaign.

In a repository checkout, fresh-plan provenance is resolved from the installed CALO-RPD package
location rather than the process working directory. Launching the native command or shortcut from
another folder therefore does not turn a fresh plan into a misleading saved-plan load failure. A
real saved-plan error and a fresh plan-generation error are reported as separate states.
The fresh-plan state includes the exact corrective reason, such as a finite evaluation budget that
is not divisible by the selected population; the application does not silently alter a scientific
input to make it pass.

The visible inputs are only the scientist-controlled portion of the plan. The generated plan also
contains deterministic member/episode identities and seeds, resource ceilings, schema and source
provenance, checkpoint/resume contracts, and exact evaluation accounting. Creating the plan does
not start training. Training completion still produces an unqualified, inactive candidate and does
not register, qualify, activate, select, or bind it automatically.

Executable compute modes are **CUDA-preferred** and **CPU-only**. CUDA admission is bounded by at
most 80% of currently free VRAM, while CPU admission is bounded by at most 80% of currently available
RAM. Intel XPU may appear only as historical or diagnostic metadata; it is not an executable mode.

Use **Qualify policy** to start or exactly resume one finite model-quality plan. Before running, the
application freezes the exact candidate architecture, parameter/schema contract, ensemble and
authenticated training-design identity, model checksum, qualification cases, paired seeds, and equal
evaluation budget. Product version and project lifecycle labels do not decide eligibility. Completed
cells are retained, so the same finite plan may be resumed any number of times without changing its
budget or selecting a more favorable rerun.

Qualification progress appears once in the persistent bottom bar. It shows live overall and current-
cell percentages while naming the separate durable-cell count. Activity logs a throttled micro step
every 500 evaluations (5% of a 10,000-evaluation formal cell), including case, run, baseline or
candidate side, best feasible objective, minimum violation, first-feasible point, throughput, and
cell ETA. The durable `qualification_events.jsonl` retains the same structured events; live in-cell
values are clearly labeled uncommitted until an authenticated checkpoint or final cell record is
acknowledged.

Use **Pause safely** in the bottom bar whenever the qualification task is running. The request is
cooperative: the current population transition finishes, then the exact optimizer population,
archives/memory, histories, evaluation counter, and random-number state are saved in an authenticated
checkpoint. The application reports **Paused** only after verifying the request, frozen-plan identity,
durable path, and SHA-256. Select the same policy and click **Qualify policy** later to restore that
checkpoint and continue the unchanged finite plan. Pause/resume may be repeated without a count
limit; it never resets or extends the frozen evaluation budget. While a run is active, task-changing
workspace controls are locked, but Activity and the bottom status bar remain usable.
If the application source changes between sessions, the library first searches for the exact
candidate's valid incomplete campaign and reopens its preserved clean source snapshot. A mere
software-version change therefore does not orphan a paused check; an architecture, schema, candidate
checksum, or frozen-plan incompatibility still fails closed.

A complete passing result is admitted automatically by the explicit qualification transaction, but
the policy remains inactive. Compare qualified policies using feasibility, objective improvement,
win rate, effect size, significance, anytime behavior, and matching evidence design. Then use
**Activate for experiments** explicitly. Finally, apply the active governing policy to experiment
settings before Power System becomes available. Qualification never performs those later actions.

**Delete model files** physically removes only an exact eligible inactive, unqualified, unreferenced
model after confirmation. Active, qualified, evidence-bound, lineage-bound, ambiguous, or
integrity-invalid models remain protected.

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
