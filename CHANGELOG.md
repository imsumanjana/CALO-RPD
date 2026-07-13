# Changelog

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