# Changelog

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
