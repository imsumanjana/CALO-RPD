# Changelog

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
