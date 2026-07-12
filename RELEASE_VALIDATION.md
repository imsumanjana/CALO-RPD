# CALO-RPD Studio 1.0.5 — Release Validation Record

## Scope of this release

Version 1.0.5 reorganizes the scientific plot controls while preserving the optimization, ORPD, CALO, guided workflow, task-status, result-storage, statistical-analysis, validation, square-preview, and high-resolution export behavior from version 1.0.4.

### Organized plot editing tools

- Every scientific plot uses a compact four-icon tool strip instead of an always-expanded formatting panel.
- **Text & labels** opens a dedicated popup for typography, title, axis labels, tick labels, legend text, and annotation styling.
- **Plot appearance** opens a dedicated popup for scales, limits, grids, axis width, line style, line width, markers, and series visibility.
- **Export figure** opens a dedicated popup for PNG, SVG, and PDF export settings.
- **Style profiles** opens a dedicated popup for save, load, reset, and apply-to-all actions.
- The visible plot toolbar contains no internal scroll area and no permanently expanded grid of editing controls.
- Plot edits continue to redraw the active Matplotlib figure immediately.

### Live Optimization plotting and export

- Live convergence preview remains an exact 1:1 square Matplotlib canvas.
- The Live Optimization content area retains vertical scrolling when screen height is insufficient.
- Live-plot export remains square for PNG, SVG, and PDF.
- PNG resolution remains selectable from 600 through 2400 DPI, with 600 DPI as the default.
- Square export locks width and height and disables tight cropping to preserve final 1:1 dimensions.

### Repository automation

- No `.github/workflows` directory is included.
- The guided scientific workflow remains entirely inside the CALO-RPD Studio desktop application.

## Validation performed in the build environment

- Python source compilation completed successfully for `calo_rpd_studio` and `tests`.
- Automated test result: **32 passed, 12 skipped**.
- The skipped tests require PyQt6 and PYPOWER, which were not installed in the build environment used for this package.
- Non-GUI square-export regression tests passed.
- GUI source files, including the popup-based plot tools, were successfully byte-compiled.
- A GUI regression test was added to verify the four focused plot tools and the absence of an internal scroll area in the compact toolbar; it is skipped when PyQt6 is unavailable.

## Recommended local verification

In the project virtual environment, run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python main.py
```

On a system with PyQt6 and PYPOWER installed, the skipped GUI and scientific cross-validation tests should run normally.
