# CALO-RPD Studio 1.0.6 — Release Validation Record

## Scope

Version 1.0.6 corrects the post-experiment result-review transition and adds selective series export to scientific plots while preserving the guided workflow, square live preview, popup plot tools, high-resolution export, optimization engine, CALO architecture, statistics, validation, and reproducibility features.

## Result-review workflow

- A selected Results Explorer row is explicitly tracked as the reviewed run.
- Confirming review marks the workflow step complete.
- Validation & Audit is unlocked immediately.
- The reviewed experiment and run are transferred to Validation & Audit.
- The application navigates to Validation & Audit automatically.

## Selective series export

- Export checkboxes are built from the currently available legend-capable series in the preview.
- Displayed legend-name overrides are reflected in checkbox text.
- Users may export all or any subset of available series.
- Non-selected series are temporarily hidden only during save.
- Export legends are rebuilt from selected visible series only.
- Preview line visibility and legend content are restored after saving.
- Empty selection is rejected when selectable series exist.

## Existing figure guarantees retained

- Live Optimization uses an exact 1:1 square preview.
- Live Optimization content scrolls vertically when screen height requires it.
- Live-plot PNG, SVG, and PDF exports remain exact square outputs.
- PNG DPI remains selectable from 600 through 2400 with 600 as the default.
- Plot tools remain separated into Text & labels, Plot appearance, Export figure, and Style profiles popups.

## Repository automation

- No `.github/workflows` directory is included.
- The guided workflow is implemented inside CALO-RPD Studio only.

## Validation performed in this build environment

- Python source compilation completed successfully for `calo_rpd_studio` and `tests`.
- Automated tests: **34 passed, 14 skipped**.
- The skipped tests require PyQt6 and PYPOWER, which are not installed in this build environment.
- Selective-series export tests passed, including temporary filtering of saved curves and legend entries followed by full preview restoration.
- Empty export selection is rejected when selectable plot series exist.
- GUI regression coverage was added for dynamic export-series checkboxes and reviewed-run navigation; these tests run when PyQt6 is available.
- No `.github/workflows` directory is present.
