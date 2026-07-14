# CALO-RPD Studio 2.0.3 — Release Validation Record

## Scope

Version 2.0.3 improves interpretation and validation throughput without modifying the frozen CALO mathematical implementation. Live convergence plots now auto-fit the currently visible data, and the Validation & Audit workspace can independently re-evaluate saved runs in bulk.

## Automated verification

- `PYTHONPATH=. pytest -q`: **84 passed, 25 skipped**.
- `python -m compileall -q calo_bootstrap calo_rpd_studio tests`: passed.
- Plot-manager regression coverage verifies zero-aware tight scaling and visible-series-only scaling.
- Bulk-validation regression coverage verifies default skipping of already verified runs and continuation after an individual run raises a validation error.
- `python -m pip wheel . --no-deps`: passed and produced the v2.0.3 wheel.
- Frozen CALO verification: passed across the original **23 frozen files**.
- Ruff was unavailable in the packaging environment and is not claimed as executed.
- PyQt6 GUI tests were skipped because PyQt6 was unavailable; three IEEE scientific tests were skipped because PYPOWER was unavailable.

## Scientific behavior

Automatic fitting changes only axis limits. It does not modify stored convergence data, algorithm ranking, objective values, constraint values, or exported raw results. For normalized constraint violation and other non-negative diagnostics, the default live view includes zero because exact feasibility is the scientific target. Users can disable automatic fitting in Plot Tools and enter manual limits when a fixed cross-figure scale is required.

Bulk validation reconstructs and independently re-evaluates each saved decision vector using the stored experiment configuration and seeds. It skips records already marked verified by default, records each new validation result in SQLite, continues after isolated processing errors, and supports cancellation between runs. Publication claims should still be based on successfully verified records.
