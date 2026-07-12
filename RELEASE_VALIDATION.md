# CALO-RPD Studio 1.0.8 — Release Validation Record

## Release focus

Version 1.0.8 corrects the Live Optimization empty-plot behavior while preserving scientifically valid convergence semantics.

The Live Optimization page now:

- defaults to **Automatic (recommended)** convergence selection;
- shows best normalized constraint violation while any monitored optimizer has not yet produced a feasible incumbent;
- switches to best-feasible objective convergence when all currently represented optimizers have feasible histories;
- displays an explicit explanatory message instead of a visually blank canvas when the selected metric has no valid data;
- reloads stored convergence histories after experiment completion or cancellation so results remain visible when the page is opened after a run.

## Automated checks in the packaging environment

- Python source compilation: PASS
- Automated test suite: 35 passed, 17 skipped
- GUI-only tests skipped because PyQt6 was not installed in the packaging environment
- PYPOWER cross-validation tests skipped because PYPOWER was not installed in the packaging environment

The skipped tests are included in the repository and are expected to run in a fully provisioned application environment.
