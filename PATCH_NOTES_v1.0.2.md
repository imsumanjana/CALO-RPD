# CALO-RPD Studio 1.0.2

This maintenance release focuses on interface quality and experiment-execution safety.

## Interface refinements

- Reworked the application shell with a sharper light/dark visual system, a compact project toolbar, and a modern navigation sidebar.
- Replaced page-level scroll containers on workspaces that already fit naturally within the application window.
- Page-level scrolling is now reserved for genuinely long configuration forms: ORPD Formulation, CALO Intelligence, and Robust Scenarios.
- Rebuilt the Dashboard with metric cards and a cleaner two-column scientific context view.
- Reworked Power System, Algorithms, Experiment Manager, Live Optimization, Statistical Analysis, Results Explorer, Validation & Audit, Publication Export, and Application Settings as normal workspaces.
- Restyled tables, tabs, inputs, buttons, status surfaces, and scrollbars for stronger contrast and a more precise desktop appearance.

## Experiment manager correction

- Starting a second experiment while another run is active no longer raises an uncaught `RuntimeError`.
- The manager now uses an explicit busy state before worker startup, preventing double-click and cross-workspace race conditions.
- Duplicate start requests are rejected safely and reported to the interface.
- Worker cleanup now occurs after the `QThread` has actually finished.

## Compatibility retained

- NumPy remains constrained to `<2.4` for PYPOWER compatibility.
- Independent PYPOWER validation continues to use direct `runpf`/`ppoption` imports rather than the broad `pypower.api` import path.
