# CALO-RPD v6.2.1 — Dashboard Layout Patch

## Scope

This patch changes only the Dashboard presentation/layout architecture. Scientific CALO, policy-training, Safe-80 compute protection, ORPD, recovery, qualification, and publication semantics remain unchanged from v6.2.0.

## Implemented dashboard changes

- The Dashboard body is now vertically scrollable, so content keeps its natural size instead of being compressed when the application window is short.
- The three previously stacked dashboard panels are now separate tabs:
  1. **System Readiness**
  2. **Training Queue**
  3. **Scientific Context**
- Every tab has its own vertical scroll area.
- The System Readiness hardware/device table keeps a usable minimum height and its native table scrolling.
- The top summary metric cards now use a 3-column, two-row grid instead of one dense 5-card row.
- Existing signal wiring, compute refresh, live governor sampling, policy readiness, training queue status, and scientific-context refresh behavior remain unchanged.

## Compatibility

No scientific data schema or policy ABI changed. Workspace Schema 3 remains unchanged. Existing v6.2.0 projects remain compatible.

## Validation

- Python source compilation: PASS.
- Dedicated dashboard-layout static regression tests: PASS.
- Existing dependency-light release/core regressions executed for this patch: recorded in `CALO-RPD-v6.2.1_DEEP_POST_GENERATION_AUDIT.txt`.
- PyQt6 is not available in the build runtime, so a physical GUI rendering test is not falsely claimed.
