# CALO-RPD Studio v2.0.3 Patch Notes

## Intelligent convergence scaling

Live Optimization now defaults to automatic visible-data fitting. The axis range is recalculated from the currently previewed series after every redraw, preventing stale limits from earlier plots or manual sessions from making small algorithm differences unreadable.

For non-negative feasibility metrics, zero remains visible by default so the scientific target of exact feasibility is not hidden. Best-feasible objective plots use a tight data-driven range instead of forcing a zero baseline.

Plot Tools → Plot appearance now includes:

- Auto-fit visible data;
- Include zero for non-negative Y data;
- Auto-fit padding.

Disabling auto-fit restores manual X/Y bounds.

## Bulk independent validation

Validation & Audit now supports:

- Validate current experiment;
- Validate all not-yet-verified runs;
- live run-by-run progress;
- passed, failed, and processing-error counts;
- cancellation between independent run reconstructions;
- a final machine-readable JSON summary.

Verified runs are skipped by default. Failed or unverified runs remain eligible for revalidation after a scientific or software correction.
