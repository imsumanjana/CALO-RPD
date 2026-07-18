# CALO-RPD Studio 3.4.3 — Release Validation Record

## Scope

Version 3.4.3 fixes publication and portfolio export completion behavior without changing the ORPD equations, CALO operators, benchmark budgets, IEEE case formulations, or accelerator scheduling semantics introduced in v3.4.2.

The reported 94% stall was traced to the final `reproducibility_bundle` task: 16 of 17 artifacts equals 94.1%, and the previous implementation emitted no progress while recursively ZIP-compressing the selected output directory. Standard publication export also ran directly on the Qt GUI thread.

## Corrections validated

- Portfolio artifact 17 now reports bounded sub-progress from 94% through 99% while packing, then reaches 100% only after the ZIP is closed and atomically committed.
- Reproducibility packaging is scoped to current portfolio evidence rather than recursively archiving arbitrary files in the selected output tree.
- Already-compressed formats such as PNG/PDF/ZIP/NPZ are stored without redundant deflate work; text/configuration artifacts use fast compression.
- Safe pause/cancel is honored during final bundle creation; the temporary archive is deleted and completed artifacts remain resumable.
- Standard verified-publication export runs in a background `QThread` and reports progress/cancellation instead of blocking the Qt event loop.
- All-NaN leading convergence-grid columns are removed before median/IQR reductions in statistical, live, portfolio, and publication-evidence paths, eliminating the warning without substituting artificial values.
- Reproducibility archives include a manifest snapshot and use atomic temporary-file replacement.

## Test evidence

The repository contains **187 tests**. Validation was executed in isolated partitions to avoid cumulative resource interference from long-running policy-training and GUI tests:

- 68 unit tests passed in the first unit partition; 1 CPU-fallback policy-training smoke test was intentionally run separately.
- 64 unit tests passed in the second unit partition.
- 54 GUI/integration/regression/scientific tests passed.
- The separately executed CPU-fallback heterogeneous policy-training smoke test passed.
- Total: **187 passed, 0 failed** across the partitions.
- Focused v3.4.3 export-completion regression suite: **4 passed**.
- IEEE 30/57/118/300 scientific and v3.4 integrity suites: **20 passed**.
- Ruff: **passed with zero findings**.
- `compileall`: **passed**.
- Frozen release manifest: **74 files verified**.

## Hardware note

The validation environment did not provide physical NVIDIA CUDA or Intel XPU hardware. v3.4.3 does not modify the v3.4.2 GPU-preferred execution policy: tensor-compatible numerical work remains CUDA-first, then XPU, then CPU fallback. Qt orchestration, file/database persistence, report generation, ZIP packaging, and independent PYPOWER validation remain host-side tasks by design.
