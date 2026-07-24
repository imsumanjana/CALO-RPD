# CALO-RPD v6.6.0 — Implementation Report

**Release name:** Remaining Audit Closure  
**Date:** 24 July 2026

## Scope

v6.6.0 closes every issue ID retained in the supplied v6.4 priority list under **Better to Resolve**. Of the 33 listed IDs, **32 required engineering changes and are resolved**; **H19 is closed as VERIFIED_NOT_DEFECT** because its premise is technically incorrect (`except Exception` does not catch `KeyboardInterrupt` or `SystemExit`). The v6.5 must-resolve fixes remain preserved.

## 1. Throughput and scalability

- **C02:** ORPD decoding uses a thread-local reusable case workspace instead of cloning the full case on every evaluation.
- **H03:** branch-angle constraints and bus-index lookup are vectorized.
- **M36:** batched Newton solves only unconverged/active rows.
- **M37:** compatible candidate×scenario work is flattened into larger deterministic Torch batches.
- **V64-N04:** real-development rollout caches ExperimentConfig and immutable source case templates per worker/chunk.
- **V64-N05:** oversized synthetic requests are deterministically split before device allocation.
- **V64-N06:** immutable Stage-B curriculum/static tensors use a bounded fingerprinted LRU cache.
- **C05/C07:** dense Torch and dense fallback paths are guarded before unsafe O(n²) allocation. Large cases fall back to the sparse CPU-reference path where supported; this release does **not** claim a new sparse-GPU Newton implementation.

## 2. Scientific consistency, ordering, statistics, and reproducibility

- **C03/M04/M05:** one carried feasibility tolerance and one canonical feasibility-first ordering are used by feasibility helpers, pairwise comparison, and sorting.
- **H10:** historical-pretraining and PPO-shuffle RNG streams are separated and their states are saved/restored independently.
- **H14:** degenerate/all-tied Friedman evidence is converted to an explicit finite neutral result instead of propagating NaN.
- **H23:** campaign ordering has a deterministic secondary key.

## 3. Edge and fallback safety

- **C06:** L-index partition, voltage, and bus-order dimensions are validated before indexing.
- **C07/M34:** dense Newton fallback is bounded; sparse Jacobian fallback handles expected construction/runtime failure classes and reports the fallback.

## 4. Bounded long-session caches

- **M16:** policy-network and broker caches are bounded LRU caches with broker cleanup on eviction.
- **V64-N06:** static Stage-B tensor caching is also bounded.

## 5. Error surfaces and diagnostics

- **H16–H18/H20:** broad silent resource/accelerator/orchestration catches were narrowed where recovery is specific; expected failures now include diagnostics. Broad handlers remain only at intentional top-level/fail-forward boundaries where errors are explicitly attached to requests, calibration state, or job status.
- **H19:** verified as an invalid audit premise rather than modified artificially.

## 6. GUI, workspace, configuration, and recovery

- **H25/H27:** workspace restoration uses semantic page keys and structured stage-aware restore errors, including missing/corrupt cases.
- **H26/L20:** validation is read-only; execution normalization is explicit; unknown configuration fields are rejected by default, including nested sections.
- **M48:** policy qualification completion state is explicit rather than inferred with `locals()`.
- **M52:** Resume Center dispatches experiments, policy training, validation, and portfolio export, with transparent deferred/unsupported reporting.
- **M54:** stale Results Explorer selections no longer raise uncaught `KeyError`.
- **L19:** corrupt portfolio manifests fail with actionable diagnostics.
- **L23:** stopping an experiment preserves verified-result counts.
- **M57:** lazy Safe-80 governor reconstruction preserves `allocation_limit_fraction`.

## Validation evidence

- New v6.6 closure suite: **22 passed**.
- Foundational/configuration/scientific regression selection: **82 passed**.
- Accelerator/continuation/v6.4/v6.5/v6.6 regression selection: **91 passed**.
- These selections overlap; they are reported separately and are **not summed as a unique-test total**.
- v6.6 release-integrity gate: **5 passed**.
- Python source/test compile check and final freeze/package-manifest verification are recorded in `RELEASE_VALIDATION.md`.

## Qualification boundary

The build runtime is dependency-light and CPU-only for PyTorch. Therefore v6.6 makes no unsupported claim of physical CUDA/XPU utilization, speedup, long-duration thermal behavior, PyQt6 interaction, or PYPOWER/commercial-reference equivalence. Those remain target-machine qualification gates.
