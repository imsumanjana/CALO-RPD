# CALO-RPD Studio v6.6.0 — Release Validation

**Release:** 6.6.0 — *Remaining Audit Closure*  
**Date:** 24 July 2026

## Remaining priority-list scope closed

v6.6 audits every issue retained in the supplied v6.4 priority list under **Better to Resolve**:

`C02, C03, C05, C06, C07, H03, H10, H14, H16, H17, H18, H19, H20, H23, H25, H26, H27, M04, M05, M16, M34, M36, M37, M48, M52, M54, M57, L19, L20, L23, V64-N04, V64-N05, V64-N06`.

- **32 actionable findings/improvements:** resolved in code.
- **H19:** `VERIFIED_NOT_DEFECT`. Its premise was incorrect because Python `except Exception` does not catch `KeyboardInterrupt` or `SystemExit`.
- All v6.5 **Must Resolve** closures remain preserved.

## Key closure groups

- Reduced hot-loop copying, vectorized branch-angle constraints, active-row batched Newton solving, cross-scenario Torch batching, cached real-development config/cases, bounded oversized synthetic requests, and bounded static-tensor reuse.
- Unified configured feasibility tolerance and deterministic feasibility-first ranking semantics.
- Separated persistent training RNG streams and hardened degenerate Friedman evidence against NaN.
- Bounded policy/network/broker caches and large dense-allocation fallback paths.
- Narrowed resource/accelerator/orchestration error handling while preserving explicit top-level fail-forward boundaries.
- Hardened semantic workspace restoration, read-only configuration validation, strict unknown-field rejection, multi-type resume dispatch, stale Results Explorer selections, corrupt portfolio manifests, verified-result preservation, and lazy Safe-80 governor reconstruction.
- Added L-index dimension/identity checks and safer sparse-to-dense Newton fallback behavior.

## Executed build-runtime evidence

- Focused v6.6 remaining-audit suite: **22 passed**.
- Foundational/configuration/scientific regression selection: **82 passed**.
- Accelerator/continuation/v6.4/v6.5/v6.6 regression selection: **91 passed**.
- The two regression selections overlap and are **not summed as a unique test count**.
- Python `compileall`: **PASS** for source and tests.
- v6.6 release-integrity gate: **5 passed**.
- v6.6 scientific/software freeze: **141 / 141 frozen files**, missing `0`, changed `0`.

## Scientific and performance boundary retained

v6.6 does **not** claim a new sparse-GPU Newton/Ybus implementation. For large cases that would exceed bounded dense accelerator allocation, the software uses a safe reference/fallback path rather than risking uncontrolled O(n²) memory use.

The release does **not** claim that every stochastic CALO controller/archive/memory operation is GPU-resident and does not guarantee a fixed GPU utilization percentage or speedup.

## Build-runtime boundaries

- PyTorch: `2.10.0+cpu`.
- Physical NVIDIA CUDA: **NOT AVAILABLE**.
- Physical Intel XPU: **NOT AVAILABLE**.
- PyQt6: **NOT AVAILABLE**.
- PYPOWER: **NOT AVAILABLE**.

Therefore the build runtime cannot certify target-laptop CUDA/XPU throughput/utilization, long-duration thermal behavior, physical PyQt6 interaction, or PYPOWER/commercial-reference equivalence.

## Target-machine qualification still required

1. Run the Stage-B parity validator on the intended CUDA/XPU runtime.
2. Execute CPU↔accelerator scientific-equivalence regressions on representative stressed ORPD cases.
3. Measure real training/evaluation throughput and device utilization under Safe-80 scheduling.
4. Run long-duration protected thermal/soak qualification.
5. Run the full PyQt6 workflow on the target Windows system.
6. Run PYPOWER-backed independent power-flow validation where required by the release protocol.

## Software freeze

- Freeze ID: `calo_v660_software_release`
- Freeze manifest: `calo_rpd_studio/data/frozen/calo_v660_freeze.json`
- Frozen files verified: **141 / 141**
- Missing: **0**
- Changed: **0**
- Canonical freeze manifest SHA-256: `02900ecc9738ee2e268a0626c5070dfe13b9be9e7e38621b741646641be208c8`
- Freeze-file SHA-256: `ecff20ebfe3126eccf518daa35490d95d70d92eddf1c448853676baadc0be2cd`

The final root package manifest lists **464 packaged files** (excluding the manifest itself and transient cache files) and is independently checked by the v6.6 release-integrity gate before ZIP delivery.
