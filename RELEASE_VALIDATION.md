# CALO-RPD Studio v6.4.0 — Release Validation

**Release:** 6.4.0 — *Stage-B Device-Resident Policy Training*  
**Date:** 23 July 2026

## Stage-B scope implemented

- FP64 device-resident synthetic curriculum population objective/constraint evaluation on admitted CUDA/direct-XPU actor lanes.
- Persistent cross-episode synthetic microbatching.
- Fail-closed NumPy-reference startup parity and periodic parity rechecks.
- Preserved curriculum RNG/task-generation semantics.
- Protected host-controller thread cap.
- Real ORPD policy-development suite exposed in CALO Intelligence.
- Default development suite: `case30, case57`.
- Default bundled development formulation: `policy_development_active_loss.yaml`.
- Exact ExperimentConfig propagation into heterogeneous real-ORPD development rollouts.
- Stage-B parity/throughput validation CLI.

## Explicit scientific boundary

v6.4 is a hybrid device-resident architecture. It does **not** claim that every stochastic CALO controller/archive/memory operation is GPU resident, nor does it guarantee a particular Task Manager utilization percentage.

## Executed build-runtime evidence

- `compileall`: **PASS** for `calo_bootstrap`, `calo_rpd_studio`, and `tests`.
- Focused Stage-B + Stage-A + v6/v5.9/CALO regression selection: **66 passed**.
- Heterogeneous training / historical pretraining / broker / audit-closure selection: **38 passed**.
- Historical competitive-training tests: **9/9 passed**, executed in two bounded groups.
- Static silent-handler AST audit: **0** broad `Exception`/`BaseException`/bare handlers whose sole action is `pass`, `continue`, or `return`.
- Stage-B dependency-light CPU validator:
  - parity verified: **true**;
  - maximum startup parity error: approximately `4.44e-15`;
  - maximum merged batch observed in the executed validation: **96 candidates**;
  - physical accelerator qualification: **false**.

A larger all-unit invocation reached late in the suite before the external command time limit. It is not represented as a complete-suite PASS.

## Build-runtime boundaries

- PyQt6: **NOT AVAILABLE**.
- PYPOWER: **NOT AVAILABLE**.
- Physical NVIDIA CUDA: **NOT AVAILABLE**.
- Physical Intel XPU: **NOT AVAILABLE**.
- Installed PyTorch build: CPU-only.

Therefore the build environment cannot certify physical RTX 4060/XPU throughput, utilization, thermal behavior, or full target-GUI behavior.

## Target-machine Stage-B qualification required

1. Run `calo-rpd-stage-b-validate --device cuda` on the target NVIDIA runtime.
2. Verify startup and periodic parity remains within the declared tolerance.
3. Measure candidate/transition throughput and microbatch merge rate.
4. Run real policy training with the case30/case57 development suite and verify the real ORPD stage reaches the intended accelerator evaluator.
5. Compare training scientific outputs/trajectory gates against the CPU reference within declared tolerances.
6. Run Safe-80/Green-Amber-Red thermal protection and multi-hour soak on the actual laptop.
7. Run PyQt6 visual/workflow validation and complete PYPOWER scientific validation.

## Software freeze

- Freeze ID: `calo_v640_software_release`
- Freeze manifest: `calo_rpd_studio/data/frozen/calo_v640_freeze.json`
- Frozen files verified: **138 / 138**
- Missing: **0**
- Changed: **0**
- Canonical freeze manifest SHA-256: `f8c3d650194724bf75c33d70fa36c4f297f23ae6fd98f8369ca0129c2592d263`
- Freeze-file SHA-256: `f02f9bc21f7afc3e1c67cc82a40cc0a4bd68be0480a1de823fcda0d95e538258`

The root package manifest is regenerated only after final release-evidence files are complete, then the delivered ZIP is extracted and both manifest and freeze are reverified independently.
