# CALO-RPD v6.9.0 Implementation Report

## Release

**Version:** 6.9.0
**Release name:** VRAM-Resident CUDA Data Plane

## Purpose

v6.9.0 implements the agreed execution model: CALO-RPD may use at most an 80%-default share of physical CUDA VRAM, while the complete active CUDA-compatible numerical data plane remains resident inside that budget. The CPU remains the asynchronous application/control/persistence plane and is removed from the Newton/backtracking hot loop and from per-minibatch PPO loss materialization.

## Implemented changes

### 1. Adaptive 80%-default CUDA VRAM governor

A new `calo_rpd_studio/accelerated/vram_residency.py` module provides:

- configurable CUDA process memory ceiling (`0.80` by default, validated range `0.10–0.95`);
- process-local enforcement through PyTorch per-process memory fraction;
- free/total memory and peak allocated/reserved telemetry;
- adaptive microbatch reduction after a genuine CUDA OOM;
- retry on the same CUDA device without silent CPU fallback;
- explicit provenance for microbatch sizes, retries and CPU-fallback count.

### 2. Device-resident ORPD population execution

`DeviceResidentORPDEvaluator` now:

- begins CUDA requests as one complete population batch;
- keeps decoded controls, scenario expansions, admittance matrices, Newton states, Jacobians, branch flows, objectives, constraints and result tensors on the assigned device;
- halves only the active microbatch if the CUDA process reaches its configured VRAM ceiling;
- concatenates completed microbatch outputs on-device;
- materializes the stable public `Evaluation` records once after the completed population request.

Static zero/near-zero branch-impedance validation is performed before device execution, removing that synchronization from the CUDA hot loop.

### 3. CPU-free CUDA Newton/backtracking hot loop

The device-resident Newton path can now use a fixed-shape masked implementation:

- converged/failed rows are neutralized by identity systems and zero right-hand sides;
- no active-row index list is transferred to the CPU;
- damping trials are fixed and mask-controlled on the device;
- convergence and acceptance decisions remain tensor operations;
- the previous host early-exit behavior remains available through configuration for non-resident/reference use.

This intentionally trades some extra masked GPU FLOPs for fewer host synchronizations and more continuous kernel scheduling.

### 4. PPO learner residency and OOM recovery

Heterogeneous CALO policy training now:

- applies the same 80%-default CUDA memory ceiling to the PPO learner process;
- keeps network, optimizer state, rollout tensors and active minibatches on the learner device;
- transfers the shuffled index vector once per PPO epoch;
- accumulates loss tensors on-device and materializes them once per epoch instead of calling `.item()` after every minibatch;
- halves and retries only the active PPO minibatch after CUDA OOM;
- never changes the learner to CPU as an OOM fallback.

### 5. GUI and configuration

Experiment Manager adds controls for:

- CUDA VRAM residency budget;
- CUDA OOM retry count;
- fixed-shape CPU-free CUDA Newton/backtracking hot loop.

The configuration schema and serialization now include:

- `cuda_vram_budget_fraction`;
- `cuda_oom_retry_count`;
- `cuda_minimum_microbatch`;
- `cuda_resident_hot_loop`.

These operational fields are excluded from scientific experiment fingerprints because they do not change the mathematical formulation.

## Preserved boundaries

v6.9 does not claim that the PyQt6 GUI, Python interpreter, SQLite, filesystem, logging, checkpoint serialization or independent CPU-reference validation execute in VRAM. Those are host responsibilities. The device-residency claim is limited to the active CUDA-compatible numerical data plane.

The build runtime has no physical NVIDIA CUDA or Intel XPU device. Therefore sustained RTX 4060 utilization, speedup, thermal behavior and real 80%-ceiling OOM recovery remain target-machine qualification steps.

## Validation performed

- Python `compileall`: **PASS**
- Focused v6.9 VRAM-residency suite: **7 passed**
- Relevant v6.9 + device-resident + v6.5/v6.6 regression selection: **53 passed**
- Relevant heterogeneous training + Stage-B + hardware/XPU + v6.9 selection: **47 passed**
- Configuration round-trip suite: **4 passed**
- v6.9 release-integrity gate: **5 passed**
- Scientific/software freeze: **150/150 files verified**
- Final package manifest: **484 packaged files**

The 53-test and 47-test selections overlap and are not summed as a unique total. PYPOWER-dependent publication-case tests were not executed in this build environment because PYPOWER is unavailable.
