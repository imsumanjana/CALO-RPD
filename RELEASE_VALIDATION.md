# CALO-RPD v6.9.0 Release Validation

> **Immutable historical v6.9 validation record.** These results do not validate the v12 source,
> Phase 4, a current policy, or current release readiness. Intel XPU references below report the old
> v6.9 scope and do not make XPU executable in v12. Do not rerun or reinterpret this file as the
> current validation plan.

## Status

- Python compileall: **PASS**
- Focused v6.9 VRAM-residency suite: **7 passed**
- Device-resident/v6.5/v6.6 regression selection: **53 passed**
- Heterogeneous-training/Stage-B/hardware/v6.9 regression selection: **47 passed**
- Configuration round-trip validation: **4 passed**
- v6.9 release-integrity gate: **5 passed**
- Scientific/software freeze: **150/150 files verified**
- Final package manifest: **484 packaged files** (independent verification performed after ZIP creation)
- Physical NVIDIA CUDA in build environment: **not available**
- Physical Intel XPU in build environment: **not available**
- PYPOWER publication-case tests: **not available in build environment**
- Target-machine RTX 4060/XPU qualification: **required**

The two regression selections overlap and are not presented as one summed unique-test count.

## Verified contracts

1. CUDA process memory is limited to an 80%-default configurable ceiling rather than unsafe full physical VRAM allocation.
2. Active CUDA-compatible ORPD tensors remain on CUDA until the completed population request is materialized.
3. Completed microbatch outputs are concatenated on-device.
4. CUDA OOM reduces the active microbatch and retries on CUDA; it does not silently move the run to CPU.
5. The resident Newton/backtracking path can avoid host convergence/active-row scalar checks by using fixed-shape masks.
6. PPO loss values are materialized once per epoch rather than after each minibatch.
7. PPO CUDA OOM reduces the active minibatch and retries without changing the learner device.
8. VRAM budget, peak allocation/reservation, microbatch sizes, OOM retries and CPU-fallback count are recorded in provenance.

## Declared boundary

The GUI, Python control plane, SQLite, file I/O, logging, checkpoint serialization and final independent CPU-reference validation remain host-side. The v6.9 residency claim is limited to the active CUDA-compatible numerical data plane. Physical speedup and sustained utilization are not claimed until measured on the target RTX 4060 laptop.
