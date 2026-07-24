# CALO-RPD Studio v6.8.0

**CALO-RPD Studio 6.8.0 — Independent CALO Intelligence & XPU Recovery** decouples CALO Intelligence policy-development/qualification workflows from Comparison/Portfolio tab constraints and hardens mixed NVIDIA+Intel XPU detection, repair, rediscovery, and readiness reporting. It preserves the v6.5-v6.7 scientific, audit, and hardware-binding closures.

## v6.8 independence and XPU recovery

- CALO Intelligence uses scientific-only policy-development validation; a valid one-run training formulation is no longer blocked by publication portfolio minimum-run rules.
- Other tabs no longer silently rehydrate CALO Intelligence controls through global `config_changed` events; applying a policy to an experiment remains explicit.
- Policy qualification uses CALO Intelligence's own scientific template and local seed rather than mutable Experiment Manager/Comparison Study state.
- Bootstrap repairs CUDA and XPU readiness per detected hardware family, so a healthy CUDA backend cannot hide a missing Intel XPU runtime.
- XPU sidecar discovery live-probes the recorded/canonical interpreter and can recover stale bootstrap state without requiring a restart after repair.
- Windows Intel detection adds PnP Display/`VEN_8086` hardware-tag fallback for hybrid-graphics laptops.
- A physical Intel GPU with no verified XPU runtime remains visible in System Readiness as detected-but-unavailable and is never scheduled until `xpu:0` passes the scientific probe.

## Prior release: v6.7 hardware-runtime closure

- NVIDIA CUDA compute discovery is now independent from optional NVML telemetry; a missing `nvidia-ml-py` can no longer erase a valid CUDA device from the scheduler.
- `nvidia-ml-py` is an explicit bootstrap/project dependency, with `nvidia-smi` retained as an independent telemetry supplement/fallback.
- NVIDIA telemetry is matched to runtime devices by UUID/PCI identity where available instead of assuming CUDA index equals `nvidia-smi` row/Windows GPU number.
- A single canonical device-binding function is used by primary, persistent CUDA, persistent XPU-sidecar, and one-shot XPU execution paths.
- XPU sidecar telemetry reports total memory and hardware identity fields and performs an explicit FP64 tensor/matmul smoke before ORPD evaluator capability is accepted.
- Every completed run records planned-vs-actual device attestation for runtime probe, evaluator, optimizer/control plane, and CALO policy inference.
- Windows adapter labels no longer imply that CIM enumeration order equals Task Manager GPU numbering.

## Prior release: CALO-RPD Studio v6.6.0

**CALO-RPD Studio 6.6.0 — Remaining Audit Closure** resolves the remaining medium-priority / “Better to Resolve” findings retained after v6.5, while preserving all v6.5 must-resolve scientific and integrity closures.

## v6.6 remaining-audit closure

- Reduces hot-loop case copying, vectorizes branch-angle constraints, removes inactive candidates from batched Newton linear solves, and fuses compatible candidate×scenario work into larger Torch batches.
- Bounds dense Torch/dense-fallback large-case memory paths and hardens sparse-to-dense fallback behavior.
- Uses one carried feasibility tolerance and one deterministic feasibility-first ordering across helper, pairwise, and bulk ranking paths.
- Separates persistent training RNG streams for PPO minibatch shuffling and historical pretraining, and makes degenerate Friedman evidence finite/non-significant instead of NaN.
- Bounds policy/network broker caches and Stage-B immutable static-tensor caches; oversized synthetic requests are split deterministically before device allocation.
- Narrows silent accelerator/resource/orchestration exception paths and reports device/profile/pool failures explicitly while retaining fail-forward scientific boundaries.
- Hardens workspace restore, configuration validation, resume-all task coverage, Results Explorer stale-run handling, portfolio manifest recovery, verified-count preservation, and Safe-80 governor reconstruction.
- Caches immutable real-development ExperimentConfig/case templates per rollout worker and preserves deterministic campaign ordering with a secondary key.

The v6.6 source-level closure suite covers every issue ID retained in the v6.4 priority list’s **Better to Resolve** section. Physical CUDA/XPU saturation, PyQt6 GUI interaction, and PYPOWER/commercial-reference validation remain target-environment qualification gates rather than simulated claims.

## v6.5 baseline — Must-Resolve Audit Closure

**CALO-RPD Studio 6.5.0 — Must-Resolve Audit Closure** closes every issue classified as **Must Resolve** in the post-v6.4 audit priority list while preserving the v6.4 Stage-B hybrid accelerator architecture.

## v6.5 must-resolve closure

- CPU-reference-style damping/backtracking is now implemented in single and batched Torch Newton–Raphson paths.
- Discrete stepped-variable generation cannot overshoot declared upper bounds.
- Zero/near-zero voltage-span normalization and near-zero policy-qualification arithmetic are stabilized.
- Single, batched, and device-resident Torch power-flow paths use one zero-impedance validity threshold.
- Policy checkpoint delete/update operations are transactional; latest-lineage registration is monotonic.
- Exact-resume checkpoints use an atomically published self-authenticating envelope, and checkpoint hashes are streamed.
- Policy and synthetic inference brokers fail pending/in-flight requests deterministically during shutdown.
- Comparison Study applies current GUI values before execution; Results Explorer tolerates malformed/incomplete JSON rows.
- Stage-B parity rejects unequal result lengths before comparison.
- Protected case118/case300 holdouts use canonical scientific identity rather than filename-only checks.

Focused must-resolve tests: **16 passed**. Combined must-resolve and accelerator/continuation regression selection: **57 passed** in the build runtime. Physical CUDA/XPU, PyQt6, and PYPOWER qualification remain target-environment gates.

## v6.4 Stage-B baseline retained

**CALO-RPD Studio 6.4.0 — Stage-B Device-Resident Policy Training** is a focused GPU/XPU training architecture upgrade on the v6.3 truthful-reporting baseline.

v6.4 does **not** pretend that every stochastic CALO controller operation has been rewritten onto the GPU. Instead, it moves the deterministic synthetic curriculum population objective/constraint kernel onto persistent FP64 PyTorch accelerator tensors, microbatches compatible requests across simultaneous rollout episodes, preserves a fail-closed NumPy-reference parity gate, and enables a real ORPD policy-development suite in the normal CALO Intelligence workflow.

## v6.4 Stage-B upgrades

### Device-resident synthetic curriculum evaluation

- Synthetic curriculum tasks are still generated by the trusted NumPy reference path so task generation and RNG semantics remain unchanged.
- On admitted CUDA/direct-XPU actor lanes, fixed task data are copied once into persistent FP64 device tensors.
- Population objective and constraint evaluation is executed in vectorized PyTorch on the accelerator.
- Compatible simultaneous episode requests are merged by a persistent cross-episode synthetic microbatch broker, producing larger accelerator batches rather than one tiny population transfer at a time.
- Candidate results are materialized back to the host once per merged microbatch for the existing stochastic CALO controller/archive/memory transition.

### Fail-closed scientific parity

Every generated accelerator-backed synthetic curriculum problem is checked against the original NumPy implementation before it is trusted. The parity gate compares:

- objective value;
- total constraint violation;
- feasibility classification;
- every constraint-component value.

A mismatch beyond the declared tolerance raises an error instead of silently switching scientific semantics. Periodic rechecks can be enabled and are on by default.

### Cross-episode batching without CPU oversubscription

- Synthetic requests from multiple simultaneous rollout episodes can be merged into one FP64 accelerator microbatch.
- Host-side controller steps remain capped by the protected per-branch rollout-worker budget.
- Stage B therefore does not reintroduce the pre-v6 problem where each branch could multiply the full CPU worker count.

### Real ORPD development suite in CALO Intelligence

The normal GUI no longer hardcodes an empty development-case tuple.

Default development suite:

- `case30`
- `case57`

Default formulation:

- `calo_rpd_studio/data/examples/policy_development_active_loss.yaml`

The real ORPD stage loads the declared `ExperimentConfig` and carries its exact:

- objective configuration;
- mixed-variable profile;
- PowerFlowOptions;
- robust objective;
- scenario construction;
- constraint tolerances.

When accelerator ORPD rollouts are enabled, this exact formulation is evaluated through the FP64 accelerator-native ORPD path on the actor device.

`case118` and `case300` are rejected by the normal development-suite selector because they remain protected as held-out/final evaluation systems in this release workflow.

### Stage-B validation command

```bash
python -m calo_rpd_studio.scripts.validate_stage_b_synthetic --device auto
```

or, after installation:

```bash
calo-rpd-stage-b-validate --device auto
```

The validator reports parity error, candidate throughput and microbatch statistics. A CPU-only run validates implementation logic but is **not** a physical CUDA/XPU qualification.

## What `CUDA 100%` means in v6.4

The Stage-A wording correction remains in force:

> **CUDA 100% means 100% of eligible rollout episodes are routed to the CUDA actor lane after protected rebinding.**

In Stage B, more of each admitted synthetic rollout is now genuinely accelerator-resident because the deterministic population evaluation kernel runs on CUDA/XPU. However, the stochastic CALO controller, archives, memory updates, candidate/controller orchestration and some state construction remain on the trusted host/reference path.

Therefore v6.4 does **not** claim:

- that every CALO operation is GPU-resident;
- that Task Manager must show a fixed utilization percentage;
- guaranteed 90–100% sustained NVIDIA utilization;
- bit-for-bit CPU↔CUDA floating-point identity on untested hardware.

The goal is higher useful accelerator work while preserving scientific parity and the v5.9+ controller semantics.

## Training progress and Safe-80 reporting

The v6.3 corrections remain:

- selected routing, recommendation, protected routing and runtime mapping are separate;
- fixed/cumulative training reports target-aware branch-epoch progress;
- exact resume separates session progress from cumulative epoch;
- last/next durable exact-safe checkpoint is shown explicitly;
- queued branches do not silently spill to CPU.

## Protected compute architecture inherited from v6.0–v6.3

- Dashboard-first CPU/XPU/GPU mapping.
- Safe-80 protected resource envelope.
- CALO Intelligence as governing intelligence before Power System.
- Power System locked until a qualified, active, runtime-compatible, integrity-verified policy is ready.
- Global Training Exclusive Lock.
- Scientific branch count separated from safe simultaneous concurrency.
- Protected branch queue and exact-resume rotation.
- One global CPU worker budget.
- No uncontrolled accelerator-to-CPU branch spillover.
- Capability-aware CUDA/direct-XPU/XPU-sidecar scheduling.
- Adaptive Green/Amber/Red compute protection and staged startup.
- Workspace schema migration, application recovery and hash-chained compute provenance.

## Canonical workflow

Dashboard → CALO Intelligence → Power System → ORPD Formulation → Algorithms → Portfolio → Robust Scenarios → Experiment → Results/Validation/Publication.

## Launch

```bash
python bootstrap.py
```

## Important validation boundary

This build environment does not provide physical NVIDIA CUDA, physical Intel XPU, PyQt6 GUI rendering, or the complete PYPOWER target stack. Therefore v6.4 includes the Stage-B implementation and dependency-light parity/regression evidence, but physical accelerator throughput/utilization and long-duration target-laptop qualification must be executed on the intended Windows machine before making hardware-performance claims.

## Release evidence

- `CALO-RPD-v6.4.0_IMPLEMENTATION_REPORT.md`
- `CALO-RPD-v6.4.0_DEEP_POST_GENERATION_AUDIT.txt`
- `FINDINGS_CLOSURE_v6.4.0.csv`
- `HARDWARE_QUALIFICATION_STATUS.json`
- `SCIENTIFIC_EQUIVALENCE_STATUS.json`
- `calo_rpd_studio/data/frozen/calo_v640_freeze.json`
- `MANIFEST.sha256`
