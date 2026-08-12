# CALO-RPD Studio v12.0.0-dev.1

**Active status: development only.** This tree starts the v12 modernization line. It is not a
release candidate or final release, its v12 final freeze and qualification gates remain open, and
historical v6.9 evidence does not qualify v12. Policy training, policy qualification,
protected-case evaluation, and release production are separate gated workflows.

Phase 4 completes the remaining production code, empty-policy behavior, integration surfaces,
documentation, and development-freeze evidence only. Every existing policy is development-only,
unqualified, inactive, non-final, and excluded from final-candidate selection and initialization.
Phase 4 does not train, evaluate, qualify, activate, register, or delete policies. After the
development freeze, old policies may be removed only through a separately authorized,
inventory-first operation. Any later policy-assisted route must train an entirely new A-E/F-off
policy from a clean policy store and qualify it independently; a policy-free Phase 5 route also
remains valid.

Phase 5 release-preparation development is implemented at the unchanged `12.0.0.dev1` identity.
It adds explicit policy-free/newly-qualified-policy scope contracts, distinct wheel/sdist member
manifests, source-bound distribution/image/SBOM/security/clean-install evidence aggregation, final
CI contract checks, and a separately authorized final-record generator. The project owner elected
to run Phase 4 and Phase 5 validation together after both coding phases. Until that combined run is
returned and accepted, the release-policy scope remains pending and this tree is not a release
candidate, release-ready, final `12.0.0`, tagged, published, or released.

The implemented Phase 4 lifecycle is fail-closed: the GUI no longer auto-discovers checkout policy
files; direct registry/GUI deletion is disabled; the Policy Center exports an exact SHA-256-bound
inventory and dry-run plan; stale policy bindings are cleared in empty-policy mode; and only a new
post-development TSH-CALO ensemble can later be activated or bound. Future training is bound to an
exact clean, empty-policy freeze report by both commit and payload SHA-256 and proves empty policy
initialization. Policy inference is pinned to
the experiment scheduler's resolved device, so CUDA-to-CPU fallback can occur only as a separately
identified full-request restart. The `calo-rpd-development-freeze` command creates a development-
only source/interface report—not a release manifest or release-readiness claim.

See [`docs/DOCUMENTATION_STATUS.md`](docs/DOCUMENTATION_STATUS.md) before using older release,
training, audit, or validation documents as instructions.

## Historical release: v6.9.0

**CALO-RPD Studio 6.9.0 — VRAM-Resident CUDA Data Plane** introduces an adaptive 80%-default CUDA VRAM ceiling while keeping the complete active CUDA-eligible ORPD and PPO numerical data plane resident on the accelerator. CPU remains the asynchronous control/persistence plane; it is removed from the Newton/backtracking hot loop and from per-minibatch PPO loss reads.

## v6.9 VRAM-resident execution

- Adds a process-local `VramResidencyGovernor` with an 80% default CUDA VRAM ceiling (configurable from 10% to 95%).
- Starts each CUDA population request as one device-resident batch and halves only the active microbatch after a genuine CUDA OOM. It retries on CUDA and never silently falls back to CPU.
- Keeps decoded controls, scenarios, admittance matrices, Newton states, Jacobians, branch flows, objectives, constraints, feasibility masks and completed microbatch outputs on the assigned device.
- Replaces device-resident Newton active-row host inspection with fixed-shape masked identity systems and fixed damping trials when **CPU-free CUDA hot loop** is enabled.
- Performs one packed host materialization only after a completed population request for the stable GUI/database/result contract.
- Keeps PPO model, optimizer, rollout tensors and active minibatches on the learner device under the same 80%-default ceiling. PPO losses are transferred once per epoch rather than once per minibatch.
- On PPO CUDA OOM, reduces the current minibatch and retries on CUDA; it does not change the learner to CPU.
- Records VRAM budget, peak allocated/reserved memory, microbatch sizes, OOM retries and CPU-fallback count in run/training provenance.

### Important boundary

The PyQt6 GUI, Python control plane, logging, SQLite, checkpoint files and final CPU-reference validation remain host responsibilities. v6.9 guarantees device residency for the active CUDA-compatible numerical data plane, not for non-CUDA desktop/application services. Physical speedup and sustained RTX 4060 utilization must be measured on the target laptop.

Build-time validation: **7 focused v6.9 tests**, **53 device-resident/audit regression tests**, **47 training/Stage-B/hardware regression tests**, **4 configuration tests**, **5 release-integrity tests**, and **150/150 frozen files verified**. The regression selections overlap.

## Prior release: v6.8 independence and XPU recovery

- CALO Intelligence remains independent from Comparison/Portfolio validation and cross-tab rehydration.
- Mixed NVIDIA+Intel hardware is repaired per accelerator and Intel XPU hardware remains visible when its runtime is unavailable.

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

The bullets in this subsection describe the v6 lineage. They are retained as release history and do
not override the v12 Phase 4 empty-policy, CUDA-preferred/CPU-only, or development-freeze boundary
stated at the top of this file.

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
