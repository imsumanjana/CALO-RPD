# CALO-RPD v6.4.0 Implementation Report

## Release identity

- Version: **6.4.0**
- Release name: **Stage-B Device-Resident Policy Training**
- Date: **23 July 2026**
- Baseline: CALO-RPD v6.3.0 Stage-A truthfulness/progress release

## Purpose

v6.4 implements the requested **Stage B** training architecture upgrade. The objective is to increase useful CUDA/direct-XPU work without silently changing CALO's scientific controller semantics.

The implementation deliberately uses a hybrid scientific architecture:

1. policy inference and PPO remain accelerator-capable;
2. deterministic synthetic curriculum population objective/constraint evaluation is now vectorized on persistent FP64 PyTorch device tensors;
3. simultaneous synthetic episode requests are cross-episode microbatched;
4. real ORPD policy-development cases are now configurable in the normal GUI and use the exact declared experiment formulation;
5. the stochastic CALO controller/archive/memory/candidate-orchestration transition remains the trusted host/reference authority.

This is a scientifically safer Stage-B step than rewriting all stochastic CALO logic onto GPU in one release and risking trajectory drift.

## 1. Device-resident synthetic curriculum kernel

Added:

`calo_rpd_studio/algorithms/calo/device_resident_synthetic.py`

### DeviceResidentCurriculumProblem

- Wraps an already-generated `CurriculumProblem`.
- Consumes **no additional RNG**, preserving original task generation and subsequent CALO random streams.
- Copies fixed task tensors once to the admitted device:
  - shift vector;
  - orthogonal rotation;
  - constraint centres/normals;
  - dimension masks;
  - mixed-variable discrete lattices.
- Evaluates complete populations in FP64 PyTorch.
- Supports CUDA, direct XPU, and CPU for dependency-light validation.

### Scientific parity gate

Before an accelerator-backed synthetic task is trusted, Stage B compares the accelerator output with the original NumPy reference for:

- objective;
- total constraint violation;
- feasibility classification;
- all named constraint components.

Default relative/scale-aware tolerance: `1e-9`.

Failure is **fail-closed**: the training request raises rather than silently accepting a scientifically different kernel.

Periodic parity rechecks are supported and enabled by default.

## 2. Cross-episode synthetic microbatching

Added `SyntheticCrossEpisodeBatchBroker`.

Compatible synthetic population requests from simultaneous rollout episodes targeting the same device are merged into one padded heterogeneous FP64 batch.

This reduces the old pattern of many tiny per-environment accelerator calls and gives CUDA/XPU larger useful batches.

The broker records:

- merged batch count;
- candidate count;
- request count;
- maximum observed merged batch;
- mean candidates per batch.

These metrics are carried into training progress/provenance.

## 3. Protected host-side controller concurrency

The stochastic CALO environment/controller step remains host-side in Stage B.

To prevent Stage B from reintroducing the historical laptop CPU-oversubscription problem, concurrent host controller stepping is capped by the protected per-branch rollout-worker budget:

`min(number_of_environments, protected_rollout_workers)`.

The accelerator broker handles the deterministic population evaluation merge; host controller concurrency does not expand without bound simply because more episodes exist.

## 4. Real ORPD development suite in the normal GUI

The previous normal GUI training path hardcoded an empty development-case tuple. v6.4 removes that limitation.

CALO Intelligence now exposes:

- Enable real ORPD development suite;
- Development cases;
- Development formulation file.

Default development cases:

- `case30`
- `case57`

Default formulation:

`calo_rpd_studio/data/examples/policy_development_active_loss.yaml`

Protected held-out/final systems `case118` and `case300` are rejected by the normal development-suite selector.

## 5. Exact real-ORPD formulation propagation

A scientific inconsistency was discovered during Stage-B implementation: the heterogeneous actor path previously constructed an accelerated ORPD problem from a case using defaults.

v6.4 corrects this.

Real ORPD policy rollouts now load the declared `ExperimentConfig` and preserve its exact:

- `ObjectiveConfig`;
- `ORPDVariableConfig`;
- `RobustObjectiveConfig`;
- `PowerFlowOptions`;
- constraint tolerances;
- scenario construction.

The selected development case replaces only `case_name`; the rest of the declared scientific formulation remains fixed.

When accelerated ORPD rollouts are enabled, `AcceleratedORPDProblem` is constructed from this exact formulation and scenario bundle.

## 6. Stage-B GUI controls and truthful execution scope

Added controls for:

- Stage-B synthetic accelerator kernel;
- real ORPD development suite;
- development cases;
- development formulation.

Stage-A reporting guarantees remain:

- selected rollout routing is separate from recommendation;
- Safe-80 protected routing is shown separately;
- runtime device mapping is explicit;
- `CUDA 100%` is not described as 100% of all code executing on CUDA.

The execution-scope text now explains that:

- policy/PPO uses the accelerator where supported;
- deterministic synthetic population evaluation can be device-resident in Stage B;
- stochastic controller/archive/memory semantics remain host/reference;
- real ORPD development evaluation can use the FP64 accelerator path when configured.

## 7. Stage-B validation CLI

Added:

`python -m calo_rpd_studio.scripts.validate_stage_b_synthetic`

Installed entry point:

`calo-rpd-stage-b-validate`

The validator measures:

- startup parity;
- maximum numerical error;
- candidate throughput;
- microbatch statistics;
- whether a physical accelerator was actually used.

A CPU run is explicitly labelled as logic validation only and cannot qualify physical CUDA/XPU performance.

## 8. CLI training support

`train_calo.py` now accepts an explicit development experiment configuration path and carries it with development cases into policy training.

## 9. Frozen scientific scope

The v6.4 freeze explicitly includes:

- `device_resident_synthetic.py`;
- `heterogeneous_training.py`;
- `training.py`;
- CALO Intelligence panel;
- Stage-B validator;
- bundled policy development formulation.

Frozen Stage-B declarations include:

- device-resident synthetic evaluation: **true**;
- cross-episode synthetic microbatching: **true**;
- fail-closed startup parity: **true**;
- real ORPD development suite configurable: **true**;
- full stochastic CALO controller GPU-resident: **false**.

## 10. Validation executed in the build environment

- Python `compileall`: **PASS**.
- Focused Stage-B + Stage-A + v6/v5.9/CALO regression group: **66 passed**.
- Heterogeneous training / historical pretraining / broker / prior audit closure group: **38 passed**.
- Historical competitive-training suite: **9/9 passed when executed in two bounded groups**.
- Static audit: **0** broad `Exception` / `BaseException` / bare handlers whose only action is `pass`, `continue`, or `return`.
- Dependency-light Stage-B CPU validator:
  - parity verified: **true**;
  - maximum startup parity error: approximately **4.44e-15**;
  - broker merged up to **96 candidates** in the executed validation;
  - physical accelerator qualification: **false** because the build runtime is CPU-only.

A larger all-unit invocation reached late in the suite without visible failures before the external command time limit; it is not represented as a complete-suite PASS.

## 11. Environment boundaries

Build runtime:

- PyQt6: unavailable;
- PYPOWER: unavailable;
- physical CUDA: unavailable;
- physical XPU: unavailable;
- PyTorch: CPU build.

Therefore this release does not claim measured RTX 4060 utilization improvement on the target laptop. The new kernels and development path must be validated on the intended Windows CUDA/XPU machine.

## 12. Expected target-machine behavior

Compared with v6.3, an admitted accelerator synthetic rollout can now perform substantially more useful device work because population objective/constraint evaluation is batched on the accelerator instead of remaining entirely NumPy-side.

After curriculum epoch 20 by default, configured real ORPD development cases can also exercise the heavier FP64 accelerator-native ORPD evaluator.

Actual utilization will still depend on:

- population size;
- number of simultaneous rollout episodes;
- batch merge rate;
- controller overhead;
- power-flow workload;
- laptop power/thermal limits.

No fixed utilization percentage is guaranteed.

## Final implementation classification

Stage B requested scope: **IMPLEMENTED**.

Full all-CALO-logic GPU residency: **NOT CLAIMED / NOT IMPLEMENTED**.

Physical target-laptop performance qualification: **PENDING TARGET HARDWARE**.
