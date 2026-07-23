# CALO-RPD v6.1.0 Implementation Report

**Release:** 6.1.0 — *Protected Training Queue and Capability-Aware Scheduling*  
**Date:** 23 July 2026  
**Baseline:** CALO-RPD v6.0.0-alpha4 / native v5.9 scientific policy ABI  
**Scope:** Requested v6 beta1–beta4 architecture upgrades only; no new optimizer-superiority claim.

## 1. Release objective

v6.1 converts the v6.0 Safe-80/policy-first foundation into a protected policy-training execution architecture. The release implements four coordinated upgrades:

1. **Global Training Exclusive Lock** across the application.
2. **Scientific branch count separated from safe simultaneous concurrency**, with a protected queue scheduler.
3. **One global CPU worker budget** and removal of uncontrolled accelerator-to-CPU branch spillover.
4. **XPU capability-aware scheduling** that distinguishes direct full-branch runtimes from sidecar actor/evaluator assistance.

The design rule is now explicit:

> **Total branch count is a scientific-diversity requirement; simultaneous branch concurrency is a protected system-resource decision.**

## 2. Beta1 — Global Training Exclusive Lock

A new application-level policy-training state is exposed through `AppState` and consumed by `MainWindow`/`WorkflowManager`.

When policy training begins:

- every scientific/configuration workspace is disabled;
- Dashboard remains available as a read-only system/training monitor;
- Open/Save mutable configuration actions are disabled;
- compute-topology refresh is blocked so the hardware identity cannot change underneath an active training session;
- application close requests an exact Safe Stop before shutdown;
- the global status-bar cancel/Safe-Stop path remains available even though the CALO Intelligence configuration page itself is disabled.

The lock is released on completed, cancelled/Safe-Stopped, failed, or launch-failed training paths.

## 3. Beta2 — Total branch count vs simultaneous concurrency

`TrainingConfig` now separates:

- `parallel_runs` — total independent scientific branches;
- `parallel_concurrency` — maximum branches admitted simultaneously;
- `branch_queue_quantum_epochs` — exact-resume rotation quantum for indefinite sessions.

### Fixed/cumulative sessions

Branches run in protected waves. Each admitted branch runs to its declared session target in one process lease, avoiding repeated accelerator-context creation. When a slot is released, the next queued scientific branch assigned to that slot is admitted.

### Indefinite sessions

A permanent queue cannot allow the first branches to monopolize finite accelerator slots. Indefinite branches therefore rotate through exact-resume lease boundaries. The minimum scientific safe cadence remains 10 epochs, and non-multiple values are rounded upward to a 10-epoch boundary.

Every branch retains its own:

- policy weights;
- optimizer state;
- Python/NumPy/Torch accelerator RNG state where supported;
- curriculum state;
- branch champion state;
- exact resume checkpoint.

No parameter averaging is introduced.

### Safe Stop with queued branches

Safe Stop establishes an exact safe state for the complete scientific branch set. Never-started queued branches are initialized under cancellation only long enough to materialize their exact starting safe state; already-started queued branches retain their existing authenticated exact safe state.

## 4. Beta3 — Global CPU worker budget

The Safe-80 profile and branch planner now share a consistent minimum host budget model:

- **2 host support worker-equivalents per active branch** for branch/accelerator/IPC support;
- **at least 1 CPU rollout worker per active branch**;
- one global protected CPU budget calculated from the machine topology and Safe-80 envelope.

The rollout portion of that global budget is divided across simultaneous branch slots. It is **not multiplied by branch count**.

Child branches also cap common native thread pools (`OMP`, `MKL`, `OpenBLAS`, `NumExpr`, PyTorch) to prevent hidden thread multiplication inside each process.

### No uncontrolled CPU fallback

If accelerator capacity is exhausted or the selected accelerator is unavailable, competitive scheduling does not silently create heavy CPU branches. The user may deliberately select CPU when the protected topology permits it, but accelerator pressure is handled by reducing concurrency/queueing or failing closed.

## 5. Beta4 — XPU capability-aware scheduling

The authoritative `ComputeTopology` classifies compute devices by role rather than treating every detected GPU as interchangeable.

### Direct CUDA

A direct CUDA runtime is eligible for full-branch scheduling only after the required runtime capability checks, including the FP64 smoke used by the scientific ORPD path.

### Direct Intel XPU

A direct `torch.xpu` runtime may be admitted as a full branch only when the direct runtime exists and the deterministic FP64 runtime smoke passes.

### XPU sidecar

An XPU sidecar remains an **auxiliary actor/evaluator runtime**, not an independent full PPO branch. It may assist an admitted branch when heterogeneous XPU share is requested, but it does not increase the full-branch concurrency count.

Competitive heterogeneous branches use strict resource binding: a requested unavailable accelerator lane is not silently redistributed to CPU. Where an admitted primary accelerator can safely absorb a missing sibling-accelerator share, the share is redirected to the already-admitted primary accelerator instead of generating hidden host load.

## 6. Authoritative protected resource plan

New module:

`calo_rpd_studio/compute/training_resources.py`

It produces a `TrainingResourcePlan` containing:

- total scientific branches;
- simultaneous protected branch limit;
- queued branch count;
- global CPU worker budget;
- host-support reserve;
- total CPU rollout-worker budget;
- per-slot primary device;
- per-slot CPU rollout budget;
- optional auxiliary XPU runtime;
- compute-topology fingerprint;
- Safe-80 protection-profile fingerprint.

The GUI performs a preflight plan using the Dashboard topology/profile. Competitive training then recalculates the live plan at launch and freezes the actual topology/profile fingerprints into provenance.

## 7. Dashboard integration

Dashboard now displays live protected-training state including:

- total scientific branches;
- safe simultaneous limit;
- active branches;
- queued branches;
- completed branches;
- global CPU worker budget;
- protected resource assignments;
- CPU/XPU/GPU runtime mapping and capability status.

Hardware refresh is disabled during the Global Training Exclusive Lock.

## 8. Exact-resume continuity additions

Where the installed runtime exposes Intel XPU RNG APIs, XPU RNG state is now persisted/restored alongside existing Python, NumPy, Torch CPU and CUDA exact-resume state.

Execution placement fields remain scientifically mutable at session boundaries because hardware may change; exact optimizer/RNG state is restored, but bit-identical numerical replay across different hardware backends is **not** claimed.

## 9. Post-generation corrections discovered during audit

The implementation audit identified and corrected additional integration issues before packaging:

- Dashboard Safe-80 CPU branch capacity initially used a two-worker minimum while the scheduler reserved two support equivalents **plus** one rollout worker; both now share one three-equivalent minimum model.
- Per-device accelerator headroom is checked when selecting a slot, preventing aggregate safe slot counts from selecting a busy first device merely because another accelerator supplied the aggregate headroom.
- Explicit missing/unavailable accelerator requests fail closed rather than falling back to CPU.
- Strict heterogeneous lane binding prevents hidden unavailable-CUDA/XPU-to-CPU redistribution.
- Current release/bootstrap fallback identity was updated to 6.1.0.
- Duplicate release-scope metadata for accelerator-to-CPU spillover was removed.
- Indefinite one-slot/two-branch queue rotation was regression-tested through the first common 10-epoch Safe-Stop boundary.

## 10. Validation evidence

Validation executed in the build environment:

- `compileall` over `calo_bootstrap`, `calo_rpd_studio`, and `tests`: **PASS**.
- v6.1 beta architecture suite: **12 passed**.
- v6.0 alpha architecture regression: **8 passed**.
- v5.9 scientific closure regression: **14 passed**.
- v5.8 audit closure regression: **10 passed**.
- CALO core suites: **11 passed**.
- selected competitive-training transaction/exact-resume/Safe-Stop regressions: **3 passed**.
- broader scientific/unit partition: **101 passed**.
- additional release/workflow/export/worker partition: **78 passed, 1 skipped** (PYPOWER-dependent bundled IEEE formulation test skipped because PYPOWER is unavailable).
- workflow/startup partition: **1 passed, 1 skipped** (PyQt6-dependent test skipped because PyQt6 is unavailable).

These runs are not presented as one deduplicated full-suite count because several targeted groups intentionally overlap historical regression coverage.

## 11. Deliberate v6.1 boundaries

v6.1 does **not** claim completion of the later thermal/power RC architecture.

Still pending for later v6 work:

- continuous Green/Amber/Red thermal/power governor;
- live temperature/power-based dynamic throttling;
- staged admission driven by measured thermal headroom;
- physical target-laptop CUDA/XPU long-duration thermal soak;
- XPU sidecar certification as a full independent PPO branch;
- complete PyQt6 GUI suite and physical accelerator validation in the target environment.

The Safe-80 engine is a protected resource-allocation/admission envelope, not a claim that software can replace firmware/hardware thermal protection.

## 12. Release assessment

v6.1 implements the requested beta1–beta4 architecture while preserving the native v5.9 scientific policy ABI and the v5.9 CALO/ORPD scientific logic. The release should be treated as a **research-validation beta architecture baseline**, not yet the final thermally qualified publication freeze.


## 13. Final source-tree release integrity

- v6.1 software freeze: **128 / 128 files verified, 0 missing, 0 changed**.
- root `MANIFEST.sha256`: **403 packaged files listed**.
- v6.1 release-integrity suite: **5 passed**.

Final ZIP extraction/hash verification is performed after archive creation and reported with the delivered artifact.
