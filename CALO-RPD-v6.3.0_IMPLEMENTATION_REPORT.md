# CALO-RPD v6.3.0 — Implementation Report

**Release name:** Training Status and Device Reporting Correctness  
**Release date:** 23 July 2026  
**Scope:** Stage A correctness/reporting upgrade on the v6.2.1 scientific baseline.

## Purpose

v6.3.0 corrects misleading training-allocation and progress information exposed by the CALO Intelligence and Dashboard interfaces. The release deliberately does **not** claim the Stage B GPU-resident CALO environment redesign. Its purpose is to make the application report, without ambiguity, what the user requested, what Safe-80 actually admitted, which physical/runtime device owns each branch, what fraction of the scientific training target has completed, and which parts of the current training stack remain host-side.

## Implemented Stage A upgrades

### 1. Selected routing is separated from recommendation

The training panel now distinguishes four concepts that were previously visually mixed:

1. **Selected rollout routing** — the current user-selected CUDA/XPU/CPU episode-routing percentages.
2. **Share planner equivalent units** — integer helper units used only to derive the selected percentages; these are explicitly not literal process counts.
3. **Recommended routing** — an advisory alternative that is never silently applied.
4. **Safe-80 protected runtime assignment** — the actual admitted primary device and effective protected routing after capability/resource rules.

Selecting CUDA 100% now synchronizes the displayed selected split with the current controls. A stale recommendation such as `CUDA 6 / XPU 2 / CPU 1` can no longer be shown as though it were the active `CUDA 9 / XPU 0 / CPU 0` selection.

The recommendation button is explicitly labelled **Apply recommendation** and is user initiated.

### 2. Misleading worker terminology corrected

The prior label `CPU actor workers` was misleading because the adjacent CUDA/XPU/CPU values represented routing/planner shares rather than literal CPU processes.

v6.3 separates:

- **CPU rollout process cap** — the actual host rollout-process cap used when a CPU lane has work;
- **Share planner (equivalent units)** — nominal CUDA/XPU/CPU units used only to derive percentages.

This prevents an integer such as `9` from simultaneously appearing to mean both “9 CPU processes” and “9 CUDA workers.”

### 3. Truthful execution-scope reporting

The Compute Status/Execution Scope area now explicitly states that:

- CUDA/XPU/CPU percentages route eligible **rollout episodes/transitions**; they are not percentages of total program instructions or guaranteed hardware-utilization percentages.
- Policy inference and PPO tensors use the admitted accelerator where supported.
- The current synthetic CALO curriculum environment remains substantially host/CPU/NumPy based in Stage A.
- The normal GUI policy-training configuration currently supplies no real ORPD development cases (`development_cases=()`), so the accelerator-native FP64 ORPD development evaluator has no development-case workload in such a run.
- Cross-episode ORPD batching becomes relevant only when development cases are explicitly configured by a workflow that supplies them.

This makes a “CUDA 100%” selection truthful: it means 100% of eligible rollout routing is assigned to the CUDA actor lane, **not** that 100% of all CALO training computation is GPU resident.

### 4. One shared protected-routing authority

A shared `protected_rollout_shares(...)` helper is now used by both:

- runtime competitive-training resource binding; and
- GUI protected-routing reporting.

This removes a class of drift where the GUI could describe one protected split while the worker executed another.

Existing protected semantics are preserved, including accelerator-share rebinding rules and prohibition of silent uncontrolled accelerator-to-CPU spillover.

### 5. Target-aware competitive training progress

The competitive coordinator no longer reports normal training progress as a hard-coded `0%` with a vague `epochs [2]` string.

A target-aware progress snapshot now reports, for fixed/cumulative sessions:

- overall completed scientific branch-epochs;
- total target scientific branch-epochs;
- percentage complete;
- branch-specific session progress;
- cumulative epoch progress where exact resume is used;
- runtime device assignment;
- last durable exact-safe epoch;
- next expected exact-safe epoch boundary.

Example for one fresh branch at epoch 2 of a 24-epoch target:

`B01 active · session 2/24 epoch(s) · epoch 2/24 · cuda:0 · last exact safe 0 · next exact safe 10`

The bottom progress bar can therefore show approximately 8% rather than 0%.

For multiple branches, overall progress is calculated on the scientific branch-epoch target, not merely the currently active branch count.

For indefinite training, progress is intentionally indeterminate while cumulative epochs and exact-safe state remain visible.

### 6. Dashboard Training Queue progress improved

The Dashboard Training Queue tab now includes dedicated fields for:

- Scientific epoch progress;
- Exact safe checkpoint;
- total/active/queued/completed branches;
- Safe-80 simultaneous limit;
- global CPU worker budget;
- protected resource assignment.

The Dashboard can therefore show both the scientific target progress and the durable exact-resume boundary without conflating them.

### 7. Runtime device mapping made explicit

The CALO Intelligence panel now shows an actual protected branch assignment preview and, while training, the resource plan returned by the coordinator.

This can distinguish for example:

- OS-visible NVIDIA GPU identity;
- CALO/PyTorch runtime identity such as `cuda:0`;
- requested rollout routing;
- effective protected rollout routing;
- actual branch-to-device assignment.

This is intentionally separate from observed Task Manager utilization, which is runtime telemetry rather than a configuration value.

## Stage B boundary — deliberately not claimed in v6.3

v6.3.0 does **not** implement or claim:

- a fully GPU-resident synthetic CALO environment;
- elimination of all per-step CPU↔GPU round trips;
- device-resident vectorization of every CALO operator/archive/memory/controller path;
- automatic configuration of a real ORPD policy-development suite in the normal GUI;
- high sustained NVIDIA utilization as a guaranteed consequence of selecting CUDA 100%.

Those are Stage B architectural/performance changes and require separate scientific-parity work.

## Scientific behavior intentionally preserved

No intentional changes were made to:

- CALO mathematical operators or feasibility-first logic;
- PPO architecture or policy ABI;
- ORPD equations/objective definitions;
- AC power-flow equations;
- robust scenario semantics;
- exact-resume scientific state;
- Safe-80 admission philosophy;
- governing-policy workflow gates.

The competitive training coordinator received progress/reporting changes and a refactor to share protected-routing semantics, not a new optimization trajectory definition.

## Validation performed in the build runtime

Build-runtime environment boundaries:

- PyQt6: unavailable;
- PYPOWER: unavailable;
- physical CUDA: unavailable (`torch` CPU build / CUDA unavailable);
- physical Intel XPU: unavailable.

Executed validation:

- Python `compileall`: PASS for `calo_bootstrap`, `calo_rpd_studio`, and `tests`.
- Focused Stage-A + v6/v5.9/CALO regression selection: **58 passed**.
- Dedicated Stage-A reporting tests include real tiny competitive-training progress-callback execution.
- Historical v6.2.1 release-identity gate is version-skipped under v6.3 rather than falsely failing the new release.
- Broad silent-handler AST audit: **0** `Exception`/`BaseException`/bare handlers whose only action is `pass`, `continue`, or `return`.
- `tests/unit/test_v56_competitive_training.py`: all **9 test bodies reported passed**, although the external command wrapper did not exit cleanly after the multiprocessing suite in this container; no lingering pytest/CALO processes remained afterward. This is recorded as evidence, not overstated as a clean process-exit PASS.

Physical target-machine checks still required before making performance claims:

- PyQt6 visual validation at the user's actual Windows scaling;
- CUDA compute-engine telemetry validation;
- NVIDIA/Intel hardware utilization and thermal soak;
- PYPOWER scientific suite;
- long-duration training with exact-resume/Safe-80 protection.

## Release conclusion

v6.3.0 Stage A makes training status, routing, device assignment, and scientific progress substantially more trustworthy. The user can now distinguish:

**requested routing → recommended routing → Safe-80 protected routing → actual runtime device assignment → observed hardware utilization**.

It also fixes the user-visible 24-epoch/`0%` inconsistency by reporting real target-aware competitive-training progress.

The release should be treated as the correct baseline for the later Stage B GPU-residency/performance redesign, not as evidence that the full CALO environment is already GPU resident.

## Final release-integrity snapshot

- v6.3 software freeze: **135 / 135 verified**, 0 missing, 0 changed.
- Freeze ID: `calo_v630_software_release`.
- Canonical freeze SHA-256: `19438e3c54fd01e7cd8671acb22a4e143e02a6c1a360241efd9b5c8b01561d6d`.
- Freeze-file SHA-256: `80d97d6d142c7bb5b10781d35f1ec6c1dd3491ef81545c68857e0dc4475be2b8`.
- Root package manifest target: **431** packaged non-cache files, excluding `MANIFEST.sha256` itself.

The delivered ZIP is independently extracted and reverified before release; its final archive SHA-256 is reported with the downloadable artifact rather than embedded here to avoid circular archive-content hashing.
