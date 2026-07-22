# CALO-RPD Studio v5.6.0 — Release Validation

**Release date:** 2026-07-22  
**Release purpose:** competitive multi-branch policy evolution, exact branch continuation, low-RAM Safe Stop, and non-destructive Base Model promotion.

## Scientific architecture implemented

v5.6 replaces the previous parallel-policy merge concept with independent competitive PPO branches.

- Independent branches are **never neural-weight averaged**.
- Branch seeds support same, incremental, decremental, and custom user-selected mixtures.
- Cumulative sessions add a fixed user-selected epoch count to the saved lifetime epoch.
- Infinite sessions have no target epoch and run until Safe Stop.
- Exact Resume restores each branch's network, optimizer, RNG, curriculum/history, branch identity, seed identity, and saved epoch.
- Base-Guided Fork is explicitly distinct from Exact Resume: it starts fresh optimizer/RNG trajectories from selected Base Model knowledge.
- Every branch separates exact resumable working state from its best scientifically evaluated Branch Champion.
- The logical Base Model is selected competitively from the previous Base and eligible Branch Champions; no parameter averaging is used.
- Formal Policy Qualification remains separate from internal Branch Champion/Base selection.

## Champion comparison

Branch promotion uses a hierarchical multi-metric comparator rather than PPO loss or a single reward:

1. mandatory policy validity and feasibility safeguards;
2. critical-metric Pareto comparison;
3. broad evidence across feasible objective statistics, convergence AUC, constraint violation, feasibility rate, time/steps to feasibility, stability, held-out validation return, and inference overhead;
4. a guard preventing small efficiency gains from compensating for a material final-feasible-objective regression.

Every epoch receives a cheap training-evidence screen. Full fixed multi-metric validation is triggered by promising screen improvement, periodic deep-validation boundaries, initial state, and cumulative-session completion. This avoids imposing a full validation suite on every non-promising epoch in extremely long campaigns.

## Safe Stop and exact checkpointing

- No permanent epoch-by-epoch snapshots are produced.
- During an active session, each branch keeps a bounded rolling window of exact-state files in a local scratch directory at 10-epoch safe boundaries.
- Safe Stop signals every child branch, selects the lowest common available previous 10th epoch, discards later work, and commits one permanent exact resume checkpoint per branch at that common epoch.
- Temporary session storage is deleted after permanent commit/base selection.
- Normal cumulative completion commits each branch at the requested exact terminal epoch.
- Branch resume state always advances after a completed session; Branch Champion/Base state advances only when the comparator supports improvement.

## Parallel-training audit corrections

The v5.6 architecture closes the main audited C/D defects:

- multiprocessing synchronization objects and child processes use one `spawn` context;
- exact branch resume paths are authoritative and actually used;
- parent cancellation sets the shared child cancellation event;
- coordinator waits for all branches rather than exiting when any one finishes;
- fatal messages and process exit codes are checked before Base selection;
- requested/started/successful/failed contributor counts are explicit;
- independent PPO network averaging was removed;
- no incoherent merged optimizer/RNG/curriculum state is created;
- coordinator returns training/champion/base-selection history;
- heterogeneous CUDA/XPU/CPU configuration is retained inside competitive branches;
- legacy curriculum conversion is schema/version-driven;
- caller global Python/NumPy/Torch/CUDA RNG state is restored after standalone training;
- deployable artifacts are immutable and logical aliases are separate;
- Safe Stop no longer overwrites a full policy with a reduced/incompatible payload.

See `FINDINGS_CLOSURE_v5.6.0.csv` for itemized status.

## Validation executed in this build environment

### Targeted v5.6 / policy-continuation partition

`40 passed`

Included:

- competitive branch seed planning;
- multi-metric scientific champion comparison;
- separate branch exact resume files and single Base selection;
- exact multi-branch resume with branch seed preservation;
- Base-Guided Fork parent immutability;
- Safe Stop exact branch commit and scratch cleanup;
- legacy curriculum conversion;
- caller global RNG restoration;
- v5 experiment/policy continuation tests;
- heterogeneous policy-training tests;
- policy registry/gating tests;
- SHA-based policy-cache and broker-timeout hardening tests;
- v5.6 release-integrity/freeze tests.

### Partitioned dependency-light unit tests

`184 passed`

The unit suite was executed in isolated partitions to avoid cross-test runtime/thread accumulation in the constrained build harness. One PYPOWER-dependent case118 test was deselected because PYPOWER is not installed. Two Qt workflow-restoration tests could not be collected because PyQt6 is not installed.

### Integration and regression

`9 passed`

### Compilation

`python -m compileall -q calo_bootstrap calo_rpd_studio tests` — **PASS**

### Current software freeze

- manifest: `calo_rpd_studio/data/frozen/calo_v560_freeze.json`
- frozen files: **99**
- verification: **PASS**
- no default neural policy is bundled or implied
- policy ABI identifiers remain `calo-v4.1` / `calo-state-v4.1-32` / existing action schema intentionally for checkpoint compatibility

## Environment-limited checks

The following were not falsely marked PASS in this build container:

- **PYPOWER scientific IEEE suite:** not run because PYPOWER is unavailable.
- **PyQt6 GUI suite/workflow restoration:** not run because PyQt6 is unavailable.
- **Ruff:** tool unavailable in this runtime.
- **Physical CUDA/XPU throughput:** no compatible accelerator hardware/runtime is exposed here.

The existing scientific solver/evaluator code outside the policy-training upgrade was not intentionally changed by v5.6. Final publication use still requires the normal independent IEEE/PYPOWER validation and paired Policy Qualification on the user's fully provisioned workstation.

## Claim discipline

v5.6 does **not** claim that CALO or any policy is universally superior. Branch Champion selection is an internal training-selection mechanism, not publication qualification. A policy must still pass the configured Candidate-vs-Reference-vs-No-AI qualification protocol and experiment-level independent validation before scientific promotion/publication use.
