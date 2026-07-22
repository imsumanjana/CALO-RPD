# CALO-RPD v5.6 Competitive Multi-Branch Policy Training

## Scientific model

CALO v5.6 treats each parallel PPO branch as an independent learning trajectory. Branch network
parameters, optimizer moments, RNG states and curriculum states are **never arithmetically averaged**.
Parallelism is used to explore several training trajectories, after which scientifically supported
champions compete to become the logical Base Model.

## Training duration and start modes

- **Cumulative:** a fixed number of epochs is added to the exact saved branch epoch in the current
  session.
- **Infinite:** training has no terminal epoch and runs until Safe Stop.
- **Exact Resume:** every branch restores its model, optimizer, RNG, curriculum/history and epoch.
  The restored branches may then continue in either Cumulative or Infinite duration mode.
- **Base-Guided Fork:** fresh optimizer/RNG trajectories start from a selected Base Model's deployable
  weights. This is not labelled Exact Resume.

## Branch seeds

The user may combine same-seed, incremental (`seed+n`), decremental (`seed-n`) and explicit custom
seed branches. Same-seed branches are not artificially perturbed; deterministic execution may produce
identical trajectories and that is accepted as valid reproducibility evidence.

## Working state, Branch Champion and Base Model

Each branch owns an exact resumable **working state**, which always advances when a session completes.
Separately, a **Branch Champion** is retained only when fixed validation evidence supports it. A frozen
Base Model acts as the promotion threshold during resumed/forked training. At session completion the
previous Base and eligible Branch Champions are compared; the Base changes only if a superior candidate
exists.

The Base Model shown in the Policy Library is a logical policy entry backed by an immutable artifact and
SHA-256. Internal branch resume checkpoints are not exposed as separate policies.

## Champion comparison

Every epoch receives a cheap Tier-1 screen from already produced training evidence. A fixed multi-metric
Tier-2 validation is run when the screen improves, at periodic deep-validation boundaries, at the initial
state and at cumulative-session completion. This avoids running a full validation bundle unnecessarily on
every non-promising epoch while still continuously comparing progress.

Tier-2 evidence includes feasibility, final feasible objective statistics, convergence AUC, constraint
violation, evaluations/steps to feasibility, stability, validation return and inference overhead.
Mandatory validity/feasibility gates and critical-metric Pareto checks precede broader multi-metric
comparison. Runtime cannot compensate for a material scientific regression. This Branch Champion
comparison is **not** formal Policy Qualification; Candidate-vs-Reference-vs-No-AI qualification remains
separate.

## Safe Stop and temporary snapshots

No permanent epoch-by-epoch policy snapshots are produced. During an active training session, each branch
keeps a bounded rolling set of exact-state files in a configurable local scratch directory at 10-epoch safe
boundaries. Faster branches may remain only a bounded number of epochs ahead of the common committed
boundary.

On Safe Stop:

1. the parent signals all child branches through the same multiprocessing spawn context;
2. workers stop without committing later partial trajectories;
3. the coordinator selects the lowest common available previous 10-epoch boundary;
4. one permanent exact resume checkpoint is committed per branch at that common epoch;
5. Branch Champions are compared with the previous Base;
6. the Base is promoted only when justified; and
7. the temporary session directory is deleted.

A completed cumulative session writes exact branch resume state at the requested terminal epoch without
requiring the terminal epoch itself to be a multiple of ten.

## Reproducibility rules

- Exact Resume restores RNG state; it does not reseed a branch from its original seed.
- Changing Cumulative vs Infinite duration after exact restore changes only the stopping horizon, not the
  restored training state.
- Base-Guided Fork uses fresh optimizer/RNG state and is recorded as a new trajectory.
- Experiment-bound policy artifacts remain immutable by SHA-256 even when a logical Base improves later.
- Policy ABI strings (`calo-v4.1`, `calo-state-v4.1-32`, action schema) remain unchanged because v5.6 changes
  training orchestration, not that checkpoint ABI.
