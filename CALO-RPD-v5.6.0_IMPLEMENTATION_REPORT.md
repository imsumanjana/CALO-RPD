# CALO-RPD Studio v5.6.0 — Implementation Report

## Scope

v5.6 implements the requested competitive policy-training architecture on top of the supplied recent repository. The release focuses on policy training/orchestration and preserves the existing ORPD evaluator, FE fairness rules, experiment continuation architecture, fail-closed policy gating, and policy ABI compatibility.

## Training model

A training job consists of independent PPO branches. User-controlled branch seeds may be the same base seed, incremental `seed+n`, decremental `seed-n`, or explicit custom values. Same-seed branches are not artificially perturbed.

Three user workflows are supported:

- **Cumulative:** fixed epoch count for the current session, accumulated from exact saved branch state.
- **Infinite:** no epoch target; continues until Safe Stop.
- **Exact Resume:** restores every saved branch exactly and then continues in Cumulative or Infinite mode.

**Base-Guided Fork** is a separate branch-start strategy. It copies deployable Base knowledge but creates fresh optimizer/RNG trajectories and is never labelled exact continuation.

## Working state vs champion/base

Each branch owns:

- exact working model/optimizer/RNG/curriculum/history state;
- a best-so-far Branch Champion selected from fixed validation evidence;
- its immutable seed/branch identity and provenance.

Working state always advances after a completed session. Champion state changes only when a candidate is superior. The previous Base is a frozen promotion threshold. At session end, the previous Base and eligible Branch Champions are compared; only a superior candidate becomes the new logical Base.

No independent branch neural parameters are averaged.

## Multi-metric champion selection

The comparator uses validity/feasibility gates, critical Pareto checks, and broad metrics including final feasible objective statistics, feasible rate, convergence AUC, constraint violation, steps to feasibility, stability/IQR, validation return, and inference overhead. Runtime is secondary and cannot compensate for a material scientific regression.

Every epoch is screened cheaply from existing training evidence. Full multi-metric validation is run when the screen improves, at periodic deep-validation boundaries, at the initial state, and at cumulative-session completion. Formal Policy Qualification remains separate.

## Safe Stop / storage

Permanent intermediate epoch snapshots are not generated. Exact branch states are written to a bounded rolling temporary disk window at 10-epoch boundaries in a configurable scratch location. Safe Stop commits the lowest common available previous 10th epoch across branches, discards later work, writes permanent trusted branch resume checkpoints, performs Base selection, and removes session scratch storage.

## GUI / policy library

The CALO Intelligence workspace exposes:

- Cumulative vs Infinite duration;
- seed-plan branch counts and custom seeds;
- New, Exact Resume, and Base-Guided Fork start semantics;
- fast scratch storage path;
- Base-guided continue and separate Exact resume branches actions;
- one logical Base policy row with branch count/seed-plan metadata rather than exposing branch resume files as policies.

Exact resume of a selected current Base asks whether the restored branches should continue as Cumulative or Infinite. Cumulative resume asks for the new session epoch count.

## Compatibility

The application release is 5.6.0. The existing `calo-v4.1` runtime/policy state/action ABI identifiers are intentionally retained because the neural policy interface did not change. Renaming those identifiers solely to match the application version would unnecessarily invalidate compatible checkpoints.
