# CALO-RPD Studio v5.0.0 — Implementation Report

## Scope

v5.0 is the continuation/provenance release planned after v4.1. It preserves the validated ORPD formulation, equal requested-function-evaluation (FE) accounting, immutable experiment-policy binding, Policy Qualification, workspace restoration, HPEM/dual-lane/precision CALO architecture, and independent validation rules. The release adds long-lived policy evolution and experiment evolution without destructively overwriting prior scientific evidence.

It does **not** claim universal CALO superiority, a fully device-resident CALO control loop, or a bit-identical policy-training/runtime transition implementation.

## 1. Policy evolution and exact training resume

Policy training now supports three continuation semantics:

- **Cumulative target**: continue until a specified total epoch count is reached.
- **Additional epochs**: add a specified number of epochs to the saved state.
- **Indefinite**: continue until Safe Stop; the exact state remains resumable across later sessions.

An exact training checkpoint stores actor/critic network state, optimizer state, cumulative epoch, training history/curriculum information, Python/NumPy/Torch RNG states, CUDA RNG states when available, training configuration and schema metadata. Exact resume rejects silent changes to scientific/training hyperparameters; intentional changes require a documented continue/fine-tune phase rather than being mislabeled exact continuation.

Checkpoint writes are atomic and SHA-verified. Deployable policy snapshots are immutable and immediately usable for Policy Qualification or runtime. Repeated stops at the same epoch never overwrite an earlier immutable snapshot.

## 2. Policy lineages

The database now tracks policy families/lineages and their checkpoints. A lineage records parent/fork relationships and distinguishes:

- **latest checkpoint** — the most recently trained artifact;
- **best-qualified checkpoint** — the checkpoint selected by recorded scientific qualification.

More training epochs never automatically imply a better policy. Existing experiments remain bound to the exact checkpoint SHA they originally used even when the lineage later continues or forks.

The CALO Intelligence Policy Center exposes lineage, cumulative epoch, role, scientific grade/status, exact-resume selection, continue/fine-tune and fork operations.

Automatic execution of expensive qualification campaigns at every configured training interval is not implemented in v5.0; the interval is advisory and qualification remains an explicit separately budgeted action.

## 3. Experiment revisions and additional independent runs

An experiment can evolve through revisions without changing its original scientific identity. Increasing the total independent-run target preserves already completed runs and schedules only new deterministic paired run indices/seeds.

Example:

`30 runs -> 50 runs`

preserves runs 1–30 and adds runs 31–50 under the same fixed scientific formulation and immutable experiment policy binding.

Each revision records the run target, FE horizon, protocol, execution strategy, publication eligibility, parent/source revision and status.

## 4. FE-horizon evolution

v5.0 distinguishes two scientifically different higher-horizon strategies.

### Exact segmented continuation

CALO supports complete optimizer-state checkpoints containing the current population/evaluations, persistent personal memory, HPEM, archives, success/context memory, variable intelligence, epsilon state, dual-lane/precision state, RNG state, counters, diagnostics and continuation provenance.

A continuation can explicitly select a preserved **source FE horizon**. The source checkpoint is loaded before any initial population is created/evaluated, so no hidden fresh-population evaluations occur. Only newly requested evaluations are consumed. Output checkpoints are revision-scoped so a new branch cannot overwrite the checkpoint of an older horizon/revision.
Exact-run checkpoints also bind a **scientific problem fingerprint** covering the case artifact/checksum, decision dimension, decoder/formulation manifest, problem configuration and scenario descriptors. A mismatched formulation is rejected rather than silently resumed under a changed scientific problem.

A segmented `5,000 -> 10,000 FE` trajectory is explicitly labeled segmented and is **not** represented as identical to a run planned for 10,000 FE from FE=0; adaptive schedules experienced the original horizon during the first segment.

### Paired recompute from original seed

For publication-safe from-start comparison at a new horizon, all algorithms can be rerun from their original paired seeds using the larger FE target. This is scientifically distinct from exact continuation and is stored as such. Unsupported baseline optimizers are never falsely labeled exact-resumed.

## 5. Publication-safety protocols

Three extension protocols are implemented:

- **All paired** — all selected algorithms and paired runs are extended/recomputed consistently; eligible for primary evidence when complete.
- **Predeclared deterministic subset** — a subset fixed before outcome inspection; eligibility depends on the recorded protocol.
- **Manual/post-hoc exploratory** — user-selected runs after seeing outcomes; always marked exploratory and excluded from unbiased primary claims.

A later exploratory long-horizon branch does not redefine the primary publication horizon. A later paired primary revision can branch from the correct preserved source.

## 6. Evidence preservation and horizon-aware analysis

Before a completed run head is replaced by a larger-horizon result, its evidence is snapshotted. v5 stores:

- run segments;
- FE-horizon snapshots;
- horizon-specific validations;
- experiment revisions;
- revision/source-horizon provenance;
- scientific fingerprints and trace references.

Results Explorer, Statistical Analysis and Portfolio Export are horizon-aware. They do not silently mix 5k/10k evidence. Publication-eligible incomplete horizons are blocked from being treated as complete primary evidence. Export directories preserve horizon/revision identity rather than overwriting older artifacts.

## 7. Workspace restoration

The v4.1 `ExperimentWorkspaceRestorer` remains authoritative. Reopening an experiment restores stored scientific configuration, algorithm/CALO intelligence selections, immutable policy binding, workflow access, historical Live Optimization plots, selected run/plot view and stored results context. v5 revision/horizon metadata is added on top of that restoration model.

## 8. Database additions

v5 adds structured persistence for:

- `policy_lineages`;
- `policy_checkpoints`;
- `experiment_revisions`;
- `run_segments`;
- `run_horizon_snapshots`;
- horizon-aware validation metadata.

Deletion/cleanup handles v5 child records and preserved trace/checkpoint artifacts while protecting referenced policy provenance.

## 9. Scientific safeguards retained

- Equal requested FE budgets remain authoritative.
- Exact cache/dedup reuse never grants extra search attempts.
- Repeated benchmark runs remain independent; runtime memories do not leak across runs.
- Historical learning remains explicit and separately controlled.
- Experiment-bound policy SHA is immutable.
- No old horizon is silently overwritten or mixed into another horizon.
- IEEE 118/300 holdout discipline remains recommended after IEEE 30/57 development/freeze.
- No policy becomes “best” merely because it has the most epochs.

## 10. Deliberately unresolved work

The following are explicitly **not** claimed solved in v5.0:

1. Full Torch/CUDA/XPU-resident CALO cognitive/control execution.
2. One bit-identical shared transition implementation for PPO training and complete runtime CALO.
3. Automatic asynchronous Policy Qualification/promotion every configured training interval.
4. Exact optimizer-state continuation for every baseline algorithm; CALO exact continuation plus paired recompute-from-seed is implemented instead.
5. Proof that the bundled/legacy policy or CALO is universally superior.

These limitations are recorded in `calo_rpd_studio/algorithms/calo/v5_disputes.py` and the v5 findings closure matrix.
