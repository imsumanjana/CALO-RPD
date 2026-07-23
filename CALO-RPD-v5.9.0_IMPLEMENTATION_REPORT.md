# CALO-RPD Studio v5.9.0 — Implementation Report

**Release name:** Scientific Closure and Exact Controller Parity  
**Release date:** 23 July 2026  
**Source basis:** CALO-RPD v5.8.0 plus the v5.8 Deep Scientific Re-Audit findings  
**Release objective:** correct the remaining high-risk policy, CALO fairness, power-system/ORPD, recovery, parity, statistics, configuration and workflow inconsistencies before any new feature expansion.

## 1. Release philosophy

v5.9 is deliberately a closure release. It does not introduce another optimizer family and does not claim that CALO or a neural policy is superior. The implementation goal is to make scientific state, action semantics, feasibility, evidence selection, backend parity and publication qualification internally consistent and fail-closed.

The central release rule is:

> A result may be called deployable/publication-grade only when the exact formulation, exact policy semantics, exact evidence protocol and required independent validation are explicitly bound and verified.

## 2. Native v5.9 policy ABI and exact controller semantics

A new semantic ABI was introduced because v5.9 changes the meaning of the policy/controller interface rather than cosmetically renaming an old schema.

- Runtime architecture: `calo-v5.9`
- State schema: `calo-state-v5.9-32`
- Action schema: `calo-action-v5.9-raw-global-4r-6o-6p`
- Training environment: `calo-training-v5.9-exact-controller`

The native state contains the historical 24-D cognitive state plus eight bounded runtime/context features. Raw neural output is now explicitly distinguished from the controller action actually executed after environment-level interventions.

### Raw versus executed provenance

v5.9 trajectory records distinguish:

- full 32-D policy input,
- raw regime/operator probabilities,
- raw sampled regime/operator,
- raw continuous parameter action,
- mapped policy parameters,
- per-learner individual regimes,
- memory/group assignments,
- precision intervention,
- forced recovery intervention,
- final executed operators.

Legacy 24-D trajectories remain readable as legacy evidence but are not silently treated as exact native-v5.9 supervision.

## 3. Training ↔ deployed CALO transition parity

During v5.9 self-audit, several hidden transition mismatches were found and corrected, including:

- archive/success-memory capacity differences,
- epsilon/progress clock differences,
- severe-stagnation context differences,
- deterministic-policy behavior leaking into environmental stochastic choices,
- probability-blending differences,
- optimizer RNG-consumption order,
- personal-best propagation differences,
- feasibility-credit differences.

A shared scientific transition semantics was then enforced.

A seeded multi-transition regression now compares the PPO training environment against deployed native-v5.9 `CALOOptimizer`. For the tested sequence it verifies exact equality of:

- population,
- personal best,
- optimizer RNG state,
- epsilon state,
- operator contextual credit,
- memory contextual credit,
- reward,
- executed operators.

This is a release gate for the tested native transition semantics; it is not presented as proof that every possible long-horizon trajectory/hardware path is already exhaustively validated.

## 4. Correct action credit and shared reward

Forced recovery is treated as an explicit executed intervention. An overridden neural operator is not credited with a recovery-generated success.

Training and runtime use the same `calculate_reward()` implementation and coefficient schema. Reward semantics are therefore versioned with the native v5.9 training/runtime contract.

## 5. Training Champion versus Deployable Scientific Base

v5.9 separates two scientifically different concepts.

### Training Champion

Synthetic curriculum tasks may be used to identify a strong training checkpoint or provisional candidate. This evidence is useful for learning progress but is not sufficient for deployment promotion.

### Deployable Scientific Base

A deployable Base requires:

- a non-empty fixed real-ORPD development suite,
- an explicit development `ExperimentConfig`,
- exact objective definition and normalization,
- exact ORPD variable/control definition,
- exact robust/scenario formulation,
- exact AC power-flow options,
- exact constraint tolerance schema,
- final-best-feasible evidence for every required development case.

Raw objectives from unrelated synthetic and real tasks are not pooled into one unit-dependent median for deployment selection. Real-case evidence is normalized by formulation-specific reference scales and retained case-wise.

Final candidate evaluation is common-bundle and order-independent. Hardware inference latency is diagnostic only.

## 6. Competitive branch transaction and stale-recovery authority

Normal multi-branch training uses staged exact-resume state and immutable generation commit semantics.

The critical v5.8 recovery-authority defect is closed by recording and enforcing:

- parent manifest SHA-256,
- parent generation ID,
- parent common resume epoch,
- validation/provenance context.

Before recovery may change authority, the current authoritative manifest must still match the exact parent SHA/generation recorded when the interrupted session began. A stale session is refused as an authority replacement and must instead be handled as an explicit fork/export.

This prevents a recovered old session from silently rolling the repository back over a newer successfully committed generation.

## 7. Safe Stop, stuck workers and long-run boundedness

v5.9 retains and strengthens:

- initial safe state before expensive progression,
- rolling common exact checkpoints,
- typed `SAFE_STOPPED` and `SAFE_STOPPED_DEGRADED`,
- cancellation grace deadline,
- forced termination of nonresponsive children,
- fatal branch diagnostic retention,
- deterministic final queue drain/terminal accounting.

Exact-resume history, coordinator messages and Champion decisions are bounded.

Telemetry is segmented and bounded by configured segment count/size rather than growing one indefinitely large JSONL/checkpoint payload.

## 8. Resource admission

Competitive branch scheduling now combines:

- explicit branch → device assignment,
- accelerator concurrency caps,
- estimated per-branch memory,
- configured memory reserve fraction,
- available-memory admission where available,
- CPU fallback rather than blind oversubscription.

Physical CUDA/XPU behavior remains a target-hardware validation requirement because memory estimates and device APIs cannot be fully qualified in the CPU-only build runtime.

## 9. Scientific formulation fingerprint

A canonical scientific fingerprint was introduced/strengthened for exact compatibility and historical transfer.

It covers, as applicable:

- case identity/checksum,
- variable/control manifest,
- objective kind/weights/scales,
- power-flow options,
- constraint tolerance schema,
- robust aggregation,
- scenario weights and callable transform identity,
- callable code/defaults/closures,
- `functools.partial` function/args/keywords,
- bound methods/callable-object scientific state,
- decoder/repair schema.

Cross-run accelerator batching also uses the full scientific signature, preventing scientifically incompatible requests from sharing a batch merely because case dimensions look similar.

## 10. CALO repair fairness

Candidate repair now has one scientific accounting authority before cache/evaluation.

The v5.8 re-audit directly demonstrated a 50% true repair-coordinate rate being reported as 25% on one path and 0% on another. v5.9 corrects this path dependence.

Regression evidence checks identical cached/uncached values for:

- repaired candidate count,
- repaired coordinate count,
- total coordinate denominator.

## 11. Explicit engineering/numerical constraint tolerances

`ConstraintToleranceConfig` is versioned (`calo_rpd_constraint_tolerance_v5.9`) and persisted in the experiment/scientific fingerprint.

It covers:

- voltage p.u.,
- generator P MW,
- generator Q MVAr,
- branch loading %,
- branch angle degrees,
- aggregate feasibility residue.

The same tolerance semantics are propagated to CPU and accelerated constraint evaluation.

## 12. Power-system / ORPD corrections

### N-1 security bundle

Branch/generator contingency builders include the intact base state by default plus selected single contingencies. Contingency-only analysis remains possible only when explicitly requested by low-level API.

### Branch angle constraints

Active `ANGMIN` / `ANGMAX` limits are enforced for in-service branches with MATPOWER-compatible sentinel handling, including the 0/0 unconstrained convention.

The constraint is propagated to:

- CPU evaluation,
- device-resident accelerator evaluation,
- parity reporting,
- robust aggregation.

### ORPD decision variables

Generator-voltage controls are exposed only for online generators on REF/PV voltage-controlled buses. PQ-bus generator `VG` values are not exposed as dead optimization variables.

### Generator limits

P/Q capability accounting is performed per individual online generator.

For multiple online generators at one bus, solved bus-level reactive requirement is post-allocated using an explicit capability-proportional convention. This is documented as a deterministic reporting/limit-accounting convention, not a unique unit-level AVR dispatch solution.

### Sparse power-flow path

The Newton Jacobian uses sparse derivative construction. `Yf`/`Yt` branch current matrices are constructed directly from sparse triplets/CSR rather than dense temporary branch-by-bus arrays.

## 13. Scientific configuration validation

Fail-fast object-boundary validation was strengthened for:

- `ObjectiveConfig`,
- `PowerFlowOptions`,
- `RobustObjectiveConfig`,
- `RobustScenarioSettings`,
- `ORPDVariableConfig`,
- shunt control definitions,
- `ConstraintToleranceConfig`,
- policy qualification thresholds.

Examples now rejected before experiment execution include:

- negative/nonfinite objective weights,
- zero/nonfinite normalization scales,
- invalid tap bounds/steps,
- invalid PF tolerance/iterations,
- NaN robust risk parameters,
- invalid qualification probabilities/effect ranges/non-inferiority margin.

## 14. Scenario structural validation

Every scenario transform must preserve the scientific structural identity required by one shared ORPD variable manifest:

- base MVA,
- bus/gen/branch matrix dimensions,
- finite numerical data,
- bus identity/order,
- generator-to-bus row identity,
- branch endpoint row identity.

Status, load and allowed operating-condition values may change, but silent topology identity remapping is rejected.

## 15. Accelerator parity strengthening

The parity problem is now bound to the exact experiment formulation, including exact PF options and constraint tolerances.

The parity battery uses deterministic structural candidates plus random evidence and compares:

- total objective,
- objective components,
- total violation,
- each constraint component,
- feasibility,
- PF convergence,
- bus types,
- voltage magnitude,
- voltage angle,
- scenario count.

A post-generation audit also found and corrected a device-resident branch-angle indexing defect where scenario indices could be confused with bus endpoint indices.

## 16. Formal superiority and non-inferiority qualification

### Superiority

Formal promotion requires:

- required paired runs per case,
- feasibility threshold,
- configured independent AC-PF validation,
- favorable direction,
- win/effect gates,
- Holm-adjusted significance.

### Non-inferiority

Non-inferiority is not inferred from “no significant difference.” v5.9 tests the declared relative degradation margin using a one-sided paired Wilcoxon test with a one-sided sign-test fallback, followed by Holm correction.

Formal qualification is gated comparator-by-case. Aggregate evidence is retained for UI summary, but a strong case cannot statistically hide a weak required case.

During post-generation audit, a bug was found where `PolicyQualifier.run()` could reference the optional raw `config` argument instead of the resolved validated `qconfig` when applying the NI margin. This was corrected before release and qualification-threshold validation was strengthened.

## 17. Robust uncertainty semantics

`load_uncertainty` and `monte_carlo` are no longer methodologically identical:

- load uncertainty: deterministic stratified load-scaling scenario set,
- Monte Carlo: seeded random load-scaling scenarios.

Declared limitations remain:

- load uncertainty is system-wide aggregate scaling, not a spatial covariance model;
- renewable uncertainty is simplified net-load uncertainty, not a complete inverter-control stochastic model.

## 18. Independent validation scope

PYPOWER validation is accurately described as **independent AC power-flow numerical cross-validation**. The exact configured PF tolerance/iteration/Q-limit settings are propagated.

It is not described as an independent reimplementation of the entire ORPD decoder, repair policy, robust formulation or optimizer.

## 19. GUI workflow

The canonical setup order is now:

Power System → ORPD → Algorithms → Portfolio Manager → CALO Intelligence (if CALO selected) → Robust Scenarios → Experiment.

Portfolio evidence planning therefore occurs immediately after algorithm selection, while final experiment execution remains locked until CALO/scenario prerequisites are complete and the evidence plan is revalidated.

## 20. Post-generation re-audit corrections

The v5.9 generation was re-audited before packaging. Additional issues found and corrected during that pass include:

1. hidden PPO-training/deployed-controller transition mismatches (clocks, RNG, memory, intervention semantics),
2. device-resident branch-angle indexing,
3. cross-run batch signature scientific identity,
4. PQ-bus dead generator-voltage controls,
5. scenario structural identity validation,
6. Experiment workspace prerequisite gating,
7. default formal-qualification `qconfig` bug,
8. case-wise statistical promotion gating,
9. incomplete v5.9 freeze coverage for newly introduced scientific modules.

## 21. Validation boundaries

The final release validation document records the exact test counts and freeze/manifest hashes.

The build environment did **not** provide:

- PyQt6,
- PYPOWER,
- physical NVIDIA CUDA,
- physical Intel XPU,
- Ruff.

Accordingly, the release does not falsely claim those target-environment gates were executed. They remain mandatory before a final publication campaign/production-scale unattended run.

## 22. Residual limitations

The following are deliberate residuals, not silently closed findings:

- deeper CALO device residency remains performance work gated by seeded trajectory parity;
- uncertainty models remain simplified as declared;
- automatic periodic formal qualification is separate from training-time Champion selection;
- independent validation is AC-PF cross-validation, not a full independent ORPD implementation;
- large orchestration modules retain maintainability debt;
- physical accelerator throughput/OOM/fault behavior requires target-hardware validation.

## 23. Release conclusion

v5.9 materially strengthens the scientific consistency of the entire CALO-RPD chain:

**policy training → deployed CALO decision → candidate repair → AC power flow → ORPD constraints/objectives → robust scenarios → backend parity → qualification → publication evidence.**

The release is suitable as the next corrected development/research-validation baseline. Final publication claims should still be made only after the target-environment validation gates, fixed benchmark protocol, predeclared ablation/evidence plan, complete independent validation, and frozen repeated-run campaign are executed.
