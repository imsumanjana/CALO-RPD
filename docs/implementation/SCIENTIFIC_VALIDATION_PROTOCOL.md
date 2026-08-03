# CALO scientific validation and preregistration protocol

Status: protocol design only; no CALO architecture change is authorized by this document.

This protocol converts the repository audit into a falsifiable evaluation program. Its purpose is
not to prove that CALO is universally best. The defensible target is to determine whether a frozen
CALO version has a practically important, reproducible advantage on a declared distribution of
constrained optimal reactive power dispatch (ORPD) tasks, and to identify where it does not.

## 1. Non-negotiable separation

Three asset classes must remain separate and be identified by checksums:

1. **Development assets** may be used for implementation, debugging, tuning and ablation.
2. **Validation assets** may select among preregistered variants but may not train policy weights.
3. **Protected test assets** are opened only after code, policy, hyperparameters, statistical analysis
   and container identity are frozen.

IEEE 30 and 57 remain development systems. IEEE 118 and 300 remain protected holdouts until a
larger family-separated split is available. A renamed or copied protected case remains protected by
content identity. Repeated load or contingency variants of the same network belong to the same
network family and must not be split across development and final test merely by changing a file
name.

Policy training is an independent workflow. It generates a candidate policy artifact. Qualification
and explicit activation produce the governing policy. Every later power-system experiment records
that governing policy's identifier and SHA-256, but the experiment workflow does not retrain it.

## 2. Claims decided before results

### 2.1 Primary claim

For each preregistered task block and evaluation budget, the primary estimand is the paired
difference between CALO and a comparator in final independently validated feasible objective. A run
that is infeasible or fails validation is retained and cannot be silently removed.

CALO earns a task-block advantage only if all of the following hold:

- feasible-run probability is non-inferior within a frozen practical margin;
- the objective effect has the preregistered favorable direction and practically important size;
- uncertainty excludes the preregistered null or non-inferiority boundary after multiplicity control;
- the result survives the paired-seed sensitivity analysis and independent power-flow validation;
- there is no unreported increase in solver failures, repairs, transfers, energy or runtime.

The aggregate claim is a distributional claim across task blocks, not a statement about every
possible optimization problem. Wins, ties, losses and invalid blocks are all reported.

### 2.2 Secondary claims

Secondary outcomes are feasibility probability, violation distribution, evaluations to first
feasible solution, feasible-objective anytime curve, final optimality/reference gap, wall time,
energy, peak RAM, peak VRAM and host-device transfer volume. They do not replace the primary
outcome after results are observed.

### 2.3 Prohibited claim patterns

- no universal or “best algorithm” language;
- no claim based only on a mean rank or uncorrected p-value;
- no removal of failed or infeasible runs;
- no use of test systems for policy training, hyperparameter selection or stopping decisions;
- no equal-wall-time claim when hardware, parallelism or fallback paths differ;
- no presentation of the current global policy action as a per-learner action;
- no promotion of a policy whose artifact, schemas or qualification scope do not match exactly.

## 3. Frozen task taxonomy

The final benchmark manifest must stratify tasks by:

| Axis | Required levels |
|---|---|
| Network family | IEEE/PYPOWER and independently sourced PGLib-OPF families |
| Size | small, medium and large under declared bus/dimension thresholds |
| Loading | light, nominal and heavy |
| Operating difficulty | ordinary, narrow-feasible and deliberately infeasible diagnostics |
| Uncertainty | deterministic, stochastic expected-value and robust/CVaR |
| Security | intact network and N-1 branch/generator contingencies |
| Controls | generator voltage, discrete tap and discrete shunt density |
| Limits | ordinary and tight voltage/reactive constraints with Q-limit switching |
| Distribution | development, validation, in-distribution test and out-of-distribution test |

Every task manifest records source URI/version, raw file checksum, normalized case checksum, base
MVA, dimensions, declared controls, absolute/incremental semantics, bounds, lattice steps, fixed
devices, scenario seed, contingency set and feasibility tolerance. A task lacking this manifest is
not eligible for confirmatory evidence.

PGLib-OPF is the intended expansion because it supplies standardized AC-OPF networks and stressed
operating variants. Incorporating it requires a licensed, checksummed data-import path and an ORPD
control-profile review; merely loading a MATPOWER file does not define a comparable ORPD problem.

## 4. Comparator hierarchy

The confirmatory comparator set must favor quality over a large count of weak methods.

### Tier 1 — mathematical references

- a mature nonlinear AC-OPF/ORPD solver with explicit formulation and tolerances;
- multi-start local optimization using the same physical task definition;
- a lower bound or relaxation where a valid comparable bound exists;
- exact or mixed-integer reference on tractable small discrete instances.

These methods need not receive an artificial black-box FE equivalence. Report their solver calls,
derivatives, termination status and optimality criteria transparently.

### Tier 2 — strong stochastic references

- a source-faithful success-history differential-evolution method such as SHADE/L-SHADE;
- CMA-ES on compatible continuous encodings, with discrete handling declared separately;
- a suitable mixed/discrete optimizer for tap and shunt lattices;
- a surrogate-assisted comparator for expensive robust tasks, with exact final reevaluation.

### Tier 3 — repository continuity references

- frozen current CALO;
- No-AI CALO as a diagnostic where its semantics are explicitly declared;
- the existing source-declared TLBO, PSO, CLPSO, MTLA-DE, QODE and other registered baselines.

Each comparator needs deterministic operator-equation tests, boundary/repair tests, exact FE
accounting, seeded short-trajectory snapshots and CPU-versus-accelerated numerical agreement before
its label can be treated as a validated implementation.

## 5. Fairness contract

All stochastic methods in a comparison block receive:

- identical case/formulation and feasibility tolerances;
- identical scenario and contingency samples;
- paired run indices and immutable seed derivation;
- the same primary black-box FE budget and checkpoints;
- the same independent final validation;
- separately disclosed tuning budget;
- one frozen implementation and parameter set for the protected test.

Parallelism, batching and GPU use may reduce time but must not change FE accounting or the numerical
problem. A CPU or staged-host fallback is a different execution stratum for timing and energy. Its
scientific outputs may be combined only after numerical-agreement gates pass, while performance
results remain stratified.

## 6. Run-count determination

The application's current study planner is a conservative planning approximation for paired
standardized mean differences. It uses a two-sided family alpha, the smallest Holm threshold across
the planned CALO-versus-baseline comparisons, target power and a failed-run allowance.

Before a confirmatory campaign:

1. obtain a small preregistered pilot from development assets only;
2. estimate the paired-difference distribution, feasibility rate and failure rate;
3. define the smallest important effect in objective-specific physical units or a frozen reference
   scale, not from the observed test result;
4. replace the normal approximation with simulation-based power for the actual planned statistic
   when discreteness, censoring, ties or non-normality matter;
5. round upward and freeze the number of initiated runs, including the failure allowance;
6. never reduce the run count after inspecting favorable interim results.

If a powered design is unavailable, label the study exploratory. Fixed labels such as 30 or 50 runs
are not a substitute for power analysis.

## 7. Statistical analysis

### 7.1 Run-level outcomes

The atomic observation is one paired run per task, method and budget. Store raw final objective,
feasibility, violation vector, first-feasible FE, checkpoint trajectory, termination state and
validation state. Keep the original unit of measurement.

### 7.2 Pairwise analysis

- report paired median/mean differences as appropriate, confidence intervals and a robust paired
  effect size;
- use a paired permutation or Wilcoxon procedure only when its assumptions and tie behavior are
  appropriate;
- control the preregistered CALO-versus-comparator family with Holm's procedure;
- use objective-specific non-inferiority margins frozen before the protected test;
- analyze feasibility with paired binary methods and confidence intervals, not by filtering to the
  runs that both methods solved.

### 7.3 Across-task analysis

Report task-block effects and uncertainty first. A Friedman/mean-rank summary may be secondary, but
it cannot conceal heterogeneous effects or substitute for magnitude. A hierarchical model or
distribution-free multiple-task analysis should include a task-family effect when enough independent
families exist.

### 7.4 Anytime performance

Preregister FE checkpoints. Report target attainment and evaluations to feasibility separately from
conditional feasible-objective progress. Do not construct a run-dependent pre-feasible AUC penalty.
If a scalar anytime endpoint is required, use frozen task-specific reference targets/scales.

## 8. CALO vNext ablation program

This section becomes executable only for architecture items explicitly approved by the user. The
minimum paired, equal-FE matrix is:

1. current frozen CALO;
2. canonical transition refactor, which must reproduce item 1;
3. graph context only;
4. hierarchical actions only;
5. graph context plus hierarchical actions;
6. uncertainty shield added;
7. contextual bandit residual added;
8. physics-informed feasibility repair added;
9. adaptive population schedule added;
10. the complete approved variant.

Every component must show incremental practical value on validation assets. A component that adds
complexity without repeatable value is removed before the final freeze. The test matrix must include
conditions where the component is expected to fail; otherwise the ablation is not falsifiable.

## 9. Independent validation

The optimizer's internal evaluator is not the final scientific authority. Every reported best
candidate is decoded from the stored vector and checked by an independent implementation or solver
path with frozen tolerances. Record:

- convergence and termination reason;
- active/reactive balance residuals;
- voltage, generator reactive, tap, shunt and thermal constraints;
- discrete-lattice validity;
- contingency/scenario outcomes;
- objective recomputation and discrepancy from the optimizer evaluator.

A mismatch beyond tolerance invalidates the candidate and triggers a diagnostic audit. It is not
repaired silently after optimization.

## 10. Resource and reproducibility evidence

Each run bundle must contain:

- source commit and clean/dirty status;
- container image digest and dependency lock checksums;
- policy ID, checksum, architecture version, state/action schema and qualification scope;
- experiment/protocol/task/scenario fingerprints;
- host OS, CPU, physical/logical memory, NVIDIA GPU and driver/runtime identity;
- requested compute mode, actual execution state and every fallback/staging transition;
- admission-time free memory, 80% allowance, peak RAM/VRAM and transfer bytes;
- seeds, raw results, failures, logs and independent-validation outputs;
- analysis code and a machine-readable claim decision.

The 80% rule is an admission ceiling based on memory free at admission, not a utilization target.
Accelerated mode keeps eligible active numerical data in VRAM when it fits. If the allowed VRAM is
insufficient, bounded staging or an explicitly recorded CPU execution state may use available system
RAM. CPU-only mode never requires NVIDIA hardware.

## 11. Freeze and opening procedure

1. pass source, unit, scientific, container and hardware qualification gates;
2. freeze code, dependencies, data/task manifests, policy, configurations and analysis scripts;
3. publish hashes and the planned run matrix before protected-test execution;
4. execute without tuning or changing stopping rules;
5. retain all initiated runs and failures;
6. run the preregistered analysis once;
7. label any additional analysis exploratory;
8. publish wins, ties, losses, invalid blocks and operational costs.

Any source or semantic change after freeze creates a new candidate and a new test opening. Historical
freeze manifests are evidence for their own release only and must not be rewritten during active
development.

## 12. Promotion and rejection gates

A CALO candidate cannot become release-qualified unless all applicable gates pass:

| Gate | Pass condition |
|---|---|
| Transition equivalence | behavior-preserving refactor matches the frozen baseline under seeded trace tests |
| Correctness | no FE overshoot, valid mixed variables, exact resume and deterministic intervention replay |
| Leakage | no protected identity or derivative used in training, tuning or model selection |
| Feasibility | no material protected-block regression in feasible probability or violations |
| Quality | powered paired effects meet frozen practical and multiplicity-controlled boundaries |
| Generalization | favorable evidence spans independent families and includes an OOD test |
| Cost | time, energy, memory and transfers are reported and not misrepresented |
| Attribution | each retained component has supporting ablation evidence |
| Reproduction | a clean CPU container reproduces science; CUDA container passes numerical agreement |

Failure of a gate means reject, revise or narrow the claim. It never means changing the threshold
after observing the protected result.

## 13. Evidence-strength mapping used by the GUI

The GUI's scientist-facing presets are planning aids, not publication guarantees:

- **Screening:** smoke checks and explicit exploratory labeling.
- **Moderate:** paired deterministic study with essential objective/feasibility outputs.
- **Rigorous:** powered multi-method study, uncertainty, corrected pairwise comparisons and complete
  reproducibility artifacts.
- **Comprehensive:** rigorous evidence plus robust/OOD blocks, strong modern and deterministic
  references, ablations, sensitivity, resource evidence and protected-test discipline.

Applying a preset writes one validated protocol across the power-system workflow. It does not alter
or retrain the governing CALO policy.

## 14. Authoritative references

- Wolpert and Macready, no-free-lunch boundary: <https://doi.org/10.1109/4235.585893>
- PGLib-OPF benchmark library: <https://github.com/power-grid-lib/pglib-opf>
- PGLib-OPF benchmark paper: <https://arxiv.org/abs/1908.02788>
- PowerModels common-formulation framework: <https://arxiv.org/abs/1711.01728>
- L-SHADE population/success-history DE: <https://ieeexplore.ieee.org/document/6900380>
- Bayesian and rank-test cautions for multiple classifier/algorithm comparisons:
  <https://jmlr.org/papers/v17/benavoli16a.html>

These references motivate the protocol; repository implementations still require their own
source-faithfulness and numerical-validation evidence.
