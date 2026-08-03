# CALO architecture change proposal — approval required

**Status:** approved on 2026-08-03 as “Approve TSH-CALO A–E, with F experimental and
evidence-gated.” A–E are authorized for careful production-candidate implementation. F is
authorized only behind an independent flag, disabled by default, and may be retained only if its
preregistered ablation and falsification evidence passes. This record does not itself claim that any
component has been implemented or scientifically validated.
**Decision owner:** user/project scientific lead
**Proposed working name:** Topology-Aware Shielded Hierarchical CALO (TSH-CALO)

## 1. Scientific boundary

CALO cannot honestly be guaranteed to outperform every optimizer on every problem. The No-Free-Lunch
result establishes why improved average performance on one problem distribution must be discussed as
specialization, not universal dominance ([Wolpert and Macready, 1997](https://doi.org/10.1109/4235.585893)).
The defensible target is narrower and stronger:

> Demonstrate that a frozen, topology-aware CALO is superior or non-inferior on a preregistered ORPD
> distribution for feasibility, final quality, robustness, and evaluation efficiency under equal
> declared budgets, while reporting runtime and energy separately.

The proposed architecture is designed for that target. It does not claim dominance outside the tested
ORPD distribution.

## 2. What the present implementation already does well

The current CALO is not a trivial metaheuristic. It already includes:

- feasibility-first and epsilon-feasible selection;
- feasible-elite and constraint-boundary archives;
- six learning operators, persistent personal bests, hierarchical elite memory, success memory,
  variable-group intelligence, precision search, and diversity recovery;
- contextual online credit and a PPO controller with four regimes, six operators, six bounded
  parameters, and a value head;
- strict function-evaluation accounting, paired seeds, exact resume, immutable policy binding,
  policy qualification, and detailed diagnostics;
- protected policy-training and held-out-case leakage guards.

These should be preserved as the baseline and ablation reference.

## 3. Material scientific weaknesses found in the current CALO

### 3.1 The neural controller is global and information-poor

The native policy sees 32 aggregate features. It does not see the power-network graph, bus-level voltage
stress, branch-level loading, generator reactive reserve, or which physical controls affect the active
constraint bottleneck. Two distinct grids can therefore produce similar 32-value policy states even when
the useful search direction is different.

The policy emits one global regime and one global operator action per generation. In the native path,
that raw operator is authoritative for ordinary learners; individual exceptions come from precision and
forced-recovery rules. This is weaker than the documented idea of independently contextual operator
choice and can correlate the whole population around a poor action.

### 3.2 Training and deployed transitions are duplicated

The runtime optimizer and policy-training environment implement closely related transition logic in
separate large code paths. One-step parity tests reduce risk but do not make the transition definition
single-source. Every future operator or selection change would need two correct implementations.

### 3.3 The learned policy has no explicit epistemic-confidence gate

The MLP always produces an action distribution, including on a topology or stress pattern far outside
training support. Runtime compatibility validates the tensor/schema contract, not whether the policy is
scientifically in-distribution. Current rule priors and recovery interventions help, but they are not a
calibrated uncertainty mechanism.

### 3.4 Online credit and neural control are not one coherent decision rule

Contextual credit exists, but the native raw neural operator remains authoritative for ordinary learners.
This means CALO has both an offline policy and online evidence without a formally defined confidence-
weighted residual rule. Dynamic multi-armed-bandit work shows why adaptive operator selection needs an
explicit exploitation/exploration mechanism under changing rewards
([Fialho et al., 2010 manuscript](https://citeseerx.ist.psu.edu/document?doi=0a82b055db761e9f7aa9689f76636b6c11da0618&repid=rep1&type=pdf)).

### 3.5 The operator set is not topology- or sensitivity-aware

All candidates are ultimately verified by the trusted AC evaluator, which is correct, but proposal
generation does not exploit local AC sensitivity to identify controls likely to reduce the currently
dominant violation. Research on sensitivity-informed AC-OPF learning reports improved generalization and
constraint satisfaction when derivative information is used
([Singh, Kekatos, and Giannakis](https://arxiv.org/abs/2103.14779)).

### 3.6 Validation is strong for a software release but insufficient for leadership claims

IEEE 30/57 training and 118/300 holdout are useful, but four named cases do not define broad ORPD
leadership. Sound optimizer benchmarking needs frozen problem instances, multiple targets/budgets,
anytime behavior, independent repeated trials, and reproducible archives; COCO formalizes these principles
for black-box optimization ([Hansen et al., 2021](https://coco-platform.org/)). Pairwise Wilcoxon tests
should not be replaced by pool-dependent mean-rank post-hoc claims
([Benavoli, Corani, and Mangili, 2016](https://jmlr.org/beta/papers/v17/benavoli16a.html)).

## 4. Proposed architecture

### Change A — one canonical CALO transition kernel

Create a pure, device-neutral transition authority consumed by both deployed optimization and policy
training. It will own candidate assignment, operator execution, exact FE accounting, personal/archive
updates, environmental selection, reward decomposition, and next-state diagnostics.

This is a prerequisite refactor. The first acceptance gate requires bit-identical seeded CPU results and
unchanged frozen baseline snapshots. No new scientific behavior is allowed in this substage.

### Change B — topology-aware graph context

Add a graph encoder alongside the existing aggregate cognitive state:

- bus nodes: voltage magnitude/angle, load, generation, reactive reserve, voltage violation;
- branch edges: impedance/admittance descriptors, tap/shift, loading, angle/thermal violation;
- control attachments: generator-voltage, tap, and shunt control identity plus remaining movement;
- scenario descriptors: load/renewable/contingency stress and aggregation role.

Message passing produces a grid embedding and three control-group embeddings. These are concatenated
with the existing 32 aggregate features, so the proven global diagnostics are retained. GNNs are a natural
inductive bias for power networks and have been studied for OPF scalability and cross-network structure
([Owerko, Gama, and Ribeiro](https://arxiv.org/abs/1910.09658)); GNN+PPO has also been explored for OPF
under topology changes ([López-Cardona et al.](https://arxiv.org/abs/2212.12470)). Those works motivate
the representation; they do not prove CALO will improve.

This changes the policy ABI and requires newly trained policies. Existing policies remain immutable
baseline artifacts and must never be silently interpreted as topology-aware.

### Change C — hierarchical group-conditioned actions

Replace one global operator action with a factored action:

1. one global search regime;
2. one operator distribution for each control group (generator voltage, transformer tap, shunt);
3. bounded continuous controls per group;
4. independent per-learner samples from the appropriate group/context distribution.

The action mask forbids operators that are invalid for a group. Forced recovery and counted precision
remain explicit, logged interventions. This resolves the current mismatch between global neural action
and claimed per-learner cognition while keeping the action space interpretable.

### Change D — uncertainty-shielded policy/bandit residual

Use a small policy ensemble or bootstrapped heads to estimate action disagreement. The executed operator
distribution becomes a declared mixture of:

- neural policy prior;
- transparent regime prior;
- sliding-window contextual-bandit residual;
- a minimum exploration floor.

Low-confidence or out-of-distribution graph states automatically reduce neural weight; they do not invent
an unverified action. The safety shield enforces valid actions, budget limits, feasibility-first ordering,
and recovery invariants. Every intervention and mixture weight is recorded for ablation and audit.

### Change E — optional physics-informed feasibility-repair operator

Add one bounded trust-region operator that uses the already computed converged AC Jacobian and current
constraint residuals to propose a control-space direction. It is a proposal only:

- it cannot declare feasibility;
- every candidate is evaluated by the same trusted full evaluator;
- every evaluation consumes the same FE budget;
- Jacobian/linear-algebra time and failures are recorded separately;
- if sensitivity is unavailable or ill-conditioned, the operator is masked rather than approximated
  silently.

Physics-informed AC-OPF work reports that embedding physical equations can reduce constraint violations
([Nellikkath and Chatzivasileiadis](https://arxiv.org/abs/2110.02672)). The proposed use is deliberately
more conservative: physics guides candidate generation but never replaces independent AC validation.

### Change F — optional evidence-driven population schedule

Permit a bounded population schedule that starts broad and contracts only when feasibility, archive
coverage, diversity, and remaining budget satisfy preregistered conditions. Linear population-size
reduction improved SHADE in L-SHADE ([Tanabe and Fukunaga](https://ieeexplore.ieee.org/document/6900380)),
but CALO should not copy a fixed linear schedule blindly. This component is accepted only if its own
ablation improves anytime performance without harming feasibility or small-budget behavior.

## 5. What will not change

- AC power-flow equations, decoder semantics, objective definitions, constraint tolerances, robust
  aggregation, and independent validation remain authoritative.
- FE accounting remains exact and equal across algorithms.
- The active policy remains immutable and checksum-bound to each experiment.
- Policy training remains an independent lifecycle; downstream experiments consume only a qualified,
  activated policy.
- Current CALO remains available as the frozen baseline until the proposed variant passes every gate.
- CUDA/CPU execution may change speed, not scientific equations or selection semantics.

## 6. Required implementation order after approval

1. Canonical transition refactor with zero-behavior-change regression gate.
2. Versioned state/action schemas and graph data contract.
3. Graph encoder and hierarchical action heads behind a new algorithm/policy version.
4. Uncertainty and online-residual controller with deterministic audit traces.
5. Physics-informed operator behind an ablation flag.
6. Population schedule behind a separate ablation flag.
7. Fresh curriculum training; no reuse of protected final-test trajectories.
8. Qualification and activation only after all scientific gates pass.

## 7. Mandatory ablation matrix

At minimum, compare with paired seeds and equal FE budgets:

1. current frozen CALO;
2. canonical-refactor CALO (must match current baseline);
3. graph context only;
4. hierarchical actions only;
5. graph + hierarchy;
6. + uncertainty shield;
7. + contextual bandit residual;
8. + physics repair;
9. + population schedule;
10. full proposed CALO.

Also compare against strong non-CALO references, including a modern success-history DE family member,
CMA-ES where variable semantics permit, deterministic/nonlinear mathematical solvers where applicable,
and the existing declared baselines. A large set of weak or redundant metaheuristics is not sufficient.

## 8. Acceptance gates

The new architecture is rejected or revised unless it passes all applicable gates:

- **Correctness:** CPU reference parity, mixed-variable lattice validity, no FE overshoot, exact resume,
  no held-out leakage, and deterministic replay of every logged intervention.
- **Feasibility:** no regression in feasible-run probability or violation distribution on any protected
  primary case; separately report failure modes.
- **Quality:** paired effect estimates with uncertainty; Holm-controlled pairwise comparisons; no claim
  from a p-value alone.
- **Generalization:** frozen unseen load/scenario/topology distributions and at least four independent
  case/formulation blocks. Training and final test identities are cryptographically separated.
- **Anytime behavior:** target attainment over multiple FE budgets, not only one final endpoint.
- **Cost:** FE, solver calls, wall time, peak RAM/VRAM, transfers, and energy reported separately.
- **Ablation:** each claimed contribution must show incremental value or be removed.
- **Reproducibility:** source/policy/config/data hashes, paired seeds, raw run records, failures, and
  complete environment/container identity retained.

## 9. Principal risks

- The GNN can overfit four named IEEE cases and appear topology-aware without generalizing.
- A larger policy increases training cost and may be worse than the current compact MLP.
- Sensitivity repair may dominate ORPD but reduce black-box generality; it must be described as a
  domain-informed operator and compared fairly on both FE and time.
- An ensemble/bandit mixture can become too complex to attribute. The ablation matrix is mandatory.
- More components create more hyperparameters and researcher degrees of freedom. Freeze them before
  final evaluation and retain all failed trials.

## 10. Approval choices

**Recorded decision (2026-08-03):** A–E approved; F experimental, independently evidence-gated,
and disabled by default. The current CALO remains the frozen baseline. TSH-CALO receives new
algorithm and policy ABI versions, never auto-activates, and cannot inherit prior superiority
evidence.

No architecture item will be implemented without an explicit decision. The recommended choices are:

- **Foundation:** approve Change A (canonical transition kernel).
- **Core vNext:** approve Changes B–D (graph context, hierarchical actions, uncertainty/bandit shield).
- **Domain operator:** separately approve or reject Change E (physics-informed repair).
- **Population control:** separately approve or reject Change F (evidence-driven population schedule).

Approval should also confirm that the current CALO remains a frozen baseline and that the new variant
cannot become active until qualification succeeds.
