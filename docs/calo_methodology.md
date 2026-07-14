# CALO Methodology — Core v2

## 1. Purpose

The **Cognitive Adaptive Learning Optimizer (CALO)** is a search-state-aware optimization architecture for constrained, mixed-variable problems such as optimal reactive power dispatch. CALO Core v2 was designed to prevent the failure mode in which a population decreases aggregate constraint violation, collapses around one low-violation point, and then remains infeasible for the rest of the evaluation budget.

CALO is not TLBO with a changed random-number distribution. Its runtime combines constraint diagnostics, adaptive epsilon-feasibility, dual archives, a multi-operator population, success-distribution memory, online operator credit, mixed-variable neighbourhood moves, and an optional hierarchical AI policy.

## 2. Exact and epsilon feasibility

Final scientific validity always uses the strict common ORPD constraints. During search, CALO may temporarily treat a candidate as epsilon-feasible when

\[
CV(\mathbf{x}) \leq \epsilon_t.
\]

The threshold decreases according to

\[
\epsilon_t = \epsilon_0\left(1-\frac{N_{FE}}{T_c}\right)^p,
\quad N_{FE}<T_c,
\]

and

\[
\epsilon_t=0,
\quad N_{FE}\geq T_c.
\]

Here `N_FE` is the objective-function evaluation count, `T_c` is a configured fraction of the total evaluation budget, and `p` controls the decay shape. This mechanism allows promising near-feasible regions to compete early while exact feasibility is enforced by the end of the control interval.

## 3. Dual archives

### 3.1 Feasible Elite Archive

The feasible archive stores high-quality exact-feasible solutions and rejects numerical duplicates. Rank-biased sampling favors strong solutions without forcing every learner toward one point.

### 3.2 Constraint Boundary Archive

The boundary archive stores diverse infeasible solutions with low total violation. Retention considers:

- total constraint quality;
- distance in normalized decision space;
- difference in the constraint-component profile.

The archive therefore preserves multiple routes toward feasibility, for example a candidate dominated by generator-Q violation and another dominated by bus-voltage violation.

## 4. Constraint decomposition

The CALO state tracks the best and mean values of:

\[
CV_V,
CV_Q,
CV_P,
CV_S,
CV_{PF},
\]

representing bus-voltage, generator reactive-power, generator active-power, branch thermal, and power-flow convergence violations. These components remain separate from the objective value.

## 5. Cognitive state

The release state vector contains 24 normalized values:

- population diversity;
- elite spread;
- exact feasible ratio;
- epsilon-feasible ratio;
- transformed mean total violation;
- transformed best total violation;
- five transformed physical constraint components;
- recent constraint improvement;
- recent feasible-objective improvement;
- constraint stagnation;
- objective stagnation;
- remaining evaluation budget;
- feasible archive fill ratio;
- boundary archive fill ratio;
- six online operator-credit values.

Objective and constraint progress are never combined into one raw scalar state.

## 6. Search regimes

The hierarchical controller assigns probabilities to four regimes:

1. **Feasibility** — emphasize boundary learning and physically meaningful device moves.
2. **Transition** — balance constraint reduction and objective quality near the feasible boundary.
3. **Objective refinement** — exploit exact-feasible elites while preserving differential information.
4. **Recovery** — temporarily rebuild diversity when objective or constraint progress stagnates.

A transparent rule-based prior is blended with the learned regime policy. Recovery is temporary and includes a cooldown; it cannot become a permanent hard override.

## 7. Per-individual operator allocation

CALO does not apply one operator to the entire population. Each learner independently samples one of six operators from a probability distribution that combines:

- hierarchical AI output;
- regime-specific prior knowledge;
- online operator credit measured during the current run.

This permits simultaneous exploration, feasibility repair, and objective refinement within one generation.

## 8. CALO Core v2 operators

### 8.1 Feasible-elite learning

\[
\mathbf{x}'_i = \mathbf{x}_i
+ \alpha r_1(\mathbf{x}_{pbest}-\mathbf{x}_i)
+ F(\mathbf{x}_{r1}-\mathbf{x}_{r2}).
\]

The first term exploits a feasible or boundary teacher. The differential term preserves directional diversity.

### 8.2 Constraint-boundary differential learning

\[
\mathbf{x}'_i = \mathbf{x}_i
+ \alpha r_1(\mathbf{x}_{C}-\mathbf{x}_i)
+ F(\mathbf{x}_{r1}-\mathbf{x}_{r2}),
\]

where `x_C` is sampled from the diverse constraint-boundary archive.

### 8.3 Cognitive teacher learning

\[
\mathbf{x}'_i = \mathbf{x}_i
+ \alpha |\mathbf{Z}_1|\odot(\mathbf{x}_{T}-\mathbf{x}_i)
+ \beta \mathbf{Z}_2\odot(\mathbf{x}_{T}-\bar{\mathbf{x}}).
\]

The teacher depends on the current regime and available archives.

### 8.4 Success-distribution memory

A bounded memory stores successful directions with objective gain, feasibility gain, operator identity, and recency. CALO samples a successful direction probabilistically rather than averaging all directions into one vector that may cancel opposing successful moves.

### 8.5 Mixed-variable neighbourhood learning

Continuous generator-voltage controls receive local continuous perturbations. Discrete transformer taps and shunts move to actual neighbouring device levels, typically ±1 admissible step and occasionally a larger local step. This prevents expensive evaluations of different normalized vectors that decode to the same physical discrete state.

### 8.6 Diversity recovery

Recovery uses opposition-guided or underexplored anchors with bounded perturbation. It does not repeatedly perturb the same current best solution.

## 9. Environmental selection

Parents and offspring are combined:

\[
P_{t+1}=\operatorname{Select}(P_t\cup Q_t).
\]

Selection uses the current epsilon-feasibility rule, preserves a quality subset, and then fills remaining positions with a quality-diversity criterion. Exact feasible elites are separately retained by the feasible archive.

## 10. Online operator credit

Each operator receives recency-weighted credit from actual current-run improvements. The final operator probability is obtained from the learned probability and online credit:

\[
P_k^{final}\propto
(P_k^{AI}+\delta)^\lambda
(P_k^{credit}+\delta)^{1-\lambda}.
\]

This allows current evidence to correct a weak learned prior.

## 11. Hierarchical AI policy

The PyTorch policy has a shared backbone and four outputs:

- regime logits;
- operator logits;
- Beta-distribution alpha parameters;
- Beta-distribution beta parameters;
- state-value estimate.

The bounded continuous action controls attraction, differential strength, exploration scale, memory contribution, diversity pressure, and recovery fraction. Both categorical and continuous actions contribute to the policy log-probability used during training.

## 12. PPO training

The training implementation uses:

- clipped PPO surrogate objective;
- old and new action log probabilities;
- generalized advantage estimation (GAE);
- value loss;
- entropy regularization;
- minibatch updates;
- multiple PPO epochs;
- gradient clipping.

The training environment imports and uses the same CALO Core v2 operator implementations, epsilon environmental selection, dual archives, cognitive state builder, success memory, operator credit, and mixed-variable moves as runtime.

Version 2.0.2 adds synchronous weighted actors. One policy snapshot is broadcast at the beginning of an epoch, complete episodes are allocated to CUDA, Intel XPU, and CPU lanes, and only matching current-policy trajectories are merged. The centralized learner then performs the PPO update. This preserves on-policy semantics while allowing accelerator-batched policy inference and concurrent CPU actors.

The built-in curriculum progresses through:

1. continuous unconstrained tasks;
2. constrained continuous tasks;
3. mixed discrete-continuous tasks;
4. narrow-feasible-region tasks;
5. optional explicitly configured ORPD development systems.

The fifth stage is enabled only when development case paths are supplied. Final publication benchmark systems are not silently used for training. Development-system identifiers are preserved in checkpoint metadata.

## 13. CALO Core v2 ablation suite

The application provides nine fixed variants:

1. Classical TLBO;
2. Legacy Gaussian MTLBO;
3. CALO Core v2 without AI;
4. CALO without epsilon-feasibility;
5. CALO without dual archives;
6. CALO without mixed-variable learning;
7. CALO without success memory;
8. CALO without diversity recovery;
9. Complete CALO.

Ablation results are stored separately from the primary 20-algorithm benchmark.

## 14. Required interpretation

CALO should not be called universally superior. The software is designed to measure:

- feasible-run rate;
- evaluations to first feasibility;
- final exact constraint violation;
- median final objective;
- runtime;
- statistical significance;
- effect size;
- problem classes where CALO gains or loses advantage.

The final evidence, not the algorithm name, determines the claim.
