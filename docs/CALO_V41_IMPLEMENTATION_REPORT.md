# CALO-RPD Studio v4.1.0 — Implementation Report

## Scope

v4.1.0 is a scientific-correction, policy-governance, diagnostics, and reproducibility release built on the v4.0 CALO architecture. It intentionally does **not** add new search operators or claim that CALO is universally superior. The release focuses on qualifying which policy is used, binding it reproducibly to experiments, restoring complete historical workspaces, and correcting misleading performance diagnostics without changing the common ORPD equations or fairness budget.

## Implemented policy system

The CALO Intelligence workspace now contains a managed Policy Center. Policies are registered with checkpoint SHA-256, runtime architecture, state/action schema, training-environment version, qualification state, grade, active/archive state, and experiment references. A policy may be compared against the selected reference policy and No-AI CALO using paired seeds and identical evaluation budgets. The primary ranking remains median final **feasible** objective; convergence AUC, feasibility probability, stability, evaluations-to-feasibility, and runtime are secondary evidence. IEEE 118 and IEEE 300 remain protected holdouts by default.

Strict experiments store an immutable policy binding containing the policy ID, artifact path, SHA-256, schemas, deterministic setting, and qualification provenance. A missing or mismatched strict policy is not silently replaced by the bundled legacy policy. Archive and safe-delete operations preserve experiments that reference older policies.

## Native v4.1 policy schema

A native v4.1 policy observes 32 features: the original 24 cognitive features plus HPEM occupancy, memory consensus, memory readiness, success-memory density, learning-lane fraction, precision activity, precision radius, and variable-group concentration. The action schema remains four regime outputs, six operator outputs, and six bounded adaptive parameters. Legacy 24-feature checkpoints are explicitly identified rather than mislabelled as native v4.1 policies.

The lightweight PPO training environment now exposes the v4.1 cognitive components and semantics, including persistent personal memory, HPEM, contextual credit, variable intelligence, adaptive epsilon, dual-lane readiness, recovery, and the 32-feature state. It is deliberately documented as a training rollout environment rather than a bit-identical duplicate of the complete runtime transition loop. Therefore a trained candidate must pass Policy Qualification in the real optimizer before it is treated as scientifically qualified.

Exact v4.1 epoch-level training resume stores actor/critic network state, optimizer state, history, Python/NumPy/Torch RNG states, generator state, CUDA RNG states when present, training configuration, and schema metadata. Legacy 24-feature training resumes are rejected for exact v4.1 continuation because silently changing the state dimension would not be scientifically exact. Long-lived policy lineages, indefinite training, and best-checkpoint lineage promotion are reserved for the planned v5.0 continuation architecture.

## Experiment workspace restoration

A centralized `ExperimentWorkspaceRestorer` now restores the authoritative `ExperimentConfig`, exact policy/intelligence binding, power-system selection, algorithm parameters, portfolio and robust-scenario settings, workflow stage state, result/statistics selections, and Live Optimization histories. Previously stored numeric convergence histories are reconstructed into plots; screenshots are not treated as scientific source data.

`LiveOptimizationPanel.load_experiment()` is now wired into historical experiment opening/resume. Lightweight view state—including selected repeated run, plot mode, portfolio preview selection, and workspace—is persisted separately from scientific configuration. Older experiments without a workspace snapshot infer completed setup gates from the fact that an experiment record could only have been created after those stages were applied, while numerical work is still revalidated through existing scientific gates.

## Performance and evaluator diagnostics

Warm-up/calibration messaging now states explicitly that it measures evaluator throughput, not complete CALO end-to-end throughput. CALO results separately record policy inference, candidate-generation, evaluator, learning-update, and control timing so host-control overhead can be diagnosed rather than hidden behind fast evaluator warm-up numbers.

Exact evaluation reuse preserves fairness: every requested candidate still consumes one function-evaluation budget unit. Exact within-batch duplicate decoded states may share one physical solve, while persistent cross-batch caching automatically disables after sufficient evidence if its hit rate is too low to justify lookup overhead.

The frozen v4.0 seeded CALO regression trajectory is preserved. An attempted optimization that changed peer-sampling RNG order was rejected and reverted. Full Torch/CUDA-native CALO cognitive control and elimination of all per-learner host logic are explicitly **not** claimed as solved in v4.1; they remain tracked performance work so scientific behavior is not changed merely for speed.

## Key tensor/state shapes

- Native policy state: `[32]`
- Population / persistent personal memory: `[P, D]`
- Canonical HPEM / hierarchical summary: `[7, D]` / `[4, D]`
- Contextual success directions: `[O, H, D]`
- Contextual operator credit: `[R, O, C]`
- Variable-group intelligence: `[R, G, K]`
- Temporary learner-memory relationships remain bounded/reused rather than retained as historical population tensors.

## Validation

The release is validated by partitioned tests because a single monolithic pytest invocation can exceed the execution wrapper timeout even when individual partitions complete normally:

- Unit: 154 passed
- GUI: 25 passed
- Integration/regression: 9 passed
- Scientific IEEE cross-checks: 4 passed
- Scientific integrity: 16 passed
- Total: **208 passed, 0 failed**
- Ruff: zero findings
- `compileall`: passed

Independent internal/PYPOWER validation passed for IEEE 30, 57, 118, and 300 with zero bus-type mismatches and zero aggregate-Q mismatches. IEEE 118 and 300 required one internal PV Q-limit switching round, as expected under the matched PV-only semantics.

## Remaining limitations carried honestly

The bundled legacy v2 policy is not claimed to be a native-v4.1 ORPD-qualified best policy. AI/rule/online-credit authority still needs paired development-case ablation. CALO control state remains primarily compact NumPy host state rather than fully Torch/CUDA resident, and some per-learner/memory operations remain host-side. A single shared bit-identical policy-training/runtime transition implementation, native device-resident CALO control, indefinite policy lineages, exact mid-run optimizer continuation, run-count extension, and evaluation-horizon extension are intentionally deferred to later work, principally the planned v5.0 continuation architecture.
