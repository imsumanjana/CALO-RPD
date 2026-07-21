# CALO-RPD Studio 4.1.0 — Release Validation Record

## Scope

Version 4.1.0 is a policy-governance, scientific-alignment, diagnostics, and reproducibility release built on the validated v4.0 CALO/ORPD core. It introduces a managed Policy Center, explicit policy qualification, immutable experiment-policy binding, native v4.1 policy schemas, adaptive exact-cache diagnostics, and complete historical workspace restoration. It deliberately does **not** claim that a bundled policy is the best ORPD policy, nor that CALO control is fully CUDA/XPU resident.

The common ORPD objective/evaluator, mixed-variable decoder, constraints, requested function-evaluation accounting, IEEE case treatment, independent PYPOWER validation, repeated-run independence, and publication-evidence gates remain common across algorithms.

## Policy qualification and provenance

- Native v4.1 policy state: 32 features (legacy 24 cognitive features plus HPEM occupancy/consensus/readiness, success-memory density, learning-lane fraction, precision activity/radius, and variable-group concentration).
- Explicit runtime/state/action/training schema metadata and SHA-256 checkpoint identity.
- Policy Library supports candidate/qualified/legacy/archived states, grades, active selection, safe archive/delete, and experiment-reference protection.
- Strict experiments store immutable policy ID/path/SHA/schema/determinism/qualification bindings; a missing or mismatched strict policy is not silently substituted.
- Policy Qualification compares Candidate vs selected Reference vs No-AI CALO with paired seeds and equal FE budgets. Primary ranking is median final feasible objective; AUC, feasibility, stability, evaluations-to-feasibility, and runtime are secondary evidence.
- IEEE 118 and IEEE 300 are protected policy-training/qualification holdouts by default unless an explicit non-final override is documented.
- The bundled legacy v2 policy is classified honestly as legacy/unqualified until separately qualified; no native-v4.1 superiority claim is made.

## Training alignment

The lightweight PPO rollout environment now uses the native 32-feature state and v4.1 cognitive components/semantics, including persistent personal memory, HPEM, contextual credit, variable-group intelligence, adaptive epsilon, dual-lane readiness, and functional recovery fraction. It is not represented as a bit-identical duplicate of the complete runtime transition loop; real-optimizer Policy Qualification is mandatory before scientific promotion.

Exact native-v4.1 epoch-level resume stores model/optimizer state and Python/NumPy/Torch/CUDA RNG state with schema metadata. Legacy 24-feature training resumes are rejected for exact native-v4.1 continuation rather than silently reshaped. Indefinite policy lineages and long-horizon best-checkpoint evolution are reserved for the planned v5.0 architecture.

## Historical experiment restoration

- A centralized `ExperimentWorkspaceRestorer` rehydrates authoritative experiment configuration, power-system and algorithm selections, CALO policy/intelligence settings, robust-scenario/portfolio state, workflow gates, and historical results.
- `LiveOptimizationPanel.load_experiment()` is now invoked on historical workspace open/resume, reconstructing stored numeric convergence histories.
- Previously selected live run/plot/preview state and last workspace are persisted as lightweight UI state, separate from scientific configuration.
- Existing experiments without a workspace snapshot infer completed setup stages from the authoritative experiment record while retaining normal scientific validation before new numerical work.

Exact mid-run optimizer continuation, persistent partial-run telemetry, run-count extension, evaluation-horizon extension, and indefinite policy lineage management are intentionally reserved for v5.0.

## Performance diagnostics

- Warm-up/calibration wording explicitly states that it measures evaluator throughput rather than complete CALO end-to-end throughput.
- CALO result metadata records policy, candidate-generation, evaluator, learning-update, and control timing for end-to-end diagnosis.
- Exact within-batch decoded-control deduplication preserves requested FE accounting. Persistent cross-batch exact caching can disable itself after sufficient evidence when hit rate is too low to justify overhead.
- The frozen seeded CALO regression behavior is preserved; an attempted host optimization that changed peer RNG ordering was rejected/reverted.
- Full Torch/CUDA-native CALO cognitive/control execution is **not** claimed as solved in v4.1. The common numerical evaluator remains accelerator-capable; host-side CALO control migration is deferred pending strict parity evidence.

## Test evidence

Validation was executed in isolated partitions because a single monolithic pytest invocation may exceed the execution-wrapper timeout even when partitions complete cleanly:

- Unit: **154 passed, 0 failed**
- GUI (Qt offscreen): **25 passed, 0 failed**
- Integration + regression: **9 passed, 0 failed**
- Scientific IEEE cross-checks: **4 passed, 0 failed**
- Scientific integrity: **16 passed, 0 failed**
- **Total: 208 passed, 0 failed**
- Ruff: **passed with zero findings**
- `compileall`: **passed**
- CALO v4.1 frozen manifest: **95 files verified**

## IEEE base-case and independent PYPOWER validation

| Case | Internal PF | Q-limit rounds | Max |ΔV| p.u. | Max |Δangle| deg | |Δloss| MW | Bus-type mismatches | Aggregate-Q mismatches |
|---|---|---:|---:|---:|---:|---:|---:|
| IEEE 30 | PASS | 0 | 6.66e-16 | 5.68e-14 | 5.91e-14 | 0 | 0 |
| IEEE 57 | PASS | 0 | 2.11e-15 | 8.53e-14 | 2.38e-13 | 0 | 0 |
| IEEE 118 | PASS | 1 | 8.88e-16 | 1.71e-13 | 8.36e-12 | 0 | 0 |
| IEEE 300 | PASS | 1 | 3.09e-14 | 3.13e-12 | 1.67e-11 | 0 | 0 |

These are solver-agreement checks, not optimizer-superiority claims.

## Audited unresolved/deferred items

The v4.1 dispute register is exposed in the CALO Intelligence workspace and stored in `FINDINGS_CLOSURE_v4.1.0.csv`. Important items not falsely closed include:

- full shared bit-identical training/runtime transition implementation: partial;
- bundled legacy policy native-v4.1 ORPD qualification: open;
- AI/rule/credit authority calibration: open;
- fully Torch/CUDA-native CALO cognitive/control path: deferred;
- remaining host-side per-learner/memory loops and policy GPU→host actions: partial/open;
- exact mid-run visual/run continuation: deferred to v5.0.

## Packaging/hardware statement

The final release is packaged as version 4.1.0. Physical NVIDIA CUDA and Intel XPU hardware were not available in this validation environment, so no fabricated physical-utilization claim is made. The numerical backend retains CUDA → XPU → CPU preference where supported; actual end-to-end CALO accelerator utilization must be measured on the target workstation.

## Release gate conclusion

**PASS for CALO-RPD Studio v4.1.0 policy-governance/reproducibility research release.**

This PASS means the implemented v4.1 changes, regression partitions, scientific base-case validation, policy-binding safeguards, historical restoration, lint/compile checks, and release integrity passed. It does **not** establish universal CALO superiority or certify any unqualified policy as best. Publication claims remain gated by frozen, paired, independently validated multi-case evidence.
