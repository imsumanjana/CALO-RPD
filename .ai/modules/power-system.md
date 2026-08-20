# power-system

**Purpose:** Power-system models, AC power flow, ORPD decoding, objectives, constraints and robustness semantics.

**Important state:** Case/formulation/scenario and solver state; protected-case identity is a scientific boundary.

**Major flow:** case/config -> formulation -> candidate decode -> power flow -> objectives/constraints -> result.

**Constraints/invariants:** Reference semantics, protected-case isolation, formulation fingerprints and tolerance consistency.

**Common failure points:** Convergence, units/tolerances, case identity, decoding and constraint inconsistencies.

## Primary files
- `calo_rpd_studio/power_system/case_model.py`
- `calo_rpd_studio/orpd/mathematical_reference.py`
- `calo_rpd_studio/orpd/problem.py`
- `calo_rpd_studio/power_system/pglib_import.py`
- `calo_rpd_studio/orpd/variable_decoder.py`
- `calo_rpd_studio/power_system/newton_raphson.py`
- `calo_rpd_studio/orpd/external_profile.py`
- `calo_rpd_studio/power_system/ac_power_flow.py`
- `calo_rpd_studio/orpd/constraints.py`
- `calo_rpd_studio/power_system/independent_validator.py`
- `calo_rpd_studio/robustness/robust_objectives.py`
- `calo_rpd_studio/orpd/objectives.py`

## Important public/entry symbols
- `ConstraintViolation` — `calo_rpd_studio/orpd/constraint_violation.py:8-16`
- `ConstraintToleranceConfig` — `calo_rpd_studio/orpd/constraints.py:15-49`
- `VariableKind` — `calo_rpd_studio/orpd/decision_variables.py:8-10`
- `DecisionVariable` — `calo_rpd_studio/orpd/decision_variables.py:14-19`
- `ReviewedORPDProfileError` — `calo_rpd_studio/orpd/external_profile.py:63-64`
- `ReviewedORPDProfile` — `calo_rpd_studio/orpd/external_profile.py:68-181`
- `SLSQPReferenceOptions` — `calo_rpd_studio/orpd/mathematical_reference.py:54-76`
- `IndependentScenarioValidation` — `calo_rpd_studio/orpd/mathematical_reference.py:80-90`
- `ReferencePoint` — `calo_rpd_studio/orpd/mathematical_reference.py:94-105`
- `SolverAccounting` — `calo_rpd_studio/orpd/mathematical_reference.py:109-116`
- `MathematicalReferenceReport` — `calo_rpd_studio/orpd/mathematical_reference.py:120-153`
- `_EvaluationCache` — `calo_rpd_studio/orpd/mathematical_reference.py:156-173`
- `ObjectiveKind` — `calo_rpd_studio/orpd/objectives.py:12-16`
- `ObjectiveConfig` — `calo_rpd_studio/orpd/objectives.py:20-64`
- `ObjectiveResult` — `calo_rpd_studio/orpd/objectives.py:68-70`
- `ORPDProblemConfig` — `calo_rpd_studio/orpd/problem.py:29-43`
- `Evaluation` — `calo_rpd_studio/orpd/problem.py:47-55`
- `ScenarioEvaluationContext` — `calo_rpd_studio/orpd/problem.py:59-65`

## Dependencies
- None confirmed

## Dependents
- `bootstrap`, `calo-policy`, `compute`, `desktop`, `experiments`, `optimization`, `persistence`, `tests`, `validation-release`

## Associated tests
- `tests/conftest.py`
- `tests/regression/test_seed_reproducibility.py`
- `tests/scientific/test_ieee_cases.py`
- `tests/scientific/test_v34_scientific_integrity.py`
- `tests/unit/test_algorithms.py`
- `tests/unit/test_calo.py`
- `tests/unit/test_calo_core_v2.py`
- `tests/unit/test_calo_v4.py`
- `tests/unit/test_calo_v41_runtime_guards.py`
- `tests/unit/test_cma_es.py`
- `tests/unit/test_lshade.py`
- `tests/unit/test_mathematical_reference.py`
- `tests/unit/test_orpd.py`
- `tests/unit/test_orpd_counted_evaluation_context.py`
- `tests/unit/test_pglib_import_and_reviewed_orpd_profile.py`

## Retrieval note
Use the machine indexes for exact locations and confidence-labelled relationships; authored invariants live in `../ARCHITECTURE.md` and `../DECISIONS.md`.
