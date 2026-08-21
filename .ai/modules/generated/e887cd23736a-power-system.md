# power-system

<!-- AUTO-GENERATED MODULE INTELLIGENCE -->
**Purpose:** Power-system models, AC power flow, ORPD decoding, objectives, constraints and robustness semantics.

**Important state:** Case/formulation/scenario and solver state; protected-case identity is a scientific boundary.

**Major flow:** case/config -> formulation -> candidate decode -> power flow -> objectives/constraints -> result.

**Constraints/invariants:** Reference semantics, protected-case isolation, formulation fingerprints and tolerance consistency.

**Common failure points:** Convergence, units/tolerances, case identity, decoding and constraint inconsistencies.

## Start here
- `calo_rpd_studio/orpd/problem.py`

## Preferred edit targets
- `calo_rpd_studio/orpd/problem.py`

## Deep/internal implementation
- None identified.

## State owners
- ORPDProblem

## Cross-module dependencies
- None confirmed.

## Dependents
- `bootstrap`
- `calo-policy`
- `compute`
- `desktop`
- `experiments`
- `optimization`
- `persistence`
- `tests`
- `validation-release`

## Related tests
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
- `tests/unit/test_power_flow.py`
- `tests/unit/test_robustness.py`
- `tests/unit/test_tsh_calo_generalization_guard.py`
- `tests/unit/test_tsh_calo_hierarchical_policy.py`
- `tests/unit/test_tsh_calo_independent_training.py`
- `tests/unit/test_tsh_calo_inference.py`
- `tests/unit/test_tsh_calo_optimizer.py`
- `tests/unit/test_tsh_calo_physics_repair.py`
- `tests/unit/test_tsh_calo_runtime_context.py`
- `tests/unit/test_tsh_calo_topology_context.py`
- `tests/unit/test_tsh_calo_training_campaign.py`
- `tests/unit/test_tsh_calo_training_environment.py`
- `tests/unit/test_tsh_calo_training_session.py`
- `tests/unit/test_tsh_calo_uncertainty_shield.py`
- `tests/unit/test_v120_phase2_contracts.py`
- `tests/unit/test_v2_benchmarking.py`
- `tests/unit/test_v33_cuda_resident.py`
- `tests/unit/test_v34_release_blockers.py`
- `tests/unit/test_v3_accelerated_backend.py`
- `tests/unit/test_v57_audit_closure_integration.py`
- `tests/unit/test_v57_previous_audit_closure.py`
- `tests/unit/test_v580_audit_closure.py`
- `tests/unit/test_v590_scientific_closure.py`
- `tests/unit/test_v5_continuation.py`
- `tests/unit/test_v640_stage_b.py`
- `tests/unit/test_v650_must_resolve.py`

## Files
- `calo_rpd_studio/orpd/AGENTS.md`
- `calo_rpd_studio/orpd/__init__.py`
- `calo_rpd_studio/orpd/constraint_violation.py`
- `calo_rpd_studio/orpd/constraints.py`
- `calo_rpd_studio/orpd/decision_variables.py`
- `calo_rpd_studio/orpd/external_profile.py`
- `calo_rpd_studio/orpd/feasibility_rules.py`
- `calo_rpd_studio/orpd/formulation_fingerprint.py`
- `calo_rpd_studio/orpd/mathematical_reference.py`
- `calo_rpd_studio/orpd/mixed_variable_handler.py`
- `calo_rpd_studio/orpd/objectives.py`
- `calo_rpd_studio/orpd/problem.py`
- `calo_rpd_studio/orpd/variable_decoder.py`
- `calo_rpd_studio/power_system/AGENTS.md`
- `calo_rpd_studio/power_system/__init__.py`
- `calo_rpd_studio/power_system/ac_power_flow.py`
- `calo_rpd_studio/power_system/branch_flows.py`
- `calo_rpd_studio/power_system/case_identity.py`
- `calo_rpd_studio/power_system/case_loader.py`
- `calo_rpd_studio/power_system/case_model.py`
- `calo_rpd_studio/power_system/case_validation.py`
- `calo_rpd_studio/power_system/independent_validator.py`
- `calo_rpd_studio/power_system/network_metrics.py`
- `calo_rpd_studio/power_system/newton_raphson.py`
- `calo_rpd_studio/power_system/pglib_import.py`
- `calo_rpd_studio/power_system/pv_pq_switching.py`
- `calo_rpd_studio/power_system/voltage_stability.py`
- `calo_rpd_studio/power_system/ybus.py`
- `calo_rpd_studio/robustness/AGENTS.md`
- `calo_rpd_studio/robustness/__init__.py`
- `calo_rpd_studio/robustness/contingencies.py`
- `calo_rpd_studio/robustness/cvar.py`
- `calo_rpd_studio/robustness/load_uncertainty.py`
- `calo_rpd_studio/robustness/monte_carlo.py`
- `calo_rpd_studio/robustness/renewable_uncertainty.py`
- `calo_rpd_studio/robustness/robust_objectives.py`
- `calo_rpd_studio/robustness/scenario.py`
- `calo_rpd_studio/robustness/scenario_generator.py`

> Generated routing is evidence, not auditing. Curated `.ai/modules/*.md` guidance remains authoritative when more specific.
