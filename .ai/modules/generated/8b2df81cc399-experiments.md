# experiments

<!-- AUTO-GENERATED MODULE INTELLIGENCE -->
**Purpose:** Experiment configuration, plans, budgets, deterministic seeds, fairness and runner orchestration.

**Important state:** Validated configuration, execution plan, seed allocation, progress and provenance.

**Major flow:** validated config -> plan -> scheduler/optimizer/evaluator -> result contracts/provenance.

**Constraints/invariants:** Fair equal budgets, exact accounting, deterministic seeds and immutable policy binding.

**Common failure points:** Configuration drift, unfair budgets, resume mismatch and provenance mismatch.

## Start here
- No public/entry surface identified automatically; use curated module guidance.

## Preferred edit targets
- `calo_rpd_studio/experiments/experiment_runner.py`

## Deep/internal implementation
- None identified.

## State owners
- None explicitly identified.

## Cross-module dependencies
- `compute`
- `core`
- `optimization`
- `persistence`
- `power-system`

## Dependents
- `bootstrap`
- `calo-policy`
- `compute`
- `core`
- `desktop`
- `persistence`
- `tests`
- `validation-release`

## Related tests
- `tests/gui/test_phase6_ribbon_workspace.py`
- `tests/gui/test_workspace_execution_ui.py`
- `tests/integration/test_database_workflow.py`
- `tests/integration/test_historical_learning.py`
- `tests/integration/test_history_deletion.py`
- `tests/integration/test_phase4_empty_policy_workflow.py`
- `tests/integration/test_workspace_execution_control.py`
- `tests/regression/test_seed_reproducibility.py`
- `tests/scientific/test_v34_scientific_integrity.py`
- `tests/unit/test_bulk_validation.py`
- `tests/unit/test_calo_v41_policy_system.py`
- `tests/unit/test_calo_v41_workflow_restore.py`
- `tests/unit/test_config.py`
- `tests/unit/test_cuda_residency_contract.py`
- `tests/unit/test_execution_plan.py`
- `tests/unit/test_mathematical_reference.py`
- `tests/unit/test_prerequisites_and_resources.py`
- `tests/unit/test_study_strength.py`
- `tests/unit/test_tsh_calo_inference.py`
- `tests/unit/test_tsh_calo_optimizer.py`
- `tests/unit/test_tsh_calo_policy_lifecycle.py`
- `tests/unit/test_v120_phase1_contracts.py`
- `tests/unit/test_v120_phase2_contracts.py`
- `tests/unit/test_v2_benchmarking.py`
- `tests/unit/test_v31_batched_throughput.py`
- `tests/unit/test_v32_portfolio_resume.py`
- `tests/unit/test_v33_cuda_resident.py`
- `tests/unit/test_v343_export_completion.py`
- `tests/unit/test_v34_release_blockers.py`
- `tests/unit/test_v541_release_integrity.py`
- `tests/unit/test_v560_release_integrity.py`
- `tests/unit/test_v570_release_integrity.py`
- `tests/unit/test_v57_previous_audit_closure.py`
- `tests/unit/test_v590_scientific_closure.py`
- `tests/unit/test_v5_continuation.py`
- `tests/unit/test_v600_release_integrity.py`
- `tests/unit/test_v610_release_integrity.py`
- `tests/unit/test_v620_release_integrity.py`
- `tests/unit/test_v621_release_integrity.py`
- `tests/unit/test_v630_release_integrity.py`

## Files
- `calo_rpd_studio/benchmarking/AGENTS.md`
- `calo_rpd_studio/benchmarking/__init__.py`
- `calo_rpd_studio/benchmarking/campaign.py`
- `calo_rpd_studio/benchmarking/evidence.py`
- `calo_rpd_studio/benchmarking/freeze.py`
- `calo_rpd_studio/benchmarking/package.py`
- `calo_rpd_studio/benchmarking/suite.py`
- `calo_rpd_studio/benchmarking/validation.py`
- `calo_rpd_studio/experiments/AGENTS.md`
- `calo_rpd_studio/experiments/__init__.py`
- `calo_rpd_studio/experiments/calo_ablation.py`
- `calo_rpd_studio/experiments/evaluation_budget.py`
- `calo_rpd_studio/experiments/execution_plan.py`
- `calo_rpd_studio/experiments/execution_plans.py`
- `calo_rpd_studio/experiments/experiment_config.py`
- `calo_rpd_studio/experiments/experiment_runner.py`
- `calo_rpd_studio/experiments/fairness_validator.py`
- `calo_rpd_studio/experiments/parallel_runner.py`
- `calo_rpd_studio/experiments/provenance.py`
- `calo_rpd_studio/experiments/result_contracts.py`
- `calo_rpd_studio/experiments/seed_manager.py`
- `calo_rpd_studio/experiments/study_strength.py`

> Generated routing is evidence, not auditing. Curated `.ai/modules/*.md` guidance remains authoritative when more specific.
