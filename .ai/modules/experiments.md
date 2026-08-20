# experiments

**Purpose:** Experiment configuration, plans, budgets, deterministic seeds, fairness and runner orchestration.

**Important state:** Validated configuration, execution plan, seed allocation, progress and provenance.

**Major flow:** validated config -> plan -> scheduler/optimizer/evaluator -> result contracts/provenance.

**Constraints/invariants:** Fair equal budgets, exact accounting, deterministic seeds and immutable policy binding.

**Common failure points:** Configuration drift, unfair budgets, resume mismatch and provenance mismatch.

## Primary files
- `calo_rpd_studio/experiments/execution_plans.py`
- `calo_rpd_studio/experiments/experiment_config.py`
- `calo_rpd_studio/experiments/experiment_runner.py`
- `calo_rpd_studio/benchmarking/suite.py`
- `calo_rpd_studio/experiments/study_strength.py`
- `calo_rpd_studio/benchmarking/campaign.py`
- `calo_rpd_studio/benchmarking/freeze.py`
- `calo_rpd_studio/benchmarking/evidence.py`
- `calo_rpd_studio/experiments/execution_plan.py`
- `calo_rpd_studio/experiments/provenance.py`
- `calo_rpd_studio/experiments/calo_ablation.py`
- `calo_rpd_studio/experiments/result_contracts.py`

## Important public/entry symbols
- `BenchmarkCampaignConfig` — `calo_rpd_studio/benchmarking/campaign.py:27-139`
- `BenchmarkTask` — `calo_rpd_studio/benchmarking/campaign.py:143-154`
- `CampaignEvidence` — `calo_rpd_studio/benchmarking/evidence.py:21-31`
- `FreezeVerification` — `calo_rpd_studio/benchmarking/freeze.py:175-198`
- `ScientificEvidencePackageBuilder` — `calo_rpd_studio/benchmarking/package.py:22-257`
- `BenchmarkStudy` — `calo_rpd_studio/benchmarking/suite.py:15-19`
- `BenchmarkSuite` — `calo_rpd_studio/benchmarking/suite.py:23-42`
- `AblationSpec` — `calo_rpd_studio/experiments/calo_ablation.py:26-29`
- `BudgetPolicy` — `calo_rpd_studio/experiments/evaluation_budget.py:7-10`
- `EvaluationBudget` — `calo_rpd_studio/experiments/evaluation_budget.py:14-25`
- `PlannedItem` — `calo_rpd_studio/experiments/execution_plan.py:11-17`
- `ControllerKind` — `calo_rpd_studio/experiments/execution_plans.py:29-32`
- `ExecutionPlanKind` — `calo_rpd_studio/experiments/execution_plans.py:35-37`
- `ExecutionLifecycle` — `calo_rpd_studio/experiments/execution_plans.py:40-52`
- `AlgorithmStage` — `calo_rpd_studio/experiments/execution_plans.py:179-259`
- `WorkspaceStudyPlan` — `calo_rpd_studio/experiments/execution_plans.py:303-482`
- `IndividualExperimentPlan` — `calo_rpd_studio/experiments/execution_plans.py:486-536`
- `RobustScenarioSettings` — `calo_rpd_studio/experiments/experiment_config.py:145-204`

## Dependencies
- `compute`, `core`, `optimization`, `persistence`, `power-system`

## Dependents
- `bootstrap`, `calo-policy`, `compute`, `core`, `desktop`, `persistence`, `tests`, `validation-release`

## Associated tests
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

## Retrieval note
Use the machine indexes for exact locations and confidence-labelled relationships; authored invariants live in `../ARCHITECTURE.md` and `../DECISIONS.md`.
