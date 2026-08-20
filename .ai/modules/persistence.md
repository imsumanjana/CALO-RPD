# persistence

**Purpose:** SQLite durable results/provenance plus resume, continuation, integrity and publication retrieval.

**Important state:** Database rows, result artifacts, checkpoints/resume envelopes and continuation contracts.

**Major flow:** execution -> durable result -> query/export; interruption -> resume validation -> restart.

**Constraints/invariants:** Atomic integrity-checked persistence, exact provenance and schema-compatible resume/extension.

**Common failure points:** Migration drift, partial writes, stale paths, corruption and incorrect cumulative reconstruction.

## Primary files
- `calo_rpd_studio/results/database.py`
- `calo_rpd_studio/resume/service.py`
- `calo_rpd_studio/continuation/experiment_evolution.py`
- `calo_rpd_studio/results/publication_export.py`
- `calo_rpd_studio/results/result_store.py`
- `calo_rpd_studio/resume/models.py`
- `calo_rpd_studio/results/comparison_engine.py`
- `calo_rpd_studio/continuation/runtime_binding.py`
- `calo_rpd_studio/results/integrity_checker.py`
- `calo_rpd_studio/results/ranking_engine.py`
- `calo_rpd_studio/results/solution_validator.py`
- `calo_rpd_studio/continuation/AGENTS.md`

## Important public/entry symbols
- `ExtensionProtocol` — `calo_rpd_studio/continuation/experiment_evolution.py:12-19`
- `ExtensionPlan` — `calo_rpd_studio/continuation/experiment_evolution.py:23-31`
- `ExperimentEvolutionService` — `calo_rpd_studio/continuation/experiment_evolution.py:34-258`
- `ResultDatabase` — `calo_rpd_studio/results/database.py:45-4306`
- `PublicationExportCancelled` — `calo_rpd_studio/results/publication_export.py:15-16`
- `PublicationExporter` — `calo_rpd_studio/results/publication_export.py:19-227`
- `ResultStore` — `calo_rpd_studio/results/result_store.py:11-62`
- `ResumeTaskType` — `calo_rpd_studio/resume/models.py:9-13`
- `ResumeStatus` — `calo_rpd_studio/resume/models.py:16-25`
- `ResumeItem` — `calo_rpd_studio/resume/models.py:29-42`
- `ResumeService` — `calo_rpd_studio/resume/service.py:20-142`
- `bind_exact_run_checkpoint` — `calo_rpd_studio/continuation/runtime_binding.py:14-56`
- `summarize_runs` — `calo_rpd_studio/results/comparison_engine.py:8-55`
- `interpret_comparison` — `calo_rpd_studio/results/comparison_engine.py:58-67`
- `_sha256_file` — `calo_rpd_studio/results/database.py:25-30`
- `_canonical_sha256` — `calo_rpd_studio/results/database.py:33-42`
- `check_run_record` — `calo_rpd_studio/results/integrity_checker.py:7-16`
- `rank_summary` — `calo_rpd_studio/results/ranking_engine.py:6-9`

## Dependencies
- `calo-policy`, `core`, `experiments`, `power-system`

## Dependents
- `bootstrap`, `compute`, `desktop`, `experiments`, `tests`, `validation-release`

## Associated tests
- `tests/integration/test_database_workflow.py`
- `tests/integration/test_historical_learning.py`
- `tests/integration/test_history_deletion.py`
- `tests/integration/test_workspace_execution_control.py`
- `tests/unit/test_calo_v41_policy_system.py`
- `tests/unit/test_tsh_calo_inference.py`
- `tests/unit/test_tsh_calo_optimizer.py`
- `tests/unit/test_tsh_calo_policy_lifecycle.py`
- `tests/unit/test_v120_phase4_development_freeze.py`
- `tests/unit/test_v120_phase4_policy_retirement.py`
- `tests/unit/test_v32_portfolio_resume.py`
- `tests/unit/test_v343_export_completion.py`
- `tests/unit/test_v34_release_blockers.py`
- `tests/unit/test_v57_audit_closure_integration.py`
- `tests/unit/test_v57_previous_audit_closure.py`

## Retrieval note
Use the machine indexes for exact locations and confidence-labelled relationships; authored invariants live in `../ARCHITECTURE.md` and `../DECISIONS.md`.
