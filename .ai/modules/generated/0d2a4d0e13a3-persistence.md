# persistence

<!-- AUTO-GENERATED MODULE INTELLIGENCE -->
**Purpose:** SQLite durable results/provenance plus resume, continuation, integrity and publication retrieval.

**Important state:** Database rows, result artifacts, checkpoints/resume envelopes and continuation contracts.

**Major flow:** execution -> durable result -> query/export; interruption -> resume validation -> restart.

**Constraints/invariants:** Atomic integrity-checked persistence, exact provenance and schema-compatible resume/extension.

**Common failure points:** Migration drift, partial writes, stale paths, corruption and incorrect cumulative reconstruction.

## Start here
- `calo_rpd_studio/results/database.py`

## Preferred edit targets
- `calo_rpd_studio/results/database.py`

## Deep/internal implementation
- None identified.

## State owners
- ResultDatabase

## Cross-module dependencies
- `calo-policy`
- `core`
- `experiments`
- `power-system`

## Dependents
- `bootstrap`
- `compute`
- `desktop`
- `experiments`
- `tests`
- `validation-release`

## Related tests
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
- `tests/unit/test_v5_continuation.py`
- `tests/unit/test_v650_must_resolve.py`

## Files
- `calo_rpd_studio/continuation/AGENTS.md`
- `calo_rpd_studio/continuation/__init__.py`
- `calo_rpd_studio/continuation/experiment_evolution.py`
- `calo_rpd_studio/continuation/runtime_binding.py`
- `calo_rpd_studio/results/AGENTS.md`
- `calo_rpd_studio/results/__init__.py`
- `calo_rpd_studio/results/comparison_engine.py`
- `calo_rpd_studio/results/database.py`
- `calo_rpd_studio/results/integrity_checker.py`
- `calo_rpd_studio/results/publication_export.py`
- `calo_rpd_studio/results/ranking_engine.py`
- `calo_rpd_studio/results/result_store.py`
- `calo_rpd_studio/results/solution_validator.py`
- `calo_rpd_studio/resume/AGENTS.md`
- `calo_rpd_studio/resume/__init__.py`
- `calo_rpd_studio/resume/models.py`
- `calo_rpd_studio/resume/service.py`

> Generated routing is evidence, not auditing. Curated `.ai/modules/*.md` guidance remains authoritative when more specific.
