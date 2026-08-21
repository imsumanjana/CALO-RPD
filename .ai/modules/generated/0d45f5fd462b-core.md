# core

<!-- AUTO-GENERATED MODULE INTELLIGENCE -->
**Purpose:** Cross-cutting CALO-RPD package utilities not owned by a narrower subsystem.

**Important state:** Varies by file; use exact symbol/dependency retrieval before assuming ownership.

**Major flow:** Cross-cutting support for narrower application/scientific subsystems.

**Constraints/invariants:** Follow the nearest AGENTS.md; avoid new global state without explicit ownership.

**Common failure points:** Hidden coupling and unclear ownership; inspect dependents before shared-helper changes.

## Start here
- No public/entry surface identified automatically; use curated module guidance.

## Preferred edit targets
- None explicitly identified; consult `.ai/architectural-semantics.json`.

## Deep/internal implementation
- None identified.

## State owners
- None explicitly identified.

## Cross-module dependencies
- `experiments`
- `optimization`

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
- `tests/unit/test_local_parameter_assistant.py`
- `tests/unit/test_packaged_gui_validator.py`
- `tests/unit/test_phase6_command_and_native_contracts.py`
- `tests/unit/test_plot_export.py`
- `tests/unit/test_statistics.py`
- `tests/unit/test_tsh_calo_automatic_qualification.py`
- `tests/unit/test_v120_phase1_contracts.py`
- `tests/unit/test_v541_release_integrity.py`
- `tests/unit/test_v560_release_integrity.py`
- `tests/unit/test_v570_release_integrity.py`
- `tests/unit/test_v57_previous_audit_closure.py`
- `tests/unit/test_v600_release_integrity.py`
- `tests/unit/test_v610_release_integrity.py`
- `tests/unit/test_v620_release_integrity.py`
- `tests/unit/test_v621_release_integrity.py`
- `tests/unit/test_v630_release_integrity.py`
- `tests/unit/test_v640_release_integrity.py`
- `tests/unit/test_v650_release_integrity.py`
- `tests/unit/test_v660_release_integrity.py`
- `tests/unit/test_v660_remaining_audit.py`
- `tests/unit/test_v670_release_integrity.py`
- `tests/unit/test_v680_release_integrity.py`
- `tests/unit/test_v690_release_integrity.py`

## Files
- `calo_rpd_studio/AGENTS.md`
- `calo_rpd_studio/__init__.py`
- `calo_rpd_studio/assistant/__init__.py`
- `calo_rpd_studio/assistant/ollama_parameter_advisor.py`
- `calo_rpd_studio/data/AGENTS.md`
- `calo_rpd_studio/data/__init__.py`
- `calo_rpd_studio/data/cases/AGENTS.md`
- `calo_rpd_studio/data/cases/__init__.py`
- `calo_rpd_studio/data/examples/AGENTS.md`
- `calo_rpd_studio/data/examples/__init__.py`
- `calo_rpd_studio/data/examples/case118_robust_mean_risk.yaml`
- `calo_rpd_studio/data/examples/case30_loss_comparison.yaml`
- `calo_rpd_studio/data/examples/policy_training_active_loss.yaml`
- `calo_rpd_studio/data/schemas/AGENTS.md`
- `calo_rpd_studio/data/schemas/__init__.py`
- `calo_rpd_studio/data/schemas/experiment_config.schema.json`
- `calo_rpd_studio/data/schemas/partial_run_failure_v2.schema.json`
- `calo_rpd_studio/data/schemas/runtime_execution_resolution_v2.schema.json`
- `calo_rpd_studio/reports/AGENTS.md`
- `calo_rpd_studio/reports/__init__.py`
- `calo_rpd_studio/reports/experiment_report.py`
- `calo_rpd_studio/reports/report_builder.py`
- `calo_rpd_studio/reports/reproducibility_bundle.py`
- `calo_rpd_studio/reports/validation_report.py`
- `calo_rpd_studio/statistics/AGENTS.md`
- `calo_rpd_studio/statistics/__init__.py`
- `calo_rpd_studio/statistics/confidence_intervals.py`
- `calo_rpd_studio/statistics/descriptive.py`
- `calo_rpd_studio/statistics/effect_sizes.py`
- `calo_rpd_studio/statistics/friedman.py`
- `calo_rpd_studio/statistics/paired.py`
- `calo_rpd_studio/statistics/posthoc.py`
- `calo_rpd_studio/statistics/rankings.py`
- `calo_rpd_studio/statistics/statistical_report.py`
- `calo_rpd_studio/statistics/wilcoxon.py`
- `calo_rpd_studio/version.py`
- `calo_rpd_studio/visualization/AGENTS.md`
- `calo_rpd_studio/visualization/__init__.py`
- `calo_rpd_studio/visualization/convergence_plots.py`
- `calo_rpd_studio/visualization/export.py`
- `calo_rpd_studio/visualization/font_preflight.py`
- `calo_rpd_studio/visualization/network_plots.py`
- `calo_rpd_studio/visualization/plot_manager.py`
- `calo_rpd_studio/visualization/plot_style.py`
- `calo_rpd_studio/visualization/publication_evidence.py`
- `calo_rpd_studio/visualization/robustness_plots.py`
- `calo_rpd_studio/visualization/statistical_plots.py`

> Generated routing is evidence, not auditing. Curated `.ai/modules/*.md` guidance remains authoritative when more specific.
