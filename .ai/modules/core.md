# core

**Purpose:** Cross-cutting CALO-RPD package utilities not owned by a narrower subsystem.

**Important state:** Varies by file; use exact symbol/dependency retrieval before assuming ownership.

**Major flow:** Cross-cutting support for narrower application/scientific subsystems.

**Constraints/invariants:** Follow the nearest AGENTS.md; avoid new global state without explicit ownership.

**Common failure points:** Hidden coupling and unclear ownership; inspect dependents before shared-helper changes.

## Primary files
- `calo_rpd_studio/visualization/publication_evidence.py`
- `calo_rpd_studio/statistics/paired.py`
- `calo_rpd_studio/visualization/plot_manager.py`
- `calo_rpd_studio/version.py`
- `calo_rpd_studio/assistant/ollama_parameter_advisor.py`
- `calo_rpd_studio/reports/report_builder.py`
- `calo_rpd_studio/visualization/font_preflight.py`
- `calo_rpd_studio/visualization/plot_style.py`
- `calo_rpd_studio/visualization/network_plots.py`
- `calo_rpd_studio/visualization/statistical_plots.py`
- `calo_rpd_studio/reports/experiment_report.py`
- `calo_rpd_studio/reports/reproducibility_bundle.py`

## Important public/entry symbols
- `LocalAssistantConfig` — `calo_rpd_studio/assistant/ollama_parameter_advisor.py:22-40`
- `LocalAssistantResponse` — `calo_rpd_studio/assistant/ollama_parameter_advisor.py:44-60`
- `OllamaParameterAdvisor` — `calo_rpd_studio/assistant/ollama_parameter_advisor.py:63-120`
- `ReportBuilder` — `calo_rpd_studio/reports/report_builder.py:6-25`
- `PairIntegrityError` — `calo_rpd_studio/statistics/paired.py:26-27`
- `ExactPair` — `calo_rpd_studio/statistics/paired.py:31-36`
- `PlotRecord` — `calo_rpd_studio/visualization/plot_manager.py:16-22`
- `PlotManager` — `calo_rpd_studio/visualization/plot_manager.py:25-330`
- `PlotStyle` — `calo_rpd_studio/visualization/plot_style.py:10-69`
- `_canonical_sha256` — `calo_rpd_studio/assistant/ollama_parameter_advisor.py:16-18`
- `build_experiment_report` — `calo_rpd_studio/reports/experiment_report.py:7-17`
- `create_bundle` — `calo_rpd_studio/reports/reproducibility_bundle.py:7-12`
- `build_validation_report` — `calo_rpd_studio/reports/validation_report.py:7-10`
- `mean_confidence_interval` — `calo_rpd_studio/statistics/confidence_intervals.py:7-16`
- `descriptive_statistics` — `calo_rpd_studio/statistics/descriptive.py:8-29`
- `cliffs_delta` — `calo_rpd_studio/statistics/effect_sizes.py:6-16`
- `friedman_test` — `calo_rpd_studio/statistics/friedman.py:11-29`
- `_record_key` — `calo_rpd_studio/statistics/paired.py:39-48`

## Dependencies
- `experiments`, `optimization`

## Dependents
- `bootstrap`, `calo-policy`, `compute`, `desktop`, `experiments`, `optimization`, `persistence`, `tests`, `validation-release`

## Associated tests
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

## Retrieval note
Use the machine indexes for exact locations and confidence-labelled relationships; authored invariants live in `../ARCHITECTURE.md` and `../DECISIONS.md`.
