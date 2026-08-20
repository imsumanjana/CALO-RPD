# bootstrap

**Purpose:** Native setup, dependency repair and application launch without coupling ordinary launch to installation.

**Important state:** Environment/prerequisite and launch state; not scientific study state.

**Major flow:** bootstrap/launcher -> prerequisite checks -> calo_rpd_studio.app.application.

**Constraints/invariants:** Ordinary launch must not install packages or perform policy lifecycle work.

**Common failure points:** Missing/incompatible prerequisites, platform launch paths and setup/launch boundary regressions.

## Primary files
- `calo_bootstrap/prerequisites.py`
- `calo_bootstrap/wizard.py`
- `calo_rpd_studio/scripts/run_mathematical_reference.py`
- `calo_bootstrap/launcher.py`
- `calo_rpd_studio/algorithms/calo/run_checkpoint.py`
- `calo_rpd_studio/scripts/run_final_benchmark.py`
- `calo_rpd_studio/scripts/run_benchmark.py`
- `Launch-CALO-RPD.ps1`
- `bootstrap.py`
- `calo_bootstrap/AGENTS.md`
- `calo_bootstrap/__init__.py`
- `run_linux.sh`

## Important public/entry symbols
- `NvidiaInfo` — `calo_bootstrap/prerequisites.py:98-103`
- `TorchInfo` — `calo_bootstrap/prerequisites.py:107-115`
- `EnvironmentReport` — `calo_bootstrap/prerequisites.py:119-137`
- `InstallProgress` — `calo_bootstrap/prerequisites.py:141-160`
- `PrerequisiteWizard` — `calo_bootstrap/wizard.py:23-381`
- `accelerator_repair_required` — `calo_bootstrap/launcher.py:16-19`
- `ensure_prerequisites` — `calo_bootstrap/launcher.py:22-37`
- `main` — `calo_bootstrap/launcher.py:40-50`
- `_run` — `calo_bootstrap/prerequisites.py:163-174`
- `_distribution_version` — `calo_bootstrap/prerequisites.py:177-181`
- `_last_json_dict` — `calo_bootstrap/prerequisites.py:184-192`
- `_core_requirement_by_label` — `calo_bootstrap/prerequisites.py:195-215`
- `detect_nvidia` — `calo_bootstrap/prerequisites.py:218-239`
- `detect_core_import_errors` — `calo_bootstrap/prerequisites.py:242-267`
- `_repair_core_import_errors` — `calo_bootstrap/prerequisites.py:270-299`
- `project_torch_requirement` — `calo_bootstrap/prerequisites.py:302-327`
- `_numeric_version_tuple` — `calo_bootstrap/prerequisites.py:330-334`
- `torch_version_satisfies_requirement` — `calo_bootstrap/prerequisites.py:337-363`

## Dependencies
- `calo-policy`, `compute`, `core`, `desktop`, `experiments`, `persistence`, `power-system`

## Dependents
- `calo-policy`, `tests`

## Associated tests
- `tests/unit/test_mathematical_reference.py`
- `tests/unit/test_prerequisites_and_resources.py`
- `tests/unit/test_v120_phase2_contracts.py`
- `tests/unit/test_v650_must_resolve.py`
- `tests/unit/test_v680_policy_independence.py`

## Retrieval note
Use the machine indexes for exact locations and confidence-labelled relationships; authored invariants live in `../ARCHITECTURE.md` and `../DECISIONS.md`.
