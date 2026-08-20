# validation-release

**Purpose:** Developer/release validators, CI, packaging, containers and evidence-generation tooling.

**Important state:** Evidence bound to exact source/artifact identities.

**Major flow:** source -> non-mutating checks/build stages -> evidence/manifests -> separately authorized gates.

**Constraints/invariants:** Pinned actions/locks, no fabricated attestations, historical evidence immutable and scoped.

**Common failure points:** Stale artifacts, unpinned dependencies/actions, identity mismatch and accidental claim elevation.

## Primary files
- `calo_rpd_studio/scripts/create_development_freeze_candidate.py`
- `calo_rpd_studio/scripts/finalize_release_records.py`
- `calo_rpd_studio/scripts/validate_phase3_workspace_accessibility.py`
- `calo_rpd_studio/scripts/accept_development_freeze.py`
- `calo_rpd_studio/scripts/validate_resource_recovery.py`
- `calo_rpd_studio/scripts/release_policy_scope.py`
- `calo_rpd_studio/scripts/validate_tsh_calo_device_equivalence.py`
- `calo_rpd_studio/scripts/create_release_preparation.py`
- `calo_rpd_studio/scripts/validate_packaged_gui.py`
- `calo_rpd_studio/scripts/_train_tsh_calo_core.py`
- `calo_rpd_studio/scripts/validate_phase3_gui_render.py`
- `calo_rpd_studio/scripts/train_tsh_calo.py`

## Important public/entry symbols
- `_TransientSettings` — `calo_rpd_studio/scripts/validate_packaged_gui.py:33-43`
- `_TransientSettings` — `calo_rpd_studio/scripts/validate_phase3_gui_render.py:16-26`
- `_TransientSettings` — `calo_rpd_studio/scripts/validate_phase3_workspace_accessibility.py:33-43`
- `LockVerification` — `calo_rpd_studio/scripts/verify_requirements_lock.py:25-38`
- `emit_training_event` — `calo_rpd_studio/scripts/_train_tsh_calo_core.py:38-44`
- `load_plan` — `calo_rpd_studio/scripts/_train_tsh_calo_core.py:47-61`
- `repository_state` — `calo_rpd_studio/scripts/_train_tsh_calo_core.py:64-83`
- `validate_repository_for_plan` — `calo_rpd_studio/scripts/_train_tsh_calo_core.py:86-105`
- `validate_development_freeze_for_plan` — `calo_rpd_studio/scripts/_train_tsh_calo_core.py:108-148`
- `_summary` — `calo_rpd_studio/scripts/_train_tsh_calo_core.py:151-183`
- `validate_training_resources` — `calo_rpd_studio/scripts/_train_tsh_calo_core.py:186-189`
- `main` — `calo_rpd_studio/scripts/_train_tsh_calo_core.py:192-363`
- `load_plan` — `calo_rpd_studio/scripts/ablate_tsh_calo.py:19-27`
- `repository_state` — `calo_rpd_studio/scripts/ablate_tsh_calo.py:30-49`
- `validate_repository_for_plan` — `calo_rpd_studio/scripts/ablate_tsh_calo.py:52-59`
- `_summary` — `calo_rpd_studio/scripts/ablate_tsh_calo.py:62-78`
- `main` — `calo_rpd_studio/scripts/ablate_tsh_calo.py:81-102`
- `_utcnow` — `calo_rpd_studio/scripts/accept_development_freeze.py:80-81`

## Dependencies
- `calo-policy`, `compute`, `core`, `desktop`, `experiments`, `persistence`, `power-system`

## Dependents
- `calo-policy`, `desktop`, `tests`

## Associated tests
- `tests/gui/test_tsh_calo_generalization_integration.py`
- `tests/unit/test_accelerator_evidence.py`
- `tests/unit/test_artifact_manifest.py`
- `tests/unit/test_config.py`
- `tests/unit/test_container_contract.py`
- `tests/unit/test_distribution_contract.py`
- `tests/unit/test_packaged_gui_validator.py`
- `tests/unit/test_phase6_command_and_native_contracts.py`
- `tests/unit/test_requirements_lock.py`
- `tests/unit/test_resource_recovery_evidence.py`
- `tests/unit/test_tsh_calo_device_equivalence.py`
- `tests/unit/test_tsh_calo_training_campaign.py`
- `tests/unit/test_v120_phase1_contracts.py`
- `tests/unit/test_v120_phase4_development_freeze.py`
- `tests/unit/test_v120_phase4_policy_retirement.py`

## Retrieval note
Use the machine indexes for exact locations and confidence-labelled relationships; authored invariants live in `../ARCHITECTURE.md` and `../DECISIONS.md`.
