# validation-release

<!-- AUTO-GENERATED MODULE INTELLIGENCE -->
**Purpose:** Developer/release validators, CI, packaging, containers and evidence-generation tooling.

**Important state:** Evidence bound to exact source/artifact identities.

**Major flow:** source -> non-mutating checks/build stages -> evidence/manifests -> separately authorized gates.

**Constraints/invariants:** Pinned actions/locks, no fabricated attestations, historical evidence immutable and scoped.

**Common failure points:** Stale artifacts, unpinned dependencies/actions, identity mismatch and accidental claim elevation.

## Start here
- No public/entry surface identified automatically; use curated module guidance.

## Preferred edit targets
- None explicitly identified; consult `.ai/architectural-semantics.json`.

## Deep/internal implementation
- `calo_rpd_studio/scripts/_train_tsh_calo_core.py`

## State owners
- None explicitly identified.

## Cross-module dependencies
- `calo-policy`
- `compute`
- `core`
- `desktop`
- `experiments`
- `persistence`
- `power-system`

## Dependents
- `calo-policy`
- `desktop`
- `tests`

## Related tests
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
- `tests/unit/test_v120_phase5_release_preparation.py`
- `tests/unit/test_v620_rc_final.py`

## Files
- `.github/workflows/AGENTS.md`
- `.github/workflows/ci.yml`
- `Dockerfile`
- `calo_rpd_studio/scripts/AGENTS.md`
- `calo_rpd_studio/scripts/__init__.py`
- `calo_rpd_studio/scripts/_train_tsh_calo_core.py`
- `calo_rpd_studio/scripts/ablate_tsh_calo.py`
- `calo_rpd_studio/scripts/accept_development_freeze.py`
- `calo_rpd_studio/scripts/audit_broad_exceptions.py`
- `calo_rpd_studio/scripts/container_smoke.py`
- `calo_rpd_studio/scripts/create_development_freeze_candidate.py`
- `calo_rpd_studio/scripts/create_release_preparation.py`
- `calo_rpd_studio/scripts/export_publication_results.py`
- `calo_rpd_studio/scripts/finalize_release_records.py`
- `calo_rpd_studio/scripts/generate_artifact_manifest.py`
- `calo_rpd_studio/scripts/generate_distribution_manifests.py`
- `calo_rpd_studio/scripts/generate_experiment_schema.py`
- `calo_rpd_studio/scripts/manage_policy_retirement.py`
- `calo_rpd_studio/scripts/migrate_legacy_resume.py`
- `calo_rpd_studio/scripts/qualify_tsh_calo.py`
- `calo_rpd_studio/scripts/release_policy_scope.py`
- `calo_rpd_studio/scripts/train_calo.py`
- `calo_rpd_studio/scripts/train_tsh_calo.py`
- `calo_rpd_studio/scripts/validate_accelerator.py`
- `calo_rpd_studio/scripts/validate_cases.py`
- `calo_rpd_studio/scripts/validate_cuda_hot_path.py`
- `calo_rpd_studio/scripts/validate_cuda_policy_hot_path.py`
- `calo_rpd_studio/scripts/validate_hardware_soak.py`
- `calo_rpd_studio/scripts/validate_packaged_gui.py`
- `calo_rpd_studio/scripts/validate_phase3_gui_render.py`
- `calo_rpd_studio/scripts/validate_phase3_workspace_accessibility.py`
- `calo_rpd_studio/scripts/validate_phase6_gui_contracts.py`
- `calo_rpd_studio/scripts/validate_resource_recovery.py`
- `calo_rpd_studio/scripts/validate_stage_b_synthetic.py`
- `calo_rpd_studio/scripts/validate_tsh_calo_device_equivalence.py`
- `calo_rpd_studio/scripts/verify_active_version.py`
- `calo_rpd_studio/scripts/verify_distribution_stage.py`
- `calo_rpd_studio/scripts/verify_phase6_distribution.py`
- `calo_rpd_studio/scripts/verify_release_ci_contract.py`
- `calo_rpd_studio/scripts/verify_requirements_lock.py`
- `calo_rpd_studio/validation/AGENTS.md`
- `calo_rpd_studio/validation/__init__.py`
- `calo_rpd_studio/validation/gui_contract.py`
- `compose.yaml`
- `containers/AGENTS.md`
- `containers/entrypoint.py`

> Generated routing is evidence, not auditing. Curated `.ai/modules/*.md` guidance remains authoritative when more specific.
