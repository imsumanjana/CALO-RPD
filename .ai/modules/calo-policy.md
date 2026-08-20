# calo-policy

**Purpose:** CALO/TSH-CALO policy artifacts, training, generalization evidence, qualification, activation and inference contracts.

**Important state:** Immutable artifact provenance, registry records, receipts and explicit active bindings.

**Major flow:** independent training -> candidate -> qualification -> explicit activation -> checksum-bound experiment use.

**Constraints/invariants:** No auto-train/qualify/activate; protected cases isolated; A-E production, F experimental/off; exact accounting.

**Common failure points:** Provenance/schema mismatch, lifecycle bypass, accounting gaps and stale policy bindings.

## Primary files
- `calo_rpd_studio/algorithms/calo/tsh_calo_qualification_campaign.py`
- `calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py`
- `calo_rpd_studio/algorithms/calo/competitive_training.py`
- `calo_rpd_studio/algorithms/calo/training.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_component_ablation.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training.py`
- `calo_rpd_studio/algorithms/calo/_policy_registry_core.py`
- `calo_rpd_studio/algorithms/calo/heterogeneous_training.py`
- `calo_rpd_studio/algorithms/calo/policy_retirement.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_automatic_qualification.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_shield.py`
- `calo_rpd_studio/algorithms/calo/ai_controller.py`

## Important public/entry symbols
- `CheckpointInfo` — `calo_rpd_studio/ai/checkpoint_manager.py:17-20`
- `CheckpointManager` — `calo_rpd_studio/ai/checkpoint_manager.py:23-108`
- `TrainingManifest` — `calo_rpd_studio/ai/training_dataset.py:7-22`
- `PolicyRecord` — `calo_rpd_studio/algorithms/calo/_policy_registry_core.py:51-94`
- `PolicyRegistry` — `calo_rpd_studio/algorithms/calo/_policy_registry_core.py:97-819`
- `TSHCALOGeneralizationGuardConfig` — `calo_rpd_studio/algorithms/calo/_tsh_calo_generalization_guard_core.py:70-152`
- `GeneralizationComparison` — `calo_rpd_studio/algorithms/calo/_tsh_calo_generalization_guard_core.py:180-190`
- `IndependentTrainingProvenance` — `calo_rpd_studio/algorithms/calo/_tsh_calo_policy_artifact_core.py:34-146`
- `TSHCALOCandidateArtifact` — `calo_rpd_studio/algorithms/calo/_tsh_calo_policy_artifact_core.py:160-203`
- `TSHCALOTrainingPauseRequested` — `calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py:94-99`
- `TSHCALOTrainingEpisodePlan` — `calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py:341-352`
- `TSHCALOTrainingMemberPlan` — `calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py:356-376`
- `TSHCALOTrainingHyperparameters` — `calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py:380-393`
- `TSHCALOEnvironmentHyperparameters` — `calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py:397-418`
- `TSHCALOTrainingCampaignPlan` — `calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py:422-673`
- `TSHCALOTrainingCampaignResult` — `calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py:691-698`
- `IndependentTSHCALOTrainingCampaign` — `calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py:701-1653`
- `TSHCALOTrainingExtensionParent` — `calo_rpd_studio/algorithms/calo/_tsh_calo_training_extension_core.py:56-62`

## Dependencies
- `bootstrap`, `compute`, `core`, `experiments`, `optimization`, `power-system`, `validation-release`

## Dependents
- `bootstrap`, `compute`, `desktop`, `optimization`, `persistence`, `tests`, `validation-release`

## Associated tests
- `tests/gui/test_phase6_ribbon_workspace.py`
- `tests/gui/test_tsh_calo_generalization_integration.py`
- `tests/gui/test_tsh_calo_policy_library_accounting.py`
- `tests/integration/test_historical_learning.py`
- `tests/integration/test_phase4_empty_policy_workflow.py`
- `tests/unit/test_calo.py`
- `tests/unit/test_calo_core_v2.py`
- `tests/unit/test_calo_numerical_robustness.py`
- `tests/unit/test_calo_v4.py`
- `tests/unit/test_calo_v41_policy_system.py`
- `tests/unit/test_calo_v41_runtime_guards.py`
- `tests/unit/test_cuda_residency_contract.py`
- `tests/unit/test_heterogeneous_policy_training.py`
- `tests/unit/test_historical_policy_pretraining.py`
- `tests/unit/test_phase6_command_and_native_contracts.py`

## Retrieval note
Use the machine indexes for exact locations and confidence-labelled relationships; authored invariants live in `../ARCHITECTURE.md` and `../DECISIONS.md`.
