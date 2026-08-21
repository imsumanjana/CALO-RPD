# calo-policy

<!-- AUTO-GENERATED MODULE INTELLIGENCE -->
**Purpose:** CALO/TSH-CALO policy artifacts, training, generalization evidence, qualification, activation and inference contracts.

**Important state:** Immutable artifact provenance, registry records, receipts and explicit active bindings.

**Major flow:** independent training -> candidate -> qualification -> explicit activation -> checksum-bound experiment use.

**Constraints/invariants:** No auto-train/qualify/activate; protected cases isolated; A-E production, F experimental/off; exact accounting.

**Common failure points:** Provenance/schema mismatch, lifecycle bypass, accounting gaps and stale policy bindings.

## Start here
- `calo_rpd_studio/algorithms/calo/policy_registry.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_generalization_guard.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_policy_artifact.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training_campaign.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training_extension.py`

## Preferred edit targets
- `calo_rpd_studio/algorithms/calo/policy_registry.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_generalization_guard.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_policy_artifact.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training_campaign.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training_extension.py`

## Deep/internal implementation
- `calo_rpd_studio/algorithms/calo/_policy_registry_core.py`
- `calo_rpd_studio/algorithms/calo/_tsh_calo_generalization_guard_core.py`
- `calo_rpd_studio/algorithms/calo/_tsh_calo_policy_artifact_core.py`
- `calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py`
- `calo_rpd_studio/algorithms/calo/_tsh_calo_training_extension_core.py`

## State owners
- None explicitly identified.

## Cross-module dependencies
- `bootstrap`
- `compute`
- `core`
- `experiments`
- `optimization`
- `power-system`
- `validation-release`

## Dependents
- `bootstrap`
- `compute`
- `desktop`
- `optimization`
- `persistence`
- `tests`
- `validation-release`

## Related tests
- `tests/gui/test_calo_intelligence_permanent_deletion.py`
- `tests/gui/test_obsolete_policy_artifact_management.py`
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
- `tests/unit/test_scientist_policy_artifact_deletion.py`
- `tests/unit/test_tsh_calo_automatic_qualification.py`
- `tests/unit/test_tsh_calo_component_ablation.py`
- `tests/unit/test_tsh_calo_device_equivalence.py`
- `tests/unit/test_tsh_calo_effective_recovery.py`
- `tests/unit/test_tsh_calo_feasibility_and_influence.py`
- `tests/unit/test_tsh_calo_generalization_guard.py`
- `tests/unit/test_tsh_calo_generalization_integration.py`
- `tests/unit/test_tsh_calo_hierarchical_policy.py`
- `tests/unit/test_tsh_calo_independent_training.py`
- `tests/unit/test_tsh_calo_inference.py`
- `tests/unit/test_tsh_calo_optimizer.py`
- `tests/unit/test_tsh_calo_parameter_response.py`
- `tests/unit/test_tsh_calo_parameter_study.py`
- `tests/unit/test_tsh_calo_parameter_trajectory.py`
- `tests/unit/test_tsh_calo_physics_repair.py`
- `tests/unit/test_tsh_calo_policy_lifecycle.py`
- `tests/unit/test_tsh_calo_population_schedule.py`
- `tests/unit/test_tsh_calo_qualification.py`
- `tests/unit/test_tsh_calo_qualification_campaign.py`
- `tests/unit/test_tsh_calo_runtime_context.py`
- `tests/unit/test_tsh_calo_runtime_transition.py`
- `tests/unit/test_tsh_calo_topology_context.py`

## Files
- `calo_rpd_studio/ai/AGENTS.md`
- `calo_rpd_studio/ai/__init__.py`
- `calo_rpd_studio/ai/checkpoint_manager.py`
- `calo_rpd_studio/ai/inference.py`
- `calo_rpd_studio/ai/model_io.py`
- `calo_rpd_studio/ai/problem_features.py`
- `calo_rpd_studio/ai/reproducibility.py`
- `calo_rpd_studio/ai/training_dataset.py`
- `calo_rpd_studio/algorithms/calo/AGENTS.md`
- `calo_rpd_studio/algorithms/calo/__init__.py`
- `calo_rpd_studio/algorithms/calo/_policy_registry_core.py`
- `calo_rpd_studio/algorithms/calo/_tsh_calo_generalization_guard_core.py`
- `calo_rpd_studio/algorithms/calo/_tsh_calo_policy_artifact_core.py`
- `calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py`
- `calo_rpd_studio/algorithms/calo/_tsh_calo_training_extension_core.py`
- `calo_rpd_studio/algorithms/calo/adaptive_epsilon.py`
- `calo_rpd_studio/algorithms/calo/ai_controller.py`
- `calo_rpd_studio/algorithms/calo/archives.py`
- `calo_rpd_studio/algorithms/calo/cognitive_state.py`
- `calo_rpd_studio/algorithms/calo/competitive_training.py`
- `calo_rpd_studio/algorithms/calo/contextual_credit.py`
- `calo_rpd_studio/algorithms/calo/device_resident_synthetic.py`
- `calo_rpd_studio/algorithms/calo/diagnostics.py`
- `calo_rpd_studio/algorithms/calo/diversity_manager.py`
- `calo_rpd_studio/algorithms/calo/dual_lane_controller.py`
- `calo_rpd_studio/algorithms/calo/environmental_selection.py`
- `calo_rpd_studio/algorithms/calo/evaluation_cache.py`
- `calo_rpd_studio/algorithms/calo/heterogeneous_training.py`
- `calo_rpd_studio/algorithms/calo/hierarchical_memory.py`
- `calo_rpd_studio/algorithms/calo/learning_operators.py`
- `calo_rpd_studio/algorithms/calo/operator_credit.py`
- `calo_rpd_studio/algorithms/calo/optimizer.py`
- `calo_rpd_studio/algorithms/calo/policy_artifact_deletion.py`
- `calo_rpd_studio/algorithms/calo/policy_lineage.py`
- `calo_rpd_studio/algorithms/calo/policy_network.py`
- `calo_rpd_studio/algorithms/calo/policy_qualification.py`
- `calo_rpd_studio/algorithms/calo/policy_qualification_admission.py`
- `calo_rpd_studio/algorithms/calo/policy_readiness.py`
- `calo_rpd_studio/algorithms/calo/policy_registry.py`
- `calo_rpd_studio/algorithms/calo/policy_retirement.py`
- `calo_rpd_studio/algorithms/calo/policy_schema.py`
- `calo_rpd_studio/algorithms/calo/precision_engine.py`
- `calo_rpd_studio/algorithms/calo/reward.py`
- `calo_rpd_studio/algorithms/calo/success_memory.py`
- `calo_rpd_studio/algorithms/calo/tensor_state.py`
- `calo_rpd_studio/algorithms/calo/topology_context.py`
- `calo_rpd_studio/algorithms/calo/training.py`
- `calo_rpd_studio/algorithms/calo/transition_kernel.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_automatic_qualification.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_component_ablation.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_evaluation_accounting.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_feasibility_assessment.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_generalization_guard.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_inference.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_optimizer.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_parameter_evidence.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_parameter_registry.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_parameter_response.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_parameter_study.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_parameter_trajectory.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_physics_repair.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_policy.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_policy_artifact.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_population_schedule.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_qualification.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_qualification_campaign.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_runtime_context.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_schema.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_shield.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training_campaign.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training_environment.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training_extension.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training_influence.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training_receipt.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training_resources.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_training_session.py`
- `calo_rpd_studio/algorithms/calo/tsh_calo_transition_kernel.py`
- `calo_rpd_studio/algorithms/calo/v41_disputes.py`
- `calo_rpd_studio/algorithms/calo/v5_disputes.py`
- `calo_rpd_studio/algorithms/calo/variable_intelligence.py`
- `calo_rpd_studio/learning/AGENTS.md`
- `calo_rpd_studio/learning/__init__.py`
- `calo_rpd_studio/learning/experience_repository.py`

> Generated routing is evidence, not auditing. Curated `.ai/modules/*.md` guidance remains authoritative when more specific.
