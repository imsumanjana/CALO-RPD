"""Cryptographic freeze manifest for final CALO benchmark campaigns.

The manifest is intentionally explicit: final benchmark execution is allowed only when the
mathematical implementation, training semantics/repository snapshot, default CALO hyperparameters,
mixed-variable decoder, and feasibility rules match the frozen hashes. A neural policy is never
implied by the software freeze; policy-assisted experiments must separately bind an explicit policy
artifact SHA-256.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

from calo_rpd_studio.version import VERSION


FREEZE_SCHEMA_VERSION = 1
DEFAULT_FREEZE_RELATIVE_PATHS = (
    "calo_rpd_studio/accelerated/device.py",
    "calo_rpd_studio/accelerated/device_resident_orpd.py",
    "calo_rpd_studio/accelerated/parity_audit.py",
    "calo_rpd_studio/accelerated/torch_decoder.py",
    "calo_rpd_studio/accelerated/torch_power_flow.py",
    "calo_rpd_studio/accelerated/torch_orpd.py",
    "calo_rpd_studio/accelerated/runtime_context.py",
    "calo_rpd_studio/accelerated/throughput_engine.py",
    "calo_rpd_studio/accelerated/scratch_pool.py",
    "calo_rpd_studio/algorithms/base_optimizer.py",
    "calo_rpd_studio/algorithms/torch_suite.py",
    "calo_rpd_studio/algorithms/calo/ai_controller.py",
    "calo_rpd_studio/algorithms/calo/archives.py",
    "calo_rpd_studio/algorithms/calo/cognitive_state.py",
    "calo_rpd_studio/algorithms/calo/diagnostics.py",
    "calo_rpd_studio/algorithms/calo/device_resident_synthetic.py",
    "calo_rpd_studio/algorithms/calo/diversity_manager.py",
    "calo_rpd_studio/algorithms/calo/environmental_selection.py",
    "calo_rpd_studio/algorithms/calo/learning_operators.py",
    "calo_rpd_studio/algorithms/calo/operator_credit.py",
    "calo_rpd_studio/algorithms/calo/optimizer.py",
    "calo_rpd_studio/algorithms/calo/policy_network.py",
    "calo_rpd_studio/algorithms/calo/policy_schema.py",
    "calo_rpd_studio/algorithms/calo/policy_registry.py",
    "calo_rpd_studio/algorithms/calo/policy_readiness.py",
    "calo_rpd_studio/algorithms/calo/policy_lineage.py",
    "calo_rpd_studio/algorithms/calo/run_checkpoint.py",
    "calo_rpd_studio/algorithms/calo/policy_qualification.py",
    "calo_rpd_studio/algorithms/calo/reward.py",
    "calo_rpd_studio/algorithms/calo/success_memory.py",
    "calo_rpd_studio/algorithms/calo/adaptive_epsilon.py",
    "calo_rpd_studio/algorithms/calo/contextual_credit.py",
    "calo_rpd_studio/algorithms/calo/dual_lane_controller.py",
    "calo_rpd_studio/algorithms/calo/evaluation_cache.py",
    "calo_rpd_studio/algorithms/calo/hierarchical_memory.py",
    "calo_rpd_studio/algorithms/calo/precision_engine.py",
    "calo_rpd_studio/algorithms/calo/tensor_state.py",
    "calo_rpd_studio/algorithms/calo/variable_intelligence.py",
    "calo_rpd_studio/algorithms/calo/v5_disputes.py",
    "calo_rpd_studio/algorithms/calo/training.py",
    "calo_rpd_studio/algorithms/calo/competitive_training.py",
    "calo_rpd_studio/algorithms/calo/heterogeneous_training.py",
    "calo_rpd_studio/algorithms/registry.py",
    "calo_rpd_studio/compute/persistent_accelerator_worker.py",
    "calo_rpd_studio/compute/persistent_accelerator_sidecar.py",
    "calo_rpd_studio/compute/persistent_training_actor.py",
    "calo_rpd_studio/compute/resource_scheduler.py",
    "calo_rpd_studio/compute/topology.py",
    "calo_rpd_studio/compute/training_resources.py",
    "calo_rpd_studio/compute/governor.py",
    "calo_rpd_studio/compute/provenance.py",
    "calo_rpd_studio/compute/soak.py",
    "calo_rpd_studio/compute/scientific_equivalence.py",
    "calo_rpd_studio/compute/training_actor_worker.py",
    "calo_rpd_studio/experiments/calo_ablation.py",
    "calo_rpd_studio/experiments/evaluation_budget.py",
    "calo_rpd_studio/experiments/seed_manager.py",
    "calo_rpd_studio/experiments/experiment_config.py",
    "calo_rpd_studio/experiments/fairness_validator.py",
    "calo_rpd_studio/experiments/experiment_runner.py",
    "calo_rpd_studio/learning/experience_repository.py",
    "calo_rpd_studio/continuation/experiment_evolution.py",
    "calo_rpd_studio/continuation/runtime_binding.py",
    "calo_rpd_studio/app/experiment_manager.py",
    "calo_rpd_studio/app/state_manager.py",
    "calo_rpd_studio/app/workspaces.py",
    "calo_rpd_studio/app/workflow_manager.py",
    "calo_rpd_studio/app/experiment_workspace_restorer.py",
    "calo_rpd_studio/app/main_window.py",
    "calo_rpd_studio/app/session_recovery.py",
    "calo_rpd_studio/portfolio/models.py",
    "calo_rpd_studio/portfolio/catalog.py",
    "calo_rpd_studio/portfolio/planner.py",
    "calo_rpd_studio/portfolio/fingerprint.py",
    "calo_rpd_studio/portfolio/exporter.py",
    "calo_rpd_studio/resume/models.py",
    "calo_rpd_studio/resume/service.py",
    "calo_rpd_studio/results/database.py",
    "calo_rpd_studio/results/result_store.py",
    "calo_rpd_studio/orpd/constraint_violation.py",
    "calo_rpd_studio/orpd/constraints.py",
    "calo_rpd_studio/orpd/feasibility_rules.py",
    "calo_rpd_studio/orpd/formulation_fingerprint.py",
    "calo_rpd_studio/orpd/mixed_variable_handler.py",
    "calo_rpd_studio/orpd/variable_decoder.py",
    "calo_rpd_studio/orpd/problem.py",
    "calo_rpd_studio/orpd/objectives.py",
    "calo_rpd_studio/robustness/cvar.py",
    "calo_rpd_studio/robustness/scenario.py",
    "calo_rpd_studio/robustness/scenario_generator.py",
    "calo_rpd_studio/robustness/load_uncertainty.py",
    "calo_rpd_studio/robustness/renewable_uncertainty.py",
    "calo_rpd_studio/robustness/monte_carlo.py",
    "calo_rpd_studio/robustness/contingencies.py",
    "calo_rpd_studio/robustness/robust_objectives.py",
    "calo_rpd_studio/power_system/case_model.py",
    "calo_rpd_studio/power_system/branch_flows.py",
    "calo_rpd_studio/power_system/case_validation.py",
    "calo_rpd_studio/power_system/case_loader.py",
    "calo_rpd_studio/power_system/ybus.py",
    "calo_rpd_studio/power_system/ac_power_flow.py",
    "calo_rpd_studio/power_system/newton_raphson.py",
    "calo_rpd_studio/power_system/voltage_stability.py",
    "calo_rpd_studio/power_system/pv_pq_switching.py",
    "calo_rpd_studio/power_system/independent_validator.py",
    "calo_rpd_studio/results/publication_export.py",
    "calo_rpd_studio/results/comparison_engine.py",
    "calo_rpd_studio/statistics/confidence_intervals.py",
    "calo_rpd_studio/statistics/effect_sizes.py",
    "calo_rpd_studio/statistics/friedman.py",
    "calo_rpd_studio/statistics/posthoc.py",
    "calo_rpd_studio/statistics/rankings.py",
    "calo_rpd_studio/statistics/wilcoxon.py",
    "calo_rpd_studio/benchmarking/campaign.py",
    "calo_rpd_studio/benchmarking/package.py",
    "calo_rpd_studio/benchmarking/evidence.py",
    "calo_rpd_studio/visualization/publication_evidence.py",
    "calo_rpd_studio/visualization/font_preflight.py",
    "calo_rpd_studio/ai/model_io.py",
    "calo_rpd_studio/ai/checkpoint_manager.py",
    "calo_rpd_studio/gui/panels/dashboard_panel.py",
    "calo_rpd_studio/gui/panels/power_system_panel.py",
    "calo_rpd_studio/gui/panels/benchmark_campaign_panel.py",
    "calo_rpd_studio/gui/panels/experiment_manager_panel.py",
    "calo_rpd_studio/gui/panels/calo_intelligence_panel.py",
    "calo_rpd_studio/gui/panels/results_explorer_panel.py",
    "calo_rpd_studio/gui/panels/resume_center_panel.py",
    "calo_rpd_studio/gui/widgets/historical_experience_widget.py",
    "calo_rpd_studio/gui/panels/live_optimization_panel.py",
    "calo_rpd_studio/scripts/train_calo.py",
    "calo_rpd_studio/scripts/run_final_benchmark.py",
    "calo_rpd_studio/scripts/migrate_legacy_resume.py",
    "calo_rpd_studio/scripts/validate_hardware_soak.py",
    "calo_rpd_studio/scripts/validate_stage_b_synthetic.py",
    "calo_rpd_studio/validation/gui_contract.py",
    "calo_rpd_studio/version.py",
    "calo_rpd_studio/data/frozen/historical_training_snapshot_v2.json",
    "calo_rpd_studio/data/examples/policy_development_active_loss.yaml",
)


@dataclass(frozen=True, slots=True)
class FreezeVerification:
    passed: bool
    manifest_path: str
    checked_files: int
    missing_files: tuple[str, ...]
    changed_files: tuple[str, ...]
    manifest_sha256: str
    message: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(payload: dict) -> str:
    clean = dict(payload)
    clean.pop("manifest_sha256", None)
    data = json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def project_root_from_module() -> Path:
    return Path(__file__).resolve().parents[2]


def create_freeze_manifest(
    destination: str | Path,
    *,
    project_root: str | Path | None = None,
    relative_paths: Iterable[str] = DEFAULT_FREEZE_RELATIVE_PATHS,
    software_version: str = VERSION,
    note: str | None = None,
) -> Path:
    root = Path(project_root) if project_root is not None else project_root_from_module()
    root = root.resolve()
    files: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    if missing:
        raise FileNotFoundError(
            "Cannot freeze CALO; required files are missing: " + ", ".join(missing)
        )

    from calo_rpd_studio.algorithms.registry import SPECS

    if note is None:
        note = (
            f"CALO-RPD v{software_version} policy-gated software architecture freeze. "
            "No default neural policy is bundled or implied; every policy-assisted experiment must bind an explicit validated policy artifact SHA-256."
        )

    payload = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "software_version": software_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "frozen_scope": {
            "mathematical_equations": True,
            "operator_definitions": True,
            "state_vector": True,
            "archive_rules": True,
            "ppo_architecture": True,
            "policy_gated_no_default_neural_policy": True,
            "immutable_experiment_policy_binding": True,
            "untrained_policy_fallback_forbidden": True,
            "training_dataset_snapshot": True,
            "hyperparameters": True,
            "constraint_handling": True,
            "accelerator_power_flow": True,
            "accelerator_constraint_evaluator": True,
            "torch_canonical_baselines": True,
            "mixed_variable_tensor_decoder": True,
            "persistent_accelerator_workers": True,
            "cross_run_batching": True,
            "automatic_batch_calibration": True,
            "measured_throughput_scheduling": True,
            "persistent_policy_training_actors": True,
            "cross_episode_policy_rollout_batching": True,
            "stage_b_device_resident_synthetic_evaluation": True,
            "stage_b_cross_episode_synthetic_microbatching": True,
            "stage_b_synthetic_startup_parity_fail_closed": True,
            "stage_b_real_orpd_development_suite_configurable": True,
            "stage_b_full_stochastic_calo_controller_gpu_resident": False,
            "portfolio_dependency_planning": True,
            "scientific_fingerprint_reuse": True,
            "campaign_resume_journal": True,
            "portfolio_artifact_resume": True,
            "device_resident_execution": True,
            "gpu_maximum_100_percent_cuda_when_available": True,
            "cuda_only_execution": False,
            "grouped_tensor_pv_pq_switching": True,
            "fractional_tail_weighted_cvar": True,
            "case_specific_formulation_profiles": True,
            "scenario_manifest_and_validation": True,
            "independent_q_limit_validation": True,
            "verified_feasible_publication_statistics": True,
            "verified_only_article_package": True,
            "asynchronous_fairness_audit": True,
            "fairness_audit_progress_and_cpu_throttling": True,
            "multi_run_live_telemetry": True,
            "progressive_portfolio_live_preview": True,
            "ieee300_matched_pypower_validation": True,
            "unclipped_reference_q_reporting": True,
            "bounded_reproducibility_bundle": True,
            "export_subprogress_and_safe_cancel": True,
            "asynchronous_standard_publication_export": True,
            "nan_safe_convergence_statistics": True,
            "persistent_personal_memory": True,
            "hpem_best_1_3_5_7": True,
            "mixed_variable_quality_diversity_memory": True,
            "max_persistent_calo_tensor_dimension_3d": True,
            "contextual_batch_operator_credit": True,
            "contextual_success_direction_memory": True,
            "variable_group_intelligence": True,
            "behavior_driven_epsilon": True,
            "single_budget_dual_lane_self_learning": True,
            "counted_cognitive_precision": True,
            "partial_recovery_without_forgetting": True,
            "exact_evaluation_dedup_fe_accounting_preserved": True,
            "strict_cross_run_memory_independence": True,
            "safe_hashed_checkpoint_loading": True,
            "policy_library_and_qualification": True,
            "native_v59_policy_schema": True,
            "experiment_workspace_restoration": True,
            "continuable_policy_training_checkpoints": True,
            "policy_lineage_latest_vs_best": True,
            "crash_safe_atomic_policy_checkpoints": False,
            "durability_hardened_atomic_policy_checkpoints": True,
            "authenticated_trusted_local_exact_resume": True,
            "experiment_revision_history": True,
            "add_more_paired_runs_same_experiment": True,
            "evaluation_horizon_evidence_snapshots": True,
            "exact_calo_run_state_continuation": True,
            "paired_recompute_from_seed_horizon_extension": True,
            "publication_safe_extension_protocols": True,
            "horizon_aware_statistics_and_export": True,
            "revision_scoped_run_checkpoints": True,
            "calo_control_fully_device_resident": False,
            "baseline_exact_optimizer_state_continuation": False,
            "automatic_periodic_policy_qualification": False,
            "formal_qualification_saved_base_artifacts_only": True,
            "robust_feasibility_default_all_scenario_max": True,
            "per_generator_active_reactive_limit_accounting": True,
            "fixed_formulation_objective_bus_partition": True,
            "verified_only_publication_fail_closed": True,
            "exact_equal_fe_budget_divisibility_rule": True,
            "font_preflight_and_metadata": True,
            "transactional_competitive_branch_generations": True,
            "bounded_infinite_training_resume_state": True,
            "typed_competitive_safe_stop_status": True,
            "competitive_session_recovery_index": True,
            "order_independent_common_bundle_base_selection": True,
            "champion_validation_bundle_fingerprint": True,
            "branch_aware_accelerator_admission": True,
            "complete_verified_publication_portfolio_required": True,
            "formal_superiority_statistical_gate": True,
            "callable_object_scientific_fingerprint": True,
            "sparse_newton_jacobian_primary_path": True,
            "stale_recovery_authority_guard": True,
            "segmented_bounded_training_telemetry": True,
            "native_v59_training_runtime_transition_parity_gate": True,
            "deployable_base_requires_exact_real_orpd_development_evidence": True,
            "n_minus_one_includes_intact_base": True,
            "branch_angle_difference_constraints": True,
            "exact_power_flow_options_in_accelerator_parity": True,
            "component_and_state_level_accelerator_parity": True,
            "single_authority_repair_accounting": True,
            "constraint_tolerance_schema": True,
            "formal_noninferiority_one_sided_holm_gate": True,
            "full_policy_trajectory_raw_vs_executed_provenance": True,
            "full_scientific_transfer_fingerprint": True,
            "scenario_transform_structural_validation": True,
            "pq_generator_voltage_dead_controls_removed": True,
            "direct_sparse_branch_current_matrices": True,
            "independent_ac_power_flow_cross_validation": True,
            "key_based_workspace_navigation": True,
            "legacy_v59_workspace_index_migration": True,
            "policy_first_governing_intelligence_gate": True,
            "dashboard_compute_topology_map": True,
            "safe80_resource_budget_engine": True,
            "safe80_global_cpu_budget": True,
            "safe80_hard_parallel_branch_ceiling": True,
            "xpu_sidecar_full_branch_certified": False,
            "xpu_capability_aware_scheduling": True,
            "direct_xpu_full_branch_requires_runtime_fp64_smoke": True,
            "xpu_sidecar_actor_evaluator_only": True,
            "dynamic_thermal_power_governor": True,
            "staged_compute_startup": True,
            "hash_chained_compute_provenance": True,
            "workspace_schema_v3_migration": True,
            "unclean_application_session_recovery": True,
            "hardware_soak_qualification_protocol": True,
            "scheduling_scientific_equivalence_protocol": True,
            "physical_multi_hour_hardware_soak_certified_in_build_runtime": False,
            "full_pyqt6_gui_validated_in_build_runtime": False,
            "queued_total_vs_concurrent_branch_scheduler": True,
            "exact_resume_queue_time_slicing": True,
            "global_training_exclusive_ui_lock": True,
            "global_cpu_worker_budget_enforced": True,
            "automatic_accelerator_to_cpu_branch_spillover": False,
        },
        "calo_default_parameters": SPECS["CALO"].default_parameters,
        "files": files,
        "benchmark_rule": (
            "No CALO tuning is permitted after TEST campaign execution begins. The software freeze "
            "does not choose or fabricate a neural policy. Policy-assisted CALO requires an explicitly "
            "imported/trained, compatible, activated and immutable experiment policy SHA-256 binding. "
            "No random/untrained/missing-policy fallback is permitted. Explicit No-AI CALO is reserved "
            "for declared research/qualification controls and is never an automatic fallback."
        ),
    }
    payload["manifest_sha256"] = _canonical_json_hash(payload)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def verify_freeze_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> FreezeVerification:
    manifest = Path(manifest_path)
    if not manifest.is_file():
        return FreezeVerification(
            False, str(manifest), 0, (), (), "", "Freeze manifest does not exist."
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    expected_manifest_hash = str(payload.get("manifest_sha256", ""))
    actual_manifest_hash = _canonical_json_hash(payload)
    root = Path(project_root) if project_root is not None else project_root_from_module()
    root = root.resolve()
    missing: list[str] = []
    changed: list[str] = []
    files = payload.get("files", {})
    for relative, meta in files.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if _sha256(path) != str(meta.get("sha256", "")):
            changed.append(relative)
    manifest_valid = bool(expected_manifest_hash) and expected_manifest_hash == actual_manifest_hash
    passed = manifest_valid and not missing and not changed
    if not manifest_valid:
        message = "Freeze manifest integrity check failed."
    elif missing:
        message = f"Freeze verification failed: {len(missing)} frozen file(s) are missing."
    elif changed:
        message = f"Freeze verification failed: {len(changed)} frozen file(s) changed."
    else:
        message = f"Frozen CALO verified across {len(files)} files."
    return FreezeVerification(
        passed,
        str(manifest),
        len(files),
        tuple(missing),
        tuple(changed),
        actual_manifest_hash,
        message,
    )
