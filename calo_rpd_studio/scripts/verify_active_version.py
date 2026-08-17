"""Fail-closed consistency check for the active v12 development identity."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tomllib

from calo_rpd_studio.version import (
    DISPLAY_VERSION,
    PRODUCT_VERSION,
    RELEASE_LINE,
    VERSION,
    VERSION_STAGE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def verify_active_version(root: Path = PROJECT_ROOT) -> dict:
    """Return a machine-readable v12 identity report without changing repository state."""

    expected = {
        "version": VERSION,
        "display_version": DISPLAY_VERSION,
        "product_version": PRODUCT_VERSION,
        "release_line": RELEASE_LINE,
        "stage": VERSION_STAGE,
    }
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package_version = str(pyproject["project"]["version"])
    checks["pyproject_matches_runtime"] = package_version == VERSION
    details["pyproject_version"] = package_version

    status = _load_json(root / "ACTIVE_DEVELOPMENT_STATUS.json")
    checks["active_status_identity"] = all(
        status.get(key) == value for key, value in expected.items()
    )
    checks["active_status_is_not_release"] = all(
        status.get(key) is False
        for key in (
            "release_candidate",
            "final_release",
            "final_freeze_available",
            "release_qualification_complete",
            "protected_case_evidence_open",
        )
    )
    checks["active_status_does_not_authorize_policy_work"] = (
        status.get("policy_training_authorized_by_status") is False
        and status.get("policy_evaluation_authorized_by_status") is False
    )
    checks["active_status_runtime_contract"] = (
        status.get("phase") == 6
        and status.get("supported_execution_modes") == ["cuda-preferred", "cpu-only"]
        and status.get("supported_execution_purposes") == ["exploratory", "formal"]
        and status.get("intel_xpu_executable") is False
        and status.get("safe_memory_admission_fraction") == 0.8
        and status.get("phase_1_validation", "").startswith("accepted_")
        and status.get("phase_2_validation") == "accepted_phase2-20260807-003828_15_of_15_passed"
        and status.get("phase_3_coding")
        == "implemented_tabbed_workspace_table_width_correction_validated_windows"
        and status.get("phase_3_initial_validation")
        == "failed_phase3-20260807-045558_11_of_18_passed"
        and status.get("phase_3_validation")
        == "windows_automated_accepted_current_tabbed_layout_linux_manually_accepted"
        and status.get("phase_3_corrections") == "implemented"
        and status.get("phase_3_revalidation") == "accepted_phase3-20260807-052047_18_of_18_passed"
        and status.get("phase_3_revalidation_policy_workflows_executed") is False
        and status.get("phase_3_remaining_windows_validation")
        == "accepted_phase3-remaining-windows-20260807-121530_10_of_10_automated_passed"
        and status.get("phase_3_remaining_windows_validation_policy_workflows_executed") is False
        and status.get("phase_3_remaining_windows_corrections")
        == "accepted_phase3-remaining-windows-20260807-121530"
        and status.get("phase_3_tabbed_layout_refinement")
        == "windows_automated_accepted_linux_xcb_manually_accepted_by_owner"
        and status.get("phase_3_tabbed_layout_refinement_windows_validation")
        == "accepted_phase3-remaining-windows-20260807-121530_10_of_10_passed"
        and status.get("phase_3_linux_rendering")
        == "manually_validated_and_accepted_by_owner_no_automated_bundle_retained"
        and status.get("phase_3_keyboard_accessibility_acceptance")
        == "accepted_windows_20260807_121530"
        and status.get("phase_3_human_reviewer_input") == "disabled_by_user_instruction"
        and status.get("phase_3_scientist_acceptance") == "not_inferred_automated_evidence_only"
        and status.get("phase_3_overall_gate") == "closed_by_owner_manual_linux_xcb_acceptance"
        and status.get("phase_4_started") is True
        and status.get("phase_4_coding")
        == "implemented_and_combined_validation_passed_before_phase6"
        and status.get("phase_4_validation")
        == "passed_phase4-20260813-000340_32_of_32_combined_validation"
        and status.get("phase_4_development_goal")
        == "completed_phase4_development_combined_validation_passed"
        and status.get("phase_5_started") is True
        and status.get("phase_5_coding")
        == "release_preparation_development_complete_combined_validation_passed"
        and status.get("phase_5_validation")
        == "passed_phase5-20260813-010531_41_of_41_combined_validation"
        and status.get("phase_5_release_policy_scope") == "pending_explicit_decision"
        and status.get("phase_5_release_candidate") is False
        and status.get("phase_5_final_release") is False
        and status.get("phase_5_publication_authorized") is False
        and status.get("phase_6_started") is True
        and status.get("phase_6_goal")
        == "individual_experiment_independence_one_way_workspace_study_and_truthful_audit_stage_run_navigation_engineering_validated"
        and status.get("phase_6_coding")
        == "implemented_and_engineering_validated_separate_mode_prerequisites_v2_individual_v3_workspace_plans_shared_singleton_controller_and_numerical_executor"
        and status.get("phase_6_training_interface")
        == "ribbon_navigation_only_tsh_calo_only_training_no_architecture_selector_optional_settings_template_per_user_default_model_directory_registered_resume_scan_locations_visible_scientific_inputs_protected_118_300_locked_visible_default_off_exact_resume_in_pane_readiness_then_explicit_confirmed_start_rule_based_calo_remains_in_algorithm_selection_no_redundant_document"
        and status.get("phase_6_training_input_help")
        == "complete_19_of_19_applicable_accessible_hover_focus_information_controls_with_directional_effects_suggested_ranges_hard_limits_and_lifecycle_boundaries_validated_phase6-20260814-131637"
        and status.get("phase_6_native_execution")
        == "validated_first_class_windows_non_docker_launch"
        and status.get("phase_6_validation")
        == "passed_phase6-20260817-224630_schema_v38_17_of_17_parity_fairness_presentation_source_stable_engineering_validation"
        and status.get("phase_6_checkbox_indicator_visibility")
        == "validated_global_palette_aware_borders_and_state_marks_light_dark_phase6-20260814-131637"
        and status.get("phase_6_new_training_recovery_presentation")
        == "validated_automatic_recovery_status_separate_from_selected_interrupted_exact_resume_phase6-20260814-131637"
        and status.get("phase_6_training_observability")
        == "implemented_structured_checkpoint_progress_durable_jsonl_detailed_activity_steps_validation_pending"
        and status.get("phase_6_checkpoint_safe_pause")
        == "implemented_authenticated_checkpoint_boundary_pause_unlimited_resume_count_validation_pending"
        and status.get("phase_6_training_evaluation_budget")
        == "finite_exact_immutable_across_pause_and_resume"
        and status.get("phase_6_unbounded_training")
        == "not_implemented_finite_exact_evaluation_plan_preserved"
        and status.get("phase_6_completed_model_extension")
        == "implemented_explicit_authenticated_repeatable_finite_segments_exact_plan_values_frozen_architecture_complete_parameter_schema_software_revision_independent_validation_pending"
        and status.get("phase_6_extension_compatibility")
        == "source_commit_and_writer_metadata_retained_as_nonadmission_provenance_architecture_resume_policy_parameter_layout_and_complete_training_parameter_field_sets_fail_closed_legacy_authority_reuse_needs_no_resupplied_paths_validation_pending"
        and status.get("phase_6_training_progress_presentation")
        == "single_persistent_bottom_bar_progress_activity_retains_checkpoint_detail_left_inputs_retain_actions_only_validation_pending"
        and status.get("phase_6_context_pane_responsiveness")
        == "training_paths_and_scroll_host_shrink_inside_fixed_left_pane_default_saved_training_path_reserves_all_wrapped_lines_without_horizontal_overflow_validation_pending"
        and status.get("phase_6_policy_library_completed_campaigns")
        == "all_completed_campaigns_visible_including_candidate_attention_states_imported_campaigns_merged_with_registry_state_validation_pending"
        and status.get("phase_6_completed_campaign_deletion")
        == "persistent_usable_delete_model_files_action_explicit_confirmed_physical_deletion_for_exact_unregistered_or_inactive_unqualified_unreferenced_imported_completed_directory_or_standalone_candidate_file_active_qualified_referenced_multi_registered_scan_root_symlink_and_incomplete_targets_refused_validation_pending"
        and status.get("phase_6_registered_policy_removal")
        == "eligible_completed_or_standalone_candidate_registration_atomically_identity_checked_suppressed_and_removed_before_exact_file_deletion_qualification_binding_lineage_active_and_checksum_guards_fail_closed_validation_pending"
        and status.get("phase_6_reviewed_policy_removal")
        == "review_policy_removal_gui_action_removed_cli_inventory_and_dry_run_workflow_retained_for_separately_governed_retirement_validation_pending"
        and status.get("phase_6_policy_activation")
        == "explicit_in_library_activation_for_independently_qualified_compatible_integrity_verified_policy_only_training_completion_never_sufficient_validation_pending"
        and status.get("phase_6_governing_policy_handoff")
        == "explicit_apply_binds_ready_immutable_policy_unlocks_and_navigates_to_power_system_without_starting_scientific_work_validation_pending"
        and status.get("phase_6_policy_library_layout")
        == "full_width_no_internal_horizontal_or_vertical_scrollbar_height_tracks_header_plus_all_current_entries_validation_pending"
        and status.get("phase_6_governing_policy_layout")
        == "full_width_expanding_policy_field_status_and_action_dynamic_page_height_bottom_clearance_manual_outer_scroll_reaches_complete_block_and_model_selection_preserves_scroll_position_validation_pending"
        and status.get("phase_6_extension_retained_state")
        == "model_optimizer_numpy_rng_torch_rng_ppo_updates_episode_receipts_device_memory_session_environment_rollout_collector_exact_accounting"
        and status.get("phase_6_extension_count_limit") == "none_each_segment_explicit_and_finite"
        and status.get("phase_6_extension_scientific_claim")
        == "no_improvement_superiority_qualification_registration_or_activation_inferred"
        and status.get("phase_6_validation_policy_or_scientific_workflows_executed") is False
        and status.get("phase_6_automated_human_acceptance") == "not_inferred"
        and status.get("phase_6_visual_refinement")
        == "completed_phase6-panel-sweep-20260813-041700_16_workspace_panels_and_4_shell_renders"
        and status.get("phase_6_workspace_navigation")
        == "complete_workspace_palette_permanent_expanded_ribbon_permanent_input_only_left_pane_scrollable_roomy_branded_preview_native_icon_controlled_separators"
        and status.get("phase_6_visual_refinement_tests")
        == "previously_passed_7_of_7_and_panel_render_contract_case_picker_icon_separator_followup_awaits_manual_validator"
    )
    checks["validation_attempt_history_contract"] = (
        status.get("phase_4_seventh_combined_validation_attempt")
        == "passed_phase4-20260812-195901_32_of_32_development_validation"
        and status.get("phase_5_seventh_combined_validation_attempt")
        == "failed_phase5-20260812-201822_5_passed_first_failure_06-types"
        and status.get("phase_5_post_seventh_attempt_corrections")
        == "pyyaml_typing_boundary_and_json_object_loader_corrected_awaiting_fresh_combined_manual_validation"
        and status.get("combined_seventh_validation_attempt")
        == "failed_phase4-phase5-20260812-195901_phase4_passed_phase5_failed_06-types"
        and status.get("phase_4_eighth_combined_validation_attempt")
        == "failed_phase4-20260812-202511_1_passed_first_failure_02-version"
        and status.get("phase_4_post_eighth_attempt_corrections")
        == "active_version_runtime_contract_aligned_to_current_status_awaiting_fresh_combined_manual_validation"
        and status.get("phase_5_eighth_combined_validation_attempt")
        == "not_started_because_phase4_failed_02-version"
        and status.get("combined_eighth_validation_attempt")
        == "failed_phase4-phase5-20260812-202511_phase4_failed_02-version_phase5_not_started"
        and status.get("phase_4_ninth_combined_validation_attempt")
        == "passed_phase4-20260812-202852_32_of_32_development_validation"
        and status.get("phase_5_ninth_combined_validation_attempt")
        == "failed_phase5-20260812-204823_23_passed_first_failure_22-cpu-build"
        and status.get("phase_5_post_ninth_attempt_corrections")
        == "docker_containerd_image_store_preflight_added_attestation_requirements_preserved_awaiting_environment_enablement_and_fresh_combined_manual_validation"
        and status.get("combined_ninth_validation_attempt")
        == "failed_phase4-phase5-20260812-202852_phase4_passed_phase5_failed_22-cpu-build"
        and status.get("phase_4_tenth_combined_validation_attempt")
        == "passed_phase4-20260813-000340_32_of_32_development_validation"
        and status.get("phase_5_tenth_combined_validation_attempt")
        == "passed_phase5-20260813-010531_41_of_41_release_preparation_validation"
        and status.get("combined_tenth_validation_attempt")
        == "passed_phase4-phase5-20260813-000340_phase4_32_of_32_phase5_41_of_41"
        and status.get("phase_6_first_validation_attempt")
        == "failed_phase6-20260813-031026_precommand_environment_identity_capture"
        and status.get("phase_6_second_validation_attempt")
        == "failed_phase6-20260813-031046_precommand_environment_identity_capture"
        and status.get("phase_6_post_second_attempt_correction")
        == "windows_powershell_environment_identity_access_corrected_fresh_phase6_rerun_required"
        and status.get("phase_6_third_validation_attempt")
        == "failed_phase6-20260813-031632_codex_sandbox_denied_python_before_command01"
        and status.get("phase_6_fourth_validation_attempt")
        == "failed_phase6-20260813-031655_9_passed_first_failure_08-format"
        and status.get("phase_6_post_fourth_attempt_correction")
        == "ruff_format_applied_to_10_reported_phase6_files_fresh_phase6_rerun_required"
        and status.get("phase_6_fifth_validation_attempt")
        == "failed_phase6-20260813-031800_11_passed_first_failure_10-phase6-gui"
        and status.get("phase_6_post_fifth_attempt_correction")
        == "unresolved_runtime_device_presented_as_not_assigned_and_compute_intent_changes_invalidate_prior_resolution_fresh_phase6_rerun_required"
        and status.get("phase_6_sixth_validation_attempt")
        == "passed_phase6-20260813-032036_19_of_19_gui_native_packaging_validation"
        and status.get("phase_6_seventh_validation_attempt")
        == "failed_phase6-20260813-183722_6_passed_first_failure_05-active-version"
        and status.get("phase_6_post_seventh_attempt_correction")
        == "product_version_added_to_active_status_identity_fresh_phase6_rerun_required"
        and status.get("phase_6_eighth_validation_attempt")
        == "failed_phase6-20260813-184612_8_passed_first_failure_07-ruff"
        and status.get("phase_6_post_eighth_attempt_correction")
        == "seven_unused_exception_bindings_and_qmessagebox_imports_removed_fresh_phase6_rerun_required"
        and status.get("phase_6_ninth_validation_attempt")
        == "failed_phase6-20260813-185633_9_passed_first_failure_08-format"
        and status.get("phase_6_post_ninth_attempt_correction")
        == "ruff_format_applied_to_27_reported_phase6_files_fresh_phase6_rerun_required"
        and status.get("phase_6_tenth_validation_attempt")
        == "failed_phase6-20260813-190343_10_passed_first_failure_09-unit"
        and status.get("phase_6_post_tenth_attempt_correction")
        == "two_stale_gui_contract_literals_aligned_to_no_redundant_document_header_and_method_verification_fresh_phase6_rerun_required"
        and status.get("phase_6_eleventh_validation_attempt")
        == "interrupted_phase6-20260813-191340_11_passed_command_10-phase6-gui_incomplete_false_pass_rejected"
        and status.get("phase_6_post_eleventh_attempt_correction")
        == "gui_tests_isolate_session_recovery_and_use_per_test_deadline_validator_requires_complete_01_through_17_sequence_fresh_phase6_rerun_required"
        and status.get("phase_6_twelfth_validation_attempt")
        == "failed_phase6-20260813-202657_11_passed_command_10_gui_18_of_21_passed_then_test18_teardown_timeout_124"
        and status.get("phase_6_post_twelfth_attempt_correction")
        == "focused_gui_fixture_uses_direct_test_owned_close_without_production_finalization_fresh_phase6_rerun_required"
        and status.get("phase_6_thirteenth_validation_attempt")
        == "failed_phase6-20260813-205516_11_passed_command_10_gui_20_of_21_passed_stale_memory_ceiling_wording_contract"
        and status.get("phase_6_post_thirteenth_attempt_correction")
        == "gui_contract_aligned_to_available_memory_safety_limit_empty_portfolio_is_prompt_not_error_fresh_phase6_rerun_required"
        and status.get("phase_6_fourteenth_validation_attempt")
        == "failed_phase6-20260813-212634_14_passed_first_failure_13-gui-render_missing_architecture_help_expectation"
        and status.get("phase_6_post_fourteenth_attempt_correction")
        == "offscreen_renderer_aligned_to_18_training_information_controls_with_explicit_key_diagnostics_fresh_phase6_rerun_required"
        and status.get("phase_6_fifteenth_validation_attempt")
        == "passed_phase6-20260813-215626_complete_19_of_19_113_tests_gui_native_packaging_source_stable"
        and status.get("phase_6_post_fifteenth_attempt_correction")
        == "visible_training_actions_exact_resume_model_library_friendly_failures_tsh_calo_only_startup_and_safe80_resource_preflight_parity_fresh_phase6_rerun_required"
        and status.get("phase_6_sixteenth_validation_attempt")
        == "failed_phase6-20260814-003900_8_passed_first_failure_07_ruff_undefined_offscreen_repository_root"
        and status.get("phase_6_post_sixteenth_attempt_correction")
        == "offscreen_resource_contract_uses_explicit_repository_root_fresh_phase6_rerun_required"
        and status.get("phase_6_seventeenth_validation_attempt")
        == "failed_phase6-20260814-004621_10_commands_73_unit_passed_22_of_24_gui_passed_stale_test_settings_and_visible_checksum_wording"
        and status.get("phase_6_post_seventeenth_attempt_correction")
        == "focused_gui_settings_cleared_per_window_and_resume_help_uses_product_integrity_language_fresh_phase6_rerun_required"
        and status.get("phase_6_eighteenth_validation_attempt")
        == "passed_phase6-20260814-004927_complete_17_stage_sequence_73_unit_24_gui_21_regression_9_integration_render_build_distribution_source_stable"
        and status.get("phase_6_nineteenth_validation_attempt")
        == "failed_phase6-20260814-130518_codex_sandbox_denied_python_before_command01"
        and status.get("phase_6_post_nineteenth_attempt_correction")
        == "complete_validator_relaunched_outside_sandbox_with_owner_authorization"
        and status.get("phase_6_twentieth_validation_attempt")
        == "failed_phase6-20260814-130547_9_passed_first_failure_08-format"
        and status.get("phase_6_post_twentieth_attempt_correction")
        == "ruff_format_applied_to_exact_5_reported_files_fresh_phase6_rerun_required"
        and status.get("phase_6_twenty_first_validation_attempt")
        == "failed_phase6-20260814-130631_10_passed_command_09_unit_73_passed_1_brittle_source_literal_failed"
        and status.get("phase_6_post_twenty_first_attempt_correction")
        == "checkbox_render_contract_aligned_to_stable_threshold_and_evidence_tokens_fresh_phase6_rerun_required"
        and status.get("phase_6_twenty_second_validation_attempt")
        == "failed_phase6-20260814-130806_13_stage_sequence_reached_gui_render_light_unchecked_perimeter_rgb_delta_79"
        and status.get("phase_6_post_twenty_second_attempt_correction")
        == "idle_checkbox_border_opacity_increased_fresh_phase6_rerun_required"
        and status.get("phase_6_twenty_third_validation_attempt")
        == "failed_phase6-20260814-131047_13_stage_sequence_reached_gui_render_light_unchecked_perimeter_rgb_delta_83"
        and status.get("phase_6_post_twenty_third_attempt_correction")
        == "idle_checkbox_border_made_fully_opaque_fresh_phase6_rerun_required"
        and status.get("phase_6_twenty_fourth_validation_attempt")
        == "failed_phase6-20260814-131435_13_stage_sequence_reached_gui_render_light_unchecked_perimeter_rgb_delta_82"
        and status.get("phase_6_post_twenty_fourth_attempt_correction")
        == "render_gate_samples_composited_host_surface_instead_of_transparent_child_pixels_fresh_phase6_rerun_required"
        and status.get("phase_6_twenty_fifth_validation_attempt")
        == "passed_phase6-20260814-131637_complete_17_stage_sequence_74_unit_25_gui_21_regression_9_integration_render_build_distribution_source_stable"
        and status.get("phase_6_twenty_sixth_validation_attempt")
        == "passed_phase6-20260814-132200_complete_17_stage_sequence_74_unit_25_gui_21_regression_9_integration_render_build_distribution_source_stable"
        and status.get("phase_6_post_twenty_sixth_source_changes")
        == "pass_superseded_by_later_progress_pause_extension_layout_removal_scroll_and_architecture_parameter_compatibility_source_fresh_validation_required"
    )

    index = _load_json(root / "STATUS_RECORD_INDEX.json")
    checks["status_index_points_to_active_record"] = (
        index.get("active_status") == "ACTIVE_DEVELOPMENT_STATUS.json"
        and index.get("active_version") == VERSION
        and "RELEASE_METADATA.json" in index.get("historical_records", [])
    )

    readme = (root / "README.md").read_text(encoding="utf-8")
    checks["readme_development_label"] = (
        readme.startswith(f"# CALO-RPD Studio v{DISPLAY_VERSION}\n")
        and "Active status: development only" in readme
    )

    gui_sources = (
        root / "calo_rpd_studio/app/main_window.py",
        root / "calo_rpd_studio/gui/navigation/sidebar.py",
        root / "calo_rpd_studio/gui/panels/application_settings_panel.py",
    )
    checks["gui_uses_product_version"] = all(
        "PRODUCT_VERSION" in path.read_text(encoding="utf-8") for path in gui_sources
    )
    checks["product_version_omits_build_stage"] = (
        PRODUCT_VERSION == VERSION.partition(".dev")[0] and "dev" not in PRODUCT_VERSION.lower()
    )

    launcher = (root / "calo_bootstrap/launcher.py").read_text(encoding="utf-8")
    checks["cli_version_is_explicit"] = all(
        marker in launcher for marker in ('"--version"', '"-V"', "DISPLAY_VERSION", "VERSION")
    )

    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    checks["container_version_label"] = (
        f"ARG APP_VERSION={VERSION}" in dockerfile
        and 'org.opencontainers.image.version="${APP_VERSION}"' in dockerfile
    )

    qualification_source = (
        root / "calo_rpd_studio/algorithms/calo/tsh_calo_qualification_campaign.py"
    ).read_text(encoding="utf-8")
    generic_qualification_source = (
        root / "calo_rpd_studio/algorithms/calo/policy_qualification.py"
    ).read_text(encoding="utf-8")
    checks["qualification_evidence_is_versioned"] = (
        "qualification-plan-v2-exact-pairs" in qualification_source
        and "PAIRED_ANALYSIS_SCHEMA_VERSION" in qualification_source
        and "RELATIVE_IMPROVEMENT_VERSION" in qualification_source
        and "source_tracked_clean" in qualification_source
        and '"source_identity": source_identity.to_dict()' in generic_qualification_source
    )

    checks["phase_allows_development_identity"] = (
        VERSION_STAGE == "development"
        and ".dev" in VERSION
        and "-dev." in DISPLAY_VERSION
        and "rc" not in VERSION
        and VERSION != "12.0.0"
    )
    passed = all(checks.values())
    return {
        "schema_version": "calo-active-version-verification-v1",
        "passed": passed,
        "expected": expected,
        "checks": checks,
        "details": details,
    }


def main() -> int:
    try:
        report = verify_active_version()
    except (OSError, KeyError, TypeError, ValueError) as exc:
        report = {
            "schema_version": "calo-active-version-verification-v1",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    sys.exit(main())
