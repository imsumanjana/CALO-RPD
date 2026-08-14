"""Phase 3 modern scientist-GUI contracts.

These tests are executed only by the user's Git-ignored Phase 3 validator.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from calo_rpd_studio.app.workspaces import (
    WORKSPACE_KEYS,
    WORKSPACE_SPECS,
    grouped_workspace_specs,
    migrate_workspace_ui,
)


ROOT = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_active_status_records_phase6_development_complete_after_combined_pass():
    import json

    payload = json.loads((ROOT / "ACTIVE_DEVELOPMENT_STATUS.json").read_text(encoding="utf-8"))
    assert payload["version"] == "12.0.0.dev1"
    assert payload["display_version"] == "12.0.0-dev.1"
    assert payload["product_version"] == "12.0.0"
    assert payload["phase"] == 6
    assert payload["phase_2_validation"] == "accepted_phase2-20260807-003828_15_of_15_passed"
    assert payload["phase_3_coding"] == (
        "implemented_tabbed_workspace_table_width_correction_validated_windows"
    )
    assert payload["phase_3_initial_validation"] == (
        "failed_phase3-20260807-045558_11_of_18_passed"
    )
    assert payload["phase_3_validation"] == (
        "windows_automated_accepted_current_tabbed_layout_linux_manually_accepted"
    )
    assert payload["phase_3_failed_validation_policy_workflows_executed"] is False
    assert payload["phase_3_corrections"] == "implemented"
    assert payload["phase_3_revalidation"] == ("accepted_phase3-20260807-052047_18_of_18_passed")
    assert payload["phase_3_revalidation_policy_workflows_executed"] is False
    assert payload["phase_3_remaining_windows_validation"] == (
        "accepted_phase3-remaining-windows-20260807-121530_10_of_10_automated_passed"
    )
    assert payload["phase_3_remaining_windows_validation_policy_workflows_executed"] is False
    assert payload["phase_3_remaining_windows_corrections"] == (
        "accepted_phase3-remaining-windows-20260807-121530"
    )
    assert payload["phase_3_tabbed_layout_refinement"] == (
        "windows_automated_accepted_linux_xcb_manually_accepted_by_owner"
    )
    assert payload["phase_3_tabbed_layout_refinement_windows_validation"] == (
        "accepted_phase3-remaining-windows-20260807-121530_10_of_10_passed"
    )
    assert payload["phase_3_linux_rendering"] == (
        "manually_validated_and_accepted_by_owner_no_automated_bundle_retained"
    )
    assert payload["phase_3_keyboard_accessibility_acceptance"] == (
        "accepted_windows_20260807_121530"
    )
    assert payload["phase_3_human_reviewer_input"] == "disabled_by_user_instruction"
    assert payload["phase_3_scientist_acceptance"] == ("not_inferred_automated_evidence_only")
    assert payload["phase_3_overall_gate"] == "closed_by_owner_manual_linux_xcb_acceptance"
    assert payload["phase_4_started"] is True
    assert payload["phase_4_coding"] == ("implemented_and_combined_validation_passed_before_phase6")
    assert payload["phase_4_validation"] == (
        "passed_phase4-20260813-000340_32_of_32_combined_validation"
    )
    assert payload["phase_4_first_combined_validation_attempt"] == (
        "interrupted_phase4-20260812-165006_partial_diagnostics_not_accepted"
    )
    assert payload["phase_4_post_attempt_corrections"] == (
        "implemented_awaiting_fresh_combined_manual_validation"
    )
    assert payload["phase_4_second_combined_validation_attempt"] == (
        "failed_phase4-20260812-182252_5_passed_first_failure_06-format"
    )
    assert payload["phase_4_post_second_attempt_corrections"] == (
        "ruff_format_applied_to_8_reported_files_awaiting_fresh_combined_manual_validation"
    )
    assert payload["phase_4_third_combined_validation_attempt"] == (
        "failed_phase4-20260812-182752_8_passed_first_failure_09-gui"
    )
    assert payload["phase_4_post_third_attempt_corrections"] == (
        "scientist_facing_development_stage_wording_removed_awaiting_fresh_combined_manual_validation"
    )
    assert payload["phase_4_fourth_combined_validation_attempt"] == (
        "failed_phase4-20260812-184454_14_passed_first_failure_14-distribution"
    )
    assert payload["phase_4_post_fourth_attempt_corrections"] == (
        "root_validation_evidence_distinguished_from_application_validation_package_awaiting_fresh_combined_manual_validation"
    )
    assert payload["phase_4_fifth_combined_validation_attempt"] == (
        "failed_phase4-20260812-185135_17_passed_first_failure_17-clean-smoke"
    )
    assert payload["phase_4_post_fifth_attempt_corrections"] == (
        "clean_install_import_assertion_scoped_to_checkout_source_and_clean_environment_awaiting_fresh_combined_manual_validation"
    )
    assert payload["phase_4_sixth_combined_validation_attempt"] == (
        "failed_phase4-20260812-190643_24_passed_first_failure_23-cuda-parity-30"
    )
    assert payload["phase_4_post_sixth_attempt_corrections"] == (
        "physical_cuda_development_evidence_tier_added_without_weakening_durable_qualification_awaiting_fresh_combined_manual_validation"
    )
    assert payload["phase_4_seventh_combined_validation_attempt"] == (
        "passed_phase4-20260812-195901_32_of_32_development_validation"
    )
    assert payload["phase_4_development_goal"] == (
        "completed_phase4_development_combined_validation_passed"
    )
    assert payload["phase_5_started"] is True
    assert payload["phase_5_coding"] == (
        "release_preparation_development_complete_combined_validation_passed"
    )
    assert payload["phase_5_validation"] == (
        "passed_phase5-20260813-010531_41_of_41_combined_validation"
    )
    assert payload["phase_5_first_combined_validation_attempt"] == (
        "not_started_because_phase4_did_not_pass"
    )
    assert payload["phase_5_second_combined_validation_attempt"] == (
        "not_started_because_phase4_failed_06-format"
    )
    assert payload["phase_5_third_combined_validation_attempt"] == (
        "not_started_because_phase4_failed_09-gui"
    )
    assert payload["phase_5_fourth_combined_validation_attempt"] == (
        "not_started_because_phase4_failed_14-distribution"
    )
    assert payload["phase_5_fifth_combined_validation_attempt"] == (
        "not_started_because_phase4_failed_17-clean-smoke"
    )
    assert payload["phase_5_sixth_combined_validation_attempt"] == (
        "not_started_because_phase4_failed_23-cuda-parity-30"
    )
    assert payload["phase_5_seventh_combined_validation_attempt"] == (
        "failed_phase5-20260812-201822_5_passed_first_failure_06-types"
    )
    assert payload["phase_5_post_seventh_attempt_corrections"] == (
        "pyyaml_typing_boundary_and_json_object_loader_corrected_awaiting_fresh_combined_manual_validation"
    )
    assert payload["combined_seventh_validation_attempt"] == (
        "failed_phase4-phase5-20260812-195901_phase4_passed_phase5_failed_06-types"
    )
    assert payload["phase_4_eighth_combined_validation_attempt"] == (
        "failed_phase4-20260812-202511_1_passed_first_failure_02-version"
    )
    assert payload["phase_4_post_eighth_attempt_corrections"] == (
        "active_version_runtime_contract_aligned_to_current_status_awaiting_fresh_combined_manual_validation"
    )
    assert payload["phase_5_eighth_combined_validation_attempt"] == (
        "not_started_because_phase4_failed_02-version"
    )
    assert payload["combined_eighth_validation_attempt"] == (
        "failed_phase4-phase5-20260812-202511_phase4_failed_02-version_phase5_not_started"
    )
    assert payload["phase_4_ninth_combined_validation_attempt"] == (
        "passed_phase4-20260812-202852_32_of_32_development_validation"
    )
    assert payload["phase_5_ninth_combined_validation_attempt"] == (
        "failed_phase5-20260812-204823_23_passed_first_failure_22-cpu-build"
    )
    assert payload["phase_5_post_ninth_attempt_corrections"] == (
        "docker_containerd_image_store_preflight_added_attestation_requirements_preserved_awaiting_environment_enablement_and_fresh_combined_manual_validation"
    )
    assert payload["combined_ninth_validation_attempt"] == (
        "failed_phase4-phase5-20260812-202852_phase4_passed_phase5_failed_22-cpu-build"
    )
    assert payload["phase_4_tenth_combined_validation_attempt"] == (
        "passed_phase4-20260813-000340_32_of_32_development_validation"
    )
    assert payload["phase_5_tenth_combined_validation_attempt"] == (
        "passed_phase5-20260813-010531_41_of_41_release_preparation_validation"
    )
    assert payload["combined_tenth_validation_attempt"] == (
        "passed_phase4-phase5-20260813-000340_phase4_32_of_32_phase5_41_of_41"
    )
    assert payload["combined_validation_harness_behavior"] == (
        "live_streaming_native_exit_code_fail_fast_first_failure_containerd_attestation_preflight"
    )
    assert payload["phase_5_release_policy_scope"] == "pending_explicit_decision"
    assert payload["phase_5_release_candidate"] is False
    assert payload["phase_5_final_release"] is False
    assert payload["phase_6_started"] is True
    assert payload["phase_6_goal"] == (
        "completed_model_finite_extension_implementation_complete_validation_pending"
    )
    assert payload["phase_6_coding"] == (
        "implemented_completed_model_repeatable_finite_extension_followup_validation_pending"
    )
    assert payload["phase_6_training_interface"] == (
        "ribbon_navigation_only_tsh_calo_only_training_no_architecture_selector_optional_settings_template_per_user_default_model_directory_registered_resume_scan_locations_visible_scientific_inputs_protected_118_300_locked_visible_default_off_exact_resume_in_pane_readiness_then_explicit_confirmed_start_rule_based_calo_remains_in_algorithm_selection_no_redundant_document"
    )
    assert payload["phase_6_training_input_help"] == (
        "complete_19_of_19_applicable_accessible_hover_focus_information_controls_with_directional_effects_suggested_ranges_hard_limits_and_lifecycle_boundaries_validated_phase6-20260814-131637"
    )
    assert payload["phase_6_native_execution"] == (
        "validated_first_class_windows_non_docker_launch"
    )
    assert payload["phase_6_validation"] == (
        "prior_pass_phase6-20260814-131637_superseded_by_progress_pause_and_completed_model_extension_followups_validation_pending"
    )
    assert payload["phase_6_checkbox_indicator_visibility"] == (
        "validated_global_palette_aware_borders_and_state_marks_light_dark_phase6-20260814-131637"
    )
    assert payload["phase_6_new_training_recovery_presentation"] == (
        "validated_automatic_recovery_status_separate_from_selected_interrupted_exact_resume_phase6-20260814-131637"
    )
    assert payload["phase_6_training_observability"] == (
        "implemented_structured_checkpoint_progress_durable_jsonl_detailed_activity_steps_validation_pending"
    )
    assert payload["phase_6_checkpoint_safe_pause"] == (
        "implemented_authenticated_checkpoint_boundary_pause_unlimited_resume_count_validation_pending"
    )
    assert payload["phase_6_training_evaluation_budget"] == (
        "finite_exact_immutable_across_pause_and_resume"
    )
    assert payload["phase_6_unbounded_training"] == (
        "not_implemented_finite_exact_evaluation_plan_preserved"
    )
    assert payload["phase_6_completed_model_extension"] == (
        "implemented_explicit_authenticated_repeatable_finite_segments_same_scientific_and_execution_plan_validation_pending"
    )
    assert payload["phase_6_extension_retained_state"] == (
        "model_optimizer_numpy_rng_torch_rng_ppo_updates_episode_receipts_device_memory_session_environment_rollout_collector_exact_accounting"
    )
    assert payload["phase_6_extension_count_limit"] == (
        "none_each_segment_explicit_and_finite"
    )
    assert payload["phase_6_extension_scientific_claim"] == (
        "no_improvement_superiority_qualification_registration_or_activation_inferred"
    )
    assert payload["phase_6_validation_policy_or_scientific_workflows_executed"] is False
    assert payload["phase_6_automated_human_acceptance"] == "not_inferred"
    assert payload["phase_6_visual_refinement"] == (
        "completed_phase6-panel-sweep-20260813-041700_16_workspace_panels_and_4_shell_renders"
    )
    assert payload["phase_6_workspace_navigation"] == (
        "complete_workspace_palette_permanent_expanded_ribbon_permanent_input_only_left_pane_scrollable_roomy_branded_preview_native_icon_controlled_separators"
    )
    assert payload["phase_6_visual_refinement_tests"] == (
        "previously_passed_7_of_7_and_panel_render_contract_case_picker_icon_separator_followup_awaits_manual_validator"
    )
    assert payload["phase_6_first_validation_attempt"] == (
        "failed_phase6-20260813-031026_precommand_environment_identity_capture"
    )
    assert payload["phase_6_second_validation_attempt"] == (
        "failed_phase6-20260813-031046_precommand_environment_identity_capture"
    )
    assert payload["phase_6_post_second_attempt_correction"] == (
        "windows_powershell_environment_identity_access_corrected_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_third_validation_attempt"] == (
        "failed_phase6-20260813-031632_codex_sandbox_denied_python_before_command01"
    )
    assert payload["phase_6_fourth_validation_attempt"] == (
        "failed_phase6-20260813-031655_9_passed_first_failure_08-format"
    )
    assert payload["phase_6_post_fourth_attempt_correction"] == (
        "ruff_format_applied_to_10_reported_phase6_files_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_fifth_validation_attempt"] == (
        "failed_phase6-20260813-031800_11_passed_first_failure_10-phase6-gui"
    )
    assert payload["phase_6_post_fifth_attempt_correction"] == (
        "unresolved_runtime_device_presented_as_not_assigned_and_compute_intent_changes_invalidate_prior_resolution_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_sixth_validation_attempt"] == (
        "passed_phase6-20260813-032036_19_of_19_gui_native_packaging_validation"
    )
    assert payload["phase_6_seventh_validation_attempt"] == (
        "failed_phase6-20260813-183722_6_passed_first_failure_05-active-version"
    )
    assert payload["phase_6_post_seventh_attempt_correction"] == (
        "product_version_added_to_active_status_identity_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_eighth_validation_attempt"] == (
        "failed_phase6-20260813-184612_8_passed_first_failure_07-ruff"
    )
    assert payload["phase_6_post_eighth_attempt_correction"] == (
        "seven_unused_exception_bindings_and_qmessagebox_imports_removed_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_ninth_validation_attempt"] == (
        "failed_phase6-20260813-185633_9_passed_first_failure_08-format"
    )
    assert payload["phase_6_post_ninth_attempt_correction"] == (
        "ruff_format_applied_to_27_reported_phase6_files_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_tenth_validation_attempt"] == (
        "failed_phase6-20260813-190343_10_passed_first_failure_09-unit"
    )
    assert payload["phase_6_post_tenth_attempt_correction"] == (
        "two_stale_gui_contract_literals_aligned_to_no_redundant_document_header_and_method_verification_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_eleventh_validation_attempt"] == (
        "interrupted_phase6-20260813-191340_11_passed_command_10-phase6-gui_incomplete_false_pass_rejected"
    )
    assert payload["phase_6_post_eleventh_attempt_correction"] == (
        "gui_tests_isolate_session_recovery_and_use_per_test_deadline_validator_requires_complete_01_through_17_sequence_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_twelfth_validation_attempt"] == (
        "failed_phase6-20260813-202657_11_passed_command_10_gui_18_of_21_passed_then_test18_teardown_timeout_124"
    )
    assert payload["phase_6_post_twelfth_attempt_correction"] == (
        "focused_gui_fixture_uses_direct_test_owned_close_without_production_finalization_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_thirteenth_validation_attempt"] == (
        "failed_phase6-20260813-205516_11_passed_command_10_gui_20_of_21_passed_stale_memory_ceiling_wording_contract"
    )
    assert payload["phase_6_post_thirteenth_attempt_correction"] == (
        "gui_contract_aligned_to_available_memory_safety_limit_empty_portfolio_is_prompt_not_error_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_fourteenth_validation_attempt"] == (
        "failed_phase6-20260813-212634_14_passed_first_failure_13-gui-render_missing_architecture_help_expectation"
    )
    assert payload["phase_6_post_fourteenth_attempt_correction"] == (
        "offscreen_renderer_aligned_to_18_training_information_controls_with_explicit_key_diagnostics_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_fifteenth_validation_attempt"] == (
        "passed_phase6-20260813-215626_complete_19_of_19_113_tests_gui_native_packaging_source_stable"
    )
    assert payload["phase_6_post_fifteenth_attempt_correction"] == (
        "visible_training_actions_exact_resume_model_library_friendly_failures_tsh_calo_only_startup_and_safe80_resource_preflight_parity_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_sixteenth_validation_attempt"] == (
        "failed_phase6-20260814-003900_8_passed_first_failure_07_ruff_undefined_offscreen_repository_root"
    )
    assert payload["phase_6_post_sixteenth_attempt_correction"] == (
        "offscreen_resource_contract_uses_explicit_repository_root_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_seventeenth_validation_attempt"] == (
        "failed_phase6-20260814-004621_10_commands_73_unit_passed_22_of_24_gui_passed_stale_test_settings_and_visible_checksum_wording"
    )
    assert payload["phase_6_post_seventeenth_attempt_correction"] == (
        "focused_gui_settings_cleared_per_window_and_resume_help_uses_product_integrity_language_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_eighteenth_validation_attempt"] == (
        "passed_phase6-20260814-004927_complete_17_stage_sequence_73_unit_24_gui_21_regression_9_integration_render_build_distribution_source_stable"
    )
    assert payload["phase_6_nineteenth_validation_attempt"] == (
        "failed_phase6-20260814-130518_codex_sandbox_denied_python_before_command01"
    )
    assert payload["phase_6_post_nineteenth_attempt_correction"] == (
        "complete_validator_relaunched_outside_sandbox_with_owner_authorization"
    )
    assert payload["phase_6_twentieth_validation_attempt"] == (
        "failed_phase6-20260814-130547_9_passed_first_failure_08-format"
    )
    assert payload["phase_6_post_twentieth_attempt_correction"] == (
        "ruff_format_applied_to_exact_5_reported_files_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_twenty_first_validation_attempt"] == (
        "failed_phase6-20260814-130631_10_passed_command_09_unit_73_passed_1_brittle_source_literal_failed"
    )
    assert payload["phase_6_post_twenty_first_attempt_correction"] == (
        "checkbox_render_contract_aligned_to_stable_threshold_and_evidence_tokens_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_twenty_second_validation_attempt"] == (
        "failed_phase6-20260814-130806_13_stage_sequence_reached_gui_render_light_unchecked_perimeter_rgb_delta_79"
    )
    assert payload["phase_6_post_twenty_second_attempt_correction"] == (
        "idle_checkbox_border_opacity_increased_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_twenty_third_validation_attempt"] == (
        "failed_phase6-20260814-131047_13_stage_sequence_reached_gui_render_light_unchecked_perimeter_rgb_delta_83"
    )
    assert payload["phase_6_post_twenty_third_attempt_correction"] == (
        "idle_checkbox_border_made_fully_opaque_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_twenty_fourth_validation_attempt"] == (
        "failed_phase6-20260814-131435_13_stage_sequence_reached_gui_render_light_unchecked_perimeter_rgb_delta_82"
    )
    assert payload["phase_6_post_twenty_fourth_attempt_correction"] == (
        "render_gate_samples_composited_host_surface_instead_of_transparent_child_pixels_fresh_phase6_rerun_required"
    )
    assert payload["phase_6_twenty_fifth_validation_attempt"] == (
        "passed_phase6-20260814-131637_complete_17_stage_sequence_74_unit_25_gui_21_regression_9_integration_render_build_distribution_source_stable"
    )


def test_phase3_grouping_preserves_all_stable_workspace_keys():
    expected_keys = (
        "dashboard",
        "calo_intelligence",
        "power_system",
        "orpd",
        "algorithms",
        "portfolio",
        "scenarios",
        "experiment",
        "live_optimization",
        "statistics",
        "results",
        "validation",
        "publication",
        "resume_center",
        "settings",
        "benchmark",
    )
    assert WORKSPACE_KEYS == expected_keys
    groups = {
        group: tuple(spec.key for _index, spec in members)
        for group, members in grouped_workspace_specs()
    }
    assert groups == {
        "Home": ("dashboard", "resume_center"),
        "Model": (
            "calo_intelligence",
            "power_system",
            "orpd",
            "algorithms",
            "scenarios",
        ),
        "Study": ("portfolio", "experiment", "live_optimization"),
        "Evidence": ("results", "statistics", "validation", "benchmark", "publication"),
        "System": ("settings",),
    }


def test_historical_workspace_restore_remains_key_authoritative():
    migrated, report = migrate_workspace_ui(
        {"workspace_schema_version": 3, "workspace_key": "results", "workspace_index": 0}
    )
    assert migrated["workspace_key"] == "results"
    assert report.target_key == "results"
    assert len(WORKSPACE_SPECS) == 16


def test_dashboard_moves_study_form_out_of_visible_tabs():
    source = _source("calo_rpd_studio/gui/panels/dashboard_panel.py")
    assert '"Next required action"' in source
    assert '"Recent work and evidence"' in source
    assert "self.legacy_study_setup = study_tab" in source
    assert 'addTab(_scrollable_tab(study_tab), "Study Setup")' not in source
    assert "self.activity_drawer = DisclosurePanel(" in source


def test_experiment_manager_uses_exact_seven_step_study_setup():
    source = _source("calo_rpd_studio/gui/panels/experiment_manager_panel.py")
    expected = (
        '"Case"',
        '"Formulation"',
        '"Algorithms"',
        '"Budget + runs"',
        '"Scenarios"',
        '"Validate + outputs"',
        '"Review + launch"',
    )
    for title in expected:
        assert title in source
    assert "self.study_setup_workflow = StudySetupWorkflow(" in source
    assert "self.evolution_drawer = DisclosurePanel(" in source
    assert "self.queue_drawer = DisclosurePanel(" in source


def test_compact_input_policy_has_bounded_scientist_controls():
    source = _source("calo_rpd_studio/gui/widgets/form_density.py")
    assert "FORM_CONTENT_MAX_WIDTH = 880" in source
    assert "ORDINARY_INPUT_MAX_WIDTH = 480" in source
    assert "SELECTOR_MAX_WIDTH = 420" in source
    assert "SCALAR_INPUT_MAX_WIDTH = 240" in source
    assert "_attach_expanded_editor(widget)" in source


def test_navigation_has_group_persistence_search_and_compact_rail():
    source = _source("calo_rpd_studio/gui/navigation/sidebar.py")
    assert 'self.search.setPlaceholderText("Find a workspace")' in source
    assert "navigation/compact" in source
    assert "navigation/group/{group_name}" in source
    assert 'self._workflow_states[index] != "locked"' in source
    assert "setAccessibleDescription" in source


def test_light_and_dark_themes_define_phase3_focus_and_semantic_surfaces():
    for relative in (
        "calo_rpd_studio/gui/themes/light.py",
        "calo_rpd_studio/gui/themes/dark.py",
    ):
        source = _source(relative)
        assert "#NavigationGroupHeader" in source
        assert "#DisclosurePanel" in source
        assert "#StudySetupWorkflow" in source
        assert "#NextActionStatus" in source
        assert "QPushButton:focus" in source
        assert "font-variant-numeric" not in source
    font_source = _source("calo_rpd_studio/gui/themes/runtime_fonts.py")
    assert "QFontDatabase.addApplicationFont" in font_source
    assert "supports_validation_sample" in font_source


def test_render_evidence_script_gates_glyphs_sizes_themes_and_compact_inputs():
    source = _source("calo_rpd_studio/scripts/validate_phase3_gui_render.py")
    assert 'SCHEMA_VERSION = "calo-phase3-gui-render-evidence-v1"' in source
    assert 'os.environ["QT_SCALE_FACTOR"]' in source
    assert "inFontUcs4" in source
    assert '"compact_input_violations"' in source
    assert '"replacement_character_hits"' in source
    assert '"application_font"' in source
    assert 'choices=("light", "dark")' in source
    assert 'choices=("offscreen", "xcb")' in source


def test_remaining_phase3_evidence_tool_is_non_scientific_and_source_bound():
    source = _source("calo_rpd_studio/scripts/validate_phase3_workspace_accessibility.py")
    assert 'SCHEMA_VERSION = "calo-phase3-workspace-accessibility-evidence-v3"' in source
    assert '"scientific_actions_executed": False' in source
    assert '"policy_workflows_executed": False' in source
    assert '"policy_training_executed": False' in source
    assert '"policy_evaluation_executed": False' in source
    assert '"protected_cases_opened": False' in source
    assert '"keyboard_interactions"' in source
    assert '"section_tab_interactions"' in source
    assert "_section_tab_interactions(window, application, output)" in source
    assert '"tree_widget_layout_checks"' in source
    assert "_tree_widget_layout_checks(" in source
    assert '"contrast_checks"' in source
    assert '"workspace_evidence"' in source
    assert '"qt_platform_matches_request"' in source
    assert '"accessible_name": widget.accessibleName()' in source
    assert '"text": " | ".join(_widget_text(widget))' in source


def test_results_and_settings_use_responsive_long_value_layouts():
    results_source = _source("calo_rpd_studio/gui/panels/results_explorer_panel.py")
    assert "filters = QGridLayout()" in results_source
    assert "filter_controls = (" in results_source
    assert "actions.addWidget(self.restore_workspace_button)" in results_source

    settings_source = _source("calo_rpd_studio/gui/panels/application_settings_panel.py")
    assert 'self.database_path.setProperty("fullWidthInput", True)' in settings_source
    assert "self.database_path.setReadOnly(True)" in settings_source
    assert "self.section_tabs.add_section(" in settings_source
    assert '"Appearance"' in settings_source
    assert '"Experiment history"' in settings_source
    assert '"Application"' in settings_source


def test_stack_heavy_workspaces_use_shared_accessible_tabs():
    tabs_source = _source("calo_rpd_studio/gui/widgets/workspace_tabs.py")
    assert "class WorkspaceTabs(QTabWidget)" in tabs_source
    assert 'self.setObjectName("WorkspaceSectionTabs")' in tabs_source
    assert "self.setTabToolTip(index, description)" in tabs_source

    expected_tabs = {
        "calo_rpd_studio/gui/panels/orpd_formulation_panel.py": (
            '"Objective"',
            '"Control variables"',
            '"Constraint policy"',
        ),
        "calo_rpd_studio/gui/panels/robust_scenarios_panel.py": (
            '"Scenario generator"',
            '"Robust objective"',
        ),
        "calo_rpd_studio/gui/panels/portfolio_manager_panel.py": (
            '"Definition"',
            '"Requested outputs"',
            '"Reuse and validation"',
            '"Derived plan"',
        ),
        "calo_rpd_studio/gui/panels/application_settings_panel.py": (
            '"Appearance"',
            '"Experiment history"',
            '"Application"',
        ),
        "calo_rpd_studio/gui/panels/benchmark_campaign_panel.py": (
            '"Method verification"',
            '"Campaign design"',
            '"Task queue"',
            '"Evidence package"',
        ),
    }
    for relative, titles in expected_tabs.items():
        source = _source(relative)
        assert "WorkspaceTabs(" in source
        assert "add_section(" in source
        for title in titles:
            assert title in source


def test_portfolio_output_tree_uses_the_full_available_width():
    source = _source("calo_rpd_studio/gui/panels/portfolio_manager_panel.py")
    assert 'self.outputs.setObjectName("PortfolioRequestedOutputs")' in source
    assert "output_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)" in source
    assert (
        "output_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)" in source
    )


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PyQt6") is None,
    reason="PyQt6 is not installed",
)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_application():
    from PyQt6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application


class _MemorySettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default=None):
        return self.values.get(key, default)

    def set_value(self, key: str, value: object) -> None:
        self.values[key] = value


def test_grouped_sidebar_hides_locked_children_and_persists_density(qt_application):
    from calo_rpd_studio.gui.navigation.sidebar import NavigationSidebar

    settings = _MemorySettings()
    sidebar = NavigationSidebar(WORKSPACE_SPECS, settings)
    try:
        assert set(sidebar.group_headers) == {"Home", "Model", "Study", "Evidence", "System"}
        assert len(sidebar.buttons) == 16
        sidebar.set_workflow_state(10, "locked", "Complete result generation first")
        assert sidebar.buttons[10].isHidden()
        assert "hidden" in sidebar.blocked_summary.text()
        sidebar.set_compact(True)
        assert settings.values["navigation/compact"] is True
        assert sidebar.maximumWidth() == 76
        sidebar.set_compact(False)
        sidebar.search.setText("publication")
        assert not sidebar.buttons[12].isHidden()
        assert sidebar.buttons[0].isHidden()
    finally:
        sidebar.close()


def test_compact_policy_caps_short_inputs_and_preserves_read_only_viewer(qt_application):
    from PyQt6.QtWidgets import QLineEdit, QPlainTextEdit, QSpinBox, QVBoxLayout, QWidget

    from calo_rpd_studio.gui.widgets.form_density import apply_compact_input_policy

    root = QWidget()
    layout = QVBoxLayout(root)
    short_text = QLineEdit()
    scalar = QSpinBox()
    report = QPlainTextEdit()
    report.setReadOnly(True)
    layout.addWidget(short_text)
    layout.addWidget(scalar)
    layout.addWidget(report)
    try:
        assert apply_compact_input_policy(root, "compact") == "compact"
        assert short_text.maximumWidth() == 480
        assert scalar.maximumWidth() == 240
        assert report.maximumHeight() == 16_777_215
        assert short_text.minimumHeight() == 40
        assert apply_compact_input_policy(root, "comfortable") == "comfortable"
        assert short_text.minimumHeight() == 44
    finally:
        root.close()
