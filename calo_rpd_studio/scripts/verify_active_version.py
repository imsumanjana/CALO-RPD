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
        == "completed_visible_in_pane_training_action_correction_validation_pending"
        and status.get("phase_6_coding")
        == "implemented_visible_check_readiness_then_start_training_action_awaiting_validation"
        and status.get("phase_6_training_interface")
        == "ribbon_navigation_only_builtin_tsh_calo_architecture_optional_existing_plan_template_visible_scientific_inputs_protected_118_300_locked_in_pane_readiness_then_explicit_confirmed_start_no_redundant_document"
        and status.get("phase_6_training_input_help")
        == "complete_18_of_18_accessible_hover_focus_information_controls_with_directional_effects_suggested_ranges_hard_limits_and_lifecycle_boundaries_manual_validation_pending"
        and status.get("phase_6_native_execution")
        == "validated_first_class_windows_non_docker_launch"
        and status.get("phase_6_validation")
        == "passed_phase6-20260813-215626_superseded_by_visible_training_action_followup_validation_pending"
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
        == "visible_persistent_in_pane_check_readiness_then_start_training_action_ribbon_navigation_only_fresh_phase6_rerun_required"
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
