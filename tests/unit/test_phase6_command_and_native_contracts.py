from __future__ import annotations

from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_phase6_command_registry_has_one_stable_authority():
    from calo_rpd_studio.gui.command_registry import COMMAND_SPECS

    identifiers = [item.command_id for item in COMMAND_SPECS]
    categories = {item.category for item in COMMAND_SPECS}
    assert len(identifiers) == len(set(identifiers))
    assert "view.context" not in identifiers
    assert "view.ribbon" not in identifiers
    assert categories == {
        "Home",
        "Workspace",
        "Experiment",
        "Algorithms",
        "Compute",
        "Results",
        "Policies",
        "View",
        "Help",
    }
    training = next(item for item in COMMAND_SPECS if item.command_id == "policies.training")
    assert training.handler == "training"
    assert "not selected for experiments automatically" in training.tooltip.lower()
    visible_command_text = " ".join(
        f"{item.label} {item.tooltip}" for item in COMMAND_SPECS
    ).lower()
    assert "a-e" not in visible_command_text
    assert "f-off" not in visible_command_text
    assert "production-candidate" not in visible_command_text
    assert "development" not in visible_command_text
    assert "legacy" not in visible_command_text
    assert "phase 4" not in visible_command_text
    assert "source-bound" not in visible_command_text


def test_product_version_hides_internal_build_stage_without_changing_build_identity():
    import json

    from calo_rpd_studio.version import DISPLAY_VERSION, PRODUCT_VERSION, VERSION_STAGE

    assert PRODUCT_VERSION == "12.0.0"
    assert "dev" not in PRODUCT_VERSION.lower()
    assert VERSION_STAGE == "development"
    assert "dev" in DISPLAY_VERSION.lower()
    status = json.loads((ROOT / "ACTIVE_DEVELOPMENT_STATUS.json").read_text(encoding="utf-8"))
    assert status["version"] == "12.0.0.dev1"
    assert status["display_version"] == DISPLAY_VERSION
    assert status["product_version"] == PRODUCT_VERSION
    assert status["stage"] == VERSION_STAGE


def test_policy_display_language_hides_internal_lifecycle_reason():
    from calo_rpd_studio.algorithms.calo.policy_readiness import (
        GoverningPolicyStatus,
        governing_policy_user_message,
    )

    technical_reason = (
        "Pre-freeze development candidate requires A-E/F-off qualification and activation."
    )
    status = GoverningPolicyStatus(False, "development_only", technical_reason)
    message = governing_policy_user_message(status).lower()

    assert status.reason == technical_reason
    assert "select a verified, compatible tsh-calo policy" in message
    for token in ("development", "candidate", "a-e", "f-off", "qualification", "activation"):
        assert token not in message


def test_independent_training_arguments_keep_check_and_start_separate():
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingLaunchModel

    model = TrainingLaunchModel()
    for key, value in {
        "plan": "plan.json",
        "output": "new-output",
    }.items():
        model.set_value(key, value)
    check = model.arguments(check=True)
    start = model.arguments(check=False)
    assert "--check" in check
    assert "--output" not in check
    assert "--check" not in start
    assert start[-2:] == ["--output", "new-output"]
    assert "qualify" not in " ".join(start).lower()
    assert "activate" not in " ".join(start).lower()
    assert "--development-freeze" not in check
    assert "--phase4-acceptance" not in check


def test_fresh_plan_uses_builtin_architecture_and_has_no_governance_path_inputs(monkeypatch):
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingLaunchModel
    from calo_rpd_studio.compute.source_identity import SourceIdentity
    import calo_rpd_studio.compute.source_identity as source_identity

    monkeypatch.setattr(
        source_identity,
        "resolve_source_identity",
        lambda: SourceIdentity("a" * 40, False, "git"),
    )
    model = TrainingLaunchModel()

    model.create_plan(
        campaign_id="fresh-gui-plan",
        development_cases=["case30", "case57"],
        member_count=2,
        master_seed=2026,
        population_size=20,
        max_evaluations=10_000,
        requested_device="auto",
        allow_cpu_fallback=True,
        training={
            "hidden_dim": 64,
            "graph_steps": 2,
            "learning_rate": 0.0003,
            "ppo_epochs": 4,
            "clip_ratio": 0.2,
            "value_weight": 0.5,
            "entropy_weight": 0.01,
            "gradient_norm": 0.5,
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
        },
    )

    assert model.plan_error == ""
    assert model.plan_payload is not None
    assert model.plan_payload["development_freeze_sha256"] == ""
    assert model.plan_payload["phase4_acceptance_sha256"] == ""
    assert model.plan_payload["feature_flags"]["allow_experimental_components"] is False
    assert model.plan_payload["feature_flags"]["population_schedule"] is False
    assert model.values["architecture"] == "tsh_calo"
    assert set(model.values) == {"architecture", "plan", "output"}
    with pytest.raises(KeyError):
        model.set_value("development_freeze", "user-selected.json")


def test_calo_architecture_does_not_route_to_the_policy_trainer():
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingLaunchModel

    model = TrainingLaunchModel()
    model.set_value("architecture", "calo")

    with pytest.raises(ValueError, match="does not require policy training"):
        model.arguments(check=True)


def test_phase6_shell_contract_keeps_inputs_and_ribbon_permanent():
    source = (ROOT / "calo_rpd_studio/app/main_window.py").read_text(encoding="utf-8")
    ribbon = (ROOT / "calo_rpd_studio/gui/widgets/ribbon_bar.py").read_text(encoding="utf-8")
    documents = (ROOT / "calo_rpd_studio/gui/widgets/document_workspace.py").read_text(
        encoding="utf-8"
    )
    application = (ROOT / "calo_rpd_studio/app/application.py").read_text(encoding="utf-8")
    context = (ROOT / "calo_rpd_studio/gui/widgets/context_pane.py").read_text(encoding="utf-8")
    assert "NoDockWidgetFeatures" in source
    assert "toggleViewAction().setEnabled(False)" in source
    assert "return False" in ribbon
    assert "MainPreviewScroll" in documents
    assert "self.tabBar().setVisible(self.count() > 1)" in documents
    assert "app.setWindowIcon(application_icon())" in application
    assert "All eligible bundled cases" in context
    assert "PROTECTED_HOLDOUT_BUS_COUNTS" in context
    assert "_TRAINING_INPUT_HELP" in context
    assert "TrainingInfoButton" in context
    assert "Training foundation" not in context
    assert "Development freeze" not in context
    assert "Phase 4 acceptance" not in context
    for relative in (
        "calo_rpd_studio/gui/themes/light.py",
        "calo_rpd_studio/gui/themes/dark.py",
    ):
        theme = (ROOT / relative).read_text(encoding="utf-8")
        assert "QMainWindow::separator" in theme


def test_native_entry_and_repository_launcher_do_not_install_on_routine_start():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    assert scripts["calo-rpd-native"] == "calo_rpd_studio.app.application:main"
    launcher = (ROOT / "Launch-CALO-RPD.ps1").read_text(encoding="utf-8")
    assert "calo_rpd_studio.app.application" in launcher
    assert ".venv\\Scripts\\python.exe" in launcher
    assert "pip install" not in launcher.lower()
    assert "bootstrap.py --setup" in launcher


def test_disabled_primary_style_is_more_specific_in_both_themes():
    for relative in (
        "calo_rpd_studio/gui/themes/light.py",
        "calo_rpd_studio/gui/themes/dark.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "QPushButton#PrimaryButton:disabled" in source
        assert source.rfind("QPushButton#PrimaryButton:disabled") > source.find(
            "QPushButton#PrimaryButton {"
        )


def test_spin_controls_use_modern_vector_arrows_and_themed_interaction_states():
    style_source = (ROOT / "calo_rpd_studio/gui/themes/modern_spin_style.py").read_text(
        encoding="utf-8"
    )
    manager_source = (ROOT / "calo_rpd_studio/gui/themes/theme_manager.py").read_text(
        encoding="utf-8"
    )
    assert "CC_SpinBox" in style_source
    assert "SC_SpinBoxUp" in style_source
    assert "SC_SpinBoxDown" in style_source
    assert "super().drawComplexControl" in style_source
    assert "self._draw_arrowhead" in style_source
    assert "QPainter.RenderHint.Antialiasing" in style_source
    assert "QPalette.ColorRole.Highlight" in style_source
    assert 'ModernSpinBoxStyle("Fusion")' in manager_source
    for relative in (
        "calo_rpd_studio/gui/themes/light.py",
        "calo_rpd_studio/gui/themes/dark.py",
    ):
        theme = (ROOT / relative).read_text(encoding="utf-8")
        assert "QSpinBox::up-button" in theme
        assert "QDoubleSpinBox::down-button" in theme
        assert "width: 24px" in theme
        assert "padding-right: 30px" in theme
        assert "up-button:hover" in theme
        assert "down-button:pressed" in theme
        assert "down-button:disabled" in theme


def test_ribbon_category_buttons_use_an_underline_without_native_tab_painting():
    ribbon = (ROOT / "calo_rpd_studio/gui/widgets/ribbon_bar.py").read_text(encoding="utf-8")
    assert "QTabBar" not in ribbon
    assert 'button.setObjectName("RibbonCategoryButton")' in ribbon
    for relative in (
        "calo_rpd_studio/gui/themes/light.py",
        "calo_rpd_studio/gui/themes/dark.py",
    ):
        theme = (ROOT / relative).read_text(encoding="utf-8")
        selected = theme.split("QPushButton#RibbonCategoryButton:checked {", 1)[1].split("}", 1)[0]
        assert "background: transparent" in selected
        assert "border-bottom: 2px solid" in selected


def test_phase6_validation_and_local_logs_remain_git_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/validation/" in ignore


def test_phase6_validator_fails_closed_on_interruption_and_gui_tests_are_bounded():
    validator = (ROOT / "validation/Validate-Phase6.ps1").read_text(encoding="utf-8")
    gui_tests = (ROOT / "tests/gui/test_phase6_ribbon_workspace.py").read_text(encoding="utf-8")

    assert "$ExpectedCommandIds" in validator
    assert "$CommandSequenceComplete" in validator
    assert "-and\n        $CommandSequenceComplete -and" in validator
    assert '"-m", "pytest", "-vv"' in validator
    assert "threading.Timer(120.0, abort_stuck_test)" in gui_tests
    assert "os._exit(124)" in gui_tests
    assert 'tmp_path / "session-recovery"' in gui_tests
    assert '"_initial_system_scan", lambda self: None' in gui_tests
    assert '"_check_unfinished_work", lambda self: None' in gui_tests
    assert '"closeEvent", lambda self, event: event.accept()' in gui_tests
    assert "before_close_func=close_focused_test_window" in gui_tests
    assert "widget.activity_center.detach_logging()" in gui_tests


def test_empty_portfolio_selection_is_an_input_prompt_not_a_logged_failure():
    source = (ROOT / "calo_rpd_studio/gui/panels/portfolio_manager_panel.py").read_text(
        encoding="utf-8"
    )
    empty_guard = source.index("if not portfolio.requested_outputs:")
    planner_call = source.index("PortfolioPlanner.plan(", empty_guard)
    technical_log = source.index('log_technical_error("portfolio planning", exc)', planner_call)

    assert empty_guard < planner_call < technical_log
    assert "Select at least one output to preview the portfolio plan." in source


def test_offscreen_renderer_requires_the_base_architecture_information_control():
    renderer = (ROOT / "calo_rpd_studio/scripts/validate_phase6_gui_contracts.py").read_text(
        encoding="utf-8"
    )
    expected_help = renderer.split("expected_help = {", 1)[1].split("}", 1)[0]

    assert '"architecture"' in expected_help
    assert "observed_help = set(training_editor.info_buttons)" in renderer
    assert "missing={sorted(expected_help - observed_help)}" in renderer
    assert "unexpected={sorted(observed_help - expected_help)}" in renderer


def test_policy_training_process_actions_are_visible_in_the_input_pane():
    context = (ROOT / "calo_rpd_studio/gui/widgets/context_pane.py").read_text(encoding="utf-8")
    main_window = (ROOT / "calo_rpd_studio/app/main_window.py").read_text(encoding="utf-8")

    assert 'self.training_action_button = QPushButton("Check readiness")' in context
    assert '"Start training", "Start the checked new-policy training run.", True' in context
    assert "self.training_controller.check_readiness()" in context
    assert "self.training_controller.start_training()" in context
    assert "_training_context_was_visible" not in main_window
    assert "_update_training_command" not in main_window
    assert "self.context_pane.activate_training()" not in main_window


def test_phase6_distribution_contract_rejects_local_evidence_and_policy_artifacts():
    from calo_rpd_studio.scripts.verify_phase6_distribution import (
        _reject_local_evidence_or_policy_artifacts,
    )

    with pytest.raises(ValueError, match="local validation evidence"):
        _reject_local_evidence_or_policy_artifacts(
            {"validation/logs/phase6/summary.json"}, label="Sdist"
        )
    with pytest.raises(ValueError, match="policy/training data"):
        _reject_local_evidence_or_policy_artifacts(
            {"calo_rpd_studio/data/trained_models/candidate/policy.pt"}, label="Wheel"
        )
