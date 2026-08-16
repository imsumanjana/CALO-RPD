from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


class _MemorySettings:
    def __init__(self):
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def set_value(self, key, value):
        self.values[key] = value


def test_phase6_command_registry_has_one_stable_authority():
    from calo_rpd_studio.gui.command_registry import COMMAND_SPECS, RIBBON_CATEGORY_ORDER

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
    assert RIBBON_CATEGORY_ORDER == (
        "Home",
        "Algorithms",
        "Workspace",
        "Experiment",
        "Compute",
        "Results",
        "Policies",
        "View",
        "Help",
    )
    assert tuple(item.label for item in COMMAND_SPECS if item.category == "Home") == (
        "Overview",
        "Open",
        "Save",
    )
    assert not {
        "home.resume",
        "home.find",
        "workspace.resume",
        "policies.resume",
        "policies.status",
        "compute.refresh",
        "results.live",
        "results.statistics",
        "workspace.dashboard",
        "workspace.calo",
        "workspace.power",
        "workspace.orpd",
        "workspace.algorithms",
        "workspace.scenarios",
        "workspace.live",
        "workspace.results",
        "workspace.statistics",
        "experiment.setup",
        "experiment.portfolio",
        "experiment.run",
        "results.validation",
        "results.benchmark",
        "results.publication",
        "help.settings",
    }.intersection(identifiers)
    labels = [item.label for item in COMMAND_SPECS]
    assert len(labels) == len(set(labels))
    assert tuple(
        item.command_id for item in COMMAND_SPECS if item.category == "Workspace"
    ) == (
        "workspace.portfolio",
        "workspace.study",
        "workspace.validation",
        "workspace.benchmark",
        "workspace.publication",
        "workspace.settings",
    )
    assert tuple(
        item.command_id for item in COMMAND_SPECS if item.category == "Experiment"
    ) == (
        "experiment.power",
        "experiment.formulation",
        "experiment.scenarios",
        "experiment.stop",
    )
    assert tuple(
        item.command_id for item in COMMAND_SPECS if item.category == "Compute"
    ) == (
        "compute.settings",
        "compute.device",
        "compute.live",
        "compute.statistics",
    )
    assert tuple(
        item.command_id for item in COMMAND_SPECS if item.category == "Results"
    ) == ("results.explorer",)
    assert tuple(
        item.command_id for item in COMMAND_SPECS if item.category == "Policies"
    ) == ("policies.training",)
    assert tuple(
        item.command_id for item in COMMAND_SPECS if item.category == "Help"
    ) == ("help.guide", "help.about")
    assert all(item.workspace != "resume_center" for item in COMMAND_SPECS)
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


def test_training_model_library_refresh_invalidates_cached_file_observations(tmp_path):
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingModelLibrary

    library = TrainingModelLibrary(_MemorySettings(), default_directory=tmp_path)
    notifications = []
    library.changed.connect(lambda: notifications.append("changed"))
    library._candidate_integrity_cache[("stale",)] = ("candidate", "", None)

    library.refresh()

    assert library._candidate_integrity_cache == {}
    assert notifications == ["changed"]


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

    technical_reason = "Runtime ABI mismatch requires A-E/F-off qualification review."
    status = GoverningPolicyStatus(False, "incompatible", technical_reason)
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
    resume = model.arguments(check=False, resume=True)
    extension_check = model.arguments(check=True, extend=True)
    extension_start = model.arguments(check=False, extend=True)
    assert "--check" in check
    assert "--output" not in check
    assert "--check" not in start
    assert start[-2:] == ["--output", "new-output"]
    assert resume[-3:] == ["--output", "new-output", "--resume"]
    assert extension_check[-4:] == ["--check", "--output", "new-output", "--extend"]
    assert extension_start[-3:] == ["--output", "new-output", "--extend"]
    assert "qualify" not in " ".join(start).lower()
    assert "activate" not in " ".join(start).lower()
    assert "--development-freeze" not in check
    assert "--phase4-acceptance" not in check


def test_fresh_plan_uses_builtin_architecture_and_has_no_governance_path_inputs(monkeypatch):
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingLaunchModel
    from calo_rpd_studio.compute.source_identity import SourceIdentity
    import calo_rpd_studio.compute.source_identity as source_identity

    source_roots = []

    def resolved_source_identity(*, cwd=None):
        source_roots.append(Path(cwd).resolve())
        return SourceIdentity("a" * 40, False, "git")

    monkeypatch.setattr(
        source_identity,
        "resolve_source_identity",
        resolved_source_identity,
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
    assert source_roots == [ROOT.resolve()]
    assert model.plan_payload["development_freeze_sha256"] == ""
    assert model.plan_payload["phase4_acceptance_sha256"] == ""
    assert model.plan_payload["feature_flags"]["allow_experimental_components"] is False
    assert model.plan_payload["feature_flags"]["population_schedule"] is False
    assert model.plan_payload["resource_envelope"]["rollout_capacity"] == 499
    model.set_resource_design(population_size=50, max_evaluations=10_000)
    assert model.plan_payload["population_size"] == 50
    assert model.plan_payload["max_evaluations"] == 10_000
    assert model.plan_payload["resource_envelope"]["maximum_population_size"] == 50
    assert model.plan_payload["resource_envelope"]["rollout_capacity"] == 199
    assert set(model.values) == {"plan", "output"}
    with pytest.raises(KeyError):
        model.set_value("architecture", "calo")
    with pytest.raises(KeyError):
        model.set_value("development_freeze", "user-selected.json")


def test_rule_based_calo_remains_an_ordinary_algorithm_not_a_training_input():
    from calo_rpd_studio.algorithms.registry import SPECS
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingLaunchModel

    assert "CALO" in SPECS
    model = TrainingLaunchModel()
    assert "architecture" not in model.values
    with pytest.raises(KeyError, match="Unknown scientist-facing training input"):
        model.set_value("architecture", "calo")


def test_training_model_library_scans_default_and_explicit_locations(tmp_path):
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingModelLibrary

    settings = _MemorySettings()
    default = tmp_path / "private-models"
    external = tmp_path / "external-models"
    resumable = external / "campaign-one"
    unsafe_interruption = external / "campaign-unsafe-window"
    completed = default / "campaign-complete"
    completed_without_candidate = default / "campaign-complete-without-candidate"
    resumable.mkdir(parents=True)
    unsafe_interruption.mkdir(parents=True)
    completed.mkdir(parents=True)
    completed_without_candidate.mkdir(parents=True)
    (resumable / "training_plan.json").write_text(
        json.dumps({"campaign_id": "campaign-one"}), encoding="utf-8"
    )
    (resumable / "training_status.json").write_text(
        json.dumps({"state": "interrupted"}), encoding="utf-8"
    )
    (unsafe_interruption / "training_plan.json").write_text(
        json.dumps({"campaign_id": "campaign-unsafe-window"}), encoding="utf-8"
    )
    (unsafe_interruption / "training_status.json").write_text(
        json.dumps(
            {
                "state": "interrupted",
                "uncommitted_cuda_window": {"starting_transition_count": 2},
            }
        ),
        encoding="utf-8",
    )
    (completed / "training_plan.json").write_text(
        json.dumps({"campaign_id": "campaign-complete"}), encoding="utf-8"
    )
    (completed / "training_status.json").write_text(
        json.dumps({"state": "completed"}), encoding="utf-8"
    )
    completed_candidate = completed / "ensemble.candidate.pt"
    completed_candidate.write_bytes(b"completed-policy-candidate")
    candidate_sha256 = hashlib.sha256(completed_candidate.read_bytes()).hexdigest()
    (completed / "training_manifest.json").write_text(
        json.dumps(
            {
                "state": "completed_unqualified",
                "ensemble_candidate": {
                    "path": completed_candidate.name,
                    "sha256": candidate_sha256,
                },
            }
        ),
        encoding="utf-8",
    )
    (completed_without_candidate / "training_plan.json").write_text(
        json.dumps({"campaign_id": "campaign-complete-without-candidate"}),
        encoding="utf-8",
    )
    (completed_without_candidate / "training_status.json").write_text(
        json.dumps({"state": "completed"}), encoding="utf-8"
    )
    (completed_without_candidate / "training_manifest.json").write_text(
        json.dumps({"state": "completed_unqualified"}), encoding="utf-8"
    )

    library = TrainingModelLibrary(settings, default_directory=default)
    assert default.is_dir()
    assert library.default_directory_error == ""
    assert library.scan_locations() == (default.resolve(),)
    library.add_scan_location(external)

    assert library.scan_locations() == (default.resolve(), external.resolve())
    assert library.resumable_campaigns() == (
        {
            "campaign_id": "campaign-one",
            "state": "interrupted",
            "directory": str(resumable.resolve()),
            "plan": str((resumable / "training_plan.json").resolve()),
        },
    )
    saved = {item["campaign_id"]: item for item in library.saved_campaigns()}
    assert set(saved) == {
        "campaign-one",
        "campaign-unsafe-window",
        "campaign-complete",
        "campaign-complete-without-candidate",
    }
    assert saved["campaign-one"]["resumable"] is True
    assert saved["campaign-unsafe-window"]["resumable"] is False
    assert saved["campaign-complete"]["resumable"] is False
    assert saved["campaign-complete"]["policy_candidate"] == str(completed_candidate.resolve())
    assert {item["campaign_id"] for item in library.completed_campaigns()} == {
        "campaign-complete",
        "campaign-complete-without-candidate",
    }
    assert library.completed_policy_candidates() == (saved["campaign-complete"],)

    deleted = library.delete_completed_campaign(completed)
    assert deleted == completed.resolve()
    assert completed.exists() is False
    assert tuple(item["campaign_id"] for item in library.completed_campaigns()) == (
        "campaign-complete-without-candidate",
    )


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
    assert "QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred" in documents
    assert "def _sync_content_height" in documents
    assert "current_widget.minimumHeight()" in documents
    assert "self.tabBar().setVisible(self.count() > 1)" in documents
    assert "app.setWindowIcon(application_icon())" in application
    assert "All eligible bundled cases" in context
    assert "PROTECTED_HOLDOUT_BUS_COUNTS" in context
    assert "_TRAINING_INPUT_HELP" in context
    assert "TrainingInfoButton" in context
    assert 'self.library_picker.addItem("New training", "")' in context
    assert 'self.fields["output"] = output_field' in context
    assert "QProgressBar" not in context
    assert "progress is shown in the bottom bar and Activity" in context
    assert "QSizePolicy.Policy.Ignored" in context
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


def test_global_checkbox_style_draws_visible_borders_and_complete_states():
    style_source = (ROOT / "calo_rpd_studio/gui/themes/modern_spin_style.py").read_text(
        encoding="utf-8"
    )
    assert "PE_IndicatorCheckBox" in style_source
    assert "_draw_checkbox_indicator" in style_source
    assert "drawRoundedRect" in style_source
    assert "State_On" in style_source
    assert "State_NoChange" in style_source
    assert "State_MouseOver" in style_source
    assert "State_HasFocus" in style_source
    assert "State_Sunken" in style_source
    assert "ColorGroup.Disabled" in style_source
    for relative in (
        "calo_rpd_studio/gui/themes/light.py",
        "calo_rpd_studio/gui/themes/dark.py",
    ):
        theme = (ROOT / relative).read_text(encoding="utf-8")
        checkbox_indicator = theme.split("QCheckBox::indicator {", 1)[1].split("}", 1)[0]
        assert "width: 16px" in checkbox_indicator
        assert "height: 16px" in checkbox_indicator
    render_source = (ROOT / "calo_rpd_studio/scripts/validate_phase6_gui_contracts.py").read_text(
        encoding="utf-8"
    )
    assert "_checkbox_border_evidence" in render_source
    assert '"unchecked", "checked", "partial", "focused", "disabled"' in render_source
    assert "if maximum_rgb_delta < 90:" in render_source
    assert '"maximum_perimeter_rgb_delta": maximum_rgb_delta' in render_source
    assert '"checkbox_borders"' in render_source


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


def test_offscreen_renderer_rejects_a_training_architecture_control():
    renderer = (ROOT / "calo_rpd_studio/scripts/validate_phase6_gui_contracts.py").read_text(
        encoding="utf-8"
    )
    expected_help = renderer.split("expected_help = {", 1)[1].split("}", 1)[0]

    assert '"library"' in expected_help
    assert '"architecture"' not in expected_help
    assert 'hasattr(training_editor, "architecture")' in renderer
    assert '"architecture" in window.training_launch_model.values' in renderer
    assert "observed_help = set(training_editor.info_buttons)" in renderer
    assert "missing={sorted(expected_help - observed_help)}" in renderer
    assert "unexpected={sorted(observed_help - expected_help)}" in renderer


def test_policy_training_process_actions_are_visible_in_the_input_pane():
    context = (ROOT / "calo_rpd_studio/gui/widgets/context_pane.py").read_text(encoding="utf-8")
    main_window = (ROOT / "calo_rpd_studio/app/main_window.py").read_text(encoding="utf-8")

    constructor = context.split("class TrainingPathEditor", 1)[1].split(
        "    def _training_activity", 1
    )[0]
    assert constructor.index("self.status = QLabel()") < constructor.index(
        "self._select_new_training()"
    )
    assert 'self.training_action_button = QPushButton("Check readiness")' in context
    assert '"Start training", "Start the checked new-policy training run.", True' in context
    assert "self.training_controller.check_readiness()" in context
    assert "self.training_controller.start_training()" in context
    assert "self.resume = self.training_controller.resume" in context
    assert 'self.resume.setText("Resume selected interrupted training")' in context
    assert 'self.automatic_recovery = QCheckBox("Automatic recovery for new training")' in context
    assert "self.automatic_recovery.setChecked(True)" in context
    assert "WA_TransparentForMouseEvents" in context
    assert "self.recovery_stack.setCurrentWidget(self.automatic_recovery)" in context
    assert "self.recovery_stack.setCurrentWidget(self.resume)" in context
    assert "output_path.exists() and not self.resume.isChecked()" in context
    controller = (ROOT / "calo_rpd_studio/gui/panels/independent_training_panel.py").read_text(
        encoding="utf-8"
    )
    assert "self.resume.toggled.connect(self._resume_intent_changed)" in controller
    assert 'self._validated_fingerprint = ""' in controller
    assert "Resume choice changed · check readiness again" in controller
    assert '"Select location for new training output"' in context
    assert '"Select interrupted training directory"' in context
    assert "while candidate.exists():" in context
    assert "TrainingModelLibrary" in controller
    campaign = (ROOT / "calo_rpd_studio/algorithms/calo/tsh_calo_training_campaign.py").read_text(
        encoding="utf-8"
    )
    extension = (ROOT / "calo_rpd_studio/algorithms/calo/tsh_calo_training_extension.py").read_text(
        encoding="utf-8"
    )
    assert "checkpoint_sha256 = session.save_resume(checkpoint_path)" in campaign
    assert 'status["session_checkpoint"] = {' in campaign
    assert 'status["state"] = "interrupted"' in campaign
    assert "request_tsh_calo_training_pause" in controller
    assert "TRAINING_PAUSE_EXIT_CODE" in controller
    assert 'if name != "checkpoint_committed":' in controller
    assert "self.state.task_status.paused(self.status.text())" in controller
    assert 'control.get("state") == "acknowledged"' in controller
    assert 'last_event.get("event") == "campaign_paused"' in controller
    assert 'EVENT_LOG_FILE = "training_events.jsonl"' in campaign
    assert "_honor_pause_after_checkpoint(status, progress)" in campaign
    assert "TSHCALOTrainingPauseRequested" in campaign
    assert 'self.pause_button = QPushButton("Pause after checkpoint")' in controller
    assert "cancellable=True" in controller
    assert "IndependentTSHCALOTrainingExtension" in extension
    assert "parent_manifest_sha256" in extension
    assert "cumulative_candidate_evaluations" in extension
    assert "same_scientific_design_required" in extension
    assert "training_compatibility_contract" in extension
    assert "policy_parameter_layout_sha256" in extension
    assert "training_parameter_schema_sha256" in campaign
    assert "TSH_CALO_NON_TRAINING_PLAN_FIELDS" in campaign
    assert "parse_tsh_calo_extension_plan" in extension
    assert "execution_source_commit" in extension
    assert 'session_id=f"{episode.session_id}:extension:' in extension
    assert "automatic_start" in extension
    assert "--extend" in controller
    assert "requires a clean non-ignored source tree" in controller
    assert "uncommitted changes" in controller
    assert "currently available cpu ram" in controller
    training_command = (ROOT / "calo_rpd_studio/scripts/train_tsh_calo.py").read_text(
        encoding="utf-8"
    )
    assert 'TRAINING_EVENT_PREFIX = "CALO_TRAINING_EVENT "' in training_command
    assert "compatible_extension=arguments.extend" in training_command
    assert "load_plan(arguments.plan, compatible_extension=arguments.extend)" in training_command
    assert "elif not arguments.extend and any(" in training_command
    assert "event_callback=emit_training_event" in training_command
    assert "TSH_CALO_TRAINING_PAUSE_EXIT_CODE" in training_command
    assert "resource_preflight = validate_training_resources(plan)" in training_command
    assert "preflight_tsh_calo_training_resources(plan.training_config(plan.members[0]))" in (
        training_command
    )
    training_core = (ROOT / "calo_rpd_studio/algorithms/calo/tsh_calo_training.py").read_text(
        encoding="utf-8"
    )
    assert training_core.count("_build_and_admit_training_network(config)") == 2
    assert "finally:\n        guard.close()" in training_core
    assert 'self.library_picker.addItem("New training", "")' in context
    assert 'self.add_library_location_button = QPushButton("Add to path")' not in context
    assert 'self.load_plan_button = QPushButton("Import settings")' not in context
    assert '("plan", "Settings template"' not in context
    assert "if selected_index == 0 and current:" in context
    assert 'self.model.set_value("plan", "")' in context
    assert "self.model.load_plan(preserve_identity=preserve_identity)" in context
    assert "self._load_plan(preserve_identity=True)" in context
    assert "self.model_library.saved_campaigns()" in context
    intelligence = (ROOT / "calo_rpd_studio/gui/panels/calo_intelligence_panel.py").read_text(
        encoding="utf-8"
    )
    assert "Import trained policy" in intelligence
    assert "completed_campaigns()" in intelligence
    assert "Activate for experiments" in intelligence
    assert 'QPushButton("Start fresh assessment")' in intelligence
    assert 'QPushButton("Assess feasibility")' not in intelligence
    assert 'QPushButton("Resume assessment")' in intelligence
    assert "self.qualification_resume_button.setVisible(False)" in intelligence
    assert "self.qualification_resume_button.clicked.connect(self.resume_selected_assessment)" in (
        intelligence
    )
    assert "inspect_verified_paused_automatic_qualification_workspace" in intelligence
    assert "discard_incomplete_automatic_qualification_workspace" in intelligence
    assert "Multiple paused assessments exist" in intelligence
    assert "Completed assessments, source snapshots" in intelligence
    assert 'QPushButton("Check formal plan")' not in intelligence
    assert 'QPushButton("Run / resume qualification")' not in intelligence
    assert 'QPushButton("Admit passed evidence")' not in intelligence
    assert 'QPushButton("Compare feasibility")' not in intelligence
    assert 'QPushButton("Select for use")' in intelligence
    assert 'QPushButton("Archive")' not in intelligence
    assert '"Scientist selection required"' not in intelligence
    assert "self.policy_activate_button.setVisible(eligible)" in intelligence
    assert "lifecycle_buttons = QHBoxLayout()" not in intelligence
    assert 'QCheckBox("Show archived")' not in intelligence
    assert "show_archived_policies" not in intelligence
    assert intelligence.count("self.state.policy_registry.list(include_archived=False)") == 2
    assert 'QGroupBox("Feasibility assessment")' in intelligence
    assert 'QGroupBox("Training-parameter influence analysis")' in intelligence
    assert "inspect_feasibility_assessment" in intelligence
    assert "admit_feasibility_assessment" in intelligence
    assert "select_assessed_policy" in intelligence
    assert "build_training_parameter_influence" in intelligence
    assert "self._policy_selection_changed()" in intelligence
    assert "_parsed_training_plan_result" in intelligence
    assert "_resize_evidence_table_to_entries" in intelligence
    assert "_reveal_influence_analysis" in intelligence
    assert "build_automatic_component_ablation_plan" not in intelligence
    assert "build_automatic_formal_qualification_plan" in intelligence
    assert "automatic_qualification_workflow_payload" in intelligence
    assert "prepare_automatic_source_snapshot" in intelligence
    assert "process.setWorkingDirectory(str(source_snapshot.root))" in intelligence
    assert "Pause safely commits the current" in intelligence
    assert "partial cell can continue later" in intelligence
    assert "this action never activates or binds the policy" in intelligence
    assert 'stage="formal"' in intelligence
    assert "_update_qualification_progress" in intelligence
    assert "request_safe_qualification_pause" in intelligence
    assert "QUALIFICATION_EVENT_PREFIX" in intelligence
    assert "cell_progress" in intelligence
    assert "live, not yet a committed cell" in intelligence
    assert "_confirmed_safe_qualification_pause" in intelligence
    assert "_retained_qualification_resume" in intelligence
    assert "inspect_tsh_calo_qualification_resume_state" in intelligence
    assert "frozen_qualification_restart_design_sha256" in intelligence
    assert "remain byte-for-byte retained" in intelligence
    assert "new corrected-source run with the unchanged frozen design" in intelligence
    assert "qualification_candidate_contract" in intelligence
    assert "source-snapshots" in intelligence
    assert "progress=0" in intelligence
    assert "Model-quality checks" in intelligence
    assert "A-E optimizer cells" not in intelligence
    assert "Feasibility assessment unavailable" in intelligence
    assert "Delete model files" in intelligence
    assert "Review policy removal" not in intelligence
    assert (
        "self.policy_delete_button.clicked.connect(self.delete_selected_model_files)"
        in intelligence
    )
    assert "unqualified_candidate_removal_blocker" in intelligence
    assert "remove_unqualified_candidate" in intelligence
    assert "ensureWidgetVisible(self.policy_controller_group" not in intelligence
    assert "_delete_standalone_policy_file" in intelligence
    assert "Permanently delete completed model files" in intelligence
    assert "Qualification required" not in intelligence
    assert "Apply governing policy and continue to Power System" in intelligence
    assert '"Training evaluations"' in intelligence
    assert "training_evaluation_count" in intelligence
    assert "Completed extension segments are included" in intelligence
    assert "qualification and experiment" in intelligence
    assert "def _resize_policy_table_to_entries" in intelligence
    assert intelligence.count("Qt.ScrollBarPolicy.ScrollBarAlwaysOff") >= 2
    assert "QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow" in intelligence
    assert "layout.addWidget(library)" in intelligence
    assert "layout.addWidget(library, 1)" not in intelligence
    assert 'self._set_workspace("power_system")' in main_window
    assert "def _apply_task_interaction_lock" in main_window
    assert "self.documents.setEnabled(enabled)" in main_window
    assert "self.context_dock.setEnabled(enabled)" in main_window
    assert "self.activity_dock.show()" in main_window
    assert "self.activity_center.setEnabled(True)" in main_window
    assert 'spec.handler not in {"cancel", "toggle_activity"}' in main_window
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
