from __future__ import annotations

import importlib.util
import faulthandler
import json
import os
import sys
import threading

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PyQt6") is None, reason="PyQt6 is not installed"
)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _phase6_gui_test_deadline(request):
    """Fail the dedicated subprocess instead of leaving one GUI contract blocked forever."""

    def abort_stuck_test():
        sys.stderr.write(f"\nPhase 6 GUI test exceeded 120 seconds: {request.node.nodeid}\n")
        sys.stderr.flush()
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        os._exit(124)

    watchdog = threading.Timer(120.0, abort_stuck_test)
    watchdog.daemon = True
    watchdog.start()
    try:
        yield
    finally:
        watchdog.cancel()


def _window(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings

    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    import calo_rpd_studio.app.main_window as main_window_module
    from calo_rpd_studio.app.session_recovery import SessionRecoveryJournal
    from calo_rpd_studio.app.settings_manager import SettingsManager
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.compute.source_identity import SourceIdentity
    import calo_rpd_studio.compute.source_identity as source_identity

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(
        source_identity,
        "resolve_source_identity",
        lambda: SourceIdentity("a" * 40, False, "test"),
    )
    monkeypatch.setattr(
        main_window_module,
        "SessionRecoveryJournal",
        lambda: SessionRecoveryJournal(tmp_path / "session-recovery"),
    )
    monkeypatch.setattr(main_window_module.MainWindow, "_initial_system_scan", lambda self: None)
    monkeypatch.setattr(main_window_module.MainWindow, "_check_unfinished_work", lambda self: None)
    monkeypatch.setattr(
        main_window_module.MainWindow, "closeEvent", lambda self, event: event.accept()
    )
    state = AppState(tmp_path / "phase6-gui.sqlite")
    window = main_window_module.MainWindow(state, ExperimentManager(state), SettingsManager())

    def close_focused_test_window(widget):
        widget.activity_center.detach_logging()

    qtbot.addWidget(window, before_close_func=close_focused_test_window)
    return state, window


def test_ribbon_is_registry_generated_and_shell_regions_are_accessible(
    qtbot, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import (
        QDockWidget,
        QPushButton,
        QStackedWidget,
        QTabWidget,
        QToolButton,
    )

    state, window = _window(qtbot, tmp_path, monkeypatch)

    assert tuple(
        window.ribbon.tabs.tabText(index) for index in range(window.ribbon.tabs.count())
    ) == (
        "Home",
        "Workspace",
        "Experiment",
        "Algorithms",
        "Compute",
        "Results",
        "Policies",
        "View",
        "Help",
    )
    assert len(window.command_registry.specs) == len(
        {item.command_id for item in window.command_registry.specs}
    )
    assert window.context_dock.accessibleName() == "Contextual input pane"
    assert window.context_dock.isHidden() is False
    assert window.context_dock.features() == QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
    assert window.context_dock.toggleViewAction().isEnabled() is False
    assert window.context_pane.maximumWidth() <= 430
    assert window.context_pane.tabs.count() == 1
    assert window.context_pane.tabs.tabText(0) == "Inputs"
    assert window.sidebar.isHidden()
    assert window.documents.document_ids() == ("scientific-workspace",)
    assert window.documents.preview_scroll.widgetResizable() is True
    assert window.documents.scientific_workspace.minimumWidth() >= 920
    assert window.documents.scientific_workspace.minimumHeight() >= 650
    assert window.documents.tabBar().isHidden() is True
    assert window.documents.cornerWidget() is None
    assert not hasattr(window, "guide")
    assert window.windowIcon().isNull() is False
    assert window.activity_center.accessibleName() == "Application activity"
    assert window.region_shortcut.key().toString() == "F6"
    assert state.task_status.busy is False
    command_ids = {item.command_id for item in window.command_registry.specs}
    assert "view.context" not in command_ids
    assert "view.ribbon" not in command_ids
    assert window.ribbon.compact is False
    assert window.ribbon.tabs.minimumHeight() >= 118
    assert window.ribbon.identity_bar.objectName() == "RibbonIdentityBar"
    assert window.ribbon.identity_bar.minimumHeight() >= 42
    assert window.ribbon.identity_bar.maximumHeight() >= 42
    assert window.ribbon.identity_bar.accessibleName() == "CALO-RPD product heading"
    assert window.ribbon.navigation_area.objectName() == "RibbonNavigationArea"
    assert window.ribbon.navigation_area.accessibleName() == "Ribbon navigation area"
    assert window.ribbon.tabs.parentWidget() is window.ribbon.navigation_area
    assert isinstance(window.ribbon.tabs, QTabWidget) is False
    category_buttons = window.ribbon.tabs.findChildren(QPushButton, "RibbonCategoryButton")
    assert len(category_buttons) == window.ribbon.tabs.count()
    assert sum(button.isChecked() for button in category_buttons) == 1
    assert window.ribbon.tabs.findChild(QStackedWidget, "RibbonPageStack") is not None
    assert window.ribbon.identity_bar.findChildren(QToolButton) == []
    assert window.ribbon.product_label.text() == "CALO-RPD Studio"
    assert window.ribbon.version_label.text() == "v12.0.0"
    assert "dev" not in window.ribbon.version_label.text().lower()
    assert window.ribbon.state_label.accessibleName() == "Application state"


def test_document_header_only_appears_for_a_real_secondary_document(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QLabel

    _state, window = _window(qtbot, tmp_path, monkeypatch)
    extra = QLabel("Secondary document")

    window.documents.open_document("secondary", "Secondary", extra)
    assert window.documents.tabBar().isVisibleTo(window.documents) is True

    window.documents._close_widget(extra)
    assert window.documents.document_ids() == ("scientific-workspace",)
    assert window.documents.tabBar().isHidden() is True


def test_inputs_and_ribbon_cannot_be_hidden(qtbot, tmp_path, monkeypatch):
    _state, window = _window(qtbot, tmp_path, monkeypatch)

    window.ribbon.set_compact(True)
    window.reset_shell_layout()
    window._save_shell_layout()

    assert window.ribbon.compact is False
    assert window.ribbon.isHidden() is False
    assert window.ribbon.tabs.isHidden() is False
    assert window.context_dock.isHidden() is False
    assert window.context_dock.isFloating() is False


def test_main_preview_owns_long_workspace_vertical_scrolling(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtCore import Qt

    from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage

    _state, window = _window(qtbot, tmp_path, monkeypatch)
    long_pages = [page for page in window.pages if isinstance(page, ScrollablePage)]

    assert window.documents.preview_scroll.verticalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert long_pages
    for page in long_pages:
        assert page.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert page.property("verticalScrollOwner") == "main-preview"
        assert page.sizeHint().height() >= page.widget().sizeHint().height()


def test_ribbon_reselection_leaves_visibility_to_qt_and_blocks_inactive_controls(
    qtbot, tmp_path, monkeypatch
):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QToolButton

    _state, window = _window(qtbot, tmp_path, monkeypatch)
    ribbon = window.ribbon

    ribbon.select_category("Compute")
    ribbon.select_category("Compute")
    current_index = ribbon.tabs.currentIndex()

    assert ribbon.tabs.tabText(current_index) == "Compute"
    for index in range(ribbon.tabs.count()):
        page = ribbon.tabs.widget(index)
        inactive = index != current_index
        assert page.isHidden() is inactive
        assert page.isEnabled() is (not inactive)
        assert page.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents) is inactive
        for button in page.findChildren(QToolButton):
            assert button.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents) is inactive
            assert button.focusPolicy() == (
                Qt.FocusPolicy.NoFocus if inactive else Qt.FocusPolicy.StrongFocus
            )
            if inactive:
                assert button.isVisibleTo(ribbon) is False

    policies_index = next(
        index for index in range(ribbon.tabs.count()) if ribbon.tabs.tabText(index) == "Policies"
    )
    training_button = next(
        button
        for button in ribbon.tabs.widget(policies_index).findChildren(QToolButton)
        if button.property("ribbonCommandId") == "policies.training"
    )
    assert training_button.isVisibleTo(ribbon) is False
    assert training_button.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents) is True


def test_ribbon_group_captions_are_contained_footer_widgets(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QGroupBox, QLabel

    _state, window = _window(qtbot, tmp_path, monkeypatch)
    expected_names = {(spec.category, spec.group) for spec in window.command_registry.specs}
    groups = window.ribbon.findChildren(QGroupBox, "RibbonGroup")
    captions = window.ribbon.findChildren(QLabel, "RibbonGroupCaption")

    assert len(groups) == len(expected_names)
    assert len(captions) == len(expected_names)
    assert {caption.text() for caption in captions} == {
        group_name for _category, group_name in expected_names
    }
    for group in groups:
        caption = group.findChild(QLabel, "RibbonGroupCaption")
        assert group.title() == ""
        assert caption is not None
        assert caption.parent() is group
        assert group.layout().indexOf(caption) == group.layout().count() - 1
        assert caption.alignment() & Qt.AlignmentFlag.AlignHCenter


def test_training_navigation_opens_independent_center_without_starting(
    qtbot, tmp_path, monkeypatch
):
    state, window = _window(qtbot, tmp_path, monkeypatch)
    policy_center = window.pages_by_key["calo_intelligence"]

    assert not hasattr(policy_center, "train_button")
    assert not hasattr(policy_center, "TrainingWorker")
    assert policy_center.new_training_button.isEnabled() is True
    assert policy_center.policy_import_button.text() == "Import policy"
    assert (
        len(
            [
                button
                for button in policy_center.findChildren(type(policy_center.new_training_button))
                if button.text() == "Import policy"
            ]
        )
        == 1
    )
    assert window.training_center.process is None
    assert window.training_center.start_button.isEnabled() is False

    window.command_registry.action("policies.training").trigger()

    assert window.documents.document_ids() == ("scientific-workspace",)
    assert window.context_pane.stack.currentWidget() is window.context_pane.training
    assert window.training_center.process is None
    assert state.policy_training_active is False
    assert state.task_status.busy is False


def test_policy_training_resume_is_prepared_only_by_independent_state_machine(
    qtbot, tmp_path, monkeypatch
):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    record = {
        "id": "independent-training-record",
        "task_type": "policy_training",
        "title": "Independent TSH-CALO campaign",
        "state": {
            "plan_path": str(tmp_path / "training-plan.json"),
            "output_directory": str(tmp_path / "candidate-output"),
        },
    }

    window.pages_by_key["resume_center"].policy_training_requested.emit(record)

    assert window.context_pane.stack.currentWidget() is window.context_pane.training
    assert window.training_launch_model.values["plan"] == record["state"]["plan_path"]
    assert window.training_launch_model.values["output"] == record["state"]["output_directory"]
    assert window.training_center.resume.isChecked() is True
    assert window.training_center.process is None
    assert window.training_center.start_button.isEnabled() is False


def test_resume_inspection_keeps_record_state_out_of_the_summary(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QPlainTextEdit

    _state, window = _window(qtbot, tmp_path, monkeypatch)
    panel = window.pages_by_key["resume_center"]
    record = {
        "id": "task-1",
        "task_type": "validation",
        "title": "Validate retained runs",
        "progress": "2/5",
        "status": "paused",
        "updated_at": "2026-08-13T00:00:00Z",
        "resumable": True,
        "state": {"private_backend_field": "must-not-be-shown"},
    }
    monkeypatch.setattr(panel, "_selected", lambda: record)
    captured = {}

    def inspect_without_opening(dialog):
        captured["dialog"] = dialog
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect_without_opening, raising=False)
    panel.inspect_selected()

    details = captured["dialog"].findChildren(QPlainTextEdit)[0]
    assert "must-not-be-shown" not in details.toPlainText()
    assert "Task ID: task-1" in details.toPlainText()


def _training_plan(tmp_path):
    from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
        TSHCALOTrainingCampaignPlan,
        TSHCALOTrainingEpisodePlan,
        TSHCALOTrainingMemberPlan,
    )
    from calo_rpd_studio.algorithms.calo.tsh_calo_training_resources import (
        TSHCALOTrainingResourceEnvelope,
    )

    plan = TSHCALOTrainingCampaignPlan(
        campaign_id="gui-campaign",
        source_commit="a" * 40,
        development_freeze_commit="a" * 40,
        development_freeze_sha256="b" * 64,
        phase4_acceptance_sha256="c" * 64,
        development_cases=("toy-development",),
        members=(
            TSHCALOTrainingMemberPlan(
                "member-1",
                101,
                (TSHCALOTrainingEpisodePlan("session-1", "toy-development", 201),),
            ),
            TSHCALOTrainingMemberPlan(
                "member-2",
                102,
                (TSHCALOTrainingEpisodePlan("session-2", "toy-development", 202),),
            ),
        ),
        resource_envelope=TSHCALOTrainingResourceEnvelope(1, 4, 8, 16, 16, 4),
        population_size=4,
        max_evaluations=8,
        requested_device="cpu",
    )
    path = tmp_path / "training-plan.json"
    path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    return path


def _complete_training_inputs(window, tmp_path):
    model = window.training_launch_model
    for key, value in {
        "plan": str(_training_plan(tmp_path)),
        "output": "candidate-output",
    }.items():
        model.set_value(key, value)
    model.load_plan()
    window.context_pane.training._load_plan_controls()
    window.context_pane.training._new_plan_mode = False
    return model


def test_training_inputs_load_active_plan_and_edits_change_launch_fingerprint(
    qtbot, tmp_path, monkeypatch
):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    model = _complete_training_inputs(window, tmp_path)
    editor = window.context_pane.training

    assert editor.plan_group.isHidden() is False
    assert editor.members.value() == 2
    assert editor.population.value() == 4
    assert model.plan_payload["development_freeze_commit"] == ""
    assert model.plan_payload["development_freeze_sha256"] == ""
    assert model.plan_payload["phase4_acceptance_sha256"] == ""
    before = model.fingerprint()
    editor.learning_rate.setValue(0.0007)

    assert model.plan_payload["training"]["learning_rate"] == pytest.approx(0.0007)
    assert model.fingerprint() != before
    assert model.arguments(check=True)[2] != str(tmp_path / "training-plan.json")


def test_training_parameters_are_available_without_an_existing_plan(qtbot, tmp_path, monkeypatch):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    editor = window.context_pane.training

    assert editor.plan_group.isHidden() is False
    assert editor.members.value() >= 2
    assert editor.population.value() >= 2
    assert editor.learning_rate.value() > 0
    assert editor.load_plan_button.text() == "Import settings"


def test_every_training_input_has_accessible_directional_information(qtbot, tmp_path, monkeypatch):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    editor = window.context_pane.training
    expected = {
        "architecture",
        "plan",
        "output",
        "campaign_id",
        "cases",
        "members",
        "master_seed",
        "population",
        "evaluations",
        "device",
        "cpu_fallback",
        "learning_rate",
        "discount",
        "gae",
        "clip",
        "ppo_epochs",
        "hidden_dim",
        "graph_steps",
    }

    assert set(editor.info_buttons) == expected
    for key, button in editor.info_buttons.items():
        assert button.text() == "i"
        assert button.toolTip()
        assert "<b>Suggested " in button.toolTip()
        assert button.accessibleName().startswith("Information about ")
        assert button.accessibleDescription()
        assert "suggested " in button.accessibleDescription().lower()
        assert "not validated optima" in button.accessibleDescription().lower()
        if key in {
            "members",
            "population",
            "evaluations",
            "learning_rate",
            "discount",
            "gae",
            "clip",
            "ppo_epochs",
            "hidden_dim",
            "graph_steps",
        }:
            description = button.accessibleDescription().lower()
            assert "increas" in description
            assert "decreas" in description
        if key in {
            "members",
            "master_seed",
            "population",
            "evaluations",
            "learning_rate",
            "discount",
            "gae",
            "clip",
            "ppo_epochs",
            "hidden_dim",
            "graph_steps",
        }:
            description = button.accessibleDescription().lower()
            assert "suggested range:" in description
            assert " to " in description

    assert "3 to 5" in editor.info_buttons["members"].accessibleDescription()
    assert "20 to 64" in editor.info_buttons["population"].accessibleDescription()
    assert "0.0001 to 0.001" in editor.info_buttons["learning_rate"].accessibleDescription()
    assert "case118 and case300" in editor.info_buttons["cases"].accessibleDescription()


def test_training_case_picker_selects_all_eligible_and_locks_protected_holdouts(
    qtbot, tmp_path, monkeypatch
):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    editor = window.context_pane.training

    assert tuple(editor.case_checks) == ("case30", "case57", "case118", "case300")
    assert editor.all_eligible_cases.isChecked() is True
    assert editor.selected_training_cases() == ["case30", "case57"]
    for protected in ("case118", "case300"):
        assert editor.case_checks[protected].isEnabled() is False
        assert editor.case_checks[protected].isChecked() is False
        assert "protected holdout" in editor.case_checks[protected].text().lower()

    editor.case_checks["case57"].setChecked(False)
    assert editor.all_eligible_cases.isChecked() is False
    assert editor.selected_training_cases() == ["case30"]
    editor.all_eligible_cases.setChecked(True)
    assert editor.selected_training_cases() == ["case30", "case57"]
    assert editor.case_checks["case118"].isChecked() is False
    assert editor.case_checks["case300"].isChecked() is False


def test_training_ribbon_opens_inputs_and_in_pane_action_checks_then_starts(
    qtbot, tmp_path, monkeypatch
):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    model = _complete_training_inputs(window, tmp_path)

    calls = []
    monkeypatch.setattr(window.training_center, "check_readiness", lambda: calls.append("check"))
    monkeypatch.setattr(window.training_center, "start_training", lambda: calls.append("start"))
    action = window.command_registry.action("policies.training")
    editor = window.context_pane.training

    action.trigger()
    assert calls == []
    assert action.text() == "Train policy"
    assert editor.action_bar.isHidden() is False
    assert editor.training_action_button.text() == "Check readiness"
    assert editor.training_action_button.isEnabled() is True

    action.trigger()
    assert calls == []
    assert action.text() == "Train policy"

    editor.training_action_button.click()
    assert calls == ["check"]

    window.training_center._validated_fingerprint = model.fingerprint()
    editor.refresh()
    assert action.text() == "Train policy"
    assert editor.training_action_button.text() == "Start training"
    assert editor.training_action_button.isEnabled() is True
    editor.training_action_button.click()
    assert calls == ["check", "start"]


def test_readiness_result_is_rejected_when_bound_paths_change(qtbot, tmp_path, monkeypatch):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    model = _complete_training_inputs(window, tmp_path)

    panel = window.training_center
    panel._operation = "check"
    panel._invocation_fingerprint = model.fingerprint()
    model.set_value("plan", "changed-plan.json")
    panel._process_finished(0, None)

    assert panel._validated_fingerprint == ""
    assert panel.start_button.isEnabled() is False
    assert "inputs changed" in panel.status.text().lower()


def test_builtin_architecture_and_governance_paths_are_not_user_inputs(
    qtbot, tmp_path, monkeypatch
):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    editor = window.context_pane.training

    assert set(editor.fields) == {"plan", "output"}
    assert editor.architecture.itemText(editor.architecture.findData("calo")) == "CALO"
    assert editor.architecture.itemText(editor.architecture.findData("tsh_calo")) == "TSH-CALO"
    assert editor.architecture.currentData() == "tsh_calo"
    assert "foundation" not in editor.info_buttons
    assert not hasattr(editor, "foundation_status")
    placeholders = " ".join(field.placeholderText().lower() for field in editor.fields.values())
    assert "development freeze" not in placeholders
    assert "phase 4" not in placeholders


def test_calo_architecture_disables_policy_training_inputs(qtbot, tmp_path, monkeypatch):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    editor = window.context_pane.training

    editor.architecture.setCurrentIndex(editor.architecture.findData("calo"))

    assert window.training_launch_model.values["architecture"] == "calo"
    assert editor.fields["plan"].isEnabled() is False
    assert editor.fields["output"].isEnabled() is False
    assert editor.campaign_id.isEnabled() is False
    assert editor.architecture.isEnabled() is True
    assert "no policy training" in editor.status.text().lower()
    assert window.training_center.check_button.isEnabled() is False
    assert window.training_center.start_button.isEnabled() is False


def test_ordinary_product_surfaces_hide_engineering_lifecycle_language(
    qtbot, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QAbstractButton, QLabel, QPlainTextEdit, QWidget

    _state, window = _window(qtbot, tmp_path, monkeypatch)
    window.show()
    window.command_registry.action("policies.training").trigger()
    window.pages_by_key["benchmark"].show()

    fragments = []
    for root in (
        window.ribbon,
        window.context_pane.training,
        window.training_center,
        window.pages_by_key["calo_intelligence"],
        window.pages_by_key["benchmark"],
    ):
        for widget in (root, *root.findChildren(QWidget)):
            if not widget.isVisibleTo(root):
                continue
            fragments.extend((widget.toolTip(), widget.statusTip(), widget.whatsThis()))
            if isinstance(widget, (QLabel, QAbstractButton)):
                fragments.append(widget.text())
            if isinstance(widget, QPlainTextEdit):
                fragments.append(widget.toPlainText())
    visible_text = " ".join(fragment for fragment in fragments if fragment).lower()
    forbidden = (
        "phase 4",
        "phase 6",
        "development freeze",
        "production-candidate",
        "source-bound",
        "feature flag",
        "post-freeze",
        "a-e/f-off",
        "runtime abi",
        "software freeze",
        "frozen calo",
        "unqualified candidate",
        "checksum",
        "sha-256",
        "fail-closed cli",
    )
    assert not {token for token in forbidden if token in visible_text}
    assert "12.0.0-dev" not in visible_text


def test_compact_editors_validate_copied_state_and_status_is_truthful(qtbot, tmp_path, monkeypatch):
    state, window = _window(qtbot, tmp_path, monkeypatch)
    editor = window.context_pane.compute
    editor.mode.setCurrentIndex(editor.mode.findData("cpu_only"))
    editor.device.setCurrentText("cuda:0")
    editor.apply()

    assert state.config.execution_backend == "cpu_only"
    assert state.config.requested_compute_device == "cpu"
    window.global_status.apply_context(state)
    assert "cpu_only" in window.global_status.compute_label.text()
    assert "not assigned" in window.global_status.device_label.text()
    assert "available-memory safety limit" in window.global_status.memory_label.toolTip().lower()
    assert "not assigned" in editor.truth.text()
    window.activity_center.refresh_context()
    assert "actual assignment: not assigned" in window.activity_center.device.text().lower()


def test_activity_uses_indeterminate_progress_without_fabricating_percentage(
    qtbot, tmp_path, monkeypatch
):
    state, window = _window(qtbot, tmp_path, monkeypatch)
    assert state.task_status.begin("Unknown-duration operation", progress=-1)

    last_row = window.activity_center.jobs.rowCount() - 1
    assert window.activity_center.jobs.item(last_row, 4).text() == "indeterminate"
    assert window.global_status.progress.minimum() == 0
    assert window.global_status.progress.maximum() == 0


def test_layout_reset_does_not_overwrite_foreground_task(qtbot, tmp_path, monkeypatch):
    state, window = _window(qtbot, tmp_path, monkeypatch)
    assert state.task_status.begin("Retained foreground task", progress=25)

    window.reset_shell_layout()

    assert state.task_status.busy is True
    assert state.task_status.title == "Retained foreground task"
    assert state.task_status.progress == 25
