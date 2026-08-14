from __future__ import annotations

import importlib.util
import faulthandler
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

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
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingModelLibrary

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
    monkeypatch.setattr(
        main_window_module,
        "TrainingModelLibrary",
        lambda settings: TrainingModelLibrary(
            settings, default_directory=tmp_path / "training-models"
        ),
    )
    monkeypatch.setattr(main_window_module.MainWindow, "_initial_system_scan", lambda self: None)
    monkeypatch.setattr(main_window_module.MainWindow, "_check_unfinished_work", lambda self: None)
    monkeypatch.setattr(
        main_window_module.MainWindow, "closeEvent", lambda self, event: event.accept()
    )
    state = AppState(tmp_path / "phase6-gui.sqlite")
    settings = SettingsManager()
    settings.settings.clear()
    settings.settings.sync()
    window = main_window_module.MainWindow(state, ExperimentManager(state), settings)

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
    from PyQt6.QtWidgets import QSizePolicy

    from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage

    _state, window = _window(qtbot, tmp_path, monkeypatch)
    long_pages = [page for page in window.pages if isinstance(page, ScrollablePage)]

    assert window.documents.preview_scroll.verticalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert window.documents.scientific_workspace.sizePolicy().verticalPolicy() == (
        QSizePolicy.Policy.Preferred
    )
    assert long_pages
    for page in long_pages:
        assert page.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert page.property("verticalScrollOwner") == "main-preview"
        assert page.sizeHint().height() >= page.widget().sizeHint().height()
        qtbot.waitUntil(lambda item=page: item.minimumHeight() >= item.widget().sizeHint().height())


def test_main_preview_can_scroll_to_dynamic_governing_policy_bottom(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QTableWidgetItem
    import torch

    from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork
    from calo_rpd_studio.algorithms.calo.policy_schema import (
        CALO_RUNTIME_ARCHITECTURE,
        POLICY_ACTION_SCHEMA,
        POLICY_STATE_DIM,
        POLICY_STATE_SCHEMA,
        TRAINING_ENVIRONMENT_VERSION,
    )

    state, window = _window(qtbot, tmp_path, monkeypatch)
    intelligence = window.pages_by_key["calo_intelligence"]
    registered = []
    for index, hidden_dim in enumerate((16, 24), start=1):
        candidate = tmp_path / f"scroll-policy-{index}.candidate.pt"
        network = CALOPolicyNetwork(input_dim=POLICY_STATE_DIM, hidden_dim=hidden_dim)
        torch.save(
            {
                "model_state_dict": network.state_dict(),
                "architecture": {
                    "input_dim": POLICY_STATE_DIM,
                    "hidden_dim": hidden_dim,
                },
                "metadata": {
                    "calo_core": "v4.1",
                    "state_dimension": POLICY_STATE_DIM,
                    "runtime_architecture_version": CALO_RUNTIME_ARCHITECTURE,
                    "state_schema_version": POLICY_STATE_SCHEMA,
                    "action_schema_version": POLICY_ACTION_SCHEMA,
                    "training_environment_version": TRAINING_ENVIRONMENT_VERSION,
                },
            },
            candidate,
        )
        registered.append(
            state.policy_registry.register(candidate, name=f"scroll-policy-{index}")
        )
    intelligence.refresh_policy_library()
    window.stack.setCurrentWidget(intelligence)
    window.resize(1120, 720)
    window.show()
    for row in range(14):
        intelligence.policy_table.insertRow(intelligence.policy_table.rowCount())
        intelligence.policy_table.setItem(
            intelligence.policy_table.rowCount() - 1,
            1,
            QTableWidgetItem(f"synthetic-layout-policy-{row:02d}"),
        )
    intelligence._resize_policy_table_to_entries()

    scroll = window.documents.preview_scroll
    qtbot.waitUntil(lambda: scroll.verticalScrollBar().maximum() > 0)
    scroll.verticalScrollBar().setValue(0)
    intelligence.policy_table.clearSelection()
    target_row = next(
        row
        for row, policy in enumerate(intelligence._policy_rows)
        if getattr(policy, "id", "") == registered[1].id
    )
    intelligence.policy_table.selectRow(target_row)
    qtbot.wait(1)
    assert intelligence._selected_policy().id == registered[1].id
    assert scroll.verticalScrollBar().value() == 0

    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    qtbot.waitUntil(
        lambda: intelligence.apply_policy_button.mapTo(
            scroll.viewport(),
            QPoint(0, intelligence.apply_policy_button.height()),
        ).y()
        <= scroll.viewport().height()
    )
    assert scroll.verticalScrollBar().value() == scroll.verticalScrollBar().maximum()


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
    from PyQt6.QtCore import Qt

    _state, window = _window(qtbot, tmp_path, monkeypatch)
    editor = window.context_pane.training
    window.context_pane.stack.setCurrentWidget(editor)
    window.resize(1120, 720)
    window.show()

    assert editor.plan_group.isHidden() is False
    assert editor.members.value() >= 2
    assert editor.population.value() >= 2
    assert editor.learning_rate.value() > 0
    assert editor.load_plan_button.text() == "Import settings"
    assert editor.library_picker.itemText(0) == "New training"
    assert Path(editor.fields["output"].text()).parent == (
        window.training_model_library.default_directory
    )
    assert editor.add_library_location_button.text() == "Add to path"
    qtbot.waitUntil(
        lambda: editor.default_library_path.height()
        >= editor.default_library_path.heightForWidth(editor.default_library_path.width())
    )
    assert editor.default_library_path.minimumWidth() == 0
    assert editor.default_library_path.toolTip() == str(
        window.training_model_library.default_directory
    )
    assert editor.recovery_stack.currentWidget() is editor.automatic_recovery
    assert editor.automatic_recovery.isChecked() is True
    assert editor.automatic_recovery.isEnabled() is True
    assert editor.automatic_recovery.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert editor.automatic_recovery.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert editor.resume.isChecked() is False
    assert "automatic recovery is on" in editor.status.text().lower()


def test_completed_training_is_visible_without_automatic_policy_registration(
    qtbot, tmp_path, monkeypatch
):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QMessageBox, QSizePolicy

    state, window = _window(qtbot, tmp_path, monkeypatch)
    editor = window.context_pane.training
    policy_center = window.pages_by_key["calo_intelligence"]
    empty_table_height = policy_center.policy_table.height()
    campaign = window.training_model_library.default_directory / "completed-campaign"
    campaign.mkdir(parents=True)
    candidate = campaign / "ensemble.candidate.pt"
    candidate.write_bytes(b"saved-completed-policy")
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    plan_payload = json.loads(_training_plan(tmp_path).read_text(encoding="utf-8"))
    plan_payload["campaign_id"] = "completed-campaign"
    (campaign / "training_plan.json").write_text(json.dumps(plan_payload), encoding="utf-8")
    (campaign / "training_status.json").write_text(
        json.dumps({"state": "completed"}), encoding="utf-8"
    )
    (campaign / "training_manifest.json").write_text(
        json.dumps(
            {
                "state": "completed_unqualified",
                "ensemble_candidate": {
                    "path": candidate.name,
                    "sha256": candidate_sha256,
                },
            }
        ),
        encoding="utf-8",
    )

    window.training_model_library.changed.emit()
    editor.refresh_model_library(window.training_model_library.default_directory)

    assert editor.library_picker.currentData()["campaign_id"] == "completed-campaign"
    assert "Training complete" in editor.library_picker.currentText()
    assert editor.resume.isChecked() is False
    assert editor.recovery_stack.currentWidget() is editor.completed_recovery
    assert editor.training_action_button.isEnabled() is False
    assert "available in the Policy library" in editor.status.text()
    matching_rows = [
        row
        for row in range(policy_center.policy_table.rowCount())
        if policy_center.policy_table.item(row, 1).text() == "completed-campaign"
    ]
    assert len(matching_rows) == 1
    assert policy_center.policy_table.horizontalHeaderItem(2).text() == (
        "Training evaluations"
    )
    assert policy_center.policy_table.item(matching_rows[0], 2).text() == "Not available"
    assert "Completed extension segments are included" in policy_center.policy_table.item(
        matching_rows[0], 2
    ).toolTip()
    assert "qualification and experiment evaluations are excluded" in (
        policy_center.policy_table.item(matching_rows[0], 2).toolTip()
    )
    assert policy_center.policy_table.height() > empty_table_height
    assert policy_center.policy_table.verticalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert policy_center.policy_table.horizontalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    expected_table_height = (
        max(
            policy_center.policy_table.horizontalHeader().height(),
            policy_center.policy_table.horizontalHeader().sizeHint().height(),
        )
        + sum(
            policy_center.policy_table.rowHeight(row)
            for row in range(policy_center.policy_table.rowCount())
        )
        + policy_center.policy_table.frameWidth() * 2
    )
    assert policy_center.policy_table.height() == expected_table_height
    assert policy_center.path.sizePolicy().horizontalPolicy() == (
        QSizePolicy.Policy.Expanding
    )
    assert policy_center.policy_controller_group.sizePolicy().horizontalPolicy() == (
        QSizePolicy.Policy.Expanding
    )
    policy_center.policy_table.selectRow(matching_rows[0])
    assert policy_center.policy_import_button.text() == "Import trained policy"
    assert policy_center.policy_import_button.isEnabled() is True
    assert policy_center.policy_activate_button.isEnabled() is False
    assert policy_center.policy_activate_button.text() == "Import before activation"
    assert policy_center.policy_delete_button.text() == "Delete model files"
    assert policy_center.policy_delete_button.isEnabled() is True
    assert not hasattr(policy_center, "policy_removal_review_button")
    assert state.policy_registry.list(include_archived=True) == []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    policy_center.delete_selected_model_files()
    assert campaign.exists() is False


def test_imported_unqualified_completed_campaign_can_be_removed_exactly(
    qtbot, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QMessageBox
    import torch

    from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork
    from calo_rpd_studio.algorithms.calo.policy_schema import (
        CALO_RUNTIME_ARCHITECTURE,
        POLICY_ACTION_SCHEMA,
        POLICY_STATE_DIM,
        POLICY_STATE_SCHEMA,
        TRAINING_ENVIRONMENT_VERSION,
    )

    state, window = _window(qtbot, tmp_path, monkeypatch)
    policy_center = window.pages_by_key["calo_intelligence"]
    campaign = window.training_model_library.default_directory / "registered-completed-campaign"
    campaign.mkdir(parents=True)
    candidate = campaign / "candidate.pt"
    network = CALOPolicyNetwork(input_dim=POLICY_STATE_DIM, hidden_dim=16)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "architecture": {"input_dim": POLICY_STATE_DIM, "hidden_dim": 16},
            "metadata": {
                "calo_core": "v4.1",
                "state_dimension": POLICY_STATE_DIM,
                "runtime_architecture_version": CALO_RUNTIME_ARCHITECTURE,
                "state_schema_version": POLICY_STATE_SCHEMA,
                "action_schema_version": POLICY_ACTION_SCHEMA,
                "training_environment_version": TRAINING_ENVIRONMENT_VERSION,
            },
        },
        candidate,
    )
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    plan_payload = json.loads(_training_plan(tmp_path).read_text(encoding="utf-8"))
    plan_payload["campaign_id"] = "registered-completed-campaign"
    (campaign / "training_plan.json").write_text(json.dumps(plan_payload), encoding="utf-8")
    (campaign / "training_status.json").write_text(
        json.dumps({"state": "completed"}), encoding="utf-8"
    )
    (campaign / "training_manifest.json").write_text(
        json.dumps(
            {
                "state": "completed_unqualified",
                "ensemble_candidate": {
                    "path": candidate.name,
                    "sha256": candidate_sha256,
                },
            }
        ),
        encoding="utf-8",
    )
    registered = state.policy_registry.register(candidate, name="registered-completed-campaign")
    window.training_model_library.changed.emit()
    matching_row = next(
        row
        for row in range(policy_center.policy_table.rowCount())
        if policy_center.policy_table.item(row, 1).text() == "registered-completed-campaign"
    )
    policy_center.policy_table.selectRow(matching_row)

    assert policy_center.policy_delete_button.isEnabled() is True
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    policy_center.delete_selected_model_files()

    assert campaign.exists() is False
    assert state.policy_registry.is_suppressed(registered.sha256) is True
    with pytest.raises(KeyError):
        state.policy_registry.get(registered.id)


def test_first_standalone_unqualified_model_can_be_deleted_exactly(
    qtbot, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QMessageBox
    import torch

    from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork
    from calo_rpd_studio.algorithms.calo.policy_schema import (
        CALO_RUNTIME_ARCHITECTURE,
        POLICY_ACTION_SCHEMA,
        POLICY_STATE_DIM,
        POLICY_STATE_SCHEMA,
        TRAINING_ENVIRONMENT_VERSION,
    )

    state, window = _window(qtbot, tmp_path, monkeypatch)
    policy_center = window.pages_by_key["calo_intelligence"]
    candidate = tmp_path / "first-standalone.candidate.pt"
    network = CALOPolicyNetwork(input_dim=POLICY_STATE_DIM, hidden_dim=16)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "architecture": {"input_dim": POLICY_STATE_DIM, "hidden_dim": 16},
            "metadata": {
                "calo_core": "v4.1",
                "state_dimension": POLICY_STATE_DIM,
                "runtime_architecture_version": CALO_RUNTIME_ARCHITECTURE,
                "state_schema_version": POLICY_STATE_SCHEMA,
                "action_schema_version": POLICY_ACTION_SCHEMA,
                "training_environment_version": TRAINING_ENVIRONMENT_VERSION,
            },
        },
        candidate,
    )
    registered = state.policy_registry.register(candidate, name="first-standalone")
    policy_center.refresh_policy_library()

    assert policy_center.policy_table.rowCount() == 1
    assert policy_center.policy_table.currentRow() == 0
    assert policy_center.policy_table.item(0, 1).text() == "first-standalone"
    assert policy_center.policy_delete_button.isEnabled() is True
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)

    policy_center.delete_selected_model_files()

    assert candidate.exists() is False
    assert state.policy_registry.is_suppressed(registered.sha256) is True
    with pytest.raises(KeyError):
        state.policy_registry.get(registered.id)


def test_authenticated_completed_training_exposes_explicit_finite_extension_action(
    qtbot, tmp_path, monkeypatch
):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    editor = window.context_pane.training
    campaign = window.training_model_library.default_directory / "extendable-campaign"
    campaign.mkdir(parents=True)
    candidate = campaign / "ensemble.candidate.pt"
    candidate.write_bytes(b"extendable-completed-policy")
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    plan_payload = json.loads(_training_plan(tmp_path).read_text(encoding="utf-8"))
    plan_payload["campaign_id"] = "extendable-campaign"
    checkpoints = []
    for member_index, member in enumerate(plan_payload["members"]):
        checkpoint = campaign / f"member-{member_index + 1:03d}.resume"
        checkpoint.write_bytes(f"checkpoint-{member_index}".encode())
        checkpoints.append(
            {
                "member_index": member_index,
                "member_id": member["member_id"],
                "path": checkpoint.name,
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "receipt_count": 1,
            }
        )
    (campaign / "training_plan.json").write_text(
        json.dumps(plan_payload), encoding="utf-8"
    )
    (campaign / "training_status.json").write_text(
        json.dumps({"state": "completed"}), encoding="utf-8"
    )
    (campaign / "training_manifest.json").write_text(
        json.dumps(
            {
                "state": "completed_unqualified",
                "continuation_checkpoints": checkpoints,
                "extension_contract": {"repeatable_finite_segments": True},
                "ensemble_candidate": {
                    "path": candidate.name,
                    "sha256": candidate_sha256,
                },
            }
        ),
        encoding="utf-8",
    )

    window.training_model_library.changed.emit()
    editor.refresh_model_library(window.training_model_library.default_directory)

    record = editor.library_picker.currentData()
    assert record["extendable"] is True
    assert window.training_center._extension_mode is True
    assert editor.training_action_button.text() == "Check extension readiness"
    assert editor.training_action_button.isEnabled() is True
    arguments = window.training_launch_model.arguments(check=True, extend=True)
    assert arguments[-3:] == ["--output", str(campaign.resolve()), "--extend"]
    assert "automatic" not in editor.status.text().lower()


def test_every_training_input_has_accessible_directional_information(qtbot, tmp_path, monkeypatch):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    editor = window.context_pane.training
    expected = {
        "library",
        "plan",
        "output",
        "resume",
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
    assert "automatic recovery stays on" in editor.info_buttons["resume"].accessibleDescription()


def test_saved_training_picker_resumes_in_its_original_registered_directory(
    qtbot, tmp_path, monkeypatch
):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    editor = window.context_pane.training
    external = tmp_path / "external-models"
    campaign = external / "interrupted-campaign"
    campaign.mkdir(parents=True)
    plan = _training_plan(tmp_path)
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    plan_payload["development_freeze_commit"] = ""
    plan_payload["development_freeze_sha256"] = ""
    plan_payload["phase4_acceptance_sha256"] = ""
    (campaign / "training_plan.json").write_text(json.dumps(plan_payload), encoding="utf-8")
    (campaign / "training_status.json").write_text(
        json.dumps({"state": "interrupted"}), encoding="utf-8"
    )

    window.training_model_library.add_scan_location(external)
    editor.refresh_model_library()
    editor.library_picker.setCurrentIndex(1)

    assert editor.resume.isHidden() is False
    assert editor.resume.isChecked() is True
    assert editor.resume.isEnabled() is True
    assert editor.recovery_stack.currentWidget() is editor.resume
    assert editor.fields["output"].text() == str(campaign.resolve())
    assert editor.fields["plan"].text() == str((campaign / "training_plan.json").resolve())
    assert editor.model.plan_payload is not None
    assert editor.model.plan_payload["source_commit"] == "a" * 40
    assert editor.campaign_id.isEnabled() is False
    assert editor.learning_rate.isEnabled() is False


def test_clean_source_readiness_failure_is_summarized_and_trace_is_debug_only(
    qtbot, tmp_path, monkeypatch
):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    model = _complete_training_inputs(window, tmp_path)
    panel = window.training_center
    messages = []
    panel.activity_message.connect(lambda severity, message: messages.append((severity, message)))
    panel._operation = "check"
    panel._invocation_fingerprint = model.fingerprint()
    raw_failure = (
        "Traceback (most recent call last):\n"
        "RuntimeError: TSH-CALO training requires a clean non-ignored source tree"
    )
    panel._process_output = [raw_failure]
    panel.activity_message.emit("DEBUG", raw_failure)

    panel._process_finished(1, None)

    assert "uncommitted changes" in panel.status.text().lower()
    assert "training was not started" in panel.status.text().lower()
    assert "uncommitted changes" in window.context_pane.training.status.text().lower()
    assert messages[-1][0] == "ERROR"
    assert "traceback" not in messages[-1][1].lower()
    assert any(severity == "DEBUG" and "Traceback" in message for severity, message in messages)
    assert "Traceback" in window.activity_center.logs.toPlainText()
    assert all(
        "Traceback" not in window.activity_center.warnings.item(index).text()
        for index in range(window.activity_center.warnings.count())
    )


def test_memory_readiness_failure_is_explained_before_training_starts(qtbot, tmp_path, monkeypatch):
    state, window = _window(qtbot, tmp_path, monkeypatch)
    model = _complete_training_inputs(window, tmp_path)
    panel = window.training_center
    panel._operation = "check"
    panel._invocation_fingerprint = model.fingerprint()
    panel._process_output = [
        "MemoryError: TSH-CALO training working set exceeds 80% of currently available CPU RAM"
    ]

    panel._process_finished(1, None)

    assert "more memory than is currently available" in panel.status.text().lower()
    assert "training was not started" in panel.status.text().lower()
    assert "more memory" in window.context_pane.training.status.text().lower()
    assert state.policy_training_active is False
    assert panel._validated_fingerprint == ""
    assert panel.start_button.isEnabled() is False


def test_new_training_rejects_existing_output_without_conflating_recovery_and_resume(
    qtbot, tmp_path, monkeypatch
):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    model = _complete_training_inputs(window, tmp_path)
    editor = window.context_pane.training
    output = tmp_path / "existing-training"
    output.mkdir()
    model.set_value("output", str(output))
    window.training_center._validated_fingerprint = model.fingerprint()

    editor.refresh()
    assert editor.resume is window.training_center.resume
    assert editor.resume.isChecked() is False
    assert editor.recovery_stack.currentWidget() is editor.automatic_recovery
    assert editor.automatic_recovery.isChecked() is True
    assert editor.training_action_button.text() == "Start training"
    assert editor.training_action_button.isEnabled() is False
    assert "already exists" in editor.status.text().lower()


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


def test_training_is_tsh_calo_only_and_governance_paths_are_not_user_inputs(
    qtbot, tmp_path, monkeypatch
):
    _state, window = _window(qtbot, tmp_path, monkeypatch)
    editor = window.context_pane.training

    assert set(editor.fields) == {"plan", "output"}
    assert not hasattr(editor, "architecture")
    assert "architecture" not in window.training_launch_model.values
    assert "architecture" not in editor.info_buttons
    assert "foundation" not in editor.info_buttons
    assert not hasattr(editor, "foundation_status")
    placeholders = " ".join(field.placeholderText().lower() for field in editor.fields.values())
    assert "development freeze" not in placeholders
    assert "phase 4" not in placeholders


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


def test_policy_training_checkpoint_progress_uses_bottom_bar_activity_and_pause_retains_it(
    qtbot, tmp_path, monkeypatch
):
    from calo_rpd_studio.gui.panels.independent_training_panel import (
        TRAINING_EVENT_PREFIX,
        TRAINING_EVENT_SCHEMA,
    )

    state, window = _window(qtbot, tmp_path, monkeypatch)
    assert state.task_status.begin(
        "Independent policy training",
        detail="Starting finite plan",
        progress=0,
        cancellable=True,
    )
    event = {
        "schema_version": TRAINING_EVENT_SCHEMA,
        "event": "checkpoint_committed",
        "progress_percent": 37,
        "member_number": 2,
        "member_count": 4,
        "case_identity": "case57",
        "episode_candidate_evaluations": 40,
        "episode_evaluation_limit": 100,
        "checkpoint_sha256": "a" * 64,
    }

    window.training_center._consume_output_line(
        f"{TRAINING_EVENT_PREFIX}{json.dumps(event, sort_keys=True)}"
    )

    assert state.task_status.progress == 37
    assert "Member 2/4" in state.task_status.detail
    assert window.global_status.progress.value() == 37
    assert window.global_status.cancel_button.text() == "Pause safely"
    assert not hasattr(window.context_pane.training, "training_progress")
    assert "checkpoint aaaaaaaaaaaa committed" in window.activity_center.logs.toPlainText()
    last_row = window.activity_center.jobs.rowCount() - 1
    assert window.activity_center.jobs.item(last_row, 4).text() == "37%"

    state.task_status.paused("Paused at a verified checkpoint")

    assert state.task_status.state == "Paused"
    assert state.task_status.progress == 37
    assert window.global_status.progress.value() == 37
    assert window.activity_center.jobs.item(
        window.activity_center.jobs.rowCount() - 1, 1
    ).text() == "Paused"


def test_layout_reset_does_not_overwrite_foreground_task(qtbot, tmp_path, monkeypatch):
    state, window = _window(qtbot, tmp_path, monkeypatch)
    assert state.task_status.begin("Retained foreground task", progress=25)

    window.reset_shell_layout()

    assert state.task_status.busy is True
    assert state.task_status.title == "Retained foreground task"
    assert state.task_status.progress == 25
