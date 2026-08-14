"""Render and verify Phase 6 shell contracts without executing scientific or policy work."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


RIBBON_CATEGORIES = (
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

PANEL_KEYS = (
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


def _save_image(window, path: Path) -> dict:
    image = window.grab()
    if image.isNull() or not image.save(str(path), "PNG"):
        raise RuntimeError(f"Could not save Phase 6 render: {path}")
    return {
        "path": str(path.resolve()),
        "width": image.width(),
        "height": image.height(),
        "bytes": path.stat().st_size,
    }


def _checkbox_border_evidence(application, output: Path, theme: str) -> dict:
    """Render global checkbox states and require a contrasting indicator perimeter."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QCheckBox,
        QStyle,
        QStyleOptionButton,
        QVBoxLayout,
        QWidget,
    )

    host = QWidget()
    host.setWindowTitle(f"{theme.title()} checkbox indicator evidence")
    layout = QVBoxLayout(host)
    probes = []
    for state_name in ("unchecked", "checked", "partial", "focused", "disabled"):
        checkbox = QCheckBox(state_name.title())
        checkbox.setTristate(state_name == "partial")
        if state_name == "checked":
            checkbox.setChecked(True)
        elif state_name == "partial":
            checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        elif state_name == "disabled":
            checkbox.setEnabled(False)
        layout.addWidget(checkbox)
        probes.append((state_name, checkbox))

    host.resize(220, host.sizeHint().height())
    host.show()
    probes[3][1].setFocus(Qt.FocusReason.OtherFocusReason)
    application.processEvents()
    rendered = _save_image(host, output / f"phase6-{theme}-checkboxes.png")
    # Sample the fully composited host render. QCheckBox has a transparent
    # background, so grabbing the child alone compares premultiplied pixels
    # with transparent black instead of the application surface users see.
    image = host.grab().toImage()
    state_evidence = {}
    for state_name, checkbox in probes:
        option = QStyleOptionButton()
        option.initFrom(checkbox)
        if checkbox.checkState() == Qt.CheckState.Checked:
            option.state |= QStyle.StateFlag.State_On
        elif checkbox.checkState() == Qt.CheckState.PartiallyChecked:
            option.state |= QStyle.StateFlag.State_NoChange
        else:
            option.state |= QStyle.StateFlag.State_Off
        indicator = checkbox.style().subElementRect(
            QStyle.SubElement.SE_CheckBoxIndicator,
            option,
            checkbox,
        )
        if indicator.width() < 16 or indicator.height() < 16:
            raise AssertionError(
                f"{theme} {state_name} checkbox indicator is undersized: {indicator}"
            )
        indicator_origin = checkbox.mapTo(host, indicator.topLeft())
        indicator_left = indicator_origin.x()
        indicator_top = indicator_origin.y()
        indicator_right = indicator_left + indicator.width() - 1
        indicator_bottom = indicator_top + indicator.height() - 1
        background_x = min(image.width() - 1, indicator_right + 3)
        background_y = indicator_top + indicator.height() // 2
        background = image.pixelColor(background_x, background_y)
        perimeter = []
        for x in range(indicator_left, indicator_right + 1):
            perimeter.extend(
                (
                    image.pixelColor(x, indicator_top),
                    image.pixelColor(x, indicator_bottom),
                )
            )
        for y in range(indicator_top, indicator_bottom + 1):
            perimeter.extend(
                (
                    image.pixelColor(indicator_left, y),
                    image.pixelColor(indicator_right, y),
                )
            )
        maximum_rgb_delta = max(
            abs(color.red() - background.red())
            + abs(color.green() - background.green())
            + abs(color.blue() - background.blue())
            for color in perimeter
        )
        if maximum_rgb_delta < 90:
            raise AssertionError(
                f"{theme} {state_name} checkbox has no visible perimeter contrast: "
                f"RGB delta {maximum_rgb_delta}"
            )
        state_evidence[state_name] = {
            "indicator_width": indicator.width(),
            "indicator_height": indicator.height(),
            "maximum_perimeter_rgb_delta": maximum_rgb_delta,
        }
    host.close()
    application.processEvents()
    return {"render": rendered, "states": state_evidence}


def validate(output: Path, *, platform: str) -> dict:
    os.environ["QT_QPA_PLATFORM"] = str(platform)
    from PyQt6.QtCore import QPoint, QSettings, Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QPushButton,
        QSizePolicy,
        QStackedWidget,
        QTabWidget,
        QToolButton,
    )

    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.main_window import MainWindow
    from calo_rpd_studio.app.settings_manager import SettingsManager
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.themes.theme_manager import apply_theme

    output.mkdir(parents=True, exist_ok=True)
    settings_directory = output / "settings"
    settings_directory.mkdir(parents=True, exist_ok=True)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(settings_directory),
    )
    application = QApplication.instance() or QApplication([])
    state = AppState(output / "phase6-render.sqlite")
    settings = SettingsManager()
    apply_theme(application, "light")
    window = MainWindow(state, ExperimentManager(state), settings)
    window.resize(1440, 900)
    window.show()
    application.processEvents()

    categories = tuple(
        window.ribbon.tabs.tabText(index) for index in range(window.ribbon.tabs.count())
    )
    if categories != RIBBON_CATEGORIES:
        raise AssertionError(f"Unexpected ribbon categories: {categories!r}")
    command_ids = [item.command_id for item in window.command_registry.specs]
    if len(command_ids) != len(set(command_ids)):
        raise AssertionError("Phase 6 command IDs are not unique")
    if {"view.context", "view.ribbon"}.intersection(command_ids):
        raise AssertionError("Input/ribbon hide commands remain exposed")
    if window.context_dock.accessibleName() != "Contextual input pane":
        raise AssertionError("Context dock accessible identity is missing")
    if window.context_dock.isHidden() or window.context_dock.toggleViewAction().isEnabled():
        raise AssertionError("The permanent input pane can be hidden")
    if window.ribbon.compact or window.ribbon.tabs.minimumHeight() < 118:
        raise AssertionError("The ribbon is not permanently expanded")
    if isinstance(window.ribbon.tabs, QTabWidget):
        raise AssertionError("The ribbon still uses the native composite tab widget")
    if window.ribbon.tabs.parentWidget() is not window.ribbon.navigation_area:
        raise AssertionError("Ribbon navigation escaped its clipping boundary")
    category_buttons = window.ribbon.tabs.findChildren(QPushButton, "RibbonCategoryButton")
    if len(category_buttons) != len(RIBBON_CATEGORIES):
        raise AssertionError("Explicit ribbon category selector is incomplete")
    if sum(button.isChecked() for button in category_buttons) != 1:
        raise AssertionError("Ribbon category selection is not exclusive")
    if window.ribbon.tabs.findChild(QStackedWidget, "RibbonPageStack") is None:
        raise AssertionError("Explicit ribbon command-page stack is absent")
    if window.ribbon.identity_bar.findChildren(QToolButton):
        raise AssertionError("A ribbon command is parented inside the product heading")
    if not window.documents.preview_scroll.widgetResizable():
        raise AssertionError("The main preview workspace is not scrollable")
    if window.documents.scientific_workspace.minimumWidth() < 920:
        raise AssertionError("The preview workspace can still be horizontally congested")
    if not window.documents.tabBar().isHidden() or window.documents.cornerWidget() is not None:
        raise AssertionError("Redundant central workspace header remains visible")
    if hasattr(window, "guide"):
        raise AssertionError("Redundant global workflow banner remains instantiated")
    if window.windowIcon().isNull():
        raise AssertionError("The native application/window icon is missing")
    if window.activity_center.accessibleName() != "Application activity":
        raise AssertionError("Activity center accessible identity is missing")
    if tuple(window.documents.document_ids()) != ("scientific-workspace",):
        raise AssertionError("Unexpected document opened during construction")
    if state.policy_training_active or state.task_status.busy:
        raise AssertionError("GUI construction performed policy or foreground work")

    light = _save_image(window, output / "phase6-light.png")
    light_checkbox_evidence = _checkbox_border_evidence(application, output, "light")
    panel_renders = []
    for key in PANEL_KEYS:
        window.stack.setCurrentWidget(window.pages_by_key[key])
        application.processEvents()
        panel_renders.append(_save_image(window, output / f"panel-{key}.png"))
    window.stack.setCurrentWidget(window.pages_by_key["dashboard"])
    application.processEvents()
    window.ribbon.select_category("Workspace")
    application.processEvents()
    workspaces = _save_image(window, output / "phase6-workspaces.png")
    window.resize(1120, 720)
    application.processEvents()
    constrained = _save_image(window, output / "phase6-constrained.png")

    window.command_registry.action("policies.training").trigger()
    application.processEvents()
    if window.training_center.process is not None:
        raise AssertionError("Training navigation started a process")
    if state.policy_training_active or state.task_status.busy:
        raise AssertionError("Training navigation changed policy/task execution state")
    if window.training_center.start_button.isEnabled():
        raise AssertionError("Training start enabled before a successful readiness check")
    if tuple(window.documents.document_ids()) != ("scientific-workspace",):
        raise AssertionError("Training action opened a redundant document")
    if window.context_pane.stack.currentWidget() is not window.context_pane.training:
        raise AssertionError("Training inputs were not selected in the context pane")
    training_editor = window.context_pane.training
    if training_editor.action_bar.isHidden():
        raise AssertionError("Training action footer is not visible with the training inputs")
    if training_editor.status.text() != (
        "Ready for validation · plan will be generated from these inputs · "
        "automatic recovery is on"
    ):
        raise AssertionError("Training inputs were refreshed before their status control was ready")
    if training_editor.training_action_button.text() != "Check readiness":
        raise AssertionError("Training inputs do not expose the readiness action")
    if not training_editor.training_action_button.isEnabled():
        raise AssertionError("Training readiness action is unavailable for valid default inputs")
    if window.command_registry.action("policies.training").text() != "Train policy":
        raise AssertionError("Training ribbon navigation was repurposed as a hidden process action")
    if set(training_editor.fields) != {"output"}:
        raise AssertionError("Scientist-facing training paths are not minimal")
    training_editor.model.plan_error = "synthetic fresh-plan generation failure"
    training_editor.refresh()
    if training_editor.status.text() != (
        "Training plan could not be generated: synthetic fresh-plan generation failure"
    ):
        raise AssertionError("Fresh-plan failures are mislabeled as saved-plan import failures")
    training_editor.model.clear_loaded_plan()
    training_editor.refresh()
    if len(training_editor._plan_controls) < 15:
        raise AssertionError("Frozen-plan training parameters are absent from the input pane")
    if training_editor.plan_group.isHidden():
        raise AssertionError("Training parameters are hidden from the input pane")
    expected_help = {
        "library",
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
    observed_help = set(training_editor.info_buttons)
    if observed_help != expected_help:
        raise AssertionError(
            "Training information-control mismatch: "
            f"missing={sorted(expected_help - observed_help)}, "
            f"unexpected={sorted(observed_help - expected_help)}"
        )
    if hasattr(training_editor, "architecture"):
        raise AssertionError("The TSH-CALO training pane still exposes an architecture selector")
    if "architecture" in window.training_launch_model.values:
        raise AssertionError("The TSH-CALO training model still accepts mutable architecture input")
    if any(
        not button.toolTip() or not button.accessibleName()
        for button in training_editor.info_buttons.values()
    ):
        raise AssertionError("Training information controls are not accessible")
    if training_editor.resume is not window.training_center.resume:
        raise AssertionError("Exact-resume choice is not bound to the training controller")
    if training_editor.recovery_stack.currentWidget() is not training_editor.automatic_recovery:
        raise AssertionError("New training does not present automatic recovery status")
    if not training_editor.automatic_recovery.isChecked():
        raise AssertionError("New-training automatic recovery is not visibly on")
    if training_editor.resume.isChecked():
        raise AssertionError("New training incorrectly requests exact resume")
    campaign_source = (
        REPOSITORY_ROOT / "calo_rpd_studio/algorithms/calo/tsh_calo_training_campaign.py"
    ).read_text(encoding="utf-8")
    if "checkpoint_sha256 = session.save_resume(checkpoint_path)" not in campaign_source:
        raise AssertionError("New training does not retain automatic recovery checkpoints")
    for token in (
        'EVENT_LOG_FILE = "training_events.jsonl"',
        "request_tsh_calo_training_pause",
        "_honor_pause_after_checkpoint(status, progress)",
        '"total_candidate_evaluations": total_evaluations',
        '"training_parameter_schema_sha256"',
        "TSH_CALO_NON_TRAINING_PLAN_FIELDS",
    ):
        if token not in campaign_source:
            raise AssertionError(f"Checkpoint-safe finite training contract is absent: {token}")
    extension_source = (
        REPOSITORY_ROOT
        / "calo_rpd_studio/algorithms/calo/tsh_calo_training_extension.py"
    ).read_text(encoding="utf-8")
    for token in (
        "IndependentTSHCALOTrainingExtension",
        "parent_manifest_sha256",
        "cumulative_candidate_evaluations",
        "same_scientific_design_required",
        "training_compatibility_contract",
        "policy_parameter_layout_sha256",
        "parse_tsh_calo_extension_plan",
        "execution_source_commit",
        "automatic_start",
    ):
        if token not in extension_source:
            raise AssertionError(f"Finite completed-training extension contract is absent: {token}")
    if training_editor.pause_training_button.accessibleName() != (
        "Pause policy training after the next checkpoint"
    ):
        raise AssertionError("The training pane has no accessible checkpoint-safe pause action")
    if hasattr(training_editor, "training_progress"):
        raise AssertionError("Training progress is duplicated inside the left input pane")
    if (
        window.global_status.progress.minimum() != 0
        or window.global_status.progress.maximum() != 100
    ):
        raise AssertionError("The bottom status bar has no finite committed-progress scale")
    for row in training_editor.path_rows:
        top_left = row.mapTo(training_editor, QPoint(0, 0))
        if top_left.x() < 0 or top_left.x() + row.width() > training_editor.width() + 2:
            raise AssertionError("A training path row overflows the left input pane")
    if training_editor.library_picker.itemText(0) != "New training":
        raise AssertionError("The resumable-model picker has no explicit new-training choice")
    for obsolete_control in (
        "add_library_location_button",
        "default_library_path",
        "load_plan_button",
    ):
        if hasattr(training_editor, obsolete_control):
            raise AssertionError(f"Obsolete training control is still visible: {obsolete_control}")
    if window.training_launch_model.values.get("plan"):
        raise AssertionError("New training still depends on an external settings-template path")
    if Path(training_editor.fields["output"].text()).parent != (
        window.training_model_library.default_directory
    ):
        raise AssertionError("Fresh training does not default to the per-user model directory")
    if training_editor.selected_training_cases() != ["case30", "case57"]:
        raise AssertionError("All eligible bundled training cases are not selected by default")
    training_command_source = (
        REPOSITORY_ROOT / "calo_rpd_studio/scripts/train_tsh_calo.py"
    ).read_text(encoding="utf-8")
    if "resource_preflight = validate_training_resources(plan)" not in training_command_source:
        raise AssertionError("Readiness does not apply the training resource-admission preflight")
    if 'TRAINING_EVENT_PREFIX = "CALO_TRAINING_EVENT "' not in training_command_source:
        raise AssertionError("The training command does not stream structured progress events")
    if "compatible_extension=arguments.extend" not in training_command_source:
        raise AssertionError("Software revision is still incorrectly used as extension identity")
    if "load_plan(arguments.plan, compatible_extension=arguments.extend)" not in (
        training_command_source
    ):
        raise AssertionError("Extension does not use its metadata-tolerant authenticated plan")
    if "elif not arguments.extend and any(" not in training_command_source:
        raise AssertionError("Completed legacy-authority extension still requires historical paths")
    if "self._rollout_capacity(population_size, max_evaluations)" not in (
        REPOSITORY_ROOT / "calo_rpd_studio/gui/panels/independent_training_panel.py"
    ).read_text(encoding="utf-8"):
        raise AssertionError("Fresh training does not bound retained rollout transitions")
    for protected_case in ("case118", "case300"):
        checkbox = training_editor.case_checks.get(protected_case)
        if checkbox is None or checkbox.isEnabled() or checkbox.isChecked():
            raise AssertionError(f"Protected holdout {protected_case} is selectable for training")
    if window.context_pane.tabs.count() != 1:
        raise AssertionError("The input pane contains workspace navigation")
    intelligence = window.pages_by_key["calo_intelligence"]
    if intelligence.policy_table.verticalScrollBarPolicy() != (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    ) or intelligence.policy_table.horizontalScrollBarPolicy() != (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    ):
        raise AssertionError("The Policy library retains a nested table scrollbar")
    expected_table_height = (
        max(
            intelligence.policy_table.horizontalHeader().height(),
            intelligence.policy_table.horizontalHeader().sizeHint().height(),
        )
        + sum(
            intelligence.policy_table.rowHeight(row)
            for row in range(intelligence.policy_table.rowCount())
        )
        + intelligence.policy_table.frameWidth() * 2
    )
    if intelligence.policy_table.height() != expected_table_height:
        raise AssertionError("The Policy library height does not match its visible entries")
    if intelligence.path.sizePolicy().horizontalPolicy() != QSizePolicy.Policy.Expanding:
        raise AssertionError("The governing-policy field does not use the full available width")
    content = intelligence.widget()
    margins = content.layout().contentsMargins()
    usable_width = content.width() - margins.left() - margins.right()
    for group in (intelligence.policy_center_group, intelligence.policy_controller_group):
        if group.width() < usable_width - 2:
            raise AssertionError("A CALO Intelligence policy section is not full width")
    if intelligence.policy_activate_button.text() != "Activate for experiments":
        raise AssertionError("The policy library has no explicit experiment-activation action")
    if intelligence.policy_delete_button.text() != "Delete model files":
        raise AssertionError("The policy library has no completed-model file deletion action")
    if hasattr(intelligence, "policy_removal_review_button"):
        raise AssertionError("The removed policy-removal review action is still visible")
    if intelligence.apply_policy_button.text() != (
        "Apply governing policy and continue to Power System"
    ):
        raise AssertionError("The governing-policy path does not expose the next Power System step")
    intelligence_source = (
        REPOSITORY_ROOT / "calo_rpd_studio/gui/panels/calo_intelligence_panel.py"
    ).read_text(encoding="utf-8")
    for token in (
        "completed_campaigns()",
        "Permanently delete completed model files",
        "Permanently delete model file",
        "Qualification required",
        "_delete_standalone_policy_file",
    ):
        if token not in intelligence_source:
            raise AssertionError(f"Completed-model policy-library contract is absent: {token}")

    window.stack.setCurrentWidget(intelligence)
    window.resize(1120, 720)
    application.processEvents()
    preview_scroll = window.documents.preview_scroll
    if preview_scroll.verticalScrollBar().maximum() <= 0:
        raise AssertionError("The main preview has no scroll range for the complete policy page")
    preview_scroll.verticalScrollBar().setValue(0)
    intelligence._policy_selection_changed()
    application.processEvents()
    application.processEvents()
    if preview_scroll.verticalScrollBar().value() != 0:
        raise AssertionError("Policy selection moved the main preview away from its scroll position")
    preview_scroll.verticalScrollBar().setValue(
        preview_scroll.verticalScrollBar().maximum()
    )
    application.processEvents()
    governing_bottom = intelligence.apply_policy_button.mapTo(
        preview_scroll.viewport(),
        QPoint(0, intelligence.apply_policy_button.height()),
    ).y()
    if governing_bottom > preview_scroll.viewport().height():
        raise AssertionError("The bottom of Governing policy is unreachable above Activity")
    if preview_scroll.verticalScrollBar().value() != preview_scroll.verticalScrollBar().maximum():
        raise AssertionError("The main preview could not reach the complete Governing policy block")
    preview_scroll.verticalScrollBar().setValue(0)

    window.ribbon.set_compact(True)
    if window.ribbon.compact or window.ribbon.tabs.minimumHeight() < 118:
        raise AssertionError("Ribbon compatibility call hid the permanent ribbon")

    apply_theme(application, "dark")
    window.ribbon.select_category("Policies")
    application.processEvents()
    dark = _save_image(window, output / "phase6-dark.png")
    dark_checkbox_evidence = _checkbox_border_evidence(application, output, "dark")
    theme_source = application.styleSheet()
    if "QPushButton#PrimaryButton:disabled" not in theme_source:
        raise AssertionError("Disabled primary-button selector is absent from the active theme")
    if "QMainWindow::separator" not in theme_source:
        raise AssertionError("Native main-window separators remain unthemed")

    result = {
        "schema": "calo-rpd-phase6-gui-validation-v1",
        "passed": True,
        "qt_platform": application.platformName(),
        "ribbon_categories": list(categories),
        "command_count": len(command_ids),
        "document_ids_after_navigation": list(window.documents.document_ids()),
        "activity_tabs": [
            window.activity_center.tabText(index) for index in range(window.activity_center.count())
        ],
        "renders": [light, workspaces, constrained, dark],
        "checkbox_borders": {
            "light": light_checkbox_evidence,
            "dark": dark_checkbox_evidence,
        },
        "panel_renders": panel_renders,
        "policy_training_executed": False,
        "policy_evaluation_executed": False,
        "policy_qualification_executed": False,
        "policy_registration_executed": False,
        "policy_activation_executed": False,
        "policy_deletion_executed": False,
        "synthetic_completed_campaign_deletion_contract_in_suite": True,
        "protected_case_work_executed": False,
        "release_executed": False,
        "scientific_workflows_executed": False,
        "human_screen_reader_acceptance_inferred": False,
        "human_usability_acceptance_inferred": False,
        "scientist_acceptance_inferred": False,
    }
    window.close()
    application.processEvents()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", default="offscreen")
    arguments = parser.parse_args(argv)
    try:
        report = validate(arguments.output.resolve(), platform=arguments.platform)
    except Exception as exc:
        report = {
            "schema": "calo-rpd-phase6-gui-validation-v1",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "policy_training_executed": False,
            "policy_evaluation_executed": False,
            "policy_qualification_executed": False,
            "policy_registration_executed": False,
            "policy_activation_executed": False,
            "policy_deletion_executed": False,
            "protected_case_work_executed": False,
            "release_executed": False,
            "scientific_workflows_executed": False,
            "human_acceptance_inferred": False,
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    sys.exit(main())
