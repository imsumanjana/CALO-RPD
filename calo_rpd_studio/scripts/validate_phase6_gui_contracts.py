"""Render and verify Phase 6 shell contracts without executing scientific or policy work."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


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


def validate(output: Path, *, platform: str) -> dict:
    os.environ["QT_QPA_PLATFORM"] = str(platform)
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import (
        QApplication,
        QPushButton,
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
    if training_editor.training_action_button.text() != "Check readiness":
        raise AssertionError("Training inputs do not expose the readiness action")
    if not training_editor.training_action_button.isEnabled():
        raise AssertionError("Training readiness action is unavailable for valid default inputs")
    if window.command_registry.action("policies.training").text() != "Train policy":
        raise AssertionError("Training ribbon navigation was repurposed as a hidden process action")
    if set(training_editor.fields) != {"plan", "output"}:
        raise AssertionError("Scientist-facing training paths are not minimal")
    if len(training_editor._plan_controls) < 15:
        raise AssertionError("Frozen-plan training parameters are absent from the input pane")
    if training_editor.plan_group.isHidden():
        raise AssertionError("Training parameters are hidden from the input pane")
    expected_help = {
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
    observed_help = set(training_editor.info_buttons)
    if observed_help != expected_help:
        raise AssertionError(
            "Training information-control mismatch: "
            f"missing={sorted(expected_help - observed_help)}, "
            f"unexpected={sorted(observed_help - expected_help)}"
        )
    if any(
        not button.toolTip() or not button.accessibleName()
        for button in training_editor.info_buttons.values()
    ):
        raise AssertionError("Training information controls are not accessible")
    if training_editor.selected_training_cases() != ["case30", "case57"]:
        raise AssertionError("All eligible bundled training cases are not selected by default")
    for protected_case in ("case118", "case300"):
        checkbox = training_editor.case_checks.get(protected_case)
        if checkbox is None or checkbox.isEnabled() or checkbox.isChecked():
            raise AssertionError(f"Protected holdout {protected_case} is selectable for training")
    if window.context_pane.tabs.count() != 1:
        raise AssertionError("The input pane contains workspace navigation")

    window.ribbon.set_compact(True)
    if window.ribbon.compact or window.ribbon.tabs.minimumHeight() < 118:
        raise AssertionError("Ribbon compatibility call hid the permanent ribbon")

    apply_theme(application, "dark")
    window.ribbon.select_category("Policies")
    application.processEvents()
    dark = _save_image(window, output / "phase6-dark.png")
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
        "panel_renders": panel_renders,
        "policy_training_executed": False,
        "policy_evaluation_executed": False,
        "policy_qualification_executed": False,
        "policy_registration_executed": False,
        "policy_activation_executed": False,
        "policy_deletion_executed": False,
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
