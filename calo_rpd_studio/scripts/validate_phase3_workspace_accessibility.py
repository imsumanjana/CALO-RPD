"""Collect non-scientific Phase 3 workspace, keyboard, and accessibility evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any


SCHEMA_VERSION = "calo-phase3-workspace-accessibility-evidence-v1"
_FORBIDDEN_VISIBLE_TEXT = (
    "journal",
    "transactions",
    "q1",
    "q2",
    "q3",
    "development",
    "developer",
    "backend",
    "schema",
    "xpu",
    "safe-80",
    "worker budget",
    "microbatch",
    "utilization",
)


class _TransientSettings:
    """In-memory preferences prevent evidence collection from changing user settings."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default=None):
        return self.values.get(key, default)

    def set_value(self, key: str, value: object) -> None:
        self.values[key] = value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_rgb(value: str) -> tuple[float, float, float]:
    normalized = value.removeprefix("#")
    return tuple(int(normalized[index : index + 2], 16) / 255 for index in (0, 2, 4))


def _relative_luminance(value: str) -> float:
    channels = []
    for channel in _hex_rgb(value):
        channels.append(
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(first: str, second: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _contrast_checks(theme: str) -> list[dict[str, object]]:
    from calo_rpd_studio.gui.themes.tokens import DARK_TOKENS, LIGHT_TOKENS

    tokens = DARK_TOKENS if theme == "dark" else LIGHT_TOKENS
    checks = (
        ("body text on background", "text", "background", 4.5),
        ("body text on surface", "text", "surface", 4.5),
        ("muted text on surface", "muted_text", "surface", 4.5),
        ("accent control on surface", "accent", "surface", 3.0),
        ("focus indicator on background", "focus", "background", 3.0),
    )
    evidence: list[dict[str, object]] = []
    for name, foreground, background, minimum in checks:
        ratio = _contrast_ratio(tokens[foreground], tokens[background])
        evidence.append(
            {
                "name": name,
                "foreground_token": foreground,
                "foreground": tokens[foreground],
                "background_token": background,
                "background": tokens[background],
                "minimum_ratio": minimum,
                "actual_ratio": round(ratio, 3),
                "passed": ratio >= minimum,
            }
        )
    return evidence


def _widget_text(widget) -> tuple[str, ...]:
    from PyQt6.QtWidgets import (
        QAbstractButton,
        QComboBox,
        QGroupBox,
        QLabel,
        QLineEdit,
        QTabWidget,
    )

    values: list[str] = []
    if isinstance(widget, (QLabel, QAbstractButton)):
        values.append(widget.text())
    if isinstance(widget, QGroupBox):
        values.append(widget.title())
    if isinstance(widget, QLineEdit):
        values.extend((widget.text(), widget.placeholderText()))
    if isinstance(widget, QComboBox):
        values.extend(widget.itemText(index) for index in range(widget.count()))
    if isinstance(widget, QTabWidget):
        values.extend(widget.tabText(index) for index in range(widget.count()))
    values.extend((widget.toolTip(), widget.statusTip(), widget.whatsThis()))
    return tuple(value for value in values if value)


def _widget_id(widget) -> str:
    return widget.objectName() or widget.metaObject().className()


def _workspace_audit(window, key: str, index: int, output: Path) -> dict[str, Any]:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QAbstractButton,
        QAbstractSpinBox,
        QComboBox,
        QFormLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QScrollArea,
        QTextEdit,
        QWidget,
    )

    page = window.pages_by_key[key]
    window.stack.setCurrentIndex(index)
    window.sidebar.set_current(index)
    window.repaint()
    visible = [page, *(widget for widget in page.findChildren(QWidget) if widget.isVisibleTo(page))]

    missing_glyphs: set[tuple[str, str, str]] = set()
    replacement_hits: set[str] = set()
    clipping: list[dict[str, object]] = []
    visible_text: list[str] = []
    for widget in visible:
        for text in _widget_text(widget):
            visible_text.append(text)
            if "\ufffd" in text:
                replacement_hits.add(text)
            for character in set(text):
                if character.isspace() or ord(character) < 32:
                    continue
                if not widget.fontMetrics().inFontUcs4(ord(character)):
                    missing_glyphs.add((_widget_id(widget), character, f"U+{ord(character):04X}"))
        if isinstance(widget, (QAbstractButton, QLabel)) and _widget_text(widget):
            if widget.width() > 0 and widget.height() > 0:
                hint = widget.sizeHint()
                if hint.width() > widget.width() + 12 and not (
                    isinstance(widget, QLabel) and widget.wordWrap()
                ):
                    clipping.append(
                        {
                            "widget": _widget_id(widget),
                            "text": " | ".join(_widget_text(widget)),
                            "accessible_name": widget.accessibleName(),
                            "tool_tip": widget.toolTip(),
                            "width": widget.width(),
                            "height": widget.height(),
                            "preferred_width": hint.width(),
                            "preferred_height": hint.height(),
                        }
                    )

    focusable_types = (
        QAbstractButton,
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
        QPlainTextEdit,
        QTextEdit,
    )
    accessible_name_violations = [
        _widget_id(widget)
        for widget in visible
        if isinstance(widget, focusable_types)
        and widget.isEnabled()
        and widget.focusPolicy() != Qt.FocusPolicy.NoFocus
        and not widget.accessibleName().strip()
    ]

    buddy_violations: list[str] = []
    for form in page.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            field_item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            label = label_item.widget() if label_item is not None else None
            field = field_item.widget() if field_item is not None else None
            if (
                isinstance(label, QLabel)
                and field is not None
                and label.isVisibleTo(page)
                and label.buddy() is not field
            ):
                buddy_violations.append(label.text())

    nested_scroll_violations: list[str] = []
    for scroll in (item for item in visible if isinstance(item, QScrollArea)):
        parent = scroll.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea) and parent.isVisibleTo(page):
                nested_scroll_violations.append(_widget_id(scroll))
                break
            parent = parent.parentWidget()

    compact_input_violations: list[dict[str, object]] = []
    for widget in visible:
        limit = None
        if isinstance(widget, QLineEdit) and not bool(widget.property("fullWidthInput")):
            limit = 480
        elif isinstance(widget, QComboBox) and not bool(widget.property("fullWidthInput")):
            limit = 420
        elif isinstance(widget, QAbstractSpinBox):
            limit = 240
        if limit is not None and widget.width() > limit:
            compact_input_violations.append(
                {"widget": _widget_id(widget), "width": widget.width(), "limit": limit}
            )

    long_editor_violations = [
        _widget_id(widget)
        for widget in visible
        if isinstance(widget, (QTextEdit, QPlainTextEdit))
        and not widget.isReadOnly()
        and not bool(widget.property("expandedLongText"))
        and (
            not bool(widget.property("compactLongText"))
            or not bool(widget.property("hasExpandDialog"))
            or widget.cornerWidget() is None
        )
    ]

    normalized_text = " ".join(visible_text).casefold()
    forbidden_hits = sorted(token for token in _FORBIDDEN_VISIBLE_TEXT if token in normalized_text)

    screenshot = output / f"workspace-{index + 1:02d}-{key}.png"
    if screenshot.exists():
        raise FileExistsError(f"Refusing to overwrite workspace screenshot: {screenshot}")
    pixmap = window.grab()
    if pixmap.isNull() or not pixmap.save(str(screenshot), "PNG"):
        raise RuntimeError(f"Could not retain workspace screenshot for {key}")
    if screenshot.stat().st_size < 10_000:
        raise RuntimeError(f"Workspace screenshot is unexpectedly small: {key}")

    failures = (
        missing_glyphs,
        replacement_hits,
        clipping,
        accessible_name_violations,
        buddy_violations,
        nested_scroll_violations,
        compact_input_violations,
        long_editor_violations,
        forbidden_hits,
    )
    return {
        "key": key,
        "index": index,
        "visible_widget_count": len(visible),
        "missing_glyphs": [
            {"widget": item[0], "character": item[1], "codepoint": item[2]}
            for item in sorted(missing_glyphs)
        ],
        "replacement_character_hits": sorted(replacement_hits),
        "clipping_candidates": clipping,
        "accessible_name_violations": accessible_name_violations,
        "form_buddy_violations": buddy_violations,
        "nested_scroll_violations": nested_scroll_violations,
        "compact_input_violations": compact_input_violations,
        "long_editor_violations": long_editor_violations,
        "forbidden_visible_text_hits": forbidden_hits,
        "screenshot": screenshot.name,
        "screenshot_size_bytes": screenshot.stat().st_size,
        "screenshot_sha256": _sha256(screenshot),
        "passed": not any(failures),
    }


def _keyboard_interactions(window, application) -> list[dict[str, object]]:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    results: list[dict[str, object]] = []

    window.activateWindow()
    window.raise_()
    QTest.keyClick(
        window,
        Qt.Key.Key_K,
        Qt.KeyboardModifier.ControlModifier,
    )
    application.processEvents()
    results.append(
        {
            "name": "Ctrl+K focuses workspace search",
            "passed": window.sidebar.search.hasFocus(),
        }
    )

    window.sidebar.search.setText("dashboard")
    application.processEvents()
    dashboard_index = list(window.pages_by_key).index("dashboard")
    results.append(
        {
            "name": "search filters to an available matching workspace",
            "passed": not window.sidebar.buttons[dashboard_index].isHidden(),
        }
    )
    window.sidebar.search.clear()

    window.sidebar.set_compact(True)
    compact_persisted = window.settings_manager.value("navigation/compact", False) is True
    window.sidebar.set_compact(False)
    results.append(
        {
            "name": "compact navigation state persists and restores",
            "passed": compact_persisted
            and window.settings_manager.value("navigation/compact", True) is False,
        }
    )

    home_before = window.sidebar.group_expanded["Home"]
    window.sidebar._set_group_expanded("Home", not home_before)
    collapsed_persisted = window.settings_manager.value("navigation/group/Home", home_before) == (
        not home_before
    )
    window.sidebar._set_group_expanded("Home", home_before)
    results.append(
        {
            "name": "navigation group collapse state persists and restores",
            "passed": collapsed_persisted,
        }
    )

    experiment_page = window.pages_by_key["experiment"]
    window.stack.setCurrentWidget(experiment_page)
    application.processEvents()
    workflow = experiment_page.study_setup_workflow
    workflow.set_step(0)
    workflow.next_button.setFocus()
    QTest.keyClick(workflow.next_button, Qt.Key.Key_Space)
    application.processEvents()
    results.append(
        {
            "name": "Space advances Study Setup without launching a study",
            "passed": workflow.current_step() == 1,
        }
    )
    workflow.step_buttons[6].setFocus()
    QTest.keyClick(workflow.step_buttons[6], Qt.Key.Key_Space)
    application.processEvents()
    results.append(
        {
            "name": "keyboard selects the Review + launch presentation step",
            "passed": workflow.current_step() == 6 and workflow.progress.text() == "Step 7 of 7",
        }
    )
    workflow.set_step(0)

    from calo_rpd_studio.gui.widgets.disclosure import DisclosurePanel

    disclosures = window.findChildren(DisclosurePanel)
    disclosure_passed = bool(disclosures)
    for disclosure in disclosures:
        for page in window.pages_by_key.values():
            if page is disclosure or page.isAncestorOf(disclosure):
                window.stack.setCurrentWidget(page)
                application.processEvents()
                break
        disclosure.set_expanded(False)
        disclosure.toggle.setFocus()
        QTest.keyClick(disclosure.toggle, Qt.Key.Key_Space)
        application.processEvents()
        disclosure_passed = disclosure_passed and disclosure.content.isVisible()
        disclosure.set_expanded(False)
    results.append(
        {
            "name": "Space opens every progressive-disclosure control",
            "control_count": len(disclosures),
            "passed": disclosure_passed,
        }
    )
    return results


def validate_workspace_accessibility(
    output_directory: str | Path,
    *,
    theme: str,
    width: int,
    height: int,
    scale_factor: float,
    qt_platform: str,
) -> dict[str, Any]:
    """Audit all workspaces without invoking any scientific or policy action."""
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase3-workspace-accessibility-evidence.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 3 evidence: {report_path}")

    runtime = output / "runtime"
    runtime.mkdir(exist_ok=True)
    os.environ["QT_QPA_PLATFORM"] = qt_platform
    os.environ["QT_SCALE_FACTOR"] = str(float(scale_factor))
    os.environ["XDG_CONFIG_HOME"] = str(runtime / "config")
    if platform.system() == "Windows":
        os.environ["APPDATA"] = str(runtime / "roaming")
        os.environ["LOCALAPPDATA"] = str(runtime / "local")

    from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    import calo_rpd_studio
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.main_window import MainWindow
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.app.workspaces import WORKSPACE_GROUP_ORDER, WORKSPACE_KEYS
    from calo_rpd_studio.gui.themes.runtime_fonts import ensure_application_font
    from calo_rpd_studio.gui.themes.theme_manager import apply_theme

    application = QApplication.instance() or QApplication([])
    apply_theme(application, theme)
    font_record = ensure_application_font(application)
    if not font_record.supports_validation_sample:
        raise RuntimeError("No resolved font supports the Phase 3 validation sample")

    settings = _TransientSettings()
    settings.set_value("navigation/compact", False)
    settings.set_value("interface_density", "comfortable")
    for group in WORKSPACE_GROUP_ORDER:
        settings.set_value(f"navigation/group/{group}", True)

    state = AppState(runtime / "phase3-workspace-accessibility.sqlite")
    window = MainWindow(state, ExperimentManager(state), settings)
    try:
        window.resize(int(width), int(height))
        window.show()
        QTest.qWait(250)  # type: ignore[call-arg, arg-type]
        application.processEvents()

        keyboard = _keyboard_interactions(window, application)
        workspaces = []
        for index, key in enumerate(WORKSPACE_KEYS):
            workspaces.append(_workspace_audit(window, key, index, output))
            application.processEvents()

        contrast = _contrast_checks(theme)
        source_root = Path(calo_rpd_studio.__file__).resolve().parent
        implementation_paths = (
            source_root / "app" / "main_window.py",
            source_root / "gui" / "navigation" / "sidebar.py",
            source_root / "gui" / "widgets" / "form_density.py",
            source_root / "gui" / "widgets" / "study_setup.py",
            source_root / "gui" / "widgets" / "disclosure.py",
            source_root / "gui" / "themes" / "tokens.py",
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "qt_platform": application.platformName(),
            "requested_qt_platform": qt_platform,
            "qt_platform_matches_request": application.platformName() == qt_platform,
            "qt_version": QT_VERSION_STR,
            "pyqt_version": PYQT_VERSION_STR,
            "theme": theme,
            "requested_logical_size": [int(width), int(height)],
            "scale_factor": float(scale_factor),
            "application_font": font_record.as_dict(),
            "workspace_count": len(WORKSPACE_KEYS),
            "stable_workspace_keys": list(WORKSPACE_KEYS),
            "keyboard_interactions": keyboard,
            "contrast_checks": contrast,
            "workspace_evidence": workspaces,
            "implementation_files_sha256": {
                str(path.relative_to(source_root.parent)): _sha256(path)
                for path in implementation_paths
            },
            "scientific_actions_executed": False,
            "policy_workflows_executed": False,
            "policy_training_executed": False,
            "policy_evaluation_executed": False,
            "qualification_campaign_executed": False,
            "benchmark_executed": False,
            "protected_cases_opened": False,
            "presentation_audit_direct_stack_access": True,
            "direct_stack_access_qualification_boundary": (
                "Direct stack selection renders locked presentation surfaces only; it does not "
                "claim legal workflow reachability or alter workflow completion state."
            ),
            "human_screen_reader_acceptance_inferred": False,
            "human_scientist_acceptance_inferred": False,
            "passed": all(item["passed"] for item in keyboard)
            and all(item["passed"] for item in contrast)
            and all(item["passed"] for item in workspaces)
            and application.platformName() == qt_platform,
        }
    finally:
        window.close()
        application.processEvents()

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        keyboard_failures = sum(not item["passed"] for item in report["keyboard_interactions"])
        contrast_failures = sum(not item["passed"] for item in report["contrast_checks"])
        workspace_failures = sum(not item["passed"] for item in report["workspace_evidence"])
        platform_failures = int(not report["qt_platform_matches_request"])
        raise RuntimeError(
            "Phase 3 workspace/accessibility evidence failed: "
            f"keyboard={keyboard_failures}, contrast={contrast_failures}, "
            f"workspaces={workspace_failures}, platform={platform_failures}"
        )
    return report


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--output", required=True)
    command.add_argument("--theme", choices=("light", "dark"), required=True)
    command.add_argument("--width", type=int, default=1440)
    command.add_argument("--height", type=int, default=900)
    command.add_argument("--scale-factor", type=float, default=1.0)
    command.add_argument(
        "--qt-platform",
        choices=("offscreen", "xcb"),
        default="offscreen",
    )
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = validate_workspace_accessibility(
        args.output,
        theme=args.theme,
        width=args.width,
        height=args.height,
        scale_factor=args.scale_factor,
        qt_platform=args.qt_platform,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
