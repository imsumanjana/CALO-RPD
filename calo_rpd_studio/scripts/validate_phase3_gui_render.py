"""Render one Phase 3 GUI matrix cell and retain clipping, glyph, and compact-input evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform


SCHEMA_VERSION = "calo-phase3-gui-render-evidence-v1"


class _TransientSettings:
    """In-memory preferences so validation never mutates the user's desktop settings."""

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


def _widget_text(widget) -> tuple[str, ...]:
    from PyQt6.QtWidgets import QAbstractButton, QComboBox, QGroupBox, QLabel, QLineEdit, QTabWidget

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


def validate_render(
    output_directory: str | Path,
    *,
    theme: str,
    width: int,
    height: int,
    scale_factor: float,
    qt_platform: str = "offscreen",
) -> dict:
    """Render and audit one isolated theme, size, and scale combination."""
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["QT_QPA_PLATFORM"] = qt_platform
    os.environ["QT_SCALE_FACTOR"] = str(float(scale_factor))

    from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import (
        QAbstractButton,
        QAbstractSpinBox,
        QApplication,
        QComboBox,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QTextEdit,
        QWidget,
    )

    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.main_window import MainWindow
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.app.workspaces import WORKSPACE_GROUP_ORDER, WORKSPACE_KEYS
    from calo_rpd_studio.gui.themes.runtime_fonts import ensure_application_font
    from calo_rpd_studio.gui.themes.theme_manager import apply_theme

    application = QApplication.instance() or QApplication([])
    apply_theme(application, theme)
    font_record = ensure_application_font(application)
    runtime = output / "runtime"
    runtime.mkdir(exist_ok=True)
    settings = _TransientSettings()
    settings.set_value("navigation/compact", False)
    settings.set_value("interface_density", "comfortable")
    for group in WORKSPACE_GROUP_ORDER:
        settings.set_value(f"navigation/group/{group}", True)
    state = AppState(runtime / "phase3-render.sqlite")
    window = MainWindow(state, ExperimentManager(state), settings)
    try:
        window.resize(int(width), int(height))
        window.show()
        QTest.qWait(250)  # type: ignore[call-arg, arg-type]
        application.processEvents()

        visible_widgets = [
            widget for widget in window.findChildren(QWidget) if widget.isVisibleTo(window)
        ]
        missing_glyphs: list[dict[str, str]] = []
        replacement_hits: list[str] = []
        clipping: list[dict[str, object]] = []
        for widget in visible_widgets:
            for text in _widget_text(widget):
                if "\ufffd" in text:
                    replacement_hits.append(text)
                for character in sorted(set(text)):
                    if character.isspace() or ord(character) < 32:
                        continue
                    if not widget.fontMetrics().inFontUcs4(ord(character)):
                        missing_glyphs.append(
                            {
                                "widget": widget.metaObject().className(),
                                "character": character,
                                "codepoint": f"U+{ord(character):04X}",
                            }
                        )
            if (
                widget.width() > 0
                and widget.height() > 0
                and (
                    isinstance(widget, QAbstractButton)
                    or (isinstance(widget, QLabel) and not widget.wordWrap())
                )
            ):
                hint = widget.sizeHint()
                if hint.width() > widget.width() + 12:
                    clipping.append(
                        {
                            "widget": widget.objectName() or widget.metaObject().className(),
                            "width": widget.width(),
                            "preferred_width": hint.width(),
                        }
                    )

        compact_violations: list[dict[str, object]] = []
        for widget in visible_widgets:
            limit = None
            if isinstance(widget, QLineEdit) and not bool(widget.property("fullWidthInput")):
                limit = 480
            elif isinstance(widget, QComboBox) and not bool(widget.property("fullWidthInput")):
                limit = 420
            elif isinstance(widget, QAbstractSpinBox):
                limit = 240
            if limit is not None and widget.width() > limit:
                compact_violations.append(
                    {
                        "widget": widget.objectName() or widget.metaObject().className(),
                        "width": widget.width(),
                        "limit": limit,
                    }
                )

        long_editor_violations = [
            widget.objectName() or widget.metaObject().className()
            for widget in visible_widgets
            if isinstance(widget, (QTextEdit, QPlainTextEdit))
            and not widget.isReadOnly()
            and not bool(widget.property("expandedLongText"))
            and widget.height() > widget.fontMetrics().lineSpacing() * 7 + 30
        ]

        screenshot = output / f"phase3-{theme}-{width}x{height}-scale-{scale_factor:g}.png"
        if screenshot.exists():
            raise FileExistsError(f"Refusing to overwrite Phase 3 render: {screenshot}")
        pixmap = window.grab()
        if pixmap.isNull() or not pixmap.save(str(screenshot), "PNG"):
            raise RuntimeError("Phase 3 GUI screenshot could not be retained")
        if screenshot.stat().st_size < 10_000:
            raise RuntimeError("Phase 3 GUI screenshot is unexpectedly small")

        report = {
            "schema_version": SCHEMA_VERSION,
            "platform": platform.platform(),
            "qt_platform": application.platformName(),
            "requested_qt_platform": qt_platform,
            "qt_platform_matches_request": application.platformName() == qt_platform,
            "qt_version": QT_VERSION_STR,
            "pyqt_version": PYQT_VERSION_STR,
            "application_font": font_record.as_dict(),
            "theme": theme,
            "requested_logical_size": [int(width), int(height)],
            "scale_factor": float(scale_factor),
            "actual_window_size": [window.width(), window.height()],
            "workspace_count": len(window.pages_by_key),
            "stable_workspace_keys": list(WORKSPACE_KEYS),
            "navigation_groups": list(window.sidebar.group_headers),
            "visible_workspace_buttons": sum(
                not button.isHidden() for button in window.sidebar.buttons
            ),
            "missing_glyphs": missing_glyphs,
            "replacement_character_hits": replacement_hits,
            "clipping_candidates": clipping,
            "compact_input_violations": compact_violations,
            "long_editor_violations": long_editor_violations,
            "screenshot": screenshot.name,
            "screenshot_size_bytes": screenshot.stat().st_size,
            "screenshot_sha256": _sha256(screenshot),
            "passed": not (
                missing_glyphs
                or replacement_hits
                or clipping
                or compact_violations
                or long_editor_violations
            )
            and application.platformName() == qt_platform,
        }
    finally:
        window.close()
        application.processEvents()

    report_path = output / "phase3-gui-render-evidence.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 3 report: {report_path}")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        failure_counts = {
            "missing_glyphs": len(report["missing_glyphs"]),
            "replacement_characters": len(report["replacement_character_hits"]),
            "clipping": len(report["clipping_candidates"]),
            "compact_inputs": len(report["compact_input_violations"]),
            "long_editors": len(report["long_editor_violations"]),
            "qt_platform_mismatch": int(not report["qt_platform_matches_request"]),
        }
        detail = ", ".join(f"{name}={count}" for name, count in failure_counts.items() if count)
        raise RuntimeError(f"Phase 3 GUI render evidence failed: {detail}")
    return report


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--output", required=True)
    command.add_argument("--theme", choices=("light", "dark"), required=True)
    command.add_argument("--width", type=int, required=True)
    command.add_argument("--height", type=int, required=True)
    command.add_argument("--scale-factor", type=float, default=1.0)
    command.add_argument(
        "--qt-platform",
        choices=("offscreen", "xcb"),
        default="offscreen",
        help="Qt platform plugin; use xcb under xvfb-run for Linux desktop evidence.",
    )
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = validate_render(
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
