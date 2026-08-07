"""Render and audit the installed CALO-RPD wheel outside its source checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform


PACKAGED_GUI_EVIDENCE_SCHEMA = "calo-rpd-packaged-gui-evidence-v1"
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
    """In-memory preferences keep a validation render isolated from desktop settings."""

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


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _visible_text(panel) -> str:
    from PyQt6.QtWidgets import (
        QAbstractButton,
        QComboBox,
        QGroupBox,
        QLabel,
        QLineEdit,
        QTabWidget,
        QWidget,
    )

    fragments: list[str] = []
    for widget in panel.findChildren(QWidget):
        if not widget.isVisibleTo(panel):
            continue
        fragments.extend((widget.toolTip(), widget.statusTip(), widget.whatsThis()))
        if isinstance(widget, (QLabel, QAbstractButton)):
            fragments.append(widget.text())
        if isinstance(widget, QGroupBox):
            fragments.append(widget.title())
        if isinstance(widget, QLineEdit):
            fragments.append(widget.placeholderText())
        if isinstance(widget, QComboBox):
            fragments.extend(widget.itemText(index) for index in range(widget.count()))
        if isinstance(widget, QTabWidget):
            fragments.extend(widget.tabText(index) for index in range(widget.count()))
    return " ".join(fragment for fragment in fragments if fragment).lower()


def validate_packaged_gui(
    output_directory: str | Path,
    *,
    forbidden_import_root: str | Path | None = None,
) -> dict:
    """Render the installed wheel and return a self-describing evidence record."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)

    import calo_rpd_studio

    package_file = Path(calo_rpd_studio.__file__).resolve()
    source_checkout_imported: bool | None = None
    if forbidden_import_root is not None:
        forbidden = Path(forbidden_import_root).resolve()
        if _is_within(package_file, forbidden):
            raise RuntimeError(
                f"Packaged GUI validation imported from forbidden checkout root: {package_file}"
            )
        source_checkout_imported = False

    from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.main_window import MainWindow
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.themes.runtime_fonts import ensure_application_font

    runtime = output / "runtime"
    runtime.mkdir(exist_ok=True)
    previous_xdg_config = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(runtime / "config")
    previous_cwd = Path.cwd()
    application = QApplication.instance() or QApplication([])
    font_record = ensure_application_font(application)
    if not font_record.supports_validation_sample:
        raise RuntimeError("Packaged GUI could not resolve a font for ordinary scientific text")
    window = None
    try:
        os.chdir(runtime)
        state = AppState(runtime / "packaged-gui.sqlite")
        settings = _TransientSettings()
        settings.set_value("navigation/compact", False)
        for group in ("Home", "Model", "Study", "Evidence", "System"):
            settings.set_value(f"navigation/group/{group}", True)
        window = MainWindow(state, ExperimentManager(state), settings)
        window.resize(1440, 900)
        window.show()
        # PyQt exposes qWait as a static method at runtime; its current stub incorrectly models an
        # instance receiver. Keep the validated runtime call and scope the ignore to that stub bug.
        QTest.qWait(300)  # type: ignore[call-arg, arg-type]
        application.processEvents()

        if window.windowTitle() != "CALO-RPD Studio":
            raise RuntimeError(f"Unexpected packaged window title: {window.windowTitle()!r}")
        if window.stack.currentWidget() is not window.pages_by_key["dashboard"]:
            raise RuntimeError("Packaged GUI did not open on the Dashboard")
        sidebar_labels = [button.text().strip() for button in window.sidebar.buttons]
        if not sidebar_labels or any(not label for label in sidebar_labels):
            raise RuntimeError("Packaged GUI has an empty sidebar navigation label")

        visible = _visible_text(window)
        forbidden_hits = sorted(token for token in _FORBIDDEN_VISIBLE_TEXT if token in visible)
        if forbidden_hits:
            raise RuntimeError(
                "Packaged GUI exposed forbidden normal-view language: " + ", ".join(forbidden_hits)
            )

        screenshot = output / "packaged-scientist-dashboard.png"
        if screenshot.exists():
            raise FileExistsError(f"Refusing to overwrite packaged GUI screenshot: {screenshot}")
        pixmap = window.grab()
        if pixmap.isNull() or pixmap.width() < 1120 or pixmap.height() < 720:
            raise RuntimeError("Packaged GUI produced an invalid or undersized render")
        if not pixmap.save(str(screenshot), "PNG") or screenshot.stat().st_size <= 10_000:
            raise RuntimeError("Packaged GUI screenshot could not be retained")

        report = {
            "schema_version": PACKAGED_GUI_EVIDENCE_SCHEMA,
            "distribution_name": "calo-rpd-studio",
            "distribution_version": importlib.metadata.version("calo-rpd-studio"),
            "package_file": str(package_file),
            "source_checkout_imported": source_checkout_imported,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "qt_platform": str(application.platformName()),
            "qt_version": QT_VERSION_STR,
            "pyqt_version": PYQT_VERSION_STR,
            "application_font": font_record.as_dict(),
            "window_title": window.windowTitle(),
            "workspace_count": len(window.pages_by_key),
            "initial_workspace": "dashboard",
            "sidebar_labels": sidebar_labels,
            "visible_forbidden_hits": forbidden_hits,
            "screenshot": screenshot.name,
            "screenshot_width": int(pixmap.width()),
            "screenshot_height": int(pixmap.height()),
            "screenshot_size_bytes": int(screenshot.stat().st_size),
            "screenshot_sha256": _sha256(screenshot),
        }
    finally:
        if window is not None:
            window.close()
            application.processEvents()
        os.chdir(previous_cwd)
        if previous_xdg_config is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = previous_xdg_config

    report_path = output / "packaged-gui-evidence.json"
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with report_path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite packaged GUI report: {report_path}") from exc
    return report


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--output", required=True)
    command.add_argument("--forbid-import-root")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = validate_packaged_gui(
        args.output,
        forbidden_import_root=args.forbid_import_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
