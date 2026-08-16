from __future__ import annotations

import faulthandler
import os
from pathlib import Path
import sys
import threading

import pytest


pytest.importorskip("PyQt6")


@pytest.fixture(autouse=True)
def _visual_gui_test_deadline(request):
    def abort_stuck_test():
        sys.stderr.write(f"\nVisual GUI test exceeded 120 seconds: {request.node.nodeid}\n")
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


def test_main_scientist_workspace_renders_to_nonempty_image(qtbot, tmp_path, monkeypatch):
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.main_window import MainWindow
    from calo_rpd_studio.app.settings_manager import SettingsManager
    from calo_rpd_studio.app.state_manager import AppState

    monkeypatch.setattr(MainWindow, "_initial_system_scan", lambda self: None)
    monkeypatch.setattr(MainWindow, "_check_unfinished_work", lambda self: None)
    monkeypatch.setattr(MainWindow, "closeEvent", lambda self, event: event.accept())
    state = AppState(tmp_path / "visual-smoke.sqlite")
    window = MainWindow(state, ExperimentManager(state), SettingsManager())
    qtbot.addWidget(
        window, before_close_func=lambda widget: widget.activity_center.detach_logging()
    )
    window.resize(1440, 900)
    window.show()
    qtbot.wait(250)

    assert window.windowTitle() == "CALO-RPD Studio"
    assert window.stack.currentWidget() is window.pages_by_key["dashboard"]
    assert all(button.text().strip() for button in window.sidebar.buttons)

    artifact_root = Path(os.environ.get("CALO_GUI_ARTIFACT_DIR", tmp_path))
    artifact_root.mkdir(parents=True, exist_ok=True)
    screenshot = artifact_root / "scientist-dashboard.png"
    pixmap = window.grab()
    assert not pixmap.isNull()
    assert pixmap.width() >= 1120
    assert pixmap.height() >= 720
    assert pixmap.save(str(screenshot), "PNG")
    assert screenshot.stat().st_size > 10_000
