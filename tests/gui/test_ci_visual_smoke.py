from __future__ import annotations

import os
from pathlib import Path

import pytest


pytest.importorskip("PyQt6")


def test_main_scientist_workspace_renders_to_nonempty_image(qtbot, tmp_path):
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.main_window import MainWindow
    from calo_rpd_studio.app.settings_manager import SettingsManager
    from calo_rpd_studio.app.state_manager import AppState

    state = AppState(tmp_path / "visual-smoke.sqlite")
    window = MainWindow(state, ExperimentManager(state), SettingsManager())
    qtbot.addWidget(window)
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
