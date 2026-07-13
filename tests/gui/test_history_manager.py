import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from calo_rpd_studio.app.settings_manager import SettingsManager
from calo_rpd_studio.app.state_manager import AppState
from calo_rpd_studio.gui.dialogs.experiment_history_dialog import ExperimentHistoryDialog
from calo_rpd_studio.gui.panels.application_settings_panel import ApplicationSettingsPanel


def test_history_manager_is_available_from_application_settings(tmp_path):
    QApplication.instance() or QApplication([])
    state = AppState(tmp_path / "results.sqlite")
    settings = SettingsManager()
    panel = ApplicationSettingsPanel(state, settings)
    assert "0 experiment(s)" in panel.history_summary.text()
    dialog = ExperimentHistoryDialog(state, panel)
    assert dialog.windowTitle() == "Manage experiment history"
    assert not dialog.delete_experiment_button.isEnabled()
    assert not dialog.clear_button.isEnabled()
