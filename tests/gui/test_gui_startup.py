from __future__ import annotations
import importlib.util,os,pytest
pytestmark=pytest.mark.skipif(importlib.util.find_spec('PyQt6') is None,reason='PyQt6 is not installed')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
def test_main_window_has_all_workspaces(qtbot,tmp_path):
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.settings_manager import SettingsManager
    from calo_rpd_studio.app.main_window import MainWindow
    state=AppState(tmp_path/'gui.sqlite');window=MainWindow(state,ExperimentManager(state),SettingsManager());qtbot.addWidget(window);assert window.stack.count()==13;assert len(window.sidebar.buttons)==13
def test_plot_toolbar_exposes_typography_controls(qtbot):
    from calo_rpd_studio.gui.plotting.scientific_plot import ScientificPlotWidget
    widget=ScientificPlotWidget();qtbot.addWidget(widget);toolbar=widget.format_toolbar;assert toolbar.font is not None;assert toolbar.size is not None;assert toolbar.bold is not None;assert toolbar.title_text is not None;assert toolbar.x_text is not None;assert toolbar.y_text is not None;assert toolbar.legend_labels is not None

def test_only_genuinely_long_workspaces_use_page_level_scrolling(qtbot, tmp_path):
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.main_window import MainWindow
    from calo_rpd_studio.app.settings_manager import SettingsManager
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage

    state = AppState(tmp_path / "scroll-layout.sqlite")
    window = MainWindow(state, ExperimentManager(state), SettingsManager())
    qtbot.addWidget(window)
    scrollable = [page for page in window.pages if isinstance(page, ScrollablePage)]
    assert len(scrollable) == 3


def test_duplicate_experiment_start_is_rejected_without_exception(qtbot, tmp_path):
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.state_manager import AppState

    state = AppState(tmp_path / "busy-state.sqlite")
    manager = ExperimentManager(state)
    messages = []
    manager.busy.connect(messages.append)
    manager._busy = True

    assert manager.start_comparison(state.config) is False
    assert messages
    assert "already running" in messages[0].lower()
