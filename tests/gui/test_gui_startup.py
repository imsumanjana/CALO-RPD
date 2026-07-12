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


def test_live_plot_is_square_and_png_dpi_range_is_600_to_2400(qtbot):
    from calo_rpd_studio.gui.plotting.scientific_plot import ScientificPlotWidget

    widget = ScientificPlotWidget(square_preview=True, square_export=True, square_preview_size=640)
    qtbot.addWidget(widget)
    assert widget.canvas.width() == widget.canvas.height() == 640
    width, height = widget.figure.get_size_inches()
    assert width == pytest.approx(height)
    toolbar = widget.format_toolbar
    assert toolbar.dpi.minimum() == 600
    assert toolbar.dpi.maximum() == 2400
    assert toolbar.dpi.value() == 600
    assert toolbar.width.value() == pytest.approx(toolbar.height.value())
    assert toolbar.tight.isEnabled() is False


def test_live_optimization_page_contains_vertical_scroll_area(qtbot, tmp_path):
    from PyQt6.QtWidgets import QScrollArea
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.panels.live_optimization_panel import LiveOptimizationPanel

    state = AppState(tmp_path / "live-square.sqlite")
    panel = LiveOptimizationPanel(state, ExperimentManager(state))
    qtbot.addWidget(panel)
    scroll_areas = panel.findChildren(QScrollArea)
    assert any(area.objectName() == "LiveOptimizationScroll" for area in scroll_areas)
    assert panel.plot.square_preview is True
    assert panel.plot.square_export is True


def test_plot_toolbar_uses_four_focused_popup_tools(qtbot):
    from PyQt6.QtWidgets import QScrollArea
    from calo_rpd_studio.gui.plotting.scientific_plot import ScientificPlotWidget

    widget = ScientificPlotWidget()
    qtbot.addWidget(widget)
    toolbar = widget.format_toolbar
    assert toolbar.text_tool_button.accessibleName() == "Text & labels"
    assert toolbar.plot_tool_button.accessibleName() == "Plot appearance"
    assert toolbar.export_tool_button.accessibleName() == "Export figure"
    assert toolbar.style_tool_button.accessibleName() == "Style profiles"
    assert toolbar.findChildren(QScrollArea) == []
    assert toolbar.text_popup.isVisible() is False
    assert toolbar.plot_popup.isVisible() is False
    assert toolbar.export_popup.isVisible() is False
    assert toolbar.style_popup.isVisible() is False
