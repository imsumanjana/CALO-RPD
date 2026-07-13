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


def test_export_popup_series_checkboxes_follow_preview_legend(qtbot):
    from calo_rpd_studio.gui.plotting.scientific_plot import ScientificPlotWidget

    widget = ScientificPlotWidget(square_preview=True, square_export=True, square_preview_size=520)
    qtbot.addWidget(widget)
    widget.plot_series(
        {"CALO": [3.0, 2.0, 1.0], "PSO": [3.2, 2.5, 2.1]},
        "Convergence",
        "Iteration",
        "Objective",
    )
    toolbar = widget.format_toolbar
    toolbar._refresh_export_series_options()
    assert set(toolbar.export_series_checks) == {"CALO", "PSO"}
    assert all(check.isChecked() for check in toolbar.export_series_checks.values())
    toolbar.export_series_checks["PSO"].setChecked(False)
    assert toolbar._selected_export_series() == {"CALO"}


def test_result_review_opens_selected_run_in_validation(qtbot, tmp_path, monkeypatch):
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.main_window import MainWindow
    from calo_rpd_studio.app.settings_manager import SettingsManager
    from calo_rpd_studio.app.state_manager import AppState

    state = AppState(tmp_path / "review-navigation.sqlite")
    window = MainWindow(state, ExperimentManager(state), SettingsManager())
    qtbot.addWidget(window)
    for key in ("power_system", "orpd", "algorithms", "calo", "scenarios"):
        window.workflow.mark_completed(key)
    window.workflow.mark_experiment_completed()
    window.workflow.mark_statistics_completed()

    selected = {}

    def fake_select_run(experiment_id, run_id):
        selected["experiment_id"] = experiment_id
        selected["run_id"] = run_id

    monkeypatch.setattr(window.pages[10], "select_run", fake_select_run)
    window.pages[9].review_completed.emit()
    window.pages[9].validation_requested.emit("experiment-1", "run-1")

    assert window.workflow.results_reviewed is True
    assert window.stack.currentIndex() == 10
    assert selected == {"experiment_id": "experiment-1", "run_id": "run-1"}


def test_live_plot_auto_mode_uses_violation_before_feasibility(qtbot, tmp_path):
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.panels.live_optimization_panel import LiveOptimizationPanel

    state = AppState(tmp_path / "live-auto.sqlite")
    panel = LiveOptimizationPanel(state, ExperimentManager(state))
    qtbot.addWidget(panel)
    panel.update_progress({
        "algorithm": "TLBO",
        "run_index": 1,
        "iteration": 1,
        "evaluations": 20,
        "best_feasible_objective": float("nan"),
        "best_constraint_violation": 0.25,
        "feasible": False,
    })
    assert panel.metric.currentText() == panel.AUTO_MODE
    assert panel.violation_series["TLBO"][1] == [0.25]
    assert panel.plot.axis.lines
    assert panel.plot.axis.get_ylabel() == "Best normalized constraint violation"


def test_live_plot_explicit_objective_without_feasibility_shows_message(qtbot, tmp_path):
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.panels.live_optimization_panel import LiveOptimizationPanel

    state = AppState(tmp_path / "live-message.sqlite")
    panel = LiveOptimizationPanel(state, ExperimentManager(state))
    qtbot.addWidget(panel)
    panel.update_progress({
        "algorithm": "PSO",
        "run_index": 1,
        "iteration": 1,
        "evaluations": 10,
        "best_feasible_objective": float("nan"),
        "best_constraint_violation": 0.4,
        "feasible": False,
    })
    panel.metric.setCurrentText(panel.OBJECTIVE_MODE)
    assert not panel.plot.axis.lines
    assert any("No feasible incumbent" in text.get_text() for text in panel.plot.axis.texts)


def test_experiment_manager_uses_guided_scrollable_order_without_compression(qtbot, tmp_path):
    from PyQt6.QtWidgets import QScrollArea
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.panels.experiment_manager_panel import ExperimentManagerPanel

    state = AppState(tmp_path / "experiment-layout.sqlite")
    panel = ExperimentManagerPanel(state, ExperimentManager(state))
    qtbot.addWidget(panel)

    assert isinstance(panel.body_scroll, QScrollArea)
    assert panel.body_scroll.objectName() == "ExperimentManagerScroll"
    assert panel.body_scroll.horizontalScrollBarPolicy().name == "ScrollBarAlwaysOff"
    assert panel.body_layout.indexOf(panel.setup_card) < panel.body_layout.indexOf(panel.fairness_card)
    assert panel.body_layout.indexOf(panel.fairness_card) < panel.body_layout.indexOf(panel.execution_card)
    assert panel.body_layout.indexOf(panel.execution_card) < panel.body_layout.indexOf(panel.queue_card)
    assert panel.compare.isEnabled() is False
    assert panel.calo.isEnabled() is False
    for widget in (panel.runs, panel.population, panel.policy, panel.budget, panel.wall, panel.maxit, panel.workers, panel.seed):
        assert widget.minimumHeight() >= 32
