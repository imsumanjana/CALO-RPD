"""Main CALO-RPD Studio window with a guided scientific workflow."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.navigation.sidebar import NavigationSidebar
from calo_rpd_studio.gui.panels.algorithms_panel import AlgorithmsPanel
from calo_rpd_studio.gui.panels.application_settings_panel import ApplicationSettingsPanel
from calo_rpd_studio.gui.panels.benchmark_campaign_panel import BenchmarkCampaignPanel
from calo_rpd_studio.gui.panels.calo_intelligence_panel import CALOIntelligencePanel
from calo_rpd_studio.gui.panels.dashboard_panel import DashboardPanel
from calo_rpd_studio.gui.panels.experiment_manager_panel import ExperimentManagerPanel
from calo_rpd_studio.gui.panels.live_optimization_panel import LiveOptimizationPanel
from calo_rpd_studio.gui.panels.orpd_formulation_panel import ORPDFormulationPanel
from calo_rpd_studio.gui.panels.power_system_panel import PowerSystemPanel
from calo_rpd_studio.gui.panels.publication_export_panel import PublicationExportPanel
from calo_rpd_studio.gui.panels.results_explorer_panel import ResultsExplorerPanel
from calo_rpd_studio.gui.panels.robust_scenarios_panel import RobustScenariosPanel
from calo_rpd_studio.gui.panels.statistical_analysis_panel import StatisticalAnalysisPanel
from calo_rpd_studio.gui.panels.validation_audit_panel import ValidationAuditPanel
from calo_rpd_studio.gui.widgets.global_status_bar import GlobalStatusBarWidget
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage
from calo_rpd_studio.gui.widgets.workflow_guide import WorkflowGuide

from .project_manager import ProjectManager
from .workflow_manager import WorkflowManager


WORKSPACES = [
    ("Dashboard", ""),
    ("Power System", ""),
    ("ORPD Formulation", ""),
    ("Algorithms", ""),
    ("CALO Intelligence", ""),
    ("Robust Scenarios", ""),
    ("Experiment Manager", ""),
    ("Live Optimization", ""),
    ("Statistical Analysis", ""),
    ("Results Explorer", ""),
    ("Validation & Audit", ""),
    ("Publication Export", ""),
    ("Application Settings", ""),
    ("Benchmark & Evidence", ""),
]


class MainWindow(QMainWindow):
    def __init__(self, state, experiment_manager, settings_manager, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.experiment_manager = experiment_manager
        self.settings_manager = settings_manager
        self.workflow = WorkflowManager(state)

        self.setWindowTitle("CALO-RPD Studio")
        self.resize(1500, 920)
        self.setMinimumSize(1120, 720)

        self.sidebar = NavigationSidebar(WORKSPACES)
        self.stack = QStackedWidget()
        self.stack.setObjectName("WorkspaceStack")
        self.pages = [
            DashboardPanel(state),
            PowerSystemPanel(state),
            ORPDFormulationPanel(state),
            AlgorithmsPanel(state),
            CALOIntelligencePanel(state, experiment_manager),
            RobustScenariosPanel(state),
            ExperimentManagerPanel(state, experiment_manager),
            LiveOptimizationPanel(state, experiment_manager),
            StatisticalAnalysisPanel(state),
            ResultsExplorerPanel(state),
            ValidationAuditPanel(state),
            PublicationExportPanel(state),
            ApplicationSettingsPanel(state, settings_manager),
            ScrollablePage(BenchmarkCampaignPanel(state, experiment_manager)),
        ]
        for page in self.pages:
            self.stack.addWidget(page)

        self.guide = WorkflowGuide()
        self.guide.next_clicked.connect(self._go_to_recommended_step)
        right = QWidget()
        right.setObjectName("WorkspaceContainer")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.guide)
        right_layout.addWidget(self.stack, 1)

        self.sidebar.page_requested.connect(self._set_workspace)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("MainSplitter")
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.sidebar)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([248, 1252])
        self.setCentralWidget(splitter)

        self._create_toolbar()
        self._create_global_status_bar()
        self._connect_workflow()
        self._refresh_workflow()

    def _connect_workflow(self) -> None:
        self.state.case_changed.connect(lambda _: self.workflow.invalidate_from("power_system"))
        self.pages[1].stage_completed.connect(
            lambda: self.workflow.mark_completed("power_system")
        )
        self.pages[2].stage_completed.connect(lambda: self.workflow.mark_completed("orpd"))
        self.pages[3].stage_completed.connect(
            lambda: self.workflow.mark_completed("algorithms")
        )
        self.pages[4].stage_completed.connect(lambda: self.workflow.mark_completed("calo"))
        self.pages[5].stage_completed.connect(
            lambda: self.workflow.mark_completed("scenarios")
        )
        self.pages[4].experiment_manager_requested.connect(lambda: self._set_workspace(6))
        self.experiment_manager.started.connect(lambda _: self.workflow.mark_experiment_started())
        self.experiment_manager.completed.connect(
            lambda _: self.workflow.mark_experiment_completed()
        )
        self.experiment_manager.cancelled.connect(lambda _: self.workflow.mark_experiment_stopped())
        self.experiment_manager.failed.connect(lambda _: self.workflow.mark_experiment_stopped())
        self.pages[8].analysis_completed.connect(self.workflow.mark_statistics_completed)
        self.pages[9].review_completed.connect(self.workflow.mark_results_reviewed)
        self.pages[9].validation_requested.connect(self._open_reviewed_run_for_validation)
        self.state.runs_changed.connect(self._refresh_verified_count)
        self.workflow.changed.connect(self._refresh_workflow)

    def _open_reviewed_run_for_validation(self, experiment_id: str, run_id: str) -> None:
        """Atomically unlock and open Validation & Audit for the reviewed run."""
        # Do not rely on signal delivery order: explicitly mark the review complete here before
        # attempting navigation to a workflow-gated workspace.
        self.workflow.mark_results_reviewed()
        self.state.current_experiment_id = experiment_id
        self.pages[10].select_run(experiment_id, run_id)
        self._refresh_workflow()
        self._set_workspace(10)
        self.state.task_status.finish("Result review confirmed; selected run is ready for independent validation")

    def _create_global_status_bar(self) -> None:
        self.global_status = GlobalStatusBarWidget()
        self.global_status.cancel_clicked.connect(self.state.task_status.cancel)
        self.state.task_status.changed.connect(self._on_task_status_changed)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addPermanentWidget(self.global_status, 1)
        self.global_status.apply_snapshot(self.state.task_status.snapshot())

    def _on_task_status_changed(self, snapshot: dict) -> None:
        self.global_status.apply_snapshot(snapshot)
        if not snapshot.get("busy") and snapshot.get("state") in {"Completed", "Failed", "Cancelled"}:
            QTimer.singleShot(4500, self.state.task_status.reset_ready)

    def _refresh_verified_count(self) -> None:
        experiment_id = self.state.current_experiment_id or None
        count = len(self.state.database.list_runs(experiment_id, verified_only=True)) if experiment_id else 0
        self.workflow.set_verified_results(count)

    def _refresh_workflow(self) -> None:
        for index in range(len(WORKSPACES)):
            state, reason = self.workflow.workspace_state(index)
            self.sidebar.set_workflow_state(index, state, reason)
        self.pages[4].set_experiment_navigation_enabled(
            self.workflow.is_workspace_enabled(6)
        )

        completed, total = self.workflow.progress()
        descriptor = self.workflow.next_descriptor()
        if descriptor is None:
            self.guide.set_guidance(
                "Workflow complete",
                "The configured workflow has no pending required step.",
                "Workflow complete",
                False,
            )
            return

        if descriptor.key in {"validation", "publication"}:
            step_text = "Post-experiment workflow"
        else:
            step_number = min(completed + 1, total)
            step_text = f"Guided workflow · step {step_number} of {total}"
        self.guide.set_guidance(
            step_text,
            f"Next: {descriptor.title}. {descriptor.instruction}",
            f"Open {WORKSPACES[descriptor.workspace_index][0]}",
            self.workflow.is_workspace_enabled(descriptor.workspace_index),
        )

    def _go_to_recommended_step(self) -> None:
        descriptor = self.workflow.next_descriptor()
        if descriptor is not None:
            self._set_workspace(descriptor.workspace_index)

    def _set_workspace(self, index: int) -> None:
        if not self.workflow.is_workspace_enabled(index):
            _, reason = self.workflow.workspace_state(index)
            QMessageBox.information(self, "Workflow step locked", reason)
            return
        self.stack.setCurrentIndex(index)
        self.sidebar.set_current(index)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Project")
        toolbar.setObjectName("TopToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        open_action = QAction("Open configuration", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_config)
        save_action = QAction("Save configuration", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_config)
        about_action = QAction("About", self)
        about_action.triggered.connect(self.about)

        toolbar.addAction(open_action)
        toolbar.addAction(save_action)
        toolbar.addSeparator()

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        context = QLabel("Cognitive Adaptive Learning Optimizer · ORPD Research Studio")
        context.setObjectName("ToolbarContext")
        toolbar.addWidget(context)
        toolbar.addSeparator()
        toolbar.addAction(about_action)

    def open_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open experiment configuration",
            "",
            "Configuration (*.yaml *.yml *.json)",
        )
        if not path:
            return
        try:
            self.state.config = ProjectManager.load(path)
            self.state.current_case = None
            self.state.current_power_flow = None
            self.workflow.reset()
            self.state.update_config()
            self.state.task_status.finish(f"Configuration loaded: {path}")
            self._set_workspace(1)
        except Exception as exc:
            QMessageBox.critical(self, "Configuration load failed", str(exc))

    def save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save experiment configuration",
            "calo_rpd_experiment.yaml",
            "YAML (*.yaml);;JSON (*.json)",
        )
        if not path:
            return
        try:
            ProjectManager.save(self.state.config, path)
            self.state.task_status.finish(f"Configuration saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Configuration save failed", str(exc))

    def about(self) -> None:
        QMessageBox.information(
            self,
            "About CALO-RPD Studio",
            "CALO-RPD Studio 3.0.0\n"
            "Cognitive Adaptive Learning Optimizer for Robust Reactive Power Dispatch\n\n"
            "Guided scientific optimization, reproducible benchmarking, validation, statistics, and publication export.",
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.experiment_manager.running:
            answer = QMessageBox.question(
                self,
                "Experiment running",
                "An experiment is active. Request safe cancellation and close the application?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.experiment_manager.cancel()
            worker = self.experiment_manager.worker
            if worker is not None:
                worker.wait(5000)
        event.accept()
