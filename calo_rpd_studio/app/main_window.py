"""Main CALO-RPD Studio window with v6 key-based, policy-first navigation."""

from __future__ import annotations

import logging

from calo_rpd_studio.version import VERSION

_LOG = logging.getLogger(__name__)

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
from calo_rpd_studio.gui.panels.portfolio_manager_panel import PortfolioManagerPanel
from calo_rpd_studio.gui.panels.resume_center_panel import ResumeCenterPanel
from calo_rpd_studio.gui.dialogs.unfinished_work_dialog import UnfinishedWorkDialog
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
from .experiment_workspace_restorer import ExperimentWorkspaceRestorer
from .session_recovery import SessionRecoveryJournal
from .workspaces import (
    WORKSPACES,
    WORKSPACE_KEYS,
    WORKSPACE_TITLE,
    migrate_workspace_ui,
    WORKSPACE_SCHEMA_VERSION,
    WORKSPACE_LAYOUT_ID,
    workspace_index_for_key,
    workspace_key_for_index,
)


class MainWindow(QMainWindow):
    def __init__(self, state, experiment_manager, settings_manager, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.experiment_manager = experiment_manager
        self.settings_manager = settings_manager
        self.workflow = WorkflowManager(state)
        self._close_when_paused = False
        self._close_when_training_stopped = False
        self._training_exclusive_active = False

        self.setWindowTitle("CALO-RPD Studio")
        self.resize(1500, 920)
        self.setMinimumSize(1120, 720)

        self.sidebar = NavigationSidebar(WORKSPACES)
        self.stack = QStackedWidget()
        self.stack.setObjectName("WorkspaceStack")
        self.pages_by_key = {
            "dashboard": DashboardPanel(state),
            "calo_intelligence": CALOIntelligencePanel(state, experiment_manager),
            "power_system": PowerSystemPanel(state),
            "orpd": ORPDFormulationPanel(state),
            "algorithms": AlgorithmsPanel(state),
            "portfolio": PortfolioManagerPanel(state),
            "scenarios": RobustScenariosPanel(state),
            "experiment": ExperimentManagerPanel(state, experiment_manager),
            "live_optimization": LiveOptimizationPanel(state, experiment_manager),
            "statistics": StatisticalAnalysisPanel(state),
            "results": ResultsExplorerPanel(state),
            "validation": ValidationAuditPanel(state),
            "publication": PublicationExportPanel(state),
            "resume_center": ResumeCenterPanel(state, experiment_manager),
            "settings": ApplicationSettingsPanel(state, settings_manager),
            "benchmark": ScrollablePage(BenchmarkCampaignPanel(state, experiment_manager)),
        }
        self.pages = [self.pages_by_key[key] for key in WORKSPACE_KEYS]
        for page in self.pages:
            self.stack.addWidget(page)
        self.restorer = ExperimentWorkspaceRestorer(self.state, self.workflow, self.pages_by_key)
        self.session_recovery = SessionRecoveryJournal()
        self._previous_unclean_session = self.session_recovery.previous_unclean()
        self.session_recovery.begin(workspace_ui={"workspace_schema_version": WORKSPACE_SCHEMA_VERSION, "workspace_layout_id": WORKSPACE_LAYOUT_ID, "workspace_key": "dashboard", "workspace_index": 0})

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
        QTimer.singleShot(150, self._initial_system_scan)
        QTimer.singleShot(350, self._check_unfinished_work)

    def _connect_workflow(self) -> None:
        self.state.case_changed.connect(lambda _: self.workflow.invalidate_from("power_system"))
        self.pages_by_key["power_system"].stage_completed.connect(lambda: self.workflow.mark_completed("power_system"))
        self.pages_by_key["orpd"].stage_completed.connect(lambda: self.workflow.mark_completed("orpd"))
        self.pages_by_key["algorithms"].stage_completed.connect(lambda: self.workflow.mark_completed("algorithms"))
        self.pages_by_key["portfolio"].stage_completed.connect(lambda: self.workflow.mark_completed("portfolio"))
        self.pages_by_key["calo_intelligence"].stage_completed.connect(self._governing_policy_event)
        self.pages_by_key["scenarios"].stage_completed.connect(lambda: self.workflow.mark_completed("scenarios"))
        self.pages_by_key["calo_intelligence"].experiment_manager_requested.connect(lambda: self._set_workspace("experiment"))
        self.experiment_manager.started.connect(lambda _: self.workflow.mark_experiment_started())
        self.experiment_manager.completed.connect(lambda _: self.workflow.mark_experiment_completed())
        self.experiment_manager.cancelled.connect(lambda _: self.workflow.mark_experiment_stopped())
        self.experiment_manager.failed.connect(lambda _: self.workflow.mark_experiment_stopped())
        self.experiment_manager.completed.connect(lambda _: self._finish_deferred_close())
        self.experiment_manager.cancelled.connect(lambda _: self._finish_deferred_close())
        self.experiment_manager.failed.connect(lambda _: self._finish_deferred_close())
        self.pages_by_key["statistics"].analysis_completed.connect(self.workflow.mark_statistics_completed)
        self.pages_by_key["results"].review_completed.connect(self.workflow.mark_results_reviewed)
        self.pages_by_key["results"].validation_requested.connect(self._open_reviewed_run_for_validation)
        self.pages_by_key["results"].experiment_restore_requested.connect(self.restore_experiment_workspace)
        self.pages_by_key["resume_center"].workspace_requested.connect(self._set_workspace)
        self.pages_by_key["resume_center"].experiment_restore_requested.connect(self.restore_experiment_workspace)
        self.pages_by_key["resume_center"].policy_training_resumed.connect(
            lambda task_id: self.pages_by_key["calo_intelligence"].resume_task_by_id(task_id)
        )
        self.pages_by_key["resume_center"].validation_resumed.connect(
            lambda task_id: self.pages_by_key["validation"].resume_task_by_id(task_id)
        )
        self.pages_by_key["resume_center"].portfolio_export_resumed.connect(
            lambda task_id: self.pages_by_key["publication"].resume_task_by_id(task_id)
        )
        self.state.runs_changed.connect(self._refresh_verified_count)
        self.state.policy_state_changed.connect(lambda _status: self.workflow.notify_governing_policy_changed())
        self.state.compute_profile_changed.connect(lambda _profile: self._refresh_workflow())
        self.state.policy_training_changed.connect(self._on_policy_training_changed)
        self.workflow.changed.connect(self._refresh_workflow)
        self.workflow.changed.connect(self._persist_workspace_state)

    def _governing_policy_event(self) -> None:
        self.state.notify_policy_state_changed()

    def _initial_system_scan(self) -> None:
        try:
            self.state.refresh_compute_profile()
            dashboard = self.pages_by_key["dashboard"]
            if hasattr(dashboard, "refresh_compute"):
                dashboard.refresh_compute()
        except Exception as exc:
            _LOG.exception("Initial compute-topology scan failed")
            self.state.task_status.fail(f"System readiness scan failed: {type(exc).__name__}: {exc}")
        finally:
            self.state.notify_policy_state_changed()

    def _open_reviewed_run_for_validation(self, experiment_id: str, run_id: str) -> None:
        self.workflow.mark_results_reviewed()
        self.state.current_experiment_id = experiment_id
        self.pages_by_key["validation"].select_run(experiment_id, run_id)
        self._refresh_workflow()
        self._set_workspace("validation")
        self.state.task_status.finish(
            "Result review confirmed; selected run is ready for independent validation"
        )

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

    def _on_policy_training_changed(self, active: bool, detail: str) -> None:
        self._training_exclusive_active = bool(active)
        # v6.1 beta1: scientific configuration is globally frozen while policy training owns the
        # compute/runtime state. Dashboard remains readable; CALO Intelligence may be viewed but its
        # widgets are disabled. Safe Stop stays available through the global status bar.
        for key, page in self.pages_by_key.items():
            page.setEnabled(not active or key == "dashboard")
        dashboard = self.pages_by_key.get("dashboard")
        if dashboard is not None and hasattr(dashboard, "set_training_exclusive_mode"):
            dashboard.set_training_exclusive_mode(bool(active), str(detail or ""))
        if hasattr(self, "open_config_action"):
            self.open_config_action.setEnabled(not active)
        if hasattr(self, "save_config_action"):
            self.save_config_action.setEnabled(not active)
        self._refresh_workflow()
        if not active and self._close_when_training_stopped:
            self._close_when_training_stopped = False
            QTimer.singleShot(0, self.close)

    def _refresh_verified_count(self) -> None:
        experiment_id = self.state.current_experiment_id or None
        count = len(self.state.database.list_runs(experiment_id, verified_only=True)) if experiment_id else 0
        self.workflow.set_verified_results(count)

    def _refresh_workflow(self) -> None:
        for index, key in enumerate(WORKSPACE_KEYS):
            state, reason = self.workflow.workspace_state_key(key)
            self.sidebar.set_workflow_state(index, state, reason)

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
        step_text = (
            "Post-experiment workflow"
            if descriptor.key in {"validation", "publication"}
            else f"Guided workflow · step {min(completed + 1, total)} of {total}"
        )
        self.guide.set_guidance(
            step_text,
            f"Next: {descriptor.title}. {descriptor.instruction}",
            f"Open {WORKSPACE_TITLE[descriptor.workspace_key]}",
            self.workflow.is_workspace_enabled(descriptor.workspace_key),
        )

    def _go_to_recommended_step(self) -> None:
        descriptor = self.workflow.next_descriptor()
        if descriptor is not None:
            self._set_workspace(descriptor.workspace_key)

    def _workspace_key(self, workspace: str | int) -> str:
        return workspace_key_for_index(workspace) if isinstance(workspace, int) else str(workspace)

    def _set_workspace(self, workspace: str | int) -> None:
        self._persist_workspace_state()
        key = self._workspace_key(workspace)
        if bool(getattr(self.state, "policy_training_active", False)) and key not in {"dashboard", "calo_intelligence"}:
            QMessageBox.information(
                self,
                "Training Exclusive Lock",
                "Policy training is running. All scientific/configuration panels are locked until training completes or Safe Stops.",
            )
            return
        if not self.workflow.is_workspace_enabled(key):
            _, reason = self.workflow.workspace_state_key(key)
            QMessageBox.information(self, "Workflow step locked", reason)
            return
        index = workspace_index_for_key(key)
        self.stack.setCurrentIndex(index)
        self.sidebar.set_current(index)

    def _check_unfinished_work(self) -> None:
        previous = dict(self._previous_unclean_session or {})
        self._previous_unclean_session = None
        if previous:
            previous_ui, migration = migrate_workspace_ui(previous.get("workspace_ui"))
            experiment_id = str(previous.get("experiment_id", "") or "")
            message = "CALO-RPD detected an unclean previous application session."
            if migration.warning:
                message += f"\n\nMigration note: {migration.warning}"
            if experiment_id:
                answer = QMessageBox.question(
                    self,
                    "Recover previous application session",
                    message + f"\n\nRestore experiment {experiment_id!r} using the saved scientific workspace?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if answer == QMessageBox.StandardButton.Yes:
                    try:
                        self.restore_experiment_workspace(experiment_id)
                    except Exception:
                        _LOG.exception("Unclean-session experiment restoration failed")
            else:
                target = str(previous_ui.get("workspace_key", "dashboard") or "dashboard")
                if self.workflow.is_workspace_enabled(target):
                    self._set_workspace(target)
        items = self.state.resume_service.unfinished()
        if not items:
            return
        dialog = UnfinishedWorkDialog(items, self)
        dialog.exec()
        if dialog.open_resume_center:
            self.pages_by_key["resume_center"].refresh()
            self._set_workspace("resume_center")

    def _persist_workspace_state(self) -> None:
        experiment_id = str(self.state.current_experiment_id or "")
        if not experiment_id:
            return
        try:
            live = self.pages_by_key["live_optimization"]
            live_state = live.view_state() if hasattr(live, "view_state") else {}
            key = workspace_key_for_index(self.stack.currentIndex())
            self.state.database.save_workspace_state(
                experiment_id,
                workflow=self.workflow.snapshot(),
                ui={
                    "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
                    "workspace_layout_id": WORKSPACE_LAYOUT_ID,
                    "workspace_key": key,
                    # Kept only for compatibility with external readers. v6 restoration uses the key.
                    "workspace_index": int(self.stack.currentIndex()),
                    "live_optimization": live_state,
                    "results_experiment_id": str(
                        getattr(self.pages_by_key["results"], "_selected_experiment_id", "") or ""
                    ),
                },
            )
            try:
                status = self.state.governing_policy_status()
                profile = getattr(self.state, "compute_protection_profile", None)
                self.session_recovery.update(
                    workspace_ui={
                        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
                        "workspace_layout_id": WORKSPACE_LAYOUT_ID,
                        "workspace_key": key,
                        "workspace_index": int(self.stack.currentIndex()),
                    },
                    experiment_id=experiment_id,
                    policy_training_active=bool(getattr(self.state, "policy_training_active", False)),
                    governing_policy_sha256=str(getattr(status, "policy_sha256", "") or ""),
                    compute_profile_fingerprint=str(getattr(profile, "topology_fingerprint", "") or ""),
                )
            except Exception:
                _LOG.exception("Failed to update application-session recovery journal")
        except Exception as exc:
            _LOG.exception("Failed to persist workspace state for experiment %s", experiment_id)
            self.state.task_status.fail(f"Workspace-state persistence failed: {type(exc).__name__}: {exc}")

    def restore_experiment_workspace(self, experiment_id: str) -> None:
        try:
            restored = self.restorer.restore(str(experiment_id))
            self._refresh_workflow()
            ui_state = dict(restored.get("ui") or {})
            live = self.pages_by_key["live_optimization"]
            if hasattr(live, "restore_view_state"):
                live.restore_view_state(ui_state.get("live_optimization"))
            results_experiment_id = str(ui_state.get("results_experiment_id", "") or "")
            if results_experiment_id:
                self.pages_by_key["results"].select_experiment(results_experiment_id)
            ui_state, migration = migrate_workspace_ui(ui_state, fallback_key="dashboard")
            target_key = str(ui_state.get("workspace_key", "dashboard") or "dashboard")
            if migration.warning:
                self.state.task_status.start(f"Workspace migration: {migration.warning}")
            if not self.workflow.is_workspace_enabled(target_key):
                target_key = "live_optimization" if self.workflow.is_workspace_enabled("live_optimization") else "experiment"
            self._set_workspace(target_key)
            self.state.task_status.finish(
                f"Restored experiment workspace · {restored['runs']} stored run(s) · {restored['campaign_status']}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Experiment restoration failed", str(exc))

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Project")
        toolbar.setObjectName("TopToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        open_action = QAction("Open configuration", self)
        self.open_config_action = open_action
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_config)
        save_action = QAction("Save configuration", self)
        self.save_config_action = save_action
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
            self, "Open experiment configuration", "", "Configuration (*.yaml *.yml *.json)"
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
            self._set_workspace("calo_intelligence")
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
            f"CALO-RPD Studio {VERSION}\n"
            "Cognitive Adaptive Learning Optimizer for Robust Reactive Power Dispatch\n\n"
            "Policy-first guided scientific optimization with Safe-80 compute protection, reproducible benchmarking, validation, statistics, and publication export.",
        )

    def _finish_deferred_close(self) -> None:
        if self._close_when_paused and not self.experiment_manager.running:
            self._close_when_paused = False
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._persist_workspace_state()
        if bool(getattr(self.state, "policy_training_active", False)):
            answer = QMessageBox.question(
                self,
                "Policy training running",
                "CALO policy training is active under the Global Training Exclusive Lock. Request an exact Safe Stop and close after training state is durably preserved?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._close_when_training_stopped = True
            panel = self.pages_by_key.get("calo_intelligence")
            if panel is not None and hasattr(panel, "request_training_safe_stop"):
                panel.request_training_safe_stop()
            event.ignore()
            return
        if self.experiment_manager.running:
            answer = QMessageBox.question(
                self,
                "Experiment running",
                "An experiment is active. Request a safe pause? New jobs will stop and the application will close after all active jobs have committed.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._close_when_paused = True
            self.experiment_manager.pause()
            event.ignore()
            return
        try:
            key = workspace_key_for_index(self.stack.currentIndex())
            status = self.state.governing_policy_status()
            profile = getattr(self.state, "compute_protection_profile", None)
            self.session_recovery.mark_clean(
                workspace_ui={
                    "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
                    "workspace_layout_id": WORKSPACE_LAYOUT_ID,
                    "workspace_key": key,
                    "workspace_index": int(self.stack.currentIndex()),
                },
                experiment_id=str(self.state.current_experiment_id or ""),
                policy_training_active=False,
                governing_policy_sha256=str(getattr(status, "policy_sha256", "") or ""),
                compute_profile_fingerprint=str(getattr(profile, "topology_fingerprint", "") or ""),
            )
        except Exception:
            _LOG.exception("Failed to finalize clean application-session recovery journal")
        event.accept()
