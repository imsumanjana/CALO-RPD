"""Main CALO-RPD Studio window with v6 key-based, policy-first navigation."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QByteArray, Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QTextBrowser,
)

from calo_rpd_studio.gui.command_registry import CommandRegistry, CommandSpec
from calo_rpd_studio.gui.dialogs.unfinished_work_dialog import UnfinishedWorkDialog
from calo_rpd_studio.gui.icons.workspace_icons import application_icon
from calo_rpd_studio.gui.navigation.sidebar import NavigationSidebar
from calo_rpd_studio.gui.panels.algorithms_panel import AlgorithmsPanel
from calo_rpd_studio.gui.panels.application_settings_panel import ApplicationSettingsPanel
from calo_rpd_studio.gui.panels.benchmark_campaign_panel import BenchmarkCampaignPanel
from calo_rpd_studio.gui.panels.calo_intelligence_panel import CALOIntelligencePanel
from calo_rpd_studio.gui.panels.dashboard_panel import DashboardPanel
from calo_rpd_studio.gui.panels.experiment_manager_panel import ExperimentManagerPanel
from calo_rpd_studio.gui.panels.independent_training_panel import (
    IndependentTrainingPanel,
    TrainingModelLibrary,
    TrainingLaunchModel,
)
from calo_rpd_studio.gui.panels.live_optimization_panel import LiveOptimizationPanel
from calo_rpd_studio.gui.panels.portfolio_manager_panel import PortfolioManagerPanel
from calo_rpd_studio.gui.panels.orpd_formulation_panel import ORPDFormulationPanel
from calo_rpd_studio.gui.panels.power_system_panel import PowerSystemPanel
from calo_rpd_studio.gui.panels.publication_export_panel import PublicationExportPanel
from calo_rpd_studio.gui.panels.results_explorer_panel import ResultsExplorerPanel
from calo_rpd_studio.gui.panels.robust_scenarios_panel import RobustScenariosPanel
from calo_rpd_studio.gui.panels.resume_center_panel import ResumeCenterPanel
from calo_rpd_studio.gui.panels.statistical_analysis_panel import StatisticalAnalysisPanel
from calo_rpd_studio.gui.panels.validation_audit_panel import ValidationAuditPanel
from calo_rpd_studio.gui.user_feedback import show_error
from calo_rpd_studio.gui.widgets.activity_center import ActivityCenter
from calo_rpd_studio.gui.widgets.context_pane import ContextPane
from calo_rpd_studio.gui.widgets.document_workspace import DocumentWorkspace
from calo_rpd_studio.gui.widgets.form_density import apply_compact_input_policy
from calo_rpd_studio.gui.widgets.global_status_bar import GlobalStatusBarWidget
from calo_rpd_studio.gui.widgets.ribbon_bar import RibbonBar
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage
from calo_rpd_studio.version import PRODUCT_VERSION

from .experiment_workspace_restorer import ExperimentWorkspaceRestorer
from .project_manager import ProjectManager
from .session_recovery import SessionRecoveryJournal
from .workflow_manager import WorkflowManager
from .workspaces import (
    WORKSPACE_KEYS,
    WORKSPACE_LAYOUT_ID,
    WORKSPACE_SCHEMA_VERSION,
    WORKSPACE_SPECS,
    migrate_workspace_ui,
    workspace_index_for_key,
    workspace_key_for_index,
)

_LOG = logging.getLogger(__name__)
_LAYOUT_VERSION = 3
_DEFAULT_CONTEXT_WIDTH = 340
_DEFAULT_ACTIVITY_HEIGHT = 135


class MainWindow(QMainWindow):
    def __init__(self, state, experiment_manager, settings_manager, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.experiment_manager = experiment_manager
        self.settings_manager = settings_manager
        self.workflow = WorkflowManager(state)
        self._close_when_paused = False
        self._training_exclusive_active = False
        self.training_model_library = TrainingModelLibrary(self.settings_manager)

        self.setWindowTitle("CALO-RPD Studio")
        self.setWindowIcon(application_icon())
        self.resize(1500, 920)
        self.setMinimumSize(1120, 720)

        self.sidebar = NavigationSidebar(WORKSPACE_SPECS, settings_manager)
        self.stack = QStackedWidget()
        self.stack.setObjectName("WorkspaceStack")
        self.pages_by_key = {
            "dashboard": DashboardPanel(state),
            "calo_intelligence": CALOIntelligencePanel(
                state, experiment_manager, self.training_model_library
            ),
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
        self.interface_density = apply_compact_input_policy(
            self.stack,
            str(self.settings_manager.value("interface_density", "comfortable")),
        )
        self.restorer = ExperimentWorkspaceRestorer(self.state, self.workflow, self.pages_by_key)
        self.session_recovery = SessionRecoveryJournal()
        self._previous_unclean_session = self.session_recovery.previous_unclean()
        self.session_recovery.begin(
            workspace_ui={
                "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
                "workspace_layout_id": WORKSPACE_LAYOUT_ID,
                "workspace_key": "dashboard",
                "workspace_index": 0,
            }
        )

        self.documents = DocumentWorkspace(self.stack)
        self.setCentralWidget(self.documents)

        self.command_registry = CommandRegistry(self)
        self.ribbon = RibbonBar(self.command_registry)
        self.setMenuWidget(self.ribbon)

        self.training_launch_model = TrainingLaunchModel(self)
        self.training_center = IndependentTrainingPanel(
            self.state, self.training_launch_model, self
        )
        self.training_center.training_completed.connect(
            self.training_model_library.record_training_output
        )
        self.context_pane = ContextPane(
            self.state,
            self.sidebar,
            self.training_launch_model,
            self.training_center,
            self.training_model_library,
            self,
        )
        self.context_dock = QDockWidget("Inputs", self)
        self.context_dock.setObjectName("Phase6ContextDock")
        self.context_dock.setAccessibleName("Contextual input pane")
        self.context_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.context_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.context_dock.toggleViewAction().setEnabled(False)
        self.context_dock.setWidget(self.context_pane)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.context_dock)

        self.activity_center = ActivityCenter(self.state, self)
        self.activity_dock = QDockWidget("Activity", self)
        self.activity_dock.setObjectName("Phase6ActivityDock")
        self.activity_dock.setAccessibleName("Jobs logs warnings device and provenance")
        self.activity_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.activity_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.activity_dock.setWidget(self.activity_center)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.activity_dock)

        self.sidebar.page_requested.connect(self._set_workspace)
        self.context_pane.workspace_requested.connect(self._context_workspace_requested)
        self.context_pane.status_message.connect(self._show_status_message)
        self.command_registry.command_triggered.connect(self._execute_command)
        self.command_registry.command_selected.connect(self._command_selected)
        self.training_center.activity_message.connect(self.activity_center.append_external)
        self.pages_by_key["calo_intelligence"].independent_training_requested.connect(
            self.command_registry.action("policies.training").trigger
        )
        self._create_compatibility_actions()
        self._install_region_shortcut()
        self._create_global_status_bar()
        self._connect_workflow()
        self._refresh_workflow()
        self._restore_shell_layout()
        self.context_dock.show()
        if not self._shell_layout_restored:
            QTimer.singleShot(0, self._apply_default_dock_sizes)
        QTimer.singleShot(150, self._initial_system_scan)
        QTimer.singleShot(350, self._check_unfinished_work)

    def _connect_workflow(self) -> None:
        self.state.case_changed.connect(lambda _: self.workflow.invalidate_from("power_system"))
        self.pages_by_key["power_system"].stage_completed.connect(
            lambda: self.workflow.mark_completed("power_system")
        )
        self.pages_by_key["orpd"].stage_completed.connect(
            lambda: self.workflow.mark_completed("orpd")
        )
        self.pages_by_key["algorithms"].stage_completed.connect(
            lambda: self.workflow.mark_completed("algorithms")
        )
        self.pages_by_key["portfolio"].stage_completed.connect(
            lambda: self.workflow.mark_completed("portfolio")
        )
        self.pages_by_key["calo_intelligence"].stage_completed.connect(self._governing_policy_event)
        self.pages_by_key["scenarios"].stage_completed.connect(
            lambda: self.workflow.mark_completed("scenarios")
        )
        self.pages_by_key["dashboard"].workspace_requested.connect(self._set_workspace)
        self.pages_by_key["experiment"].workspace_requested.connect(self._set_workspace)
        self.pages_by_key["settings"].density_changed.connect(self._apply_interface_density)
        self.experiment_manager.started.connect(lambda _: self.workflow.mark_experiment_started())
        self.experiment_manager.completed.connect(
            lambda _: self.workflow.mark_experiment_completed()
        )
        self.experiment_manager.cancelled.connect(lambda _: self.workflow.mark_experiment_stopped())
        self.experiment_manager.failed.connect(lambda _: self.workflow.mark_experiment_stopped())
        self.experiment_manager.completed.connect(lambda _: self._finish_deferred_close())
        self.experiment_manager.cancelled.connect(lambda _: self._finish_deferred_close())
        self.experiment_manager.failed.connect(lambda _: self._finish_deferred_close())
        self.pages_by_key["statistics"].analysis_completed.connect(
            self.workflow.mark_statistics_completed
        )
        self.pages_by_key["results"].review_completed.connect(self.workflow.mark_results_reviewed)
        self.pages_by_key["results"].validation_requested.connect(
            self._open_reviewed_run_for_validation
        )
        self.pages_by_key["results"].experiment_restore_requested.connect(
            self.restore_experiment_workspace
        )
        self.pages_by_key["resume_center"].workspace_requested.connect(self._set_workspace)
        self.pages_by_key["resume_center"].experiment_restore_requested.connect(
            self.restore_experiment_workspace
        )
        self.pages_by_key["resume_center"].policy_training_requested.connect(
            self._prepare_independent_training_resume
        )
        self.pages_by_key["resume_center"].validation_resumed.connect(
            lambda task_id: self.pages_by_key["validation"].resume_task_by_id(task_id)
        )
        self.pages_by_key["resume_center"].portfolio_export_resumed.connect(
            lambda task_id: self.pages_by_key["publication"].resume_task_by_id(task_id)
        )
        self.state.runs_changed.connect(self._refresh_verified_count)
        self.state.policy_state_changed.connect(
            lambda _status: self.workflow.notify_governing_policy_changed()
        )
        self.state.compute_profile_changed.connect(lambda _profile: self._refresh_workflow())
        self.state.policy_training_changed.connect(self._on_policy_training_changed)
        self.workflow.changed.connect(self._refresh_workflow)
        self.workflow.changed.connect(self._persist_workspace_state)

    def _governing_policy_event(self) -> None:
        self.state.notify_policy_state_changed()

    def _apply_interface_density(self, density: str) -> None:
        self.interface_density = apply_compact_input_policy(self.stack, density)

    def _initial_system_scan(self) -> None:
        try:
            self.state.refresh_compute_profile()
            dashboard = self.pages_by_key["dashboard"]
            if hasattr(dashboard, "refresh_compute"):
                dashboard.refresh_compute()
        except Exception:
            _LOG.exception("Initial compute-topology scan failed")
            self.state.task_status.fail("System readiness scan stopped")
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
        state = str(snapshot.get("state", "Ready"))
        title = str(snapshot.get("title", "") or snapshot.get("detail", ""))
        self.ribbon.set_summary(f"{state} · {title}" if title else state)
        self.command_registry.set_available(
            "experiment.stop",
            bool(snapshot.get("busy")) and bool(snapshot.get("cancellable")),
            "The current task does not expose safe cancellation.",
        )
        if not snapshot.get("busy") and snapshot.get("state") in {
            "Completed",
            "Failed",
            "Cancelled",
        }:
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

    def _refresh_verified_count(self) -> None:
        experiment_id = self.state.current_experiment_id or None
        count = (
            len(self.state.database.list_runs(experiment_id, verified_only=True))
            if experiment_id
            else 0
        )
        self.workflow.set_verified_results(count)

    def _refresh_workflow(self) -> None:
        for index, key in enumerate(WORKSPACE_KEYS):
            state, reason = self.workflow.workspace_state_key(key)
            self.sidebar.set_workflow_state(index, state, reason)
        for spec in self.command_registry.specs:
            if spec.handler != "workspace" or not spec.workspace:
                continue
            enabled = self.workflow.is_workspace_enabled(spec.workspace)
            _, reason = self.workflow.workspace_state_key(spec.workspace)
            self.command_registry.set_available(spec.command_id, enabled, reason)
        training_active = bool(getattr(self.state, "policy_training_active", False))
        self.command_registry.set_available(
            "project.open",
            not training_active,
            "Configuration is locked while training owns runtime state.",
        )
        self.command_registry.set_available(
            "project.save",
            not training_active,
            "Configuration is locked while training owns runtime state.",
        )
        self.command_registry.set_available("policies.training", True)
        self.command_registry.set_available("policies.resume", True)
        task = self.state.task_status.snapshot()
        self.command_registry.set_available(
            "experiment.stop",
            bool(task.get("busy")) and bool(task.get("cancellable")),
            "The current task does not expose safe cancellation.",
        )
        self.activity_center.refresh_context()
        self.global_status.apply_context(self.state)

        descriptor = self.workflow.next_descriptor()
        if descriptor is None:
            dashboard = self.pages_by_key.get("dashboard")
            if dashboard is not None and hasattr(dashboard, "set_next_action"):
                dashboard.set_next_action(
                    "Review retained evidence",
                    "The configured workflow has no pending required step.",
                    "results",
                    True,
                )
            return
        dashboard = self.pages_by_key.get("dashboard")
        if dashboard is not None and hasattr(dashboard, "set_next_action"):
            dashboard.set_next_action(
                descriptor.title,
                descriptor.instruction,
                descriptor.workspace_key,
                self.workflow.is_workspace_enabled(descriptor.workspace_key),
            )

    def _workspace_key(self, workspace: str | int) -> str:
        return workspace_key_for_index(workspace) if isinstance(workspace, int) else str(workspace)

    def _set_workspace(self, workspace: str | int) -> None:
        self._persist_workspace_state()
        key = self._workspace_key(workspace)
        if bool(getattr(self.state, "policy_training_active", False)) and key not in {
            "dashboard",
            "calo_intelligence",
        }:
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
        self.settings_manager.set_value("phase6/last_workspace_key", key)
        self.documents.focus_scientific_workspace()
        related = next(
            (
                spec
                for spec in self.command_registry.specs
                if spec.handler == "workspace" and spec.workspace == key
            ),
            None,
        )
        if related is not None:
            self.command_registry.select(related.command_id)

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
                    message
                    + f"\n\nRestore experiment {experiment_id!r} using the saved scientific workspace?",
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
                    policy_training_active=bool(
                        getattr(self.state, "policy_training_active", False)
                    ),
                    governing_policy_sha256=str(getattr(status, "policy_sha256", "") or ""),
                    compute_profile_fingerprint=str(
                        getattr(profile, "topology_fingerprint", "") or ""
                    ),
                )
            except Exception:
                _LOG.exception("Failed to update application-session recovery journal")
        except Exception:
            _LOG.exception("Failed to persist workspace state for experiment %s", experiment_id)
            self.state.task_status.fail("Workspace state could not be saved")

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
                target_key = (
                    "live_optimization"
                    if self.workflow.is_workspace_enabled("live_optimization")
                    else "experiment"
                )
            self._set_workspace(target_key)
            self.state.task_status.finish(
                f"Restored experiment workspace · {restored['runs']} stored run(s) · {restored['campaign_status']}"
            )
        except Exception as exc:
            show_error(
                self,
                "Experiment could not be restored",
                "The saved experiment workspace could not be opened.",
                exc,
                source="experiment restoration",
            )

    def _create_compatibility_actions(self) -> None:
        """Retain established public action attributes while the registry owns the actions."""
        self.open_config_action = self.command_registry.action("project.open")
        self.save_config_action = self.command_registry.action("project.save")

    def _install_region_shortcut(self) -> None:
        self._focus_regions = (
            self.ribbon.tabs,
            self.context_pane.tabs,
            self.documents,
            self.activity_center,
        )
        self._focus_region_index = -1
        self.region_shortcut = QShortcut(QKeySequence("F6"), self)
        self.region_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.region_shortcut.activated.connect(self._focus_next_region)

    def _focus_next_region(self) -> None:
        self._focus_region_index = (self._focus_region_index + 1) % len(self._focus_regions)
        target = self._focus_regions[self._focus_region_index]
        if target is self.context_pane.tabs:
            self.context_dock.show()
        elif target is self.activity_center:
            self.activity_dock.show()
        target.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _command_selected(self, spec: CommandSpec) -> None:
        self.context_pane.show_command(spec)
        self.context_dock.show()

    def _execute_command(self, command_id: str) -> None:
        spec = self.command_registry.spec(command_id)
        handlers = {
            "open": self.open_config,
            "save": self.save_config,
            "find": self._find_workspace,
            "cancel": self.state.task_status.cancel,
            "training": self._open_training_center,
            "toggle_activity": lambda: self.activity_dock.setVisible(
                not self.activity_dock.isVisible()
            ),
            "toggle_theme": self._toggle_theme,
            "reset_layout": self.reset_shell_layout,
            "guide": self._open_user_guide,
            "about": self.about,
        }
        if spec.handler == "workspace":
            self._set_workspace(spec.workspace)
            return
        handler = handlers.get(spec.handler)
        if handler is not None:
            handler()

    def _context_workspace_requested(self, target: str) -> None:
        if target == "independent-training":
            self._open_training_center()
        elif target:
            self._set_workspace(target)

    def _show_status_message(self, message: str) -> None:
        self.statusBar().showMessage(str(message), 5000)

    def _find_workspace(self) -> None:
        self.context_dock.show()
        self.ribbon.select_category("Workspace")
        self.ribbon.tabs.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _open_training_center(self) -> None:
        self.context_dock.show()
        self.context_pane.show_command(self.command_registry.spec("policies.training"))
        self.context_pane.training.refresh()

    def _prepare_independent_training_resume(self, record: dict) -> None:
        """Open and prefill independent resume inputs without starting policy work."""

        self.context_dock.show()
        self.context_pane.show_command(self.command_registry.spec("policies.training"))
        try:
            self.context_pane.prepare_training_resume(record)
        except Exception as exc:
            show_error(
                self,
                "Training resume could not be prepared",
                "This record is not compatible with independent policy training.",
                exc,
                source="independent training resume",
            )

    def _open_user_guide(self) -> None:
        guide = getattr(self, "_guide_document", None)
        if guide is None:
            guide = QTextBrowser()
            guide.setObjectName("NativeAndContainerGuide")
            guide.setAccessibleName("CALO-RPD operating guide")
            guide.setOpenExternalLinks(True)
            candidates = (
                Path(__file__).resolve().parents[2] / "docs" / "NATIVE_WINDOWS_GUIDE.md",
                Path(__file__).resolve().parents[2] / "README.md",
            )
            source = next((item for item in candidates if item.is_file()), None)
            if source is None:
                guide.setMarkdown(
                    "# CALO-RPD Studio operation\n\n"
                    "This installed build uses `calo-rpd-native` for direct routine desktop "
                    "launch. The direct entry does not install dependencies or perform policy "
                    "work. Use the **Compute** ribbon to distinguish configured CUDA-preferred "
                    "or CPU-only intent from the actual assigned device; Safe-80 values are "
                    "admission ceilings, not measured use.\n\n"
                    "Repository checkouts also provide `Launch-CALO-RPD.ps1` and the complete "
                    "native/Docker guide under `docs/NATIVE_WINDOWS_GUIDE.md`."
                )
            else:
                guide.setMarkdown(source.read_text(encoding="utf-8"))
            self._guide_document = guide
        self.documents.open_document(
            "operating-guide",
            "Operating guide",
            guide,
            tooltip="Native Windows and Docker guidance",
        )

    def _toggle_theme(self) -> None:
        target = "dark" if str(self.state.theme) != "dark" else "light"
        self.settings_manager.set_value("appearance", target)
        self.state.set_theme(target)

    def reset_shell_layout(self) -> None:
        self.ribbon.set_compact(False)
        self.context_dock.setFloating(False)
        self.activity_dock.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.context_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.activity_dock)
        self.context_dock.show()
        self.activity_dock.show()
        self.resizeDocks((self.context_dock,), (_DEFAULT_CONTEXT_WIDTH,), Qt.Orientation.Horizontal)
        self.resizeDocks(
            (self.activity_dock,), (_DEFAULT_ACTIVITY_HEIGHT,), Qt.Orientation.Vertical
        )
        self.documents.focus_scientific_workspace()
        self.settings_manager.set_value("phase6/layout_version", _LAYOUT_VERSION)
        self.settings_manager.set_value("phase6/main_window_state", QByteArray())
        self._show_status_message("Application layout restored to the default")

    def _apply_default_dock_sizes(self) -> None:
        self.resizeDocks((self.context_dock,), (_DEFAULT_CONTEXT_WIDTH,), Qt.Orientation.Horizontal)
        self.resizeDocks(
            (self.activity_dock,), (_DEFAULT_ACTIVITY_HEIGHT,), Qt.Orientation.Vertical
        )

    def _restore_shell_layout(self) -> None:
        self.ribbon.set_compact(False, emit=False)
        try:
            version = int(self.settings_manager.value("phase6/layout_version", 0) or 0)
        except (TypeError, ValueError):
            version = 0
        state = self.settings_manager.value("phase6/main_window_state", QByteArray())
        restored = False
        if version == _LAYOUT_VERSION and isinstance(state, QByteArray) and not state.isEmpty():
            restored = bool(self.restoreState(state, _LAYOUT_VERSION))
        self._shell_layout_restored = restored
        if not restored:
            self.context_dock.show()
            self.activity_dock.show()
            self.resizeDocks(
                (self.context_dock,), (_DEFAULT_CONTEXT_WIDTH,), Qt.Orientation.Horizontal
            )
            self.resizeDocks(
                (self.activity_dock,), (_DEFAULT_ACTIVITY_HEIGHT,), Qt.Orientation.Vertical
            )
        self.context_dock.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.context_dock)
        self.context_dock.show()
        last_workspace = str(
            self.settings_manager.value("phase6/last_workspace_key", "dashboard") or "dashboard"
        )
        if last_workspace in WORKSPACE_KEYS and self.workflow.is_workspace_enabled(last_workspace):
            self._set_workspace(last_workspace)

    def _save_shell_layout(self) -> None:
        self.ribbon.set_compact(False, emit=False)
        self.context_dock.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.context_dock)
        self.context_dock.show()
        self.settings_manager.set_value("phase6/layout_version", _LAYOUT_VERSION)
        self.settings_manager.set_value("phase6/main_window_state", self.saveState(_LAYOUT_VERSION))

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
            show_error(
                self,
                "Configuration could not be opened",
                "The selected configuration is invalid or unavailable.",
                exc,
                source="configuration open",
            )

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
            show_error(
                self,
                "Configuration could not be saved",
                "Choose a writable location and try again.",
                exc,
                source="configuration save",
            )

    def about(self) -> None:
        QMessageBox.information(
            self,
            "About CALO-RPD Studio",
            f"CALO-RPD Studio {PRODUCT_VERSION}\n"
            "Cognitive Adaptive Learning Optimizer for Robust Reactive Power Dispatch\n\n"
            "Scientific optimization with optional policy guidance, available-memory protection, reproducible benchmarking, validation, statistics, and publication export.",
        )

    def _finish_deferred_close(self) -> None:
        if self._close_when_paused and not self.experiment_manager.running:
            self._close_when_paused = False
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._persist_workspace_state()
        if self.training_center.process is not None:
            QMessageBox.information(
                self,
                "Independent policy process is active",
                "An independent readiness or training process is still active. Keep CALO-RPD "
                "Studio open until it finishes; closing will not silently terminate it or accept "
                "an incomplete result.",
            )
            event.ignore()
            return
        if bool(getattr(self.state, "policy_training_active", False)):
            QMessageBox.information(
                self,
                "Independent policy training is active",
                "Keep CALO-RPD Studio open until the independent training process finishes. "
                "Closing will not silently stop or accept an incomplete training result.",
            )
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
        self._save_shell_layout()
        self.activity_center.detach_logging()
        event.accept()
