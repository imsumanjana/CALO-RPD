"""Sequential durable orchestration of immutable Workspace study cells."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from calo_rpd_studio.experiments.execution_plans import ExecutionLifecycle, ExecutionPlanKind


class WorkspaceCampaignCoordinator(QObject):
    """Feed one immutable Workspace cell at a time to the shared ExperimentManager."""

    changed = pyqtSignal(object)
    finished = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    paused = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, state, experiment_manager, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.manager = experiment_manager
        self.plan_id = ""
        self.current_cell_id = ""
        self._remaining_cells: list[dict] = []
        self._had_failures = False
        self._terminal_cancel_requested = False
        self._advance_when_idle = False
        experiment_manager.started.connect(self._on_experiment_started)
        experiment_manager.completed.connect(self._on_experiment_completed)
        experiment_manager.paused.connect(self._on_experiment_paused)
        experiment_manager.cancelled.connect(self._on_experiment_cancelled)
        experiment_manager.failed.connect(self._on_experiment_failed)
        experiment_manager.idle.connect(self._on_manager_idle)

    @property
    def active(self) -> bool:
        return bool(self.plan_id)

    def run(self, plan_id: str, *, resume: bool = False) -> None:
        if self.active or self.manager.running:
            raise RuntimeError("Another Workspace cell or experiment is already running")
        plan = self.state.database.get_execution_plan(str(plan_id))
        if plan is None or str(plan["plan_kind"]) != ExecutionPlanKind.WORKSPACE.value:
            raise RuntimeError("Workspace execution requires an exact Workspace study plan")
        if resume:
            self.state.execution_control.resume(str(plan_id), ExecutionPlanKind.WORKSPACE)
        else:
            self.state.execution_control.begin_run(str(plan_id))
        self.plan_id = str(plan_id)
        self._terminal_cancel_requested = False
        self._had_failures = False
        self._remaining_cells = [
            row
            for row in self.state.database.list_workspace_plan_cells(self.plan_id)
            if str(row["lifecycle_state"])
            in {"planned", "queued", "paused", "interrupted_resumable", "failed"}
        ]
        self.state.notify_execution_state_changed()
        try:
            self._start_next_cell()
        except Exception as exc:
            if self.active:
                current = self.state.database.get_execution_plan(self.plan_id)
                if current is not None and str(current["lifecycle_state"]) == "running":
                    self.state.execution_control.transition(
                        self.plan_id,
                        expected=(ExecutionLifecycle.RUNNING.value,),
                        new_state=ExecutionLifecycle.INTERRUPTED_RESUMABLE.value,
                        message=f"Workspace cell admission was interrupted: {type(exc).__name__}: {exc}",
                    )
                self._clear()
                self.state.notify_execution_state_changed()
            raise

    def _start_next_cell(self) -> None:
        if not self.plan_id:
            return
        if not self._remaining_cells:
            lifecycle = (
                ExecutionLifecycle.COMPLETED_WITH_FAILURES
                if self._had_failures
                else ExecutionLifecycle.COMPLETED
            )
            plan_id = self.plan_id
            self.state.execution_control.commit_terminal(
                plan_id,
                lifecycle=lifecycle,
                message=(
                    "Workspace campaign completed with retained failed cells"
                    if self._had_failures
                    else "Workspace campaign completed"
                ),
            )
            self._clear()
            self.state.notify_execution_state_changed()
            self.finished.emit(plan_id)
            return
        cell = self._remaining_cells.pop(0)
        self.current_cell_id = str(cell["id"])
        config = self.state.execution_control.plan_configuration(
            self.plan_id, cell_id=self.current_cell_id
        )
        root = Path(config.output_directory)
        config.output_directory = str(
            root / "workspace_campaigns" / self.plan_id / self.current_cell_id
        )
        self.state.database.update_workspace_plan_cell(
            self.current_cell_id,
            lifecycle_state="running",
            campaign_id=str(cell.get("campaign_id", "") or ""),
            experiment_id=str(cell.get("experiment_id", "") or ""),
            message="Submitted to the shared ExperimentManager",
        )
        prior_campaign = str(cell.get("campaign_id", "") or "")
        if prior_campaign:
            self.state.execution_control.verify_campaign_binding(
                self.plan_id, prior_campaign
            )
        started = (
            self.manager.resume_campaign(prior_campaign, update_workspace=False)
            if prior_campaign
            else self.manager.start_comparison(config)
        )
        if not started:
            self.state.database.update_workspace_plan_cell(
                self.current_cell_id,
                lifecycle_state="interrupted_resumable",
                campaign_id=prior_campaign,
                message="The shared ExperimentManager did not accept this cell",
            )
            plan_id = self.plan_id
            self.state.execution_control.transition(
                plan_id,
                expected=(ExecutionLifecycle.RUNNING.value,),
                new_state=ExecutionLifecycle.INTERRUPTED_RESUMABLE.value,
                message="The shared ExperimentManager rejected the Workspace cell",
            )
            self._clear()
            self.state.notify_execution_state_changed()
            raise RuntimeError("The shared ExperimentManager did not accept the Workspace cell")
        self.changed.emit(self.snapshot())

    def pause_safely(self) -> None:
        if not self.active or not self.manager.running:
            raise RuntimeError("No running Workspace campaign can be paused")
        self.state.execution_control.request_pause(self.plan_id)
        self.state.notify_execution_state_changed()
        self.manager.pause()

    def cancel_remaining(self) -> None:
        if not self.active:
            raise RuntimeError("No Workspace campaign is active")
        self._terminal_cancel_requested = True
        if self.manager.running:
            self.manager.cancel()
            return
        self._finish_cancelled()

    def _on_experiment_started(self, experiment_id: str) -> None:
        if not self.active or not self.current_cell_id:
            return
        worker = self.manager.worker
        campaign_id = str(getattr(worker, "campaign_id", "") or "")
        self.state.execution_control.transition(
            self.plan_id,
            expected=(ExecutionLifecycle.RUNNING.value,),
            new_state=ExecutionLifecycle.RUNNING.value,
            message="Workspace plan bound to the active cell campaign",
            campaign_id=campaign_id,
        )
        self.state.database.update_workspace_plan_cell(
            self.current_cell_id,
            lifecycle_state="running",
            campaign_id=campaign_id,
            experiment_id=str(experiment_id),
            message="Workspace cell numerical execution started",
        )
        self.changed.emit(self.snapshot())

    def _on_experiment_completed(self, experiment_id: str) -> None:
        if not self.active or not self.current_cell_id:
            return
        cells = {
            str(row["id"]): row
            for row in self.state.database.list_workspace_plan_cells(self.plan_id)
        }
        row = cells[self.current_cell_id]
        campaign_id = str(row.get("campaign_id", "") or "")
        failures = []
        if campaign_id:
            failures = [
                item
                for item in self.state.database.list_campaign_tasks(campaign_id)
                if str(item["status"]) == "failed"
            ]
        state = "completed_with_failures" if failures else "completed"
        self._had_failures = self._had_failures or bool(failures)
        self.state.database.update_workspace_plan_cell(
            self.current_cell_id,
            lifecycle_state=state,
            campaign_id=campaign_id,
            experiment_id=str(experiment_id),
            message=(
                f"Completed with {len(failures)} retained failed job(s)"
                if failures
                else "Completed"
            ),
        )
        self.current_cell_id = ""
        self._advance_when_idle = True

    def _on_manager_idle(self) -> None:
        if not self.active or not self._advance_when_idle:
            return
        self._advance_when_idle = False
        if self._terminal_cancel_requested:
            self._finish_cancelled()
        else:
            self._start_next_cell()

    def _on_experiment_paused(self, experiment_id: str) -> None:
        if not self.active or not self.current_cell_id:
            return
        worker = self.manager.worker
        campaign_id = str(getattr(worker, "campaign_id", "") or "")
        self.state.database.update_workspace_plan_cell(
            self.current_cell_id,
            lifecycle_state="paused",
            campaign_id=campaign_id,
            experiment_id=str(experiment_id),
            message="Paused safely; exact checkpoint or deterministic restart boundary retained",
        )
        plan_id = self.plan_id
        self.state.execution_control.commit_paused(plan_id, campaign_id=campaign_id)
        self._clear()
        self.state.notify_execution_state_changed()
        self.paused.emit(plan_id)

    def _on_experiment_cancelled(self, experiment_id: str) -> None:
        if not self.active:
            return
        if self.current_cell_id:
            self.state.database.update_workspace_plan_cell(
                self.current_cell_id,
                lifecycle_state="cancelled",
                experiment_id=str(experiment_id),
                message="Cancelled terminally; completed evidence retained",
            )
        self._finish_cancelled()

    def _finish_cancelled(self) -> None:
        plan_id = self.plan_id
        for row in self._remaining_cells:
            self.state.database.update_workspace_plan_cell(
                str(row["id"]),
                lifecycle_state="cancelled",
                message="Cancelled before numerical admission",
            )
        self.state.execution_control.commit_terminal(
            plan_id,
            lifecycle=ExecutionLifecycle.CANCELLED,
            message="Workspace remaining work cancelled terminally; completed evidence retained",
        )
        self._clear()
        self.state.notify_execution_state_changed()
        self.cancelled.emit(plan_id)

    def _on_experiment_failed(self, message: str) -> None:
        if not self.active:
            return
        plan_id = self.plan_id
        self.state.execution_control.transition(
            plan_id,
            expected=("running", "pausing"),
            new_state=ExecutionLifecycle.INTERRUPTED_RESUMABLE.value,
            message=f"Workspace cell interrupted: {message}",
        )
        if self.current_cell_id:
            self.state.database.update_workspace_plan_cell(
                self.current_cell_id,
                lifecycle_state="interrupted_resumable",
                message=str(message),
            )
        self._clear()
        self.state.notify_execution_state_changed()
        self.failed.emit(str(message))

    def snapshot(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "current_cell_id": self.current_cell_id,
            "remaining_cells": len(self._remaining_cells),
            "terminal_cancel_requested": self._terminal_cancel_requested,
        }

    def _clear(self) -> None:
        self.plan_id = ""
        self.current_cell_id = ""
        self._remaining_cells = []
        self._terminal_cancel_requested = False
        self._advance_when_idle = False
