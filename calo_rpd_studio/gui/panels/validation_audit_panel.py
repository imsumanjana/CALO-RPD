"""Independent solution validation and provenance audit."""
from __future__ import annotations

import json

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from calo_rpd_studio.benchmarking.validation import (
    select_runs_for_validation,
    validate_runs,
)
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage
from calo_rpd_studio.results.integrity_checker import check_run_record
from calo_rpd_studio.results.solution_validator import validate_stored_run
from calo_rpd_studio.resume.models import ResumeStatus, ResumeTaskType


class _BulkValidationWorker(QThread):
    progress = pyqtSignal(object)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, database, rows: list[dict], parent=None) -> None:
        super().__init__(parent)
        self.database = database
        self.rows = list(rows)

    def run(self) -> None:
        try:
            summary = validate_runs(
                self.database,
                self.rows,
                progress_callback=lambda payload: self.progress.emit(payload),
                cancel_callback=self.isInterruptionRequested,
            )
            self.completed.emit(summary)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ValidationAuditPanel(WorkspacePage):
    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Validation & Audit",
            "Reload saved decisions, independently rerun the physical model, recompute objectives and constraints, and record result integrity individually or in bulk.",
            parent,
        )
        self.state = state
        self._bulk_worker: _BulkValidationWorker | None = None
        self._bulk_resume_task_id = ""

        controls = QHBoxLayout()
        self.experiment = QComboBox()
        self.run = QComboBox()
        refresh = QPushButton("Refresh")
        validate = QPushButton("Validate selected run")
        validate.setObjectName("PrimaryButton")
        refresh.clicked.connect(self.refresh_experiments)
        validate.clicked.connect(self.validate)
        self.experiment.currentIndexChanged.connect(self.refresh_runs)
        controls.addWidget(QLabel("Experiment"))
        controls.addWidget(self.experiment, 1)
        controls.addWidget(QLabel("Run"))
        controls.addWidget(self.run, 1)
        controls.addWidget(refresh)
        controls.addWidget(validate)
        self.layout_root.addLayout(controls)

        bulk_box = QGroupBox("Bulk independent validation")
        bulk_layout = QVBoxLayout(bulk_box)
        bulk_actions = QHBoxLayout()
        self.bulk_current_button = QPushButton("Validate current experiment")
        self.bulk_current_button.setObjectName("PrimaryButton")
        self.bulk_all_button = QPushButton("Validate all not-yet-verified runs")
        self.bulk_cancel_button = QPushButton("Cancel bulk validation")
        self.bulk_cancel_button.setEnabled(False)
        self.bulk_resume_button = QPushButton("Resume bulk validation")
        self.bulk_current_button.clicked.connect(self.validate_current_experiment)
        self.bulk_all_button.clicked.connect(self.validate_all_unverified)
        self.bulk_cancel_button.clicked.connect(self.cancel_bulk_validation)
        self.bulk_resume_button.clicked.connect(self.resume_bulk_validation)
        bulk_actions.addWidget(self.bulk_current_button)
        bulk_actions.addWidget(self.bulk_all_button)
        bulk_actions.addWidget(self.bulk_cancel_button)
        bulk_actions.addWidget(self.bulk_resume_button)
        bulk_actions.addStretch(1)
        bulk_layout.addLayout(bulk_actions)

        self.bulk_status = QLabel(
            "Bulk validation skips runs already marked verified. Failed or unverified runs can be checked again."
        )
        self.bulk_status.setWordWrap(True)
        self.bulk_status.setObjectName("InfoText")
        bulk_layout.addWidget(self.bulk_status)
        self.bulk_progress = QProgressBar()
        self.bulk_progress.setRange(0, 100)
        self.bulk_progress.setValue(0)
        self.bulk_progress.setFormat("Ready")
        bulk_layout.addWidget(self.bulk_progress)
        self.layout_root.addWidget(bulk_box)

        box = QGroupBox("Audit record")
        layout = QVBoxLayout(box)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)
        self.layout_root.addWidget(box, 1)

        state.runs_changed.connect(self.refresh_experiments)
        self.refresh_experiments()

    def refresh_experiments(self) -> None:
        current = self.experiment.currentData()
        self.experiment.blockSignals(True)
        self.experiment.clear()
        for experiment in self.state.database.list_experiments():
            self.experiment.addItem(
                f"{experiment['created_at']} — {experiment['name']}",
                experiment["id"],
            )
        index = self.experiment.findData(current)
        self.experiment.setCurrentIndex(max(index, 0))
        self.experiment.blockSignals(False)
        self.refresh_runs()

    def refresh_runs(self) -> None:
        current = self.run.currentData()
        self.run.blockSignals(True)
        self.run.clear()
        experiment_id = self.experiment.currentData()
        if experiment_id:
            for row in self.state.database.list_runs(experiment_id):
                self.run.addItem(
                    f"Run {row['run_index'] + 1} — {row['algorithm']} — {row['validation_status']}",
                    row["id"],
                )
        index = self.run.findData(current)
        self.run.setCurrentIndex(max(index, 0))
        self.run.blockSignals(False)

    def select_run(self, experiment_id: str, run_id: str) -> None:
        """Select a reviewed run so validation opens on the intended record."""
        self.refresh_experiments()
        experiment_index = self.experiment.findData(experiment_id)
        if experiment_index >= 0:
            self.experiment.setCurrentIndex(experiment_index)
        self.refresh_runs()
        run_index = self.run.findData(run_id)
        if run_index >= 0:
            self.run.setCurrentIndex(run_index)

    def validate(self) -> None:
        run_id = self.run.currentData()
        if not run_id or self._bulk_worker is not None:
            return
        task = self.state.task_status
        if not task.begin("Validating stored solution", detail="Independent physical-model re-evaluation"):
            return
        QApplication.processEvents()
        try:
            record = self.state.database.get_run(run_id)
            basic = check_run_record(record)
            result = validate_stored_run(self.state.database, run_id)
            self.output.setPlainText(
                json.dumps(
                    {
                        "record_integrity": basic,
                        "independent_validation": result,
                    },
                    indent=2,
                    allow_nan=True,
                )
            )
            self.state.runs_changed.emit()
            if result.get("passed"):
                task.finish("Independent result validation passed")
            else:
                task.fail("Independent result validation failed")
        except Exception as exc:
            task.fail(str(exc))
            QMessageBox.critical(self, "Validation failed", str(exc))

    def validate_current_experiment(self) -> None:
        experiment_id = self.experiment.currentData()
        if not experiment_id:
            return
        rows = select_runs_for_validation(
            self.state.database,
            experiment_id=experiment_id,
            only_unverified=True,
        )
        self._start_bulk_validation(rows, scope="current experiment")

    def validate_all_unverified(self) -> None:
        rows = select_runs_for_validation(
            self.state.database,
            only_unverified=True,
        )
        self._start_bulk_validation(rows, scope="all experiments")

    def _start_bulk_validation(self, rows: list[dict], *, scope: str) -> None:
        if self._bulk_worker is not None:
            return
        if not rows:
            self.bulk_status.setText(f"No not-yet-verified runs were found in {scope}.")
            self.bulk_progress.setValue(100)
            self.bulk_progress.setFormat("Nothing to validate")
            return
        task = self.state.task_status
        if not task.begin(
            "Bulk independent validation",
            detail=f"0/{len(rows)} runs completed · {scope}",
            progress=0,
        ):
            return

        self.bulk_current_button.setEnabled(False)
        self.bulk_all_button.setEnabled(False)
        self.bulk_cancel_button.setEnabled(True)
        self.bulk_progress.setValue(0)
        self.bulk_progress.setFormat(f"0/{len(rows)} · 0%")
        self.bulk_status.setText(
            f"Independently validating {len(rows)} saved runs from {scope}. Each run is reconstructed and physically re-evaluated."
        )

        state_payload = {"scope": scope, "run_ids": [str(row["id"]) for row in rows]}
        if not self._bulk_resume_task_id:
            self._bulk_resume_task_id = self.state.resume_service.register(
                ResumeTaskType.VALIDATION,
                f"Bulk validation — {scope}",
                state_payload,
                total=len(rows),
                status=ResumeStatus.RUNNING,
            )
        else:
            self.state.resume_service.update(
                self._bulk_resume_task_id,
                status=ResumeStatus.RUNNING,
                current=0,
                total=len(rows),
                state=state_payload,
                resumable=True,
            )
        worker = _BulkValidationWorker(self.state.database, rows, self)
        self._bulk_worker = worker
        worker.progress.connect(self._bulk_progressed)
        worker.completed.connect(self._bulk_completed)
        worker.failed.connect(self._bulk_failed)
        worker.finished.connect(self._bulk_worker_finished)
        worker.start()

    def _bulk_progressed(self, payload: dict) -> None:
        completed = int(payload.get("completed", 0))
        total = int(payload.get("total", 0))
        percent = int(payload.get("percent", 0))
        passed = int(payload.get("passed", 0))
        failed = int(payload.get("failed", 0))
        errors = int(payload.get("errors", 0))
        algorithm = str(payload.get("algorithm", ""))
        run_index = int(payload.get("run_index", -1)) + 1
        self.bulk_progress.setValue(percent)
        self.bulk_progress.setFormat(f"{completed}/{total} · {percent}%")
        self.bulk_status.setText(
            f"Checking {algorithm} run {run_index} · passed {passed} · failed {failed} · errors {errors}"
        )
        self.state.task_status.update(
            percent,
            f"{completed}/{total} runs · passed {passed} · failed {failed} · errors {errors}",
        )
        if self._bulk_resume_task_id:
            self.state.resume_service.update(
                self._bulk_resume_task_id,
                status=ResumeStatus.RUNNING,
                current=completed,
                total=total,
            )

    def _bulk_completed(self, summary: dict) -> None:
        self.output.setPlainText(json.dumps({"bulk_validation": summary}, indent=2, allow_nan=True))
        self.state.runs_changed.emit()
        if summary.get("cancelled"):
            self.bulk_status.setText(
                f"Bulk validation cancelled after {summary.get('validated', 0)} completed checks."
            )
            self.bulk_progress.setFormat("Cancelled")
            self.state.task_status.cancelled("Bulk validation cancelled")
            if self._bulk_resume_task_id:
                self.state.resume_service.update(self._bulk_resume_task_id, status=ResumeStatus.PAUSED, resumable=True)
        else:
            passed = int(summary.get("passed", 0))
            failed = int(summary.get("failed", 0))
            errors = int(summary.get("errors", 0))
            self.bulk_status.setText(
                f"Bulk validation complete · passed {passed} · failed {failed} · errors {errors}."
            )
            self.bulk_progress.setValue(100)
            self.bulk_progress.setFormat("Completed")
            if failed or errors:
                self.state.task_status.finish(
                    f"Bulk validation completed with {failed} failed validations and {errors} processing errors"
                )
            else:
                self.state.task_status.finish(f"Bulk validation passed for {passed} runs")
            if self._bulk_resume_task_id:
                self.state.resume_service.update(self._bulk_resume_task_id, status=ResumeStatus.COMPLETED, current=100, total=100, resumable=False)

    def _bulk_failed(self, message: str) -> None:
        self.bulk_status.setText(f"Bulk validation stopped: {message}")
        self.bulk_progress.setFormat("Failed")
        self.state.task_status.fail(message)
        if self._bulk_resume_task_id:
            self.state.resume_service.update(self._bulk_resume_task_id, status=ResumeStatus.INTERRUPTED, resumable=True)
        QMessageBox.critical(self, "Bulk validation failed", message)

    def resume_bulk_validation(self) -> None:
        items = [item for item in self.state.resume_service.unfinished() if item.task_type == ResumeTaskType.VALIDATION.value]
        if not items:
            QMessageBox.information(self, "Bulk validation resume", "No resumable bulk-validation task was found.")
            return
        item = items[0]
        rows = []
        for run_id in item.state.get("run_ids", []):
            row = self.state.database.get_run(str(run_id))
            if row is not None and row.get("validation_status") != "verified":
                rows.append(row)
        if not rows:
            self.state.resume_service.update(item.id, status=ResumeStatus.COMPLETED, resumable=False)
            QMessageBox.information(self, "Bulk validation resume", "All runs in the saved validation queue are already verified.")
            return
        self._bulk_resume_task_id = item.id
        self._start_bulk_validation(rows, scope=str(item.state.get("scope", "saved validation queue")))

    def cancel_bulk_validation(self) -> None:
        if self._bulk_worker is None:
            return
        self._bulk_worker.requestInterruption()
        self.bulk_cancel_button.setEnabled(False)
        self.bulk_status.setText(
            "Cancellation requested. The current physical re-evaluation will finish, then the remaining runs will stay queued for later validation."
        )

    def _bulk_worker_finished(self) -> None:
        worker = self._bulk_worker
        self._bulk_worker = None
        self.bulk_current_button.setEnabled(True)
        self.bulk_all_button.setEnabled(True)
        self.bulk_cancel_button.setEnabled(False)
        if worker is not None:
            worker.deleteLater()
