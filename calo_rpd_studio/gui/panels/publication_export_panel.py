"""Publication and resumable article-portfolio export workspace."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTextEdit,
)

from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage
from calo_rpd_studio.portfolio.exporter import PortfolioExporter
from calo_rpd_studio.results.publication_export import PublicationExporter
from calo_rpd_studio.resume.models import ResumeStatus, ResumeTaskType


class _StandardExportWorker(QThread):
    progress = pyqtSignal(dict)
    completed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, database, experiment_id: str, directory: str) -> None:
        super().__init__()
        self.database = database
        self.experiment_id = str(experiment_id)
        self.directory = str(directory)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            path = PublicationExporter(self.database).export(
                self.experiment_id,
                self.directory,
                progress_callback=self.progress.emit,
                cancel_callback=lambda: self._cancel,
            )
            if self._cancel:
                self.cancelled.emit(str(path))
            else:
                self.completed.emit(str(path))
        except Exception as exc:
            if self._cancel:
                self.cancelled.emit(self.directory)
            else:
                self.failed.emit(f"{type(exc).__name__}: {exc}")


class _PortfolioExportWorker(QThread):
    progress = pyqtSignal(dict)
    completed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, database, experiment_id: str, directory: str) -> None:
        super().__init__()
        self.database = database
        self.experiment_id = str(experiment_id)
        self.directory = str(directory)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            path = PortfolioExporter(self.database).export(
                self.experiment_id,
                self.directory,
                progress_callback=self.progress.emit,
                cancel_callback=lambda: self._cancel,
            )
            if self._cancel:
                self.cancelled.emit(str(path))
            else:
                self.completed.emit(str(path))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class PublicationExportPanel(WorkspacePage):
    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Publication & Portfolio Export",
            "Generate verified numerical packages or resume an article-oriented portfolio containing only the figures, tables, traces, and evidence selected in Portfolio Manager.",
            parent,
        )
        self.state = state
        self.worker: _StandardExportWorker | _PortfolioExportWorker | None = None
        self.resume_task_id = ""

        card = SectionCard(
            "Export selection",
            "The standard package exports verified records. The portfolio engine follows the stored portfolio plan, skips unavailable evidence with an explicit reason, and resumes artifact-by-artifact.",
        )
        experiment_row = QHBoxLayout()
        self.experiment = QComboBox()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        experiment_row.addWidget(QLabel("Experiment"))
        experiment_row.addWidget(self.experiment, 1)
        experiment_row.addWidget(refresh)
        card.layout_root.addLayout(experiment_row)

        output_row = QHBoxLayout()
        self.directory = QLineEdit("publication_export")
        choose = QPushButton("Choose directory")
        choose.clicked.connect(self.choose)
        output_row.addWidget(self.directory, 1)
        output_row.addWidget(choose)
        card.layout_root.addLayout(output_row)

        actions = QHBoxLayout()
        standard = QPushButton("Export verified publication package")
        standard.clicked.connect(self.export_standard)
        portfolio = QPushButton("Generate selected portfolio")
        portfolio.setObjectName("PrimaryButton")
        portfolio.clicked.connect(lambda: self.export_portfolio(resume=False))
        resume = QPushButton("Resume portfolio generation")
        resume.clicked.connect(lambda: self.export_portfolio(resume=True))
        self.cancel_button = QPushButton("Pause export safely")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_export)
        for button in (standard, portfolio, resume, self.cancel_button):
            actions.addWidget(button)
        actions.addStretch(1)
        card.layout_root.addLayout(actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        card.layout_root.addWidget(self.progress)
        self.layout_root.addWidget(card)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setMinimumHeight(200)
        self.layout_root.addWidget(self.status, 1)

        state.runs_changed.connect(self.refresh)
        state.task_status.cancel_requested.connect(self.cancel_export)
        self.refresh()

    def refresh(self) -> None:
        current = self.experiment.currentData()
        self.experiment.clear()
        for experiment in self.state.database.list_experiments():
            self.experiment.addItem(
                f"{experiment['created_at']} — {experiment['name']}",
                experiment["id"],
            )
        index = self.experiment.findData(current)
        self.experiment.setCurrentIndex(max(index, 0))

    def choose(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select publication export directory",
            self.directory.text() or ".",
        )
        if path:
            self.directory.setText(path)

    def export_standard(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Publication export", "An export is already running.")
            return
        experiment_id = self.experiment.currentData()
        if not experiment_id:
            return
        task = self.state.task_status
        if not task.begin(
            "Exporting verified publication package",
            detail="Collecting independently verified records",
            progress=0,
            cancellable=True,
        ):
            return
        directory = self.directory.text().strip() or "publication_export"
        self.progress.setValue(0)
        self.cancel_button.setEnabled(True)
        self.status.append("Starting verified publication export in the background...")
        self.worker = _StandardExportWorker(self.state.database, str(experiment_id), directory)
        self.worker.progress.connect(self._standard_progress)
        self.worker.completed.connect(self._standard_completed)
        self.worker.cancelled.connect(self._standard_cancelled)
        self.worker.failed.connect(self._standard_failed)
        self.worker.finished.connect(self._portfolio_finished)
        self.worker.start()

    def _standard_progress(self, payload: dict) -> None:
        percent = int(payload.get("percent", 0))
        artifact = str(payload.get("artifact", "artifact"))
        status = str(payload.get("status", "working"))
        self.progress.setValue(percent)
        self.status.append(f"{percent}% · {artifact}: {status}")
        self.state.task_status.update(percent, f"Publication export: {artifact} ({status})")

    def _standard_completed(self, directory: str) -> None:
        experiment_id = self.experiment.currentData()
        count = len(self.state.database.list_runs(experiment_id, verified_only=True)) if experiment_id else 0
        self.progress.setValue(100)
        self.status.append(
            f"Standard publication export completed. Verified runs exported: {count}. Directory: {Path(directory).resolve()}"
        )
        self.state.task_status.finish(f"Publication package exported with {count} verified run(s)")

    def _standard_cancelled(self, directory: str) -> None:
        self.status.append(f"Publication export cancelled safely. Directory: {Path(directory).resolve()}")
        self.state.task_status.cancelled("Publication export cancelled safely")

    def _standard_failed(self, message: str) -> None:
        self.status.append(f"Publication export failed: {message}")
        self.state.task_status.fail(message)
        QMessageBox.critical(self, "Publication export failed", message)

    def _resume_record(self, experiment_id: str) -> tuple[str, dict] | tuple[str, None]:
        for item in self.state.resume_service.unfinished():
            if item.task_type == ResumeTaskType.PORTFOLIO_EXPORT.value and str(item.state.get("experiment_id", "")) == str(experiment_id):
                return item.id, item.state
        return "", None

    def export_portfolio(self, *, resume: bool) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Portfolio export", "A portfolio export is already running.")
            return
        experiment_id = self.experiment.currentData()
        if not experiment_id:
            return
        directory = self.directory.text().strip() or "publication_export"
        record_id, record_state = self._resume_record(str(experiment_id))
        if resume:
            if record_state is None:
                manifest = Path(directory) / "portfolio_manifest.json"
                if not manifest.is_file():
                    QMessageBox.information(self, "Resume portfolio", "No unfinished portfolio record or manifest was found for this experiment and directory.")
                    return
            elif record_state.get("directory"):
                directory = str(record_state["directory"])
                self.directory.setText(directory)
        elif record_state is not None:
            answer = QMessageBox.question(
                self,
                "Unfinished portfolio detected",
                "An unfinished portfolio exists for this experiment. Resume it instead of starting another export?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                directory = str(record_state.get("directory") or directory)
                self.directory.setText(directory)
                resume = True

        if not self.state.task_status.begin(
            "Generating article result portfolio",
            detail="Planning selected figures, tables, and reproducibility artifacts",
            progress=0,
            cancellable=True,
        ):
            return

        self.resume_task_id = record_id or f"portfolio-export-{experiment_id}"
        state_payload = {"experiment_id": str(experiment_id), "directory": directory}
        if record_id:
            self.state.resume_service.update(
                self.resume_task_id,
                status=ResumeStatus.RUNNING,
                state=state_payload,
                resumable=True,
            )
        else:
            self.state.resume_service.register(
                ResumeTaskType.PORTFOLIO_EXPORT,
                f"Portfolio export — {experiment_id}",
                state_payload,
                task_id=self.resume_task_id,
                status=ResumeStatus.RUNNING,
            )

        self.progress.setValue(0)
        self.cancel_button.setEnabled(True)
        self.worker = _PortfolioExportWorker(self.state.database, str(experiment_id), directory)
        self.worker.progress.connect(self._portfolio_progress)
        self.worker.completed.connect(self._portfolio_completed)
        self.worker.cancelled.connect(self._portfolio_cancelled)
        self.worker.failed.connect(self._portfolio_failed)
        self.worker.finished.connect(self._portfolio_finished)
        self.worker.start()

    def _portfolio_progress(self, payload: dict) -> None:
        percent = int(payload.get("percent", 0))
        current = int(payload.get("completed", 0))
        total = int(payload.get("total", 0))
        artifact = str(payload.get("artifact", "artifact"))
        status = str(payload.get("status", "working"))
        self.progress.setValue(percent)
        self.status.append(f"{current}/{total} · {artifact}: {status}")
        self.state.task_status.update(percent, f"Portfolio artifact {current}/{total}: {artifact} ({status})")
        self.state.resume_service.update(
            self.resume_task_id,
            current=current,
            total=total,
            status=ResumeStatus.RUNNING,
        )

    def cancel_export(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.state.task_status.update(detail="Safe pause requested; finishing the current artifact")
            self.state.resume_service.update(self.resume_task_id, status=ResumeStatus.PAUSING, resumable=True)
            self.worker.cancel()
            self.cancel_button.setEnabled(False)

    def _portfolio_completed(self, directory: str) -> None:
        self.progress.setValue(100)
        self.status.append(f"Portfolio completed. Directory: {Path(directory).resolve()}")
        self.state.resume_service.update(self.resume_task_id, status=ResumeStatus.COMPLETED, resumable=False)
        self.state.task_status.finish("Selected article portfolio generated")

    def _portfolio_cancelled(self, directory: str) -> None:
        self.status.append(f"Portfolio paused. Completed artifacts remain reusable. Directory: {Path(directory).resolve()}")
        self.state.resume_service.update(self.resume_task_id, status=ResumeStatus.PAUSED, resumable=True)
        self.state.task_status.cancelled("Portfolio paused safely; resume will generate only missing artifacts")

    def _portfolio_failed(self, message: str) -> None:
        self.status.append(f"Portfolio export failed: {message}")
        self.state.resume_service.update(self.resume_task_id, status=ResumeStatus.INTERRUPTED, resumable=True)
        self.state.task_status.fail(message)
        QMessageBox.critical(self, "Portfolio export failed", message)

    def _portfolio_finished(self) -> None:
        self.cancel_button.setEnabled(False)
        self.worker = None
