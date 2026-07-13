"""Experiment history and persisted trace-data management dialog."""
from __future__ import annotations

import json

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _format_bytes(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TB"


class ExperimentHistoryDialog(QDialog):
    """Manage old experiments without mixing destructive actions into the main workflow."""

    def __init__(self, state, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Manage experiment history")
        self.resize(1000, 650)
        self.setMinimumSize(820, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QLabel("Experiment history and stored traces")
        title.setObjectName("PageTitle")
        root.addWidget(title)
        description = QLabel(
            "Delete completed runs or entire experiments together with their validation records, failure logs, "
            "and referenced compressed convergence/population array files. External publication exports are independent copies and are not deleted automatically."
        )
        description.setWordWrap(True)
        description.setObjectName("PageSubtitle")
        root.addWidget(description)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setObjectName("InfoText")
        root.addWidget(self.summary)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        experiment_container = QWidget()
        experiment_layout = QVBoxLayout(experiment_container)
        experiment_layout.setContentsMargins(0, 0, 0, 0)
        experiment_layout.setSpacing(6)
        experiment_layout.addWidget(QLabel("Stored experiments"))
        self.experiments = QTableWidget(0, 7)
        self.experiments.setHorizontalHeaderLabels(
            ["Created", "Name", "Runs", "Failed", "Verified", "Trace files", "Trace storage"]
        )
        self.experiments.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.experiments.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.experiments.verticalHeader().setVisible(False)
        self.experiments.setAlternatingRowColors(True)
        self.experiments.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.experiments.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in range(2, 7):
            self.experiments.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.experiments.itemSelectionChanged.connect(self._refresh_runs)
        experiment_layout.addWidget(self.experiments)
        splitter.addWidget(experiment_container)

        run_container = QWidget()
        run_layout = QVBoxLayout(run_container)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.setSpacing(6)
        run_layout.addWidget(QLabel("Completed runs in the selected experiment"))
        self.runs = QTableWidget(0, 6)
        self.runs.setHorizontalHeaderLabels(
            ["Algorithm", "Run", "Objective", "Feasible", "Validation", "Trace file"]
        )
        self.runs.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.runs.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.runs.verticalHeader().setVisible(False)
        self.runs.setAlternatingRowColors(True)
        self.runs.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 6):
            self.runs.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.runs.itemSelectionChanged.connect(self._update_buttons)
        run_layout.addWidget(self.runs)
        splitter.addWidget(run_container)
        splitter.setSizes([330, 220])
        root.addWidget(splitter, 1)

        actions = QHBoxLayout()
        self.delete_run_button = QPushButton("Delete selected run")
        self.delete_experiment_button = QPushButton("Delete selected experiment")
        self.clear_button = QPushButton("Delete all experiment history")
        self.delete_run_button.clicked.connect(self._delete_selected_run)
        self.delete_experiment_button.clicked.connect(self._delete_selected_experiment)
        self.clear_button.clicked.connect(self._clear_all)
        actions.addWidget(self.delete_run_button)
        actions.addWidget(self.delete_experiment_button)
        actions.addStretch(1)
        actions.addWidget(self.clear_button)
        root.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._experiment_rows: list[dict] = []
        self._run_rows: list[dict] = []
        self.state.task_status.changed.connect(lambda _snapshot: self._update_buttons())
        self.refresh()

    def selected_experiment_id(self) -> str:
        row = self.experiments.currentRow()
        if 0 <= row < len(self._experiment_rows):
            return str(self._experiment_rows[row]["id"])
        return ""

    def selected_run_id(self) -> str:
        row = self.runs.currentRow()
        if 0 <= row < len(self._run_rows):
            return str(self._run_rows[row]["id"])
        return ""

    def refresh(self, preferred_experiment_id: str = "") -> None:
        current = preferred_experiment_id or self.selected_experiment_id()
        self._experiment_rows = self.state.database.list_experiments()
        self.experiments.setRowCount(len(self._experiment_rows))
        selected_row = -1
        for row_index, experiment in enumerate(self._experiment_rows):
            summary = self.state.database.experiment_storage_summary(experiment["id"])
            values = [
                experiment["created_at"],
                experiment["name"],
                summary["runs"],
                summary["failures"],
                summary["verified_runs"],
                summary["trace_files"],
                _format_bytes(summary["trace_bytes"]),
            ]
            for column, value in enumerate(values):
                self.experiments.setItem(row_index, column, QTableWidgetItem(str(value)))
            if experiment["id"] == current:
                selected_row = row_index
        history = self.state.database.history_storage_summary()
        self.summary.setText(
            f"Stored history: {history['experiments']} experiment(s), {history['runs']} completed run(s), "
            f"{history['failures']} failed run record(s), {history['validations']} validation record(s), "
            f"{history['trace_files']} referenced trace file(s), {_format_bytes(history['trace_bytes'])} trace storage."
        )
        if self._experiment_rows:
            self.experiments.selectRow(selected_row if selected_row >= 0 else 0)
        else:
            self._run_rows = []
            self.runs.setRowCount(0)
        self._refresh_runs()
        self._update_buttons()

    def _refresh_runs(self) -> None:
        experiment_id = self.selected_experiment_id()
        self._run_rows = self.state.database.list_runs(experiment_id) if experiment_id else []
        self.runs.setRowCount(len(self._run_rows))
        for row_index, row in enumerate(self._run_rows):
            data = json.loads(row["result_json"])
            values = [
                row["algorithm"],
                int(row["run_index"]) + 1,
                data.get("best_objective"),
                data.get("feasible"),
                row["validation_status"],
                row["arrays_path"] or "—",
            ]
            for column, value in enumerate(values):
                self.runs.setItem(row_index, column, QTableWidgetItem(str(value)))
        if self._run_rows:
            self.runs.selectRow(0)
        self._update_buttons()

    def _update_buttons(self) -> None:
        busy = bool(self.state.task_status.busy)
        self.delete_run_button.setEnabled(bool(self.selected_run_id()) and not busy)
        self.delete_experiment_button.setEnabled(bool(self.selected_experiment_id()) and not busy)
        self.clear_button.setEnabled(bool(self._experiment_rows) and not busy)
        if busy:
            self.summary.setToolTip("History deletion is disabled while a scientific task is active.")
        else:
            self.summary.setToolTip("")

    def _notify_changed(self, deleted_experiment_id: str = "") -> None:
        if deleted_experiment_id and self.state.current_experiment_id == deleted_experiment_id:
            self.state.current_experiment_id = ""
        self.state.runs_changed.emit()

    def _delete_selected_run(self) -> None:
        run_id = self.selected_run_id()
        if not run_id:
            return
        row = self.state.database.get_run(run_id)
        if row is None:
            self.refresh()
            return
        experiment_id = str(row["experiment_id"])
        answer = QMessageBox.warning(
            self,
            "Delete stored run",
            "Delete the selected completed run?\n\n"
            f"Algorithm: {row['algorithm']}\n"
            f"Run: {int(row['run_index']) + 1}\n\n"
            "Its validation record and referenced compressed trace-array file will also be removed. This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        summary = self.state.database.delete_run(run_id)
        self._notify_changed()
        self.refresh(experiment_id)
        QMessageBox.information(
            self,
            "Run deleted",
            f"Deleted {summary['runs_deleted']} run, {summary['validations_deleted']} validation record(s), "
            f"and {summary['trace_files_deleted']} trace file(s). Reclaimed {_format_bytes(summary['trace_bytes_reclaimed'])}.",
        )

    def _delete_selected_experiment(self) -> None:
        experiment_id = self.selected_experiment_id()
        if not experiment_id:
            return
        summary = self.state.database.experiment_storage_summary(experiment_id)
        answer = QMessageBox.warning(
            self,
            "Delete experiment and traces",
            "Delete this experiment and all of its stored application history?\n\n"
            f"Name: {summary['name']}\n"
            f"Created: {summary['created_at']}\n"
            f"Completed runs: {summary['runs']}\n"
            f"Failed-run records: {summary['failures']}\n"
            f"Validation records: {summary['validations']}\n"
            f"Referenced trace storage: {_format_bytes(summary['trace_bytes'])}\n\n"
            "The database records and referenced compressed trace-array files will be removed. External publication exports are not modified. This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        result = self.state.database.delete_experiment(experiment_id)
        self._notify_changed(experiment_id)
        self.refresh()
        QMessageBox.information(
            self,
            "Experiment deleted",
            f"Deleted {result['runs_deleted']} completed run(s), {result['failures_deleted']} failure record(s), "
            f"{result['validations_deleted']} validation record(s), and {result['trace_files_deleted']} trace file(s). "
            f"Reclaimed {_format_bytes(result['trace_bytes_reclaimed'])}.",
        )

    def _clear_all(self) -> None:
        summary = self.state.database.history_storage_summary()
        text, accepted = QInputDialog.getText(
            self,
            "Delete all experiment history",
            "This will remove all stored experiment records and all referenced trace-array files.\n"
            f"Current history: {summary['experiments']} experiment(s), {summary['runs']} completed run(s), {_format_bytes(summary['trace_bytes'])} trace storage.\n\n"
            "Type DELETE ALL to continue:",
        )
        if not accepted or text.strip() != "DELETE ALL":
            return
        result = self.state.database.clear_history()
        self.state.current_experiment_id = ""
        self._notify_changed()
        self.refresh()
        QMessageBox.information(
            self,
            "Experiment history cleared",
            f"Deleted {result['experiments_deleted']} experiment(s), {result['runs_deleted']} completed run(s), "
            f"{result['failures_deleted']} failure record(s), {result['validations_deleted']} validation record(s), "
            f"and {result['trace_files_deleted']} trace file(s). Reclaimed {_format_bytes(result['trace_bytes_reclaimed'])}.",
        )
