"""Filterable raw result explorer."""
from __future__ import annotations

import json

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from calo_rpd_studio.gui.dialogs.experiment_history_dialog import ExperimentHistoryDialog
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


class ResultsExplorerPanel(WorkspacePage):
    """Inspect stored runs and explicitly hand the selected run to validation."""

    review_completed = pyqtSignal()
    validation_requested = pyqtSignal(str, str)
    experiment_restore_requested = pyqtSignal(str)

    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Results Explorer",
            "Filter experiments and inspect objective values, feasibility, constraint violations, runtime, controls, convergence, and final physical system state.",
            parent,
        )
        self.state = state
        self._rows: list[dict] = []
        self._selected_experiment_id = ""
        self._selected_run_id = ""

        filters = QHBoxLayout()
        self.experiment = QComboBox()
        self.algorithm = QComboBox()
        self.algorithm.addItem("All algorithms", "")
        self.validation = QComboBox()
        self.validation.addItem("All validation states", "")
        self.validation.addItems(["unverified", "verified", "failed"])
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        manage_history = QPushButton("Manage history")
        manage_history.clicked.connect(self._manage_history)
        restore_workspace = QPushButton("Open experiment workspace")
        restore_workspace.setToolTip("Restore this experiment's saved parameters, CALO intelligence, workflow access, and stored plots.")
        restore_workspace.clicked.connect(self._restore_selected_experiment)
        filters.addWidget(QLabel("Experiment"))
        filters.addWidget(self.experiment, 1)
        filters.addWidget(QLabel("Algorithm"))
        filters.addWidget(self.algorithm)
        filters.addWidget(QLabel("Validation"))
        filters.addWidget(self.validation)
        filters.addWidget(refresh)
        filters.addWidget(restore_workspace)
        filters.addWidget(manage_history)
        self.layout_root.addLayout(filters)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            [
                "Run ID",
                "Algorithm",
                "Run",
                "Seed",
                "Objective",
                "Feasible",
                "Violation",
                "Runtime (s)",
                "Validation",
            ]
        )
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.show_selected)
        self.table.cellClicked.connect(lambda *_: self.show_selected())
        self.layout_root.addWidget(self.table, 2)

        details_box = QGroupBox("Selected run details")
        details_layout = QVBoxLayout(details_box)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(170)
        details_layout.addWidget(self.details)
        self.layout_root.addWidget(details_box, 1)

        self.review_button = QPushButton("Confirm result review and continue to validation")
        self.review_button.setObjectName("PrimaryButton")
        self.review_button.setEnabled(False)
        self.review_button.clicked.connect(self._confirm_review)
        self.layout_root.addWidget(self.review_button)

        state.runs_changed.connect(self.refresh_experiments)
        self.experiment.currentIndexChanged.connect(self.refresh)
        self.algorithm.currentIndexChanged.connect(self.refresh)
        self.validation.currentIndexChanged.connect(self.refresh)
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
        self.refresh()

    def select_experiment(self, experiment_id: str) -> None:
        self.refresh_experiments()
        index = self.experiment.findData(str(experiment_id))
        if index >= 0:
            self.experiment.setCurrentIndex(index)
        if hasattr(self, "refresh"):
            self.refresh()

    def refresh(self) -> None:
        experiment_id = self.experiment.currentData()
        self._selected_experiment_id = ""
        self._selected_run_id = ""
        self.details.clear()
        self.review_button.setEnabled(False)
        if not experiment_id:
            self.table.setRowCount(0)
            self._rows = []
            return

        rows = self.state.database.list_runs(experiment_id)
        names = sorted({row["algorithm"] for row in rows})
        current = self.algorithm.currentData()
        self.algorithm.blockSignals(True)
        self.algorithm.clear()
        self.algorithm.addItem("All algorithms", "")
        for name in names:
            self.algorithm.addItem(name, name)
        index = self.algorithm.findData(current)
        self.algorithm.setCurrentIndex(max(index, 0))
        self.algorithm.blockSignals(False)

        algorithm_filter = self.algorithm.currentData()
        validation_filter = self.validation.currentText() if self.validation.currentIndex() > 0 else ""
        rows = [
            row
            for row in rows
            if (not algorithm_filter or row["algorithm"] == algorithm_filter)
            and (not validation_filter or row["validation_status"] == validation_filter)
        ]
        self._rows = rows
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            data = json.loads(row["result_json"])
            seed = json.loads(row["seed_json"])["algorithm_seed"]
            values = [
                row["id"],
                row["algorithm"],
                row["run_index"] + 1,
                seed,
                data["best_objective"],
                data["feasible"],
                data["total_constraint_violation"],
                data["runtime_seconds"],
                row["validation_status"],
            ]
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(str(value)))

        # A visible first row is an actionable selection. This avoids the previous state where a
        # cell appeared highlighted but selectedRows() returned nothing because selection behavior
        # was cell-based.
        if rows:
            self.table.selectRow(0)
            self.show_selected()

    def show_selected(self) -> None:
        row_index = self.table.currentRow()
        if row_index < 0 or row_index >= len(self._rows):
            selected = self.table.selectionModel().selectedRows()
            row_index = selected[0].row() if selected else -1
        if row_index < 0 or row_index >= len(self._rows):
            self._selected_experiment_id = ""
            self._selected_run_id = ""
            self.details.clear()
            self.review_button.setEnabled(False)
            return

        row = self._rows[row_index]
        data = json.loads(row["result_json"])
        self._selected_experiment_id = str(row["experiment_id"])
        self._selected_run_id = str(row["id"])
        self.review_button.setEnabled(True)
        self.details.setPlainText(
            json.dumps(
                {
                    "run_id": row["id"],
                    "algorithm": row["algorithm"],
                    "run": int(row["run_index"]) + 1,
                    "objective": data.get("best_objective"),
                    "feasible": data.get("feasible"),
                    "total_constraint_violation": data.get("total_constraint_violation"),
                    "decoded_controls": data.get("decoded_controls", {}),
                    "objective_components": data.get("objective_components", {}),
                    "termination_reason": data.get("termination_reason"),
                    "metadata": data.get("metadata", {}),
                },
                indent=2,
                allow_nan=True,
            )
        )



    def select_run(self, experiment_id: str, run_id: str) -> None:
        """Select an exact stored run, preserving compatibility with linked validation views."""
        self.refresh_experiments()
        index = self.experiment.findData(str(experiment_id))
        if index >= 0:
            self.experiment.setCurrentIndex(index)
            self.refresh()
        for row_index, row in enumerate(self._rows):
            if str(row.get("id")) == str(run_id):
                self.table.selectRow(row_index)
                self.show_selected()
                return
        raise KeyError(f"Run {run_id!r} is not available in experiment {experiment_id!r}")


    def _restore_selected_experiment(self) -> None:
        experiment_id = str(self.experiment.currentData() or "")
        if experiment_id:
            self.experiment_restore_requested.emit(experiment_id)

    def _manage_history(self) -> None:
        """Open the dedicated destructive-history manager and refresh this workspace afterwards."""
        dialog = ExperimentHistoryDialog(self.state, self)
        dialog.exec()
        self.refresh_experiments()

    def _confirm_review(self) -> None:
        """Complete review and immediately request validation of the exact selected run."""
        if not self._selected_experiment_id or not self._selected_run_id:
            self.show_selected()
        if not self._selected_experiment_id or not self._selected_run_id:
            QMessageBox.information(
                self,
                "Select a result",
                "Select one completed run before continuing to independent validation.",
            )
            return
        self.state.current_experiment_id = self._selected_experiment_id
        self.review_completed.emit()
        self.validation_requested.emit(self._selected_experiment_id, self._selected_run_id)
