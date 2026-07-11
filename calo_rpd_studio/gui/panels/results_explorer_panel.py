"""Filterable raw result explorer."""
from __future__ import annotations

import json

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


class ResultsExplorerPanel(WorkspacePage):
    review_completed = pyqtSignal()

    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Results Explorer",
            "Filter experiments and inspect objective values, feasibility, constraint violations, runtime, controls, convergence, and final physical system state.",
            parent,
        )
        self.state = state
        self._rows: list[dict] = []

        filters = QHBoxLayout()
        self.experiment = QComboBox()
        self.algorithm = QComboBox()
        self.algorithm.addItem("All algorithms", "")
        self.validation = QComboBox()
        self.validation.addItem("All validation states", "")
        self.validation.addItems(["unverified", "verified", "failed"])
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        filters.addWidget(QLabel("Experiment"))
        filters.addWidget(self.experiment, 1)
        filters.addWidget(QLabel("Algorithm"))
        filters.addWidget(self.algorithm)
        filters.addWidget(QLabel("Validation"))
        filters.addWidget(self.validation)
        filters.addWidget(refresh)
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
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.show_selected)
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
        self.review_button.clicked.connect(self.review_completed.emit)
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

    def refresh(self) -> None:
        experiment_id = self.experiment.currentData()
        if not experiment_id:
            self.table.setRowCount(0)
            self._rows = []
            self.review_button.setEnabled(False)
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

    def show_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        row = self._rows[rows[0].row()]
        data = json.loads(row["result_json"])
        self.review_button.setEnabled(True)
        self.details.setPlainText(
            json.dumps(
                {
                    "run_id": row["id"],
                    "algorithm": row["algorithm"],
                    "decoded_controls": data.get("decoded_controls", {}),
                    "objective_components": data.get("objective_components", {}),
                    "termination_reason": data.get("termination_reason"),
                    "metadata": data.get("metadata", {}),
                },
                indent=2,
                allow_nan=True,
            )
        )
