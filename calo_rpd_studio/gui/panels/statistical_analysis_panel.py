"""Descriptive statistics and editable comparison figures."""
from __future__ import annotations

import json

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
)

from calo_rpd_studio.gui.plotting.scientific_plot import ScientificPlotWidget
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage
from calo_rpd_studio.statistics.descriptive import descriptive_statistics


class StatisticalAnalysisPanel(WorkspacePage):
    analysis_completed = pyqtSignal()

    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Statistical Analysis",
            "Compute repeated-run descriptive statistics and inspect publication-formatted distribution and convergence figures.",
            parent,
        )
        self.state = state

        controls = QHBoxLayout()
        self.experiment = QComboBox()
        refresh = QPushButton("Refresh experiments")
        analyze = QPushButton("Analyze selected experiment")
        analyze.setObjectName("PrimaryButton")
        refresh.clicked.connect(self.refresh_experiments)
        analyze.clicked.connect(self.analyze)
        controls.addWidget(QLabel("Experiment"))
        controls.addWidget(self.experiment, 1)
        controls.addWidget(refresh)
        controls.addWidget(analyze)
        self.layout_root.addLayout(controls)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            [
                "Algorithm",
                "Count",
                "Best",
                "Mean",
                "Median",
                "Worst",
                "Std",
                "IQR",
                "CV",
                "CI low",
                "CI high",
            ]
        )
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.layout_root.addWidget(self.table, 1)

        self.tabs = QTabWidget()
        self.boxplot = ScientificPlotWidget(
            title="Objective distribution",
            xlabel="Algorithm",
            ylabel="Objective",
        )
        self.convergence = ScientificPlotWidget(
            title="Median best-feasible convergence",
            xlabel="Objective-function evaluations",
            ylabel="Best feasible objective",
        )
        self.tabs.addTab(self.boxplot, "Distribution")
        self.tabs.addTab(self.convergence, "Convergence")
        self.layout_root.addWidget(self.tabs, 2)

        state.runs_changed.connect(self.refresh_experiments)
        self.refresh_experiments()

    def refresh_experiments(self) -> None:
        current = self.experiment.currentData()
        self.experiment.clear()
        for experiment in self.state.database.list_experiments():
            self.experiment.addItem(
                f"{experiment['created_at']} — {experiment['name']}",
                experiment["id"],
            )
        index = self.experiment.findData(current)
        self.experiment.setCurrentIndex(max(index, 0))

    def select_experiment(self, experiment_id: str) -> None:
        self.refresh_experiments()
        index = self.experiment.findData(str(experiment_id))
        if index >= 0:
            self.experiment.setCurrentIndex(index)
        if hasattr(self, "refresh"):
            self.refresh()

    def analyze(self) -> None:
        experiment_id = self.experiment.currentData()
        if not experiment_id:
            return
        task = self.state.task_status
        if not task.begin("Computing statistical analysis", detail="Loading repeated-run results"):
            return
        QApplication.processEvents()
        try:
            rows = self.state.database.list_runs(experiment_id)
            if not rows:
                raise ValueError("The selected experiment contains no completed optimization runs.")
            groups: dict[str, list[float]] = {}
            convergence: dict[str, list[tuple[list[int], list[float]]]] = {}
            for row in rows:
                data = json.loads(row["result_json"])
                groups.setdefault(row["algorithm"], []).append(float(data["best_objective"]))
                metadata = data.get("metadata", {})
                evaluations = metadata.get("convergence_evaluations", [])
                feasible_history = metadata.get("best_feasible_objective_history", [])
                if evaluations and feasible_history and len(evaluations) == len(feasible_history):
                    convergence.setdefault(row["algorithm"], []).append(
                        ([int(x) for x in evaluations], [float(y) for y in feasible_history])
                    )

            self.table.setRowCount(len(groups))
            for row_index, (algorithm, values) in enumerate(sorted(groups.items())):
                stats = descriptive_statistics(values)
                data = [
                    algorithm,
                    stats.get("count"),
                    stats.get("best"),
                    stats.get("mean"),
                    stats.get("median"),
                    stats.get("worst"),
                    stats.get("std"),
                    stats.get("iqr"),
                    stats.get("coefficient_of_variation"),
                    stats.get("confidence_low"),
                    stats.get("confidence_high"),
                ]
                for column, value in enumerate(data):
                    self.table.setItem(row_index, column, QTableWidgetItem(str(value)))

            self.boxplot.axis.clear()
            self.boxplot.axis.boxplot(
                list(groups.values()),
                tick_labels=list(groups.keys()),
            )
            metadata = self.boxplot.manager.records[self.boxplot.plot_id].metadata
            metadata.update(
                {
                    "title": "Objective distribution",
                    "xlabel": "Algorithm",
                    "ylabel": "Objective",
                }
            )
            self.boxplot.manager.apply(self.boxplot.plot_id, self.boxplot.style)

            median_series: dict[str, tuple[list[int], list[float]]] = {}
            for algorithm, runs in convergence.items():
                if not runs:
                    continue
                max_evaluation = max((max(xs) for xs, _ in runs if xs), default=0)
                if max_evaluation <= 0:
                    continue
                # Use a common evaluation grid and carry the most recent recorded feasible best
                # forward. NaNs before first feasibility are excluded from the cross-run median.
                grid = np.unique(
                    np.concatenate([np.asarray(xs, dtype=int) for xs, _ in runs if xs])
                )
                aligned = []
                for xs, ys in runs:
                    x_arr = np.asarray(xs, dtype=int)
                    y_arr = np.asarray(ys, dtype=float)
                    values = np.full(grid.shape, np.nan, dtype=float)
                    for gi, gx in enumerate(grid):
                        pos = np.searchsorted(x_arr, gx, side="right") - 1
                        if pos >= 0:
                            values[gi] = y_arr[pos]
                    aligned.append(values)
                matrix = np.vstack(aligned)
                valid_columns = np.any(np.isfinite(matrix), axis=0)
                if not np.any(valid_columns):
                    continue
                local_grid = grid[valid_columns]
                local_matrix = matrix[:, valid_columns]
                median = np.nanmedian(local_matrix, axis=0)
                valid = np.isfinite(median)
                if np.any(valid):
                    median_series[algorithm] = (local_grid[valid].tolist(), median[valid].tolist())
            self.convergence.plot_xy_series(
                median_series,
                "Median best-feasible convergence",
                "Objective-function evaluations",
                "Best feasible objective",
            )
            task.finish(f"Statistical analysis completed for {len(groups)} algorithm group(s)")
            self.analysis_completed.emit()
        except Exception as exc:
            task.fail(str(exc))
            QMessageBox.critical(self, "Statistical analysis failed", str(exc))
