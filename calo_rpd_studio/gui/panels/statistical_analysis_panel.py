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
        self.horizon = QComboBox()
        self.horizon_status = QLabel("Select an experiment/evidence horizon")
        self.horizon_status.setWordWrap(True)
        refresh = QPushButton("Refresh experiments")
        analyze = QPushButton("Analyze selected experiment")
        analyze.setObjectName("PrimaryButton")
        refresh.clicked.connect(self.refresh_experiments)
        analyze.clicked.connect(self.analyze)
        controls.addWidget(QLabel("Experiment"))
        controls.addWidget(self.experiment, 1)
        controls.addWidget(QLabel("Evidence horizon"))
        controls.addWidget(self.horizon)
        controls.addWidget(refresh)
        controls.addWidget(analyze)
        self.layout_root.addLayout(controls)
        self.layout_root.addWidget(self.horizon_status)
        self.experiment.currentIndexChanged.connect(self.refresh_horizons)
        self.horizon.currentIndexChanged.connect(self._update_horizon_status)

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
        self.refresh_horizons()

    def refresh_horizons(self, *_args) -> None:
        experiment_id = self.experiment.currentData()
        current = self.horizon.currentData()
        self.horizon.blockSignals(True)
        self.horizon.clear()
        if experiment_id:
            for value in self.state.database.list_experiment_horizons(str(experiment_id)):
                self.horizon.addItem(f"{int(value):,} FE", int(value))
        index = self.horizon.findData(current)
        self.horizon.setCurrentIndex(index if index >= 0 else max(self.horizon.count() - 1, 0))
        self.horizon.blockSignals(False)
        self._update_horizon_status()

    def _update_horizon_status(self, *_args) -> None:
        experiment_id = self.experiment.currentData()
        horizon = self.horizon.currentData()
        if not experiment_id or horizon is None:
            self.horizon_status.setText("No stored evidence horizon is available.")
            return
        status = self.state.database.experiment_horizon_status(str(experiment_id), int(horizon))
        revision = status.get("revision") or {}
        eligibility = (
            "primary-stat eligible"
            if status.get("publication_eligible")
            else "exploratory/unclassified"
        )
        completeness = (
            "complete"
            if status.get("complete")
            else f"provisional {status.get('available_count', 0)}/{status.get('expected_count', 0)}"
        )
        self.horizon_status.setText(
            f"{int(horizon):,} FE · {completeness} · {eligibility}"
            + (
                f" · revision {revision.get('revision_number')} ({revision.get('extension_mode')})"
                if revision
                else ""
            )
            + ". Statistics below use only this exact FE horizon; horizons are never mixed."
        )

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
            horizon = self.horizon.currentData()
            if horizon is None:
                raise ValueError("Select a stored evidence horizon before statistical analysis.")
            status = self.state.database.experiment_horizon_status(str(experiment_id), int(horizon))
            rows = list(status.get("rows", []))
            if not rows:
                raise ValueError(
                    "The selected experiment contains no completed runs at this FE horizon."
                )
            if status.get("publication_eligible") and not status.get("complete"):
                raise ValueError(
                    f"The publication-eligible {int(horizon):,}-FE revision is incomplete "
                    f"({status.get('available_count', 0)}/{status.get('expected_count', 0)} paired runs). "
                    "Complete/resume the revision before primary statistical analysis."
                )
            groups: dict[str, list[float]] = {}
            convergence: dict[str, list[tuple[list[int], list[float]]]] = {}
            for row in rows:
                data = json.loads(row["result_json"])
                objective = float(data.get("best_objective", np.nan))
                if bool(data.get("feasible")) and np.isfinite(objective):
                    groups.setdefault(row["algorithm"], []).append(objective)
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
            if groups:
                self.boxplot.axis.boxplot(
                    list(groups.values()),
                    tick_labels=list(groups.keys()),
                )
            else:
                self.boxplot.axis.text(
                    0.5,
                    0.5,
                    "No feasible completed runs at this horizon",
                    ha="center",
                    va="center",
                    transform=self.boxplot.axis.transAxes,
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
            task.finish(
                f"Statistical analysis completed at {int(horizon):,} FE for {len(groups)} feasible algorithm group(s)"
            )
            self.analysis_completed.emit()
        except Exception as exc:
            task.fail(str(exc))
            QMessageBox.critical(self, "Statistical analysis failed", str(exc))
