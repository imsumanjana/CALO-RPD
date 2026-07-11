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
            title="Mean convergence",
            xlabel="Recorded iteration",
            ylabel="Best objective",
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
            convergence: dict[str, list[list[float]]] = {}
            for row in rows:
                data = json.loads(row["result_json"])
                groups.setdefault(row["algorithm"], []).append(float(data["best_objective"]))
                convergence.setdefault(row["algorithm"], []).append(data["convergence_history"])

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

            mean_series: dict[str, list[float]] = {}
            for algorithm, runs in convergence.items():
                if not runs:
                    continue
                length = min(len(history) for history in runs)
                mean_series[algorithm] = np.mean(
                    [history[:length] for history in runs],
                    axis=0,
                ).tolist()
            self.convergence.plot_series(
                mean_series,
                "Mean convergence",
                "Recorded iteration",
                "Best objective",
            )
            task.finish(f"Statistical analysis completed for {len(groups)} algorithm group(s)")
            self.analysis_completed.emit()
        except Exception as exc:
            task.fail(str(exc))
            QMessageBox.critical(self, "Statistical analysis failed", str(exc))
