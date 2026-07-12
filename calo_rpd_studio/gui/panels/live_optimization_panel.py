"""Live optimization telemetry and scientifically interpretable square convergence preview."""
from __future__ import annotations

import json
import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from calo_rpd_studio.gui.plotting.scientific_plot import ScientificPlotWidget
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


class LiveOptimizationPanel(WorkspacePage):
    """Live optimization workspace with a square, evaluation-based convergence plot.

    A raw objective value can rise while a feasibility-first optimizer replaces an infeasible
    incumbent by a lower-violation point. Such a trace must not be labelled as ordinary
    best-objective convergence. The live view therefore separates:

    * best feasible objective (monotonic once feasibility is reached), and
    * best normalized constraint violation (monotonic non-increasing).

    The x-axis is objective-function evaluations rather than iteration count, which is the fair
    axis when optimizers use different numbers of evaluations per iteration.
    """

    AUTO_MODE = "Automatic (recommended)"
    OBJECTIVE_MODE = "Best feasible objective"
    VIOLATION_MODE = "Best constraint violation"

    def __init__(self, state, manager, parent=None) -> None:
        super().__init__(
            "Live Optimization",
            "Monitor objective quality, feasibility, evaluation count, CALO cognitive telemetry, and scientifically comparable convergence without blocking the interface.",
            parent,
        )
        self.state = state
        self.manager = manager
        self.current_run_index: int | None = None
        self.objective_series: dict[str, tuple[list[int], list[float]]] = {}
        self.violation_series: dict[str, tuple[list[int], list[float]]] = {}

        scroll = QScrollArea()
        scroll.setObjectName("LiveOptimizationScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        content.setObjectName("LiveOptimizationContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 8)
        content_layout.setSpacing(16)

        telemetry = SectionCard("Current telemetry")
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)
        telemetry.layout_root.addLayout(grid)
        self.labels: dict[str, QLabel] = {}
        names = [
            "Algorithm",
            "Repeated run",
            "Iteration",
            "Evaluations",
            "Best feasible objective",
            "Best constraint violation",
            "Feasible incumbent",
            "CALO operator",
            "Population diversity",
            "Feasible population ratio",
            "Reward",
        ]
        for index, name in enumerate(names):
            row = index % 6
            col = (index // 6) * 2
            key = QLabel(name)
            key.setObjectName("MetricLabel")
            value = QLabel("—")
            value.setObjectName("ContextValue")
            self.labels[name] = value
            grid.addWidget(key, row, col)
            grid.addWidget(value, row, col + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        content_layout.addWidget(telemetry)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Convergence metric"))
        self.metric = QComboBox()
        self.metric.addItems([self.AUTO_MODE, self.OBJECTIVE_MODE, self.VIOLATION_MODE])
        self.metric.currentTextChanged.connect(self._redraw_plot)
        mode_row.addWidget(self.metric)
        self.metric_note = QLabel(
            "Automatic mode shows constraint-violation convergence until the monitored algorithms have reached feasibility, then switches to best-feasible objective."
        )
        self.metric_note.setWordWrap(True)
        self.metric_note.setObjectName("HelpText")
        mode_row.addWidget(self.metric_note, 1)
        content_layout.addLayout(mode_row)

        self.plot = ScientificPlotWidget(
            title="Live convergence",
            xlabel="Objective-function evaluations",
            ylabel=self.OBJECTIVE_MODE,
            square_preview=True,
            square_export=True,
            square_preview_size=720,
        )
        content_layout.addWidget(self.plot, 0, Qt.AlignmentFlag.AlignHCenter)
        content_layout.addStretch(1)

        scroll.setWidget(content)
        self.layout_root.addWidget(scroll, 1)

        manager.progress.connect(self.update_progress)
        manager.started.connect(lambda _: self.reset())
        manager.completed.connect(self.load_experiment)
        manager.cancelled.connect(self.load_experiment)

        self._show_waiting_message()

    def reset(self) -> None:
        self.current_run_index = None
        self.objective_series = {}
        self.violation_series = {}
        if hasattr(self, "metric"):
            self.metric.blockSignals(True)
            self.metric.setCurrentText(self.AUTO_MODE)
            self.metric.blockSignals(False)
        self._show_waiting_message()

    def _show_waiting_message(self) -> None:
        self.plot.show_message(
            "Waiting for optimizer telemetry. Start or resume an experiment to populate live convergence.",
            title="Live convergence",
            xlabel="Objective-function evaluations",
            ylabel="Convergence metric",
        )

    @staticmethod
    def _append_point(store: dict[str, tuple[list[int], list[float]]], label: str, x: int, y: float) -> None:
        xs, ys = store.setdefault(label, ([], []))
        if xs and x < xs[-1]:
            # A new run should be handled by the caller; never draw a backwards evaluation axis.
            return
        if xs and x == xs[-1]:
            ys[-1] = y
        else:
            xs.append(x)
            ys.append(y)

    @staticmethod
    def _has_points(series: dict[str, tuple[list[int], list[float]]]) -> bool:
        return any(bool(xs) and bool(ys) for xs, ys in series.values())

    def _automatic_mode(self) -> str:
        seen = {label for label, (xs, ys) in self.violation_series.items() if xs and ys}
        feasible = {label for label, (xs, ys) in self.objective_series.items() if xs and ys}
        # Use objective convergence only when every optimizer currently represented in the
        # monitored repeated run has produced at least one feasible incumbent. Until then, the
        # constraint-violation trace is the only scientifically comparable non-empty metric.
        if seen and seen.issubset(feasible):
            return self.OBJECTIVE_MODE
        return self.VIOLATION_MODE

    def _redraw_plot(self) -> None:
        requested = self.metric.currentText() if hasattr(self, "metric") else self.AUTO_MODE
        mode = self._automatic_mode() if requested == self.AUTO_MODE else requested
        run_suffix = "" if self.current_run_index is None else f" — repeated run {self.current_run_index}"

        if mode == self.VIOLATION_MODE:
            if self._has_points(self.violation_series):
                self.plot.plot_xy_series(
                    self.violation_series,
                    f"Constraint-violation convergence{run_suffix}",
                    "Objective-function evaluations",
                    "Best normalized constraint violation",
                )
                self.metric_note.setText(
                    "Showing normalized constraint-violation convergence because feasibility has not yet been reached by every monitored optimizer."
                    if requested == self.AUTO_MODE
                    else "Constraint violation is the correct convergence metric before a feasible incumbent exists."
                )
            else:
                self.plot.show_message(
                    "No convergence telemetry has been received yet.",
                    title=f"Constraint-violation convergence{run_suffix}",
                    xlabel="Objective-function evaluations",
                    ylabel="Best normalized constraint violation",
                )
        else:
            if self._has_points(self.objective_series):
                self.plot.plot_xy_series(
                    self.objective_series,
                    f"Best-feasible objective convergence{run_suffix}",
                    "Objective-function evaluations",
                    "Best feasible objective",
                )
                self.metric_note.setText(
                    "Showing monotonic best-feasible objective convergence against objective-function evaluations."
                )
            else:
                self.plot.show_message(
                    "No feasible incumbent has been reached yet. Use Automatic mode or Best constraint violation to monitor progress.",
                    title=f"Best-feasible objective convergence{run_suffix}",
                    xlabel="Objective-function evaluations",
                    ylabel="Best feasible objective",
                )

    def load_experiment(self, experiment_id: str) -> None:
        """Restore convergence histories after an experiment finishes or is cancelled.

        Live telemetry is transient by nature, so the completed run metadata is reloaded from
        SQLite. This also makes the page useful when the user opens Live Optimization after the
        numerical run has already completed.
        """
        if not experiment_id:
            return
        rows = self.state.database.list_runs(experiment_id)
        if not rows:
            return
        latest_index = max(int(row.get("run_index", 0)) for row in rows)
        latest_rows = [row for row in rows if int(row.get("run_index", 0)) == latest_index]
        objective_series: dict[str, tuple[list[int], list[float]]] = {}
        violation_series: dict[str, tuple[list[int], list[float]]] = {}
        for row in latest_rows:
            try:
                result = json.loads(row["result_json"])
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            metadata = result.get("metadata") or {}
            evaluations = [int(v) for v in metadata.get("convergence_evaluations", [])]
            feasible_history = metadata.get("best_feasible_objective_history", [])
            violation_history = metadata.get("best_constraint_violation_history", [])
            label = str(row.get("algorithm") or result.get("algorithm") or "Optimizer")

            fx, fy = [], []
            for x, y in zip(evaluations, feasible_history):
                try:
                    value = float(y)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    fx.append(int(x)); fy.append(value)
            vx, vy = [], []
            for x, y in zip(evaluations, violation_history):
                try:
                    value = float(y)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    vx.append(int(x)); vy.append(value)
            if fx:
                objective_series[label] = (fx, fy)
            if vx:
                violation_series[label] = (vx, vy)

        self.current_run_index = latest_index + 1
        self.objective_series = objective_series
        self.violation_series = violation_series
        self._redraw_plot()

    def update_progress(self, data: dict) -> None:
        if data.get("phase") in {"run_completed", "run_failed"}:
            return

        algorithm = str(data.get("algorithm", "—"))
        run_index = int(data.get("run_index", 1))
        evaluations = int(data.get("evaluations", 0))

        # Show one repeated-run comparison at a time. This prevents unrelated repeated runs from
        # being concatenated into one false convergence curve.
        if self.current_run_index is None:
            self.current_run_index = run_index
        elif run_index != self.current_run_index:
            self.current_run_index = run_index
            self.objective_series = {}
            self.violation_series = {}

        self.labels["Algorithm"].setText(algorithm)
        self.labels["Repeated run"].setText(str(run_index))
        self.labels["Iteration"].setText(str(data.get("iteration", "—")))
        self.labels["Evaluations"].setText(str(evaluations if evaluations else "—"))

        feasible_best = data.get("best_feasible_objective")
        if isinstance(feasible_best, (int, float)) and math.isfinite(float(feasible_best)):
            feasible_best = float(feasible_best)
            self.labels["Best feasible objective"].setText(f"{feasible_best:.10g}")
            self._append_point(self.objective_series, algorithm, evaluations, feasible_best)
        else:
            self.labels["Best feasible objective"].setText("Not reached")

        violation = data.get("best_constraint_violation")
        if isinstance(violation, (int, float)) and math.isfinite(float(violation)):
            violation = float(violation)
            self.labels["Best constraint violation"].setText(f"{violation:.10g}")
            self._append_point(self.violation_series, algorithm, evaluations, violation)
        else:
            self.labels["Best constraint violation"].setText("—")

        self.labels["Feasible incumbent"].setText(str(data.get("feasible", "—")))
        self.labels["CALO operator"].setText(str(data.get("calo_operator", "—")))
        self.labels["Population diversity"].setText(
            f"{data['diversity']:.5g}" if "diversity" in data else "—"
        )
        self.labels["Feasible population ratio"].setText(
            f"{data['feasible_ratio']:.5g}" if "feasible_ratio" in data else "—"
        )
        self.labels["Reward"].setText(
            f"{data['reward']:.5g}" if "reward" in data else "—"
        )
        self._redraw_plot()
