"""Live optimization telemetry and square convergence preview."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from calo_rpd_studio.gui.plotting.scientific_plot import ScientificPlotWidget
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


class LiveOptimizationPanel(WorkspacePage):
    """Live optimization workspace with a square publication-oriented plot.

    The telemetry and plot are placed inside a vertical scroll area so the 1:1 preview
    remains physically square instead of being compressed on shorter displays.
    """

    def __init__(self, state, manager, parent=None) -> None:
        super().__init__(
            "Live Optimization",
            "Monitor real objective values, feasibility, evaluation count, CALO cognitive telemetry, and convergence without blocking the interface.",
            parent,
        )
        self.state = state
        self.manager = manager
        self.series: dict[str, list[float]] = {}

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
            "Iteration",
            "Evaluations",
            "Best objective",
            "Feasible",
            "CALO operator",
            "Population diversity",
            "Feasible population ratio",
            "Reward",
        ]
        for index, name in enumerate(names):
            row = index % 5
            col = (index // 5) * 2
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

        self.plot = ScientificPlotWidget(
            title="Live convergence",
            xlabel="Recorded iteration",
            ylabel="Best objective",
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

    def reset(self) -> None:
        self.series = {}
        self.plot.plot_series({}, "Live convergence", "Recorded iteration", "Best objective")

    def update_progress(self, data: dict) -> None:
        algorithm = str(data.get("algorithm", "—"))
        self.labels["Algorithm"].setText(algorithm)
        self.labels["Iteration"].setText(str(data.get("iteration", "—")))
        self.labels["Evaluations"].setText(str(data.get("evaluations", "—")))
        best = data.get("best_objective")
        self.labels["Best objective"].setText(
            f"{best:.10g}" if isinstance(best, (int, float)) else "—"
        )
        self.labels["Feasible"].setText(str(data.get("feasible", "—")))
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
        if isinstance(best, (int, float)):
            self.series.setdefault(algorithm, []).append(float(best))
            self.plot.plot_series(
                self.series,
                "Live convergence",
                "Recorded iteration",
                "Best objective",
            )
