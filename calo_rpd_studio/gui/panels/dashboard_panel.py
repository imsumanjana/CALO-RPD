"""Modern application dashboard."""
from __future__ import annotations

from PyQt6.QtWidgets import QGridLayout, QLabel, QSizePolicy

from calo_rpd_studio.gui.widgets.section_card import MetricCard, SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage
from calo_rpd_studio.power_system.network_metrics import summarize_case


class DashboardPanel(WorkspacePage):
    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Dashboard",
            "Current power-system context, optimization protocol, scenario mode, and verified experiment activity.",
            parent,
        )
        self.state = state

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(12)
        self.case_metric = MetricCard("Power system", "No case loaded", "Load a reference case to begin")
        self.objective_metric = MetricCard("ORPD objective", "Active power loss", "Common evaluator for every optimizer")
        self.algorithm_metric = MetricCard("Selected optimizers", "3", "CALO, TLBO, PSO")
        self.verified_metric = MetricCard("Verified results", "0", "Independent validation required for export")
        cards = [
            self.case_metric,
            self.objective_metric,
            self.algorithm_metric,
            self.verified_metric,
        ]
        for index, card in enumerate(cards):
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            metrics.addWidget(card, 0, index)
        self.layout_root.addLayout(metrics)

        context = SectionCard(
            "Scientific context",
            "The dashboard reflects only loaded case data and persisted experiment records; no performance values are pre-filled.",
        )
        grid = QGridLayout()
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(9)
        context.layout_root.addLayout(grid)
        self.labels: dict[str, QLabel] = {}
        names = [
            "Power-system case",
            "Buses",
            "Generators",
            "Branches",
            "Transformers",
            "Shunt buses",
            "ORPD objective",
            "Primary algorithms",
            "Scenario mode",
            "Completed experiments",
            "Verified results",
        ]
        for index, name in enumerate(names):
            row = index % 6
            col = (index // 6) * 2
            key = QLabel(name)
            key.setObjectName("MetricLabel")
            value = QLabel("—")
            value.setWordWrap(True)
            value.setObjectName("ContextValue")
            self.labels[name] = value
            grid.addWidget(key, row, col)
            grid.addWidget(value, row, col + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self.layout_root.addWidget(context)
        self.layout_root.addStretch(1)

        state.case_changed.connect(lambda _: self.refresh())
        state.config_changed.connect(lambda _: self.refresh())
        state.runs_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        case = self.state.current_case
        if case:
            metrics = summarize_case(case)
            self.labels["Power-system case"].setText(case.name)
            self.labels["Buses"].setText(str(metrics["buses"]))
            self.labels["Generators"].setText(str(metrics["generators"]))
            self.labels["Branches"].setText(str(metrics["branches"]))
            self.labels["Transformers"].setText(str(metrics["transformers"]))
            self.labels["Shunt buses"].setText(str(metrics["shunt_buses"]))
            self.case_metric.set_metric(case.name, f'{metrics["buses"]} buses · {metrics["branches"]} branches')
        else:
            for name in [
                "Power-system case",
                "Buses",
                "Generators",
                "Branches",
                "Transformers",
                "Shunt buses",
            ]:
                self.labels[name].setText("—")
            self.case_metric.set_metric("No case loaded", "Load a reference case to begin")

        objective = self.state.config.objective.kind.value
        algorithms = list(self.state.config.algorithms)
        self.labels["ORPD objective"].setText(objective)
        self.labels["Primary algorithms"].setText(", ".join(algorithms))
        self.labels["Scenario mode"].setText(self.state.config.scenarios.mode)

        experiments = self.state.database.list_experiments()
        verified = sum(
            1
            for experiment in experiments
            for run in self.state.database.list_runs(experiment["id"])
            if run["validation_status"] == "verified"
        )
        self.labels["Completed experiments"].setText(str(len(experiments)))
        self.labels["Verified results"].setText(str(verified))

        self.objective_metric.set_metric(objective.replace("_", " ").title(), "Common ORPD objective")
        self.algorithm_metric.set_metric(str(len(algorithms)), ", ".join(algorithms[:4]) + ("…" if len(algorithms) > 4 else ""))
        self.verified_metric.set_metric(str(verified), f"{len(experiments)} experiment record(s)")
