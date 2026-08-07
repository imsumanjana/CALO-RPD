"""Robust ORPD scenario configuration."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.widgets.page_header import PageHeader
from calo_rpd_studio.gui.widgets.chip_editor import IntegerChipEditor
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage
from calo_rpd_studio.gui.widgets.workspace_tabs import WorkspaceTabs
from calo_rpd_studio.robustness.robust_objectives import RobustAggregation


class RobustScenariosPanel(ScrollablePage):
    stage_completed = pyqtSignal()
    MODES = [
        "deterministic",
        "load_uncertainty",
        "monte_carlo",
        "renewable_uncertainty",
        "branch_contingency",
        "generator_contingency",
    ]

    def __init__(self, state, parent=None):
        content = QWidget()
        super().__init__(content, parent)
        self.state = state
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(
            PageHeader(
                "Robust Scenarios",
                "Define deterministic, uncertainty, Monte Carlo, renewable-injection, or contingency scenarios and the robust objective aggregation.",
            )
        )

        self.mode = QComboBox()
        self.mode.addItems(self.MODES)
        self.count = QSpinBox()
        self.count.setRange(1, 100000)
        self.pstd = self._spin(0, 1, 0.05)
        self.qstd = self._spin(0, 1, 0.05)
        self.branch = IntegerChipEditor("Zero-based branch indices")
        self.gen = IntegerChipEditor("Zero-based generator indices")
        self.renew_bus = QSpinBox()
        self.renew_bus.setRange(0, 100000)
        self.renew_mw = self._spin(0, 1e9, 0)
        self.cf_mean = self._spin(0, 1, 0.5)
        self.cf_std = self._spin(0, 1, 0.15)

        generator_page = QWidget()
        generator_layout = QHBoxLayout(generator_page)
        generator_layout.setContentsMargins(18, 18, 18, 18)
        generator_layout.setSpacing(16)
        sampling = QGroupBox("Sampling and load uncertainty")
        sampling_form = QFormLayout(sampling)
        sampling_form.addRow("Mode", self.mode)
        sampling_form.addRow("Scenario count", self.count)
        sampling_form.addRow("Active-load standard deviation", self.pstd)
        sampling_form.addRow("Reactive-load standard deviation", self.qstd)
        contingency = QGroupBox("Contingency and renewable inputs")
        contingency_form = QFormLayout(contingency)
        contingency_form.addRow("Branch outage indices", self.branch)
        contingency_form.addRow("Generator outage indices", self.gen)
        contingency_form.addRow("Renewable bus number", self.renew_bus)
        contingency_form.addRow("Renewable rated power (MW)", self.renew_mw)
        contingency_form.addRow("Mean capacity factor", self.cf_mean)
        contingency_form.addRow("Capacity-factor standard deviation", self.cf_std)
        generator_layout.addWidget(sampling, 1)
        generator_layout.addWidget(contingency, 1)

        self.aggregation = QComboBox()
        for item in RobustAggregation:
            self.aggregation.addItem(item.value, item)
        self.risk = self._spin(0, 100, 1)
        self.alpha = self._spin(0.5, 0.9999, 0.95)

        objective_page = QWidget()
        objective_layout = QHBoxLayout(objective_page)
        objective_layout.setContentsMargins(18, 18, 18, 18)
        objective_layout.setSpacing(16)
        robust = QGroupBox("Aggregation controls")
        robust_form = QFormLayout(robust)
        robust_form.addRow("Aggregation", self.aggregation)
        robust_form.addRow("Mean-risk coefficient", self.risk)
        robust_form.addRow("CVaR confidence level", self.alpha)
        guidance = QGroupBox("Interpretation")
        guidance_layout = QVBoxLayout(guidance)
        guidance_text = QLabel(
            "Expected aggregation summarizes the scenario mean. Mean-risk adds a dispersion "
            "penalty, while CVaR focuses on the configured upper-tail confidence level."
        )
        guidance_text.setWordWrap(True)
        guidance_layout.addWidget(guidance_text)
        guidance_layout.addStretch(1)
        objective_layout.addWidget(robust, 1)
        objective_layout.addWidget(guidance, 1)

        self.section_tabs = WorkspaceTabs("Robust scenario sections")
        self.section_tabs.add_section(
            "Scenario generator",
            generator_page,
            "Configure sampling, uncertainty, contingency, and renewable inputs.",
        )
        self.section_tabs.add_section(
            "Robust objective",
            objective_page,
            "Configure and interpret robust objective aggregation.",
        )
        self.section_tabs.setMinimumHeight(440)
        layout.addWidget(self.section_tabs, 1)

        apply_button = QPushButton("Apply scenario configuration and continue")
        apply_button.setObjectName("PrimaryButton")
        apply_button.clicked.connect(self.apply)
        layout.addWidget(apply_button)
        self.state.config_changed.connect(lambda _: self.refresh())
        self.refresh()

    def _spin(self, low, high, value):
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(6)
        spin.setValue(value)
        return spin

    def _indices(self, text):
        return [int(item.strip()) for item in text.split(",") if item.strip()]

    def refresh(self):
        scenarios = self.state.config.scenarios
        self.mode.setCurrentText(scenarios.mode)
        self.count.setValue(scenarios.count)
        self.pstd.setValue(scenarios.active_load_std)
        self.qstd.setValue(scenarios.reactive_load_std)
        self.branch.setText(",".join(map(str, scenarios.branch_outages)))
        self.gen.setText(",".join(map(str, scenarios.generator_outages)))
        self.renew_bus.setValue(scenarios.renewable_bus)
        self.renew_mw.setValue(scenarios.renewable_rated_mw)
        self.cf_mean.setValue(scenarios.renewable_mean_capacity_factor)
        self.cf_std.setValue(scenarios.renewable_std_capacity_factor)
        index = self.aggregation.findData(self.state.config.robust_objective.aggregation)
        self.aggregation.setCurrentIndex(max(index, 0))
        self.risk.setValue(self.state.config.robust_objective.risk_lambda)
        self.alpha.setValue(self.state.config.robust_objective.cvar_alpha)

    def load_from_config(self, config) -> None:
        self.refresh()

    def apply(self):
        try:
            scenarios = self.state.config.scenarios
            scenarios.mode = self.mode.currentText()
            scenarios.count = self.count.value()
            scenarios.active_load_std = self.pstd.value()
            scenarios.reactive_load_std = self.qstd.value()
            scenarios.branch_outages = self._indices(self.branch.text())
            scenarios.generator_outages = self._indices(self.gen.text())
            scenarios.renewable_bus = self.renew_bus.value()
            scenarios.renewable_rated_mw = self.renew_mw.value()
            scenarios.renewable_mean_capacity_factor = self.cf_mean.value()
            scenarios.renewable_std_capacity_factor = self.cf_std.value()
            self.state.config.robust_objective.aggregation = self.aggregation.currentData()
            self.state.config.robust_objective.risk_lambda = self.risk.value()
            self.state.config.robust_objective.cvar_alpha = self.alpha.value()
            self.state.config.validate()
            self.state.update_config()
            self.stage_completed.emit()
        except Exception as exc:
            QMessageBox.critical(self, "Scenario configuration error", str(exc))
