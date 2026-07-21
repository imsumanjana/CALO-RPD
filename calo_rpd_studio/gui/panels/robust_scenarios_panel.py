"""Robust ORPD scenario configuration."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.widgets.page_header import PageHeader
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage
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

        box = QGroupBox("Scenario generator")
        form = QFormLayout(box)
        self.mode = QComboBox()
        self.mode.addItems(self.MODES)
        self.count = QSpinBox()
        self.count.setRange(1, 100000)
        self.pstd = self._spin(0, 1, 0.05)
        self.qstd = self._spin(0, 1, 0.05)
        self.branch = QLineEdit()
        self.branch.setPlaceholderText("Zero-based branch indices, comma separated")
        self.gen = QLineEdit()
        self.gen.setPlaceholderText("Zero-based generator indices, comma separated")
        self.renew_bus = QSpinBox()
        self.renew_bus.setRange(0, 100000)
        self.renew_mw = self._spin(0, 1e9, 0)
        self.cf_mean = self._spin(0, 1, 0.5)
        self.cf_std = self._spin(0, 1, 0.15)
        form.addRow("Mode", self.mode)
        form.addRow("Scenario count", self.count)
        form.addRow("Active-load standard deviation", self.pstd)
        form.addRow("Reactive-load standard deviation", self.qstd)
        form.addRow("Branch outage indices", self.branch)
        form.addRow("Generator outage indices", self.gen)
        form.addRow("Renewable bus number", self.renew_bus)
        form.addRow("Renewable rated power (MW)", self.renew_mw)
        form.addRow("Mean capacity factor", self.cf_mean)
        form.addRow("Capacity-factor standard deviation", self.cf_std)
        layout.addWidget(box)

        robust = QGroupBox("Robust objective")
        robust_form = QFormLayout(robust)
        self.aggregation = QComboBox()
        for item in RobustAggregation:
            self.aggregation.addItem(item.value, item)
        self.risk = self._spin(0, 100, 1)
        self.alpha = self._spin(0.5, 0.9999, 0.95)
        robust_form.addRow("Aggregation", self.aggregation)
        robust_form.addRow("Mean-risk coefficient", self.risk)
        robust_form.addRow("CVaR confidence level", self.alpha)
        layout.addWidget(robust)

        apply_button = QPushButton("Apply scenario configuration and continue")
        apply_button.setObjectName("PrimaryButton")
        apply_button.clicked.connect(self.apply)
        layout.addWidget(apply_button)
        layout.addStretch(1)
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
