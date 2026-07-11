"""ORPD variables, objectives, mixed-variable settings, and constraint policy."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.widgets.page_header import PageHeader
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage
from calo_rpd_studio.orpd.objectives import ObjectiveKind


class ORPDFormulationPanel(ScrollablePage):
    stage_completed = pyqtSignal()

    def __init__(self, state, parent=None):
        content = QWidget()
        super().__init__(content, parent)
        self.state = state
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(
            PageHeader(
                "ORPD Formulation",
                "Define one common physical search space, objective, discrete device behavior, and feasibility-first comparison policy for every optimizer.",
            )
        )

        objective = QGroupBox("Objective function")
        grid = QGridLayout(objective)
        self.kind = QComboBox()
        for kind in ObjectiveKind:
            self.kind.addItem(kind.value, kind)
        self.wloss = self._spin(0, 100, 1)
        self.wvd = self._spin(0, 100, 0)
        self.wli = self._spin(0, 100, 0)
        grid.addWidget(QLabel("Objective"), 0, 0)
        grid.addWidget(self.kind, 0, 1)
        grid.addWidget(QLabel("Loss weight"), 1, 0)
        grid.addWidget(self.wloss, 1, 1)
        grid.addWidget(QLabel("Voltage-deviation weight"), 2, 0)
        grid.addWidget(self.wvd, 2, 1)
        grid.addWidget(QLabel("L-index weight"), 3, 0)
        grid.addWidget(self.wli, 3, 1)
        layout.addWidget(objective)

        variables = QGroupBox("Control variables and mixed-variable decoding")
        vg = QGridLayout(variables)
        self.gen_v = QCheckBox("Generator voltage magnitudes")
        self.taps = QCheckBox("Transformer tap settings")
        self.shunts = QCheckBox("Shunt reactive compensation")
        self.discrete_taps = QCheckBox("Discrete transformer taps")
        self.discrete_shunts = QCheckBox("Discrete shunt steps")
        self.tap_min = self._spin(0.5, 1.5, 0.9)
        self.tap_max = self._spin(0.5, 1.5, 1.1)
        self.tap_step = self._spin(0.0001, 0.2, 0.0125)
        for i, widget in enumerate(
            [self.gen_v, self.taps, self.shunts, self.discrete_taps, self.discrete_shunts]
        ):
            vg.addWidget(widget, i, 0, 1, 2)
        vg.addWidget(QLabel("Tap minimum"), 5, 0)
        vg.addWidget(self.tap_min, 5, 1)
        vg.addWidget(QLabel("Tap maximum"), 6, 0)
        vg.addWidget(self.tap_max, 6, 1)
        vg.addWidget(QLabel("Tap step"), 7, 0)
        vg.addWidget(self.tap_step, 7, 1)
        layout.addWidget(variables)

        policy = QGroupBox("Constraint treatment")
        policy_layout = QVBoxLayout(policy)
        text = QLabel(
            "Feasibility-first ranking is applied independently of the objective: feasible candidates dominate infeasible candidates; feasible candidates are ordered by objective; infeasible candidates are ordered by normalized total violation. Voltage, generator P/Q, device, branch thermal, and power-flow convergence checks remain explicit."
        )
        text.setWordWrap(True)
        policy_layout.addWidget(text)
        layout.addWidget(policy)

        save = QPushButton("Apply ORPD formulation and continue")
        save.setObjectName("PrimaryButton")
        save.clicked.connect(self.apply)
        layout.addWidget(save)
        layout.addStretch(1)
        state.config_changed.connect(lambda _: self.refresh())
        self.refresh()

    def _spin(self, low, high, value):
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(6)
        spin.setValue(value)
        return spin

    def refresh(self):
        config = self.state.config
        index = self.kind.findData(config.objective.kind)
        self.kind.setCurrentIndex(max(index, 0))
        self.wloss.setValue(config.objective.weight_loss)
        self.wvd.setValue(config.objective.weight_voltage_deviation)
        self.wli.setValue(config.objective.weight_l_index)
        self.gen_v.setChecked(config.variables.generator_voltages)
        self.taps.setChecked(config.variables.transformer_taps)
        self.shunts.setChecked(config.variables.shunt_compensation)
        self.discrete_taps.setChecked(config.variables.discrete_transformer_taps)
        self.discrete_shunts.setChecked(config.variables.discrete_shunts)
        self.tap_min.setValue(config.variables.transformer_minimum)
        self.tap_max.setValue(config.variables.transformer_maximum)
        self.tap_step.setValue(config.variables.transformer_step)

    def apply(self):
        config = self.state.config
        config.objective.kind = self.kind.currentData()
        config.objective.weight_loss = self.wloss.value()
        config.objective.weight_voltage_deviation = self.wvd.value()
        config.objective.weight_l_index = self.wli.value()
        config.variables.generator_voltages = self.gen_v.isChecked()
        config.variables.transformer_taps = self.taps.isChecked()
        config.variables.shunt_compensation = self.shunts.isChecked()
        config.variables.discrete_transformer_taps = self.discrete_taps.isChecked()
        config.variables.discrete_shunts = self.discrete_shunts.isChecked()
        config.variables.transformer_minimum = self.tap_min.value()
        config.variables.transformer_maximum = self.tap_max.value()
        config.variables.transformer_step = self.tap_step.value()
        try:
            config.validate()
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "ORPD formulation error", str(exc))
            return
        self.state.update_config()
        self.stage_completed.emit()
