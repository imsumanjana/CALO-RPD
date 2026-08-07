"""ORPD variables, objectives, mixed-variable settings, and constraint policy."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.widgets.page_header import PageHeader
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage
from calo_rpd_studio.gui.widgets.workspace_tabs import WorkspaceTabs
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

        self.kind = QComboBox()
        for kind in ObjectiveKind:
            self.kind.addItem(kind.value, kind)
        self.wloss = self._spin(0, 100, 1)
        self.wvd = self._spin(0, 100, 0)
        self.wli = self._spin(0, 100, 0)

        objective_page = QWidget()
        objective_layout = QHBoxLayout(objective_page)
        objective_layout.setContentsMargins(18, 18, 18, 18)
        objective_layout.setSpacing(16)
        objective_selection = QGroupBox("Objective selection")
        objective_form = QFormLayout(objective_selection)
        objective_form.addRow("Objective", self.kind)
        objective_form.addRow("Loss weight", self.wloss)
        objective_weights = QGroupBox("Additional objective weights")
        weight_form = QFormLayout(objective_weights)
        weight_form.addRow("Voltage-deviation weight", self.wvd)
        weight_form.addRow("L-index weight", self.wli)
        objective_layout.addWidget(objective_selection, 1)
        objective_layout.addWidget(objective_weights, 1)

        self.gen_v = QCheckBox("Generator voltage magnitudes")
        self.taps = QCheckBox("Transformer tap settings")
        self.shunts = QCheckBox("Shunt reactive compensation")
        self.discrete_taps = QCheckBox("Discrete transformer taps")
        self.discrete_shunts = QCheckBox("Discrete shunt steps")
        self.tap_min = self._spin(0.5, 1.5, 0.9)
        self.tap_max = self._spin(0.5, 1.5, 1.1)
        self.tap_step = self._spin(0.0001, 0.2, 0.0125)

        controls_page = QWidget()
        controls_layout = QHBoxLayout(controls_page)
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setSpacing(16)
        included_controls = QGroupBox("Included controls")
        included_layout = QVBoxLayout(included_controls)
        for widget in (
            self.gen_v,
            self.taps,
            self.shunts,
            self.discrete_taps,
            self.discrete_shunts,
        ):
            included_layout.addWidget(widget)
        included_layout.addStretch(1)
        tap_decoding = QGroupBox("Transformer tap decoding")
        tap_form = QFormLayout(tap_decoding)
        tap_form.addRow("Tap minimum", self.tap_min)
        tap_form.addRow("Tap maximum", self.tap_max)
        tap_form.addRow("Tap step", self.tap_step)
        controls_layout.addWidget(included_controls, 1)
        controls_layout.addWidget(tap_decoding, 1)

        policy_page = QWidget()
        policy_layout = QVBoxLayout(policy_page)
        policy_layout.setContentsMargins(18, 18, 18, 18)
        text = QLabel(
            "Feasibility-first ranking is applied independently of the objective: feasible candidates dominate infeasible candidates; feasible candidates are ordered by objective; infeasible candidates are ordered by normalized total violation. Voltage, generator P/Q, device, branch thermal, and power-flow convergence checks remain explicit."
        )
        text.setWordWrap(True)
        policy_layout.addWidget(text)
        policy_layout.addStretch(1)

        self.section_tabs = WorkspaceTabs("ORPD formulation sections")
        self.section_tabs.add_section(
            "Objective",
            objective_page,
            "Choose the objective and its weighting terms.",
        )
        self.section_tabs.add_section(
            "Control variables",
            controls_page,
            "Choose physical control variables and transformer tap decoding.",
        )
        self.section_tabs.add_section(
            "Constraint policy",
            policy_page,
            "Review the common feasibility-first comparison policy.",
        )
        self.section_tabs.setMinimumHeight(360)
        layout.addWidget(self.section_tabs, 1)

        save = QPushButton("Apply ORPD formulation and continue")
        save.setObjectName("PrimaryButton")
        save.clicked.connect(self.apply)
        layout.addWidget(save)
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
