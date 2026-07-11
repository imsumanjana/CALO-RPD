"""Case loading, data inspection, base power flow, and cross-validation."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
)

from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.power_system.case_validation import validate_case
from calo_rpd_studio.power_system.independent_validator import validate_against_pypower
from calo_rpd_studio.power_system.network_metrics import summarize_case


class PowerSystemPanel(WorkspacePage):
    stage_completed = pyqtSignal()

    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Power System",
            "Follow the validation sequence: load a reference case, solve the base AC power flow, then complete the independent cross-check.",
            parent,
        )
        self.state = state

        controls = QGroupBox("Case selection and validation sequence")
        row = QHBoxLayout(controls)
        self.case_combo = QComboBox()
        self.case_combo.addItems(CaseLoader.available_cases())
        self.case_combo.setMinimumWidth(180)
        self.load_button = QPushButton("1. Load case")
        self.load_button.setObjectName("PrimaryButton")
        self.power_flow_button = QPushButton("2. Run base AC power flow")
        self.cross_check_button = QPushButton("3. Cross-check with PYPOWER")
        self.power_flow_button.setEnabled(False)
        self.cross_check_button.setEnabled(False)
        self.load_button.clicked.connect(self.load_case)
        self.power_flow_button.clicked.connect(self.run_pf)
        self.cross_check_button.clicked.connect(self.cross_validate)
        row.addWidget(self.case_combo)
        row.addWidget(self.load_button)
        row.addWidget(self.power_flow_button)
        row.addWidget(self.cross_check_button)
        row.addStretch(1)
        self.layout_root.addWidget(controls)

        self.summary = QLabel("No case loaded. Start with step 1.")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("InfoText")
        self.layout_root.addWidget(self.summary)

        self.tabs = QTabWidget()
        self.bus_table = QTableWidget()
        self.gen_table = QTableWidget()
        self.branch_table = QTableWidget()
        for table in (self.bus_table, self.gen_table, self.branch_table):
            table.setAlternatingRowColors(True)
            table.setSortingEnabled(False)
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.bus_table, "Bus data")
        self.tabs.addTab(self.gen_table, "Generator data")
        self.tabs.addTab(self.branch_table, "Branch data")
        self.layout_root.addWidget(self.tabs, 1)

        self.result = QLabel("Complete the three validation steps in order.")
        self.result.setWordWrap(True)
        self.result.setObjectName("ResultBanner")
        self.layout_root.addWidget(self.result)

    def _fill(self, table: QTableWidget, array, prefix: str) -> None:
        table.setRowCount(array.shape[0])
        table.setColumnCount(array.shape[1])
        table.setHorizontalHeaderLabels([f"{prefix}{i}" for i in range(array.shape[1])])
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                table.setItem(row, column, QTableWidgetItem(f"{array[row, column]:.8g}"))

    def load_case(self) -> None:
        task = self.state.task_status
        if not task.begin("Loading power-system case", detail=self.case_combo.currentText()):
            QMessageBox.information(self, "Task busy", "Wait for the active task to finish first.")
            return
        QApplication.processEvents()
        try:
            case = CaseLoader.load(self.case_combo.currentText())
            report = validate_case(case)
            if not report.valid:
                raise ValueError("\n".join(report.errors))
            self.state.set_case(case)
            self.state.config.case_name = case.name
            self.state.update_config()
            metrics = summarize_case(case)
            self.summary.setText(
                f"{case.name} · {metrics['buses']} buses · {metrics['generators']} online generators · "
                f"{metrics['branches']} active branches · {metrics['transformers']} transformers · "
                f"checksum {metrics['checksum']}"
            )
            self._fill(self.bus_table, case.bus, "B")
            self._fill(self.gen_table, case.gen, "G")
            self._fill(self.branch_table, case.branch, "L")
            self.power_flow_button.setEnabled(True)
            self.cross_check_button.setEnabled(False)
            self.result.setText("Case validation passed. Continue with step 2: run the base AC power flow.")
            task.finish("Case loaded and validated")
        except Exception as exc:
            self.power_flow_button.setEnabled(False)
            self.cross_check_button.setEnabled(False)
            task.fail(str(exc))
            QMessageBox.critical(self, "Case load failed", str(exc))

    def run_pf(self) -> None:
        if self.state.current_case is None:
            QMessageBox.information(self, "Load a case first", "Complete step 1 before running the power flow.")
            return
        task = self.state.task_status
        if not task.begin("Running base AC power flow", detail=self.state.current_case.name):
            QMessageBox.information(self, "Task busy", "Wait for the active task to finish first.")
            return
        QApplication.processEvents()
        try:
            power_flow = run_ac_power_flow(self.state.current_case)
            self.state.current_power_flow = power_flow
            message = (
                f"Converged: {power_flow.converged} · Newton iterations: {power_flow.iterations} · "
                f"Q-limit switching rounds: {power_flow.q_limit_rounds} · "
                f"maximum mismatch: {power_flow.max_mismatch:.3e} · "
                f"active-power loss: {power_flow.total_loss_mw:.8f} MW"
            )
            if power_flow.warnings:
                message += " · " + " ".join(power_flow.warnings)
            self.result.setText(message)
            if power_flow.converged:
                self.cross_check_button.setEnabled(True)
                task.finish("Base AC power flow converged")
            else:
                self.cross_check_button.setEnabled(False)
                task.fail("Base AC power flow did not converge")
        except Exception as exc:
            self.cross_check_button.setEnabled(False)
            task.fail(str(exc))
            QMessageBox.critical(self, "Power-flow execution failed", str(exc))

    def cross_validate(self) -> None:
        if self.state.current_power_flow is None or not self.state.current_power_flow.converged:
            QMessageBox.information(self, "Run power flow first", "Complete step 2 before cross-validation.")
            return
        task = self.state.task_status
        if not task.begin("Cross-validating AC power flow", detail="Independent PYPOWER comparison"):
            QMessageBox.information(self, "Task busy", "Wait for the active task to finish first.")
            return
        QApplication.processEvents()
        try:
            result = validate_against_pypower(
                self.state.current_case,
                self.state.current_power_flow,
            )
            self.result.setText(
                f"{result.message} · available: {result.available} · passed: {result.passed} · "
                f"max |ΔV|={result.max_vm_difference:.3e} p.u. · "
                f"max |Δangle|={result.max_va_difference_deg:.3e}° · "
                f"|Δloss|={result.loss_difference_mw:.3e} MW"
            )
            if result.available and result.passed:
                task.finish("Independent cross-validation passed")
                self.stage_completed.emit()
            else:
                task.fail("Independent cross-validation did not pass")
        except Exception as exc:
            task.fail(str(exc))
            QMessageBox.warning(self, "Cross-validation unavailable", str(exc))
