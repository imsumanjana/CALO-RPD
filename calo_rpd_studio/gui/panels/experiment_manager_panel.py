"""Experiment configuration, fairness audit, queue status, and execution."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.experiments.evaluation_budget import BudgetPolicy
from calo_rpd_studio.experiments.fairness_validator import validate_fairness
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


class ExperimentManagerPanel(WorkspacePage):
    def __init__(self, state, manager, parent=None) -> None:
        super().__init__(
            "Experiment Manager",
            "Configure repeated seeded experiments, audit fairness, execute primary comparisons, and track queued, completed, failed, or cancelled runs.",
            parent,
        )
        self.state = state
        self.manager = manager
        self.completed_runs = 0
        self.failed_runs = 0
        self.expected_runs = 0
        self.fairness_passed = False

        setup = SectionCard(
            "Execution configuration",
            "Publication comparisons default to equal objective-function evaluation budgets.",
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        setup.layout_root.addLayout(grid)

        self.runs = QSpinBox()
        self.runs.setRange(1, 10_000)
        self.population = QSpinBox()
        self.population.setRange(2, 100_000)
        self.policy = QComboBox()
        self.policy.addItem(
            "Equal objective evaluations",
            BudgetPolicy.EQUAL_EVALUATIONS.value,
        )
        self.policy.addItem(
            "Equal wall-clock time",
            BudgetPolicy.EQUAL_WALL_CLOCK.value,
        )
        self.policy.addItem(
            "Algorithm-native limits",
            BudgetPolicy.ALGORITHM_NATIVE.value,
        )
        self.budget = QSpinBox()
        self.budget.setRange(1, 2_000_000_000)
        self.wall = QDoubleSpinBox()
        self.wall.setRange(0.1, 604_800)
        self.wall.setSuffix(" s")
        self.maxit = QSpinBox()
        self.maxit.setRange(1, 10_000_000)
        self.workers = QSpinBox()
        self.workers.setRange(1, 256)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.output = QLineEdit()
        choose = QPushButton("Choose")
        choose.clicked.connect(self.choose_output)
        output_widget = QWidget()
        output_layout = QHBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(7)
        output_layout.addWidget(self.output, 1)
        output_layout.addWidget(choose)
        self.selected = QLabel()
        self.selected.setWordWrap(True)

        fields = [
            ("Independent runs", self.runs),
            ("Population size", self.population),
            ("Budget policy", self.policy),
            ("Objective evaluations", self.budget),
            ("Wall-clock budget", self.wall),
            ("Iteration safety limit", self.maxit),
            ("Parallel workers", self.workers),
            ("Master seed", self.seed),
        ]
        for index, (label, widget) in enumerate(fields):
            pair_column = (index % 2) * 2
            row = index // 2
            key = QLabel(label)
            key.setObjectName("MetricLabel")
            grid.addWidget(key, row, pair_column)
            grid.addWidget(widget, row, pair_column + 1)
        grid.addWidget(QLabel("Result array directory"), 4, 0)
        grid.addWidget(output_widget, 4, 1, 1, 3)
        grid.addWidget(QLabel("Primary algorithms"), 5, 0)
        grid.addWidget(self.selected, 5, 1, 1, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self.layout_root.addWidget(setup)

        execution = SectionCard("Execution")
        buttons = QHBoxLayout()
        self.compare = QPushButton("Run Comparative Experiment")
        self.compare.setObjectName("PrimaryButton")
        self.calo = QPushButton("Run CALO Analysis")
        self.cancel = QPushButton("Cancel")
        self.cancel.setEnabled(False)
        self.compare.setEnabled(False)
        self.calo.setEnabled(False)
        self.compare.clicked.connect(self.start_comparison)
        self.calo.clicked.connect(self.start_calo)
        self.cancel.clicked.connect(self.cancel_requested)
        buttons.addWidget(self.compare)
        buttons.addWidget(self.calo)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel)
        execution.layout_root.addLayout(buttons)
        self.status = QLabel("Run the fairness audit before starting an experiment. Global task progress is shown in the bottom status bar.")
        self.status.setWordWrap(True)
        self.status.setObjectName("InfoText")
        execution.layout_root.addWidget(self.status)
        self.layout_root.addWidget(execution)

        details = QTabWidget()
        fairness_page = QWidget()
        fairness_layout = QVBoxLayout(fairness_page)
        fairness_layout.setContentsMargins(12, 12, 12, 12)
        self.audit = QPlainTextEdit()
        self.audit.setReadOnly(True)
        self.audit_button = QPushButton("1. Run fairness audit")
        self.audit_button.setObjectName("PrimaryButton")
        self.audit_button.clicked.connect(self.run_fairness_audit)
        fairness_layout.addWidget(self.audit, 1)
        fairness_layout.addWidget(self.audit_button)
        details.addTab(fairness_page, "Fairness audit")

        queue_page = QWidget()
        queue_layout = QVBoxLayout(queue_page)
        queue_layout.setContentsMargins(12, 12, 12, 12)
        self.queue = QTableWidget(0, 3)
        self.queue.setHorizontalHeaderLabels(
            ["Run", "Algorithm / CALO variant", "Status"]
        )
        self.queue.setAlternatingRowColors(True)
        self.queue.verticalHeader().setVisible(False)
        self.queue.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.queue.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.Stretch,
        )
        self.queue.horizontalHeader().setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        queue_layout.addWidget(self.queue)
        details.addTab(queue_page, "Run queue")
        self.layout_root.addWidget(details, 1)

        manager.started.connect(self.on_started)
        manager.run_completed.connect(self.on_run_completed)
        manager.run_failed.connect(self.on_run_failed)
        manager.completed.connect(self.on_completed)
        manager.cancelled.connect(self.on_cancelled)
        manager.failed.connect(self.on_failed)
        manager.busy.connect(self.on_busy)
        self.policy.currentIndexChanged.connect(self._controls)
        for widget in (self.runs, self.population, self.policy, self.budget, self.wall, self.maxit, self.workers, self.seed):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._invalidate_fairness)
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._invalidate_fairness)
        self.output.textChanged.connect(self._invalidate_fairness)
        state.config_changed.connect(lambda _: self.refresh())
        self.refresh()
        self._set_running(manager.running)

    def _invalidate_fairness(self, *_args) -> None:
        if self.manager.running:
            return
        self.fairness_passed = False
        self.compare.setEnabled(False)
        self.calo.setEnabled(False)
        self.status.setText("Configuration changed. Run the fairness audit before starting an experiment.")

    def refresh(self) -> None:
        config = self.state.config
        self.runs.setValue(config.runs)
        self.population.setValue(config.population_size)
        index = self.policy.findData(config.budget.policy.value)
        self.policy.setCurrentIndex(max(index, 0))
        self.budget.setValue(config.budget.max_evaluations)
        self.wall.setValue(config.budget.wall_clock_seconds or 60)
        self.maxit.setValue(config.max_iterations)
        self.workers.setValue(config.parallel_workers)
        self.seed.setValue(config.master_seed)
        self.output.setText(config.output_directory)
        self.selected.setText(", ".join(config.algorithms))
        self._controls()

    def _controls(self) -> None:
        policy = BudgetPolicy(self.policy.currentData())
        self.budget.setEnabled(policy is not BudgetPolicy.EQUAL_WALL_CLOCK)
        self.wall.setEnabled(policy is BudgetPolicy.EQUAL_WALL_CLOCK)
        self.maxit.setEnabled(policy is BudgetPolicy.ALGORITHM_NATIVE)

    def apply(self) -> None:
        config = self.state.config
        config.runs = self.runs.value()
        config.population_size = self.population.value()
        config.budget.policy = BudgetPolicy(self.policy.currentData())
        config.budget.max_evaluations = self.budget.value()
        config.budget.wall_clock_seconds = (
            self.wall.value()
            if config.budget.policy is BudgetPolicy.EQUAL_WALL_CLOCK
            else None
        )
        config.max_iterations = self.maxit.value()
        config.parallel_workers = self.workers.value()
        config.master_seed = self.seed.value()
        config.output_directory = self.output.text().strip() or "results_data"
        config.validate()
        self.state.update_config()

    def choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select result array directory",
            self.output.text() or ".",
        )
        if path:
            self.output.setText(path)

    def run_fairness_audit(self) -> bool:
        task = self.state.task_status
        if task.busy:
            self.audit.setPlainText("Wait for the active scientific task to finish before running the fairness audit.")
            return False
        task.begin("Auditing experiment fairness", detail="Checking common comparison protocol")
        try:
            self.apply()
            report = validate_fairness(self.state.config)
        except Exception as exc:
            self.fairness_passed = False
            self.compare.setEnabled(False)
            self.calo.setEnabled(False)
            self.audit.setPlainText(str(exc))
            task.fail(str(exc))
            return False
        lines = [
            "PASS: comparative protocol is internally consistent."
            if report.fair
            else "FAIL: comparative protocol requires correction."
        ]
        lines.extend(f"ERROR: {message}" for message in report.errors)
        lines.extend(f"NOTICE: {message}" for message in report.warnings)
        self.audit.setPlainText("\n".join(lines))
        self.fairness_passed = bool(report.fair)
        self.compare.setEnabled(self.fairness_passed and not self.manager.running)
        self.calo.setEnabled(self.fairness_passed and not self.manager.running)
        if self.fairness_passed:
            self.status.setText("Fairness audit passed. Step 2 is now available: run the comparison or CALO analysis.")
            task.finish("Fairness audit passed")
        else:
            self.status.setText("Fairness audit failed. Correct the reported issues before execution.")
            task.fail("Fairness audit failed")
        return self.fairness_passed

    def _populate_queue(self, labels: list[str]) -> None:
        rows = [
            (run_index, label)
            for run_index in range(self.state.config.runs)
            for label in labels
        ]
        self.queue.setRowCount(len(rows))
        for row, (run_index, label) in enumerate(rows):
            self.queue.setItem(row, 0, QTableWidgetItem(str(run_index + 1)))
            self.queue.setItem(row, 1, QTableWidgetItem(label))
            self.queue.setItem(row, 2, QTableWidgetItem("Queued"))

    def _manager_available(self) -> bool:
        if not self.manager.running:
            return True
        QMessageBox.information(
            self,
            "Experiment already running",
            "An experiment is already running. Wait for it to finish or request safe cancellation before starting another run.",
        )
        return False

    def start_comparison(self) -> None:
        if not self._manager_available():
            return
        if not self.run_fairness_audit():
            QMessageBox.critical(
                self,
                "Fairness audit failed",
                "Correct the reported configuration errors before running the comparison.",
            )
            return
        labels = list(self.state.config.algorithms)
        self._populate_queue(labels)
        self.expected_runs = self.state.config.runs * len(labels)
        self.completed_runs = 0
        self.failed_runs = 0
        self._set_running(True)
        if not self.manager.start_comparison(self.state.config):
            self._set_running(self.manager.running)

    def start_calo(self) -> None:
        if not self._manager_available():
            return
        if not self.fairness_passed and not self.run_fairness_audit():
            QMessageBox.critical(
                self,
                "Fairness audit failed",
                "Correct the reported configuration errors before running CALO analysis.",
            )
            return
        try:
            self.apply()
        except Exception as exc:
            QMessageBox.critical(self, "Configuration error", str(exc))
            return
        labels = [
            "Classical TLBO",
            "Legacy Gaussian MTLBO",
            "CALO without AI",
            "CALO without success memory",
            "CALO without stagnation recovery",
            "CALO without diversity feedback",
            "Complete CALO",
        ]
        self._populate_queue(labels)
        self.expected_runs = self.state.config.runs * len(labels)
        self.completed_runs = 0
        self.failed_runs = 0
        self._set_running(True)
        if not self.manager.start_calo_analysis(self.state.config):
            self._set_running(self.manager.running)

    def _set_running(self, running: bool) -> None:
        self.compare.setEnabled((not running) and self.fairness_passed)
        self.calo.setEnabled((not running) and self.fairness_passed)
        self.audit_button.setEnabled(not running)
        self.cancel.setEnabled(running)

    def _mark_next(self, algorithm: str, status: str) -> None:
        for row in range(self.queue.rowCount()):
            algorithm_item = self.queue.item(row, 1)
            status_item = self.queue.item(row, 2)
            if (
                algorithm_item is not None
                and status_item is not None
                and algorithm_item.text() == algorithm
                and status_item.text() in ("Queued", "Active")
            ):
                status_item.setText(status)
                return

    def on_started(self, experiment_id: str) -> None:
        self._set_running(True)
        self.status.setText(f"Experiment {experiment_id} is running.")

    def on_run_completed(self, run_id: str, algorithm: str) -> None:
        self.completed_runs += 1
        self._mark_next(algorithm, "Completed")
        self._update_status(f"Latest completed: {algorithm}.")

    def on_run_failed(self, failure_id: str, algorithm: str) -> None:
        self.failed_runs += 1
        self._mark_next(algorithm, "Failed")
        self._update_status(
            f"Latest failed: {algorithm}; failure record {failure_id[:8]}."
        )

    def _update_status(self, suffix: str) -> None:
        finished = self.completed_runs + self.failed_runs
        self.status.setText(
            f"Finished {finished} of {self.expected_runs} runs: "
            f"{self.completed_runs} completed, {self.failed_runs} failed. {suffix}"
        )

    def cancel_requested(self) -> None:
        self.state.task_status.cancel()
        for row in range(self.queue.rowCount()):
            item = self.queue.item(row, 2)
            if item is not None and item.text() == "Queued":
                item.setText("Cancelled")
        self.status.setText(
            "Cancellation requested. The active numerical step will finish safely before execution stops."
        )

    def on_completed(self, experiment_id: str) -> None:
        self._set_running(False)
        self.status.setText(
            f"Experiment {experiment_id} finished: {self.completed_runs} completed and "
            f"{self.failed_runs} failed runs. All outcomes are stored with provenance."
        )

    def on_cancelled(self, experiment_id: str) -> None:
        self._set_running(False)
        self.status.setText(
            f"Experiment {experiment_id} was cancelled safely. Completed runs remain stored with provenance."
        )

    def on_failed(self, message: str) -> None:
        self._set_running(False)
        self.status.setText(
            "Experiment stopped because an execution or configuration error occurred."
        )
        QMessageBox.critical(self, "Experiment execution failed", message)

    def on_busy(self, message: str) -> None:
        self.status.setText(message)
