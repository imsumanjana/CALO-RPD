"""Experiment configuration, fairness audit, queue status, and execution."""
from __future__ import annotations

import os

import psutil

from PyQt6.QtCore import Qt
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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.experiments.evaluation_budget import BudgetPolicy
from calo_rpd_studio.experiments.execution_plan import ABLATION_MODE, COMPARISON_MODE, labels_for_mode, planned_item_count
from calo_rpd_studio.experiments.fairness_validator import validate_fairness
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


class ExperimentManagerPanel(WorkspacePage):
    """Guided experiment workflow with a scrollable body for compact screens."""

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

        # This workspace is genuinely taller than a typical laptop viewport.  Keep the
        # page header fixed and scroll only the workflow body so controls retain their
        # normal size instead of being vertically compressed by Qt's layout engine.
        self.body_scroll = QScrollArea()
        self.body_scroll.setObjectName("ExperimentManagerScroll")
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.body_scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.body_content = QWidget()
        self.body_content.setObjectName("ExperimentManagerContent")
        self.body_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.body_layout = QVBoxLayout(self.body_content)
        self.body_layout.setContentsMargins(0, 0, 8, 8)
        self.body_layout.setSpacing(16)
        self.body_scroll.setWidget(self.body_content)
        self.layout_root.addWidget(self.body_scroll, 1)

        self.setup_card = SectionCard(
            "1. Experiment configuration",
            "Set the repeated-run protocol and compute resources. The fairness audit uses these exact values.",
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        self.setup_card.layout_root.addLayout(grid)

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
        self.recommended_workers = self._recommended_worker_count()
        self.workers.setToolTip(
            "Independent optimizer jobs are executed in separate CPU processes when this value is greater than one."
        )
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
        self.plan_summary = QLabel()
        self.plan_summary.setWordWrap(True)
        self.plan_summary.setObjectName("InfoText")
        self.execution_note = QLabel()
        self.execution_note.setWordWrap(True)
        self.execution_note.setObjectName("HelpText")

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
            widget.setMinimumHeight(32)
            pair_column = (index % 2) * 2
            row = index // 2
            key = QLabel(label)
            key.setObjectName("MetricLabel")
            grid.addWidget(key, row, pair_column)
            grid.addWidget(widget, row, pair_column + 1)
        self.output.setMinimumHeight(32)
        choose.setMinimumHeight(32)
        use_recommended = QPushButton(f"Use recommended ({self.recommended_workers})")
        use_recommended.setMinimumHeight(32)
        use_recommended.setToolTip("Set a conservative CPU-process count based on available physical cores.")
        use_recommended.clicked.connect(lambda: self.workers.setValue(self.recommended_workers))
        grid.addWidget(QLabel("CPU execution"), 4, 0)
        grid.addWidget(use_recommended, 4, 1)
        grid.addWidget(QLabel("Result array directory"), 5, 0)
        grid.addWidget(output_widget, 5, 1, 1, 3)
        grid.addWidget(QLabel("Primary algorithms"), 6, 0)
        grid.addWidget(self.selected, 6, 1, 1, 3)
        grid.addWidget(self.plan_summary, 7, 0, 1, 4)
        grid.addWidget(self.execution_note, 8, 0, 1, 4)
        grid.setColumnMinimumWidth(0, 130)
        grid.setColumnMinimumWidth(2, 150)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self.body_layout.addWidget(self.setup_card)

        # Fairness is intentionally placed before execution.  The disabled run buttons
        # below make the required order unambiguous: configure -> audit -> execute.
        self.fairness_card = SectionCard(
            "2. Fairness audit",
            "Verify that all selected algorithms use the same case, objective, scenarios, constraints, seeds, and comparison budget before execution.",
        )
        audit_actions = QHBoxLayout()
        self.audit_button = QPushButton("Run fairness audit")
        self.audit_button.setObjectName("PrimaryButton")
        self.audit_button.setMinimumHeight(36)
        self.audit_button.clicked.connect(self.run_fairness_audit)
        self.audit_state = QLabel("Required before execution")
        self.audit_state.setObjectName("InfoText")
        self.audit_state.setWordWrap(True)
        audit_actions.addWidget(self.audit_button)
        audit_actions.addWidget(self.audit_state, 1)
        self.fairness_card.layout_root.addLayout(audit_actions)
        self.audit = QPlainTextEdit()
        self.audit.setReadOnly(True)
        self.audit.setMinimumHeight(150)
        self.audit.setPlaceholderText("The fairness report will appear here.")
        self.fairness_card.layout_root.addWidget(self.audit)
        self.body_layout.addWidget(self.fairness_card)

        self.execution_card = SectionCard(
            "3. Run study",
            "Execution becomes available only after the fairness audit passes for the current configuration.",
        )
        buttons = QHBoxLayout()
        self.compare = QPushButton("Run Primary Algorithm Comparison")
        self.compare.setObjectName("PrimaryButton")
        self.calo = QPushButton("Run CALO Ablation Study")
        self.compare.setToolTip("Run exactly the primary algorithms selected on the Algorithms page.")
        self.calo.setToolTip("Run seven fixed CALO/TLBO ablation variants. Primary algorithm checkboxes are not used by this study.")
        self.cancel = QPushButton("Cancel")
        self.cancel.setEnabled(False)
        self.compare.setEnabled(False)
        self.calo.setEnabled(False)
        for button in (self.compare, self.calo, self.cancel):
            button.setMinimumHeight(36)
        self.compare.clicked.connect(self.start_comparison)
        self.calo.clicked.connect(self.start_calo)
        self.cancel.clicked.connect(self.cancel_requested)
        buttons.addWidget(self.compare)
        buttons.addWidget(self.calo)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel)
        self.execution_card.layout_root.addLayout(buttons)
        self.status = QLabel("Complete the fairness audit above before starting an experiment. Global task progress is shown in the bottom status bar.")
        self.status.setWordWrap(True)
        self.status.setObjectName("InfoText")
        self.execution_card.layout_root.addWidget(self.status)
        self.body_layout.addWidget(self.execution_card)

        self.queue_card = SectionCard(
            "Run queue",
            "The exact algorithm/run jobs for the active study are listed here.",
        )
        self.queue = QTableWidget(0, 3)
        self.queue.setMinimumHeight(280)
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
        self.queue_card.layout_root.addWidget(self.queue)
        self.body_layout.addWidget(self.queue_card)
        self.body_layout.addStretch(1)

        manager.started.connect(self.on_started)
        manager.progress.connect(self.on_progress)
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
                widget.valueChanged.connect(self._update_plan_summary)
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._invalidate_fairness)
                widget.currentIndexChanged.connect(self._update_plan_summary)
        self.output.textChanged.connect(self._invalidate_fairness)
        state.config_changed.connect(lambda _: self.refresh())
        self.refresh()
        self._set_running(manager.running)

    @staticmethod
    def _recommended_worker_count() -> int:
        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        # Leave one physical core responsive where possible and cap the default to avoid excessive
        # per-process memory use from NumPy/SciPy/PyTorch imports.
        return max(1, min(12, physical - 1 if physical > 2 else physical))

    def _update_plan_summary(self, *_args) -> None:
        runs = int(self.runs.value())
        selected_count = len(self.state.config.algorithms)
        comparison_jobs = runs * selected_count
        ablation_jobs = runs * len(labels_for_mode(self.state.config, ABLATION_MODE))
        self.plan_summary.setText(
            f"Planned primary comparison: {selected_count} selected algorithms × {runs} runs = {comparison_jobs} jobs. "
            f"CALO ablation study: 7 fixed variants × {runs} runs = {ablation_jobs} jobs."
        )
        workers = int(self.workers.value())
        if workers <= 1:
            self.execution_note.setText(
                "Single-worker mode is scientifically valid but will leave most CPU cores idle. "
                f"For faster throughput, use approximately {self.recommended_workers} workers. GPU and disk activity are expected to remain low because AC power flow and the baseline metaheuristics are CPU-bound; CALO's policy network is intentionally small."
            )
        else:
            self.execution_note.setText(
                f"Parallel throughput mode will run up to {workers} independent optimizer jobs in separate CPU processes. "
                "This improves benchmark throughput. Per-run wall-clock times are then affected by CPU contention, so use one worker for strict publication-quality runtime comparisons. GPU activity is not expected to be high for this workload."
            )

    def _invalidate_fairness(self, *_args) -> None:
        if self.manager.running:
            return
        self.fairness_passed = False
        self.compare.setEnabled(False)
        self.calo.setEnabled(False)
        self.audit_state.setText("Configuration changed — audit required")
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
        self.selected.setText(f"{len(config.algorithms)} selected: " + ", ".join(config.algorithms))
        self._controls()
        self._update_plan_summary()

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
            self.audit_state.setText("Audit could not be completed")
            task.fail(str(exc))
            return False
        lines = [
            "PASS: comparative protocol is internally consistent."
            if report.fair
            else "FAIL: comparative protocol requires correction.",
            f"PRIMARY COMPARISON PLAN: {len(self.state.config.algorithms)} selected algorithms × {self.state.config.runs} runs = {planned_item_count(self.state.config, COMPARISON_MODE)} jobs.",
            f"CALO ABLATION PLAN: 7 fixed variants × {self.state.config.runs} runs = {planned_item_count(self.state.config, ABLATION_MODE)} jobs; this study intentionally ignores the primary algorithm checkbox selection.",
        ]
        lines.extend(f"ERROR: {message}" for message in report.errors)
        lines.extend(f"NOTICE: {message}" for message in report.warnings)
        self.audit.setPlainText("\n".join(lines))
        self.fairness_passed = bool(report.fair)
        self.compare.setEnabled(self.fairness_passed and not self.manager.running)
        self.calo.setEnabled(self.fairness_passed and not self.manager.running)
        if self.fairness_passed:
            self.audit_state.setText("Passed — study execution unlocked")
            self.status.setText("Fairness audit passed. Step 3 is now available: run the primary comparison or CALO ablation study.")
            task.finish("Fairness audit passed")
        else:
            self.audit_state.setText("Failed — correct the reported issues")
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
        labels = list(labels_for_mode(self.state.config, COMPARISON_MODE))
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
        answer = QMessageBox.question(
            self,
            "Run CALO ablation study",
            "This is not the 20-algorithm comparison. It runs seven fixed CALO/TLBO ablation variants and intentionally ignores the primary algorithm checkbox selection. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        labels = list(labels_for_mode(self.state.config, ABLATION_MODE))
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
        if running:
            self.audit_state.setText("Locked while experiment is running")
        elif self.fairness_passed:
            self.audit_state.setText("Passed — study execution unlocked")

    def _mark_job(self, run_index: int, algorithm: str, status: str) -> None:
        for row in range(self.queue.rowCount()):
            run_item = self.queue.item(row, 0)
            algorithm_item = self.queue.item(row, 1)
            status_item = self.queue.item(row, 2)
            if (
                run_item is not None
                and algorithm_item is not None
                and status_item is not None
                and run_item.text() == str(run_index)
                and algorithm_item.text() == algorithm
            ):
                status_item.setText(status)
                return

    def on_progress(self, data: dict) -> None:
        if data.get("phase") in {"run_completed", "run_failed"}:
            return
        algorithm = str(data.get("algorithm", ""))
        run_index = int(data.get("run_index", 0) or 0)
        if algorithm and run_index > 0:
            self._mark_job(run_index, algorithm, "Active")

    def on_started(self, experiment_id: str) -> None:
        self._set_running(True)
        workers = self.state.config.parallel_workers
        self.status.setText(
            f"Experiment {experiment_id} is running with {workers} CPU worker{'s' if workers != 1 else ''}. "
            f"Planned jobs: {self.expected_runs}."
        )

    def on_run_completed(self, run_id: str, algorithm: str, run_index: int) -> None:
        self.completed_runs += 1
        self._mark_job(run_index, algorithm, "Completed")
        self._update_status(f"Latest completed: {algorithm}.")

    def on_run_failed(self, failure_id: str, algorithm: str, run_index: int) -> None:
        self.failed_runs += 1
        self._mark_job(run_index, algorithm, "Failed")
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
