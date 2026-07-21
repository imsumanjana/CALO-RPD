"""Frozen v2 benchmark campaign and Transactions evidence workspace."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from calo_rpd_studio.algorithms.registry import primary_algorithm_names
from calo_rpd_studio.benchmarking.campaign import (
    BenchmarkCampaignConfig,
    build_campaign,
    write_campaign_plan,
)
from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.benchmarking.package import TransactionsPackageBuilder
from calo_rpd_studio.benchmarking.validation import validate_campaign
from calo_rpd_studio.benchmarking.suite import standard_benchmark_suite
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


class BenchmarkCampaignPanel(WorkspacePage):
    """Configure and execute the final frozen 20-algorithm benchmark campaign."""

    def __init__(self, state, experiment_manager, parent=None) -> None:
        super().__init__(
            "Benchmark & Evidence",
            "Freeze CALO, execute the full 20-algorithm final TEST campaign, and generate Transactions-level statistical and reproducibility evidence.",
            parent,
        )
        self.state = state
        self.experiment_manager = experiment_manager
        self.suite = standard_benchmark_suite()
        self._tasks = []
        self._task_cursor = -1
        self._campaign_active = False
        self._manifest_path: Path | None = None
        self._current_experiment_id = ""

        freeze_card = SectionCard(
            "A. Frozen CALO gate",
            "Final TEST execution is blocked unless the release freeze manifest matches the CALO equations, operators, state, archives, PPO architecture, policy checkpoint, training snapshot, hyperparameters, decoder, and feasibility rules.",
        )
        freeze_row = QHBoxLayout()
        self.freeze_path = QLineEdit(
            str(Path(__file__).resolve().parents[2] / "data" / "frozen" / "calo_v500_freeze.json")
        )
        self.freeze_status = QLabel("Not verified")
        verify = QPushButton("Verify frozen CALO")
        verify.setObjectName("PrimaryButton")
        verify.clicked.connect(self.verify_freeze)
        freeze_row.addWidget(self.freeze_path, 1)
        freeze_row.addWidget(verify)
        freeze_row.addWidget(self.freeze_status)
        freeze_card.layout_root.addLayout(freeze_row)
        self.layout_root.addWidget(freeze_card)

        design_card = SectionCard(
            "B. Final benchmark design",
            "The final campaign always uses all 20 primary algorithms, equal objective-function evaluation budgets, shared run seeds within each task, and 30–50 independent runs.",
        )
        grid = QGridLayout()
        self.case_checks: dict[str, QCheckBox] = {}
        case_box = QGroupBox("Benchmark systems")
        case_layout = QVBoxLayout(case_box)
        for case in self.suite.cases:
            check = QCheckBox(case.upper())
            check.setChecked(True)
            self.case_checks[case] = check
            case_layout.addWidget(check)
        grid.addWidget(case_box, 0, 0)

        study_box = QGroupBox("Study matrix")
        study_layout = QVBoxLayout(study_box)
        default_studies = {
            "deterministic",
            "mixed",
            "load_mean_risk",
            "renewable_cvar",
            "branch_worst_case",
        }
        self.study_checks: dict[str, QCheckBox] = {}
        for study in self.suite.studies:
            check = QCheckBox(study.label)
            check.setToolTip(study.description)
            check.setChecked(study.key in default_studies)
            self.study_checks[study.key] = check
            study_layout.addWidget(check)
        grid.addWidget(study_box, 0, 1)

        numeric_box = QGroupBox("Campaign controls")
        numeric = QGridLayout(numeric_box)
        self.runs = QSpinBox()
        self.runs.setRange(30, 50)
        self.runs.setValue(30)
        self.evaluations = QSpinBox()
        self.evaluations.setRange(100, 10_000_000)
        self.evaluations.setValue(5000)
        self.population = QSpinBox()
        self.population.setRange(5, 10000)
        self.population.setValue(50)
        self.master_seed = QSpinBox()
        self.master_seed.setRange(0, 2_147_483_647)
        self.master_seed.setValue(2026)
        self.workers = QSpinBox()
        self.workers.setRange(1, 256)
        self.workers.setValue(max(1, int(self.state.config.parallel_workers)))
        self.output_directory = QLineEdit("benchmark_v500")
        numeric.addWidget(QLabel("Independent runs / algorithm / task"), 0, 0)
        numeric.addWidget(self.runs, 0, 1)
        numeric.addWidget(QLabel("Evaluation budget"), 1, 0)
        numeric.addWidget(self.evaluations, 1, 1)
        numeric.addWidget(QLabel("Population size"), 2, 0)
        numeric.addWidget(self.population, 2, 1)
        numeric.addWidget(QLabel("Campaign master seed"), 3, 0)
        numeric.addWidget(self.master_seed, 3, 1)
        numeric.addWidget(QLabel("Parallel workers"), 4, 0)
        numeric.addWidget(self.workers, 4, 1)
        numeric.addWidget(QLabel("Output directory"), 5, 0)
        numeric.addWidget(self.output_directory, 5, 1)
        grid.addWidget(numeric_box, 0, 2)
        design_card.layout_root.addLayout(grid)

        buttons = QHBoxLayout()
        self.plan_button = QPushButton("Build frozen campaign plan")
        self.plan_button.clicked.connect(self.build_plan)
        self.start_button = QPushButton("Start full 20-algorithm TEST campaign")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_campaign)
        self.cancel_button = QPushButton("Cancel after active jobs stop safely")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.experiment_manager.cancel)
        buttons.addWidget(self.plan_button)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        design_card.layout_root.addLayout(buttons)
        self.layout_root.addWidget(design_card)

        queue_card = SectionCard(
            "C. Campaign task queue",
            "Each row is a complete 20-algorithm repeated-run experiment. Final TEST experiments are automatically locked out of historical learning.",
        )
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["#", "Task", "Case", "Study", "Jobs", "Status", "Experiment ID"]
        )
        self.table.setMinimumHeight(260)
        queue_card.layout_root.addWidget(self.table)
        self.layout_root.addWidget(queue_card)

        package_card = SectionCard(
            "D. Transactions-level research package",
            "Generate verified tables, advanced publication figures, global nonparametric statistics, evidence-based interpretation, raw run records, experiment configurations, validation status, frozen CALO manifest, and a reproducibility archive.",
        )
        package_row = QHBoxLayout()
        self.package_manifest = QLineEdit("benchmark_v500/campaign_manifest.json")
        browse = QPushButton("Load campaign manifest")
        browse.clicked.connect(self.choose_manifest)
        validate_button = QPushButton("Validate completed campaign")
        validate_button.clicked.connect(self.validate_completed_campaign)
        build_package = QPushButton("Generate Transactions research package")
        build_package.setObjectName("PrimaryButton")
        build_package.clicked.connect(self.generate_package)
        package_row.addWidget(self.package_manifest, 1)
        package_row.addWidget(browse)
        package_row.addWidget(validate_button)
        package_row.addWidget(build_package)
        package_card.layout_root.addLayout(package_row)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        package_card.layout_root.addWidget(self.log)
        self.layout_root.addWidget(package_card)

        experiment_manager.started.connect(self._on_experiment_started)
        experiment_manager.completed.connect(self._on_experiment_completed)
        experiment_manager.cancelled.connect(self._on_experiment_cancelled)
        experiment_manager.failed.connect(self._on_experiment_failed)

    def _selected_cases(self) -> tuple[str, ...]:
        return tuple(key for key, check in self.case_checks.items() if check.isChecked())

    def _selected_studies(self) -> tuple[str, ...]:
        return tuple(key for key, check in self.study_checks.items() if check.isChecked())

    def campaign_config(self) -> BenchmarkCampaignConfig:
        return BenchmarkCampaignConfig(
            cases=self._selected_cases(),
            study_keys=self._selected_studies(),
            runs=self.runs.value(),
            max_evaluations=self.evaluations.value(),
            population_size=self.population.value(),
            master_seed=self.master_seed.value(),
            output_directory=self.output_directory.text().strip() or "benchmark_v500",
            parallel_workers=self.workers.value(),
            execution_backend=self.state.config.execution_backend,
            freeze_manifest=self.freeze_path.text().strip(),
            algorithms=primary_algorithm_names(),
        )

    def verify_freeze(self) -> bool:
        result = verify_freeze_manifest(self.freeze_path.text().strip())
        self.freeze_status.setText("VERIFIED" if result.passed else "FAILED")
        self.freeze_status.setToolTip(result.message)
        self.log.append(result.message)
        return result.passed

    def build_plan(self) -> None:
        try:
            if not self._selected_cases() or not self._selected_studies():
                raise ValueError("Select at least one benchmark system and one study.")
            campaign = self.campaign_config()
            tasks = build_campaign(
                campaign, base_config=deepcopy(self.state.config), suite=self.suite
            )
            output = Path(campaign.output_directory)
            output.mkdir(parents=True, exist_ok=True)
            self._manifest_path = write_campaign_plan(
                campaign, tasks, output / "campaign_manifest.json"
            )
            self.package_manifest.setText(str(self._manifest_path))
            self._tasks = tasks
            self._task_cursor = -1
            self.table.setRowCount(len(tasks))
            for row, task in enumerate(tasks):
                values = [
                    task.task_index + 1,
                    task.task_id,
                    task.case_name,
                    task.study_label,
                    task.planned_jobs,
                    "Planned",
                    "",
                ]
                for column, value in enumerate(values):
                    self.table.setItem(row, column, QTableWidgetItem(str(value)))
            total_jobs = sum(task.planned_jobs for task in tasks)
            self.log.append(
                f"Campaign plan created: {len(tasks)} tasks, {len(primary_algorithm_names())} algorithms, {campaign.runs} runs per algorithm/task, {total_jobs:,} independent optimizer jobs."
            )
            self.start_button.setEnabled(True)
        except Exception as exc:
            self.start_button.setEnabled(False)
            QMessageBox.critical(self, "Campaign plan failed", str(exc))

    def start_campaign(self) -> None:
        if not self._tasks:
            self.build_plan()
            if not self._tasks:
                return
        if not self.verify_freeze():
            QMessageBox.critical(
                self,
                "Frozen CALO verification failed",
                "The final TEST campaign cannot start until the frozen CALO manifest verifies successfully.",
            )
            return
        answer = QMessageBox.question(
            self,
            "Start final TEST campaign",
            "This starts the frozen final benchmark. All created experiments will be classified and locked as TEST and will be ineligible for historical learning. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._campaign_active = True
        self._task_cursor = -1
        self.start_button.setEnabled(False)
        self.plan_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._start_next_task()

    def _start_next_task(self) -> None:
        if not self._campaign_active:
            return
        self._task_cursor += 1
        if self._task_cursor >= len(self._tasks):
            self._campaign_active = False
            self.cancel_button.setEnabled(False)
            self.plan_button.setEnabled(True)
            self.log.append(
                "Final benchmark campaign completed. Generate the Transactions research package after independent validation is complete."
            )
            return
        task = self._tasks[self._task_cursor]
        self.table.setItem(self._task_cursor, 5, QTableWidgetItem("Starting"))
        self.state.config = deepcopy(task.config)
        self.state.update_config()
        started = self.experiment_manager.start_comparison(task.config)
        if not started:
            self._campaign_active = False
            self.table.setItem(self._task_cursor, 5, QTableWidgetItem("Blocked"))
            self.cancel_button.setEnabled(False)
            self.plan_button.setEnabled(True)

    def _update_manifest_task(
        self, *, experiment_id: str | None = None, status: str | None = None
    ) -> None:
        if self._manifest_path is None or self._task_cursor < 0:
            return
        payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        task = payload["tasks"][self._task_cursor]
        if experiment_id is not None:
            task["experiment_id"] = experiment_id
        if status is not None:
            task["status"] = status
        self._manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _on_experiment_started(self, experiment_id: str) -> None:
        if not self._campaign_active or self._task_cursor < 0:
            return
        self._current_experiment_id = experiment_id
        self.state.database.set_experiment_learning_role(
            experiment_id,
            "test",
            eligible=False,
            locked=True,
        )
        self.table.setItem(self._task_cursor, 5, QTableWidgetItem("Running · TEST locked"))
        self.table.setItem(self._task_cursor, 6, QTableWidgetItem(experiment_id))
        self._update_manifest_task(experiment_id=experiment_id, status="running")

    def _on_experiment_completed(self, experiment_id: str) -> None:
        if not self._campaign_active or experiment_id != self._current_experiment_id:
            return
        failures = self.state.database.list_failures(experiment_id)
        status = "completed" if not failures else "completed_with_failures"
        label = "Completed" if not failures else f"Completed with {len(failures)} failure(s)"
        self.table.setItem(self._task_cursor, 5, QTableWidgetItem(label))
        self._update_manifest_task(experiment_id=experiment_id, status=status)
        self.log.append(
            f"Completed task {self._tasks[self._task_cursor].task_id} · {experiment_id} · {label}"
        )
        QTimer.singleShot(250, self._start_next_task)

    def _on_experiment_cancelled(self, experiment_id: str) -> None:
        if not self._campaign_active or experiment_id != self._current_experiment_id:
            return
        self._campaign_active = False
        self.table.setItem(self._task_cursor, 5, QTableWidgetItem("Cancelled"))
        self._update_manifest_task(experiment_id=experiment_id, status="cancelled")
        self.cancel_button.setEnabled(False)
        self.plan_button.setEnabled(True)
        self.start_button.setEnabled(True)

    def _on_experiment_failed(self, message: str) -> None:
        if not self._campaign_active:
            return
        self._campaign_active = False
        if self._task_cursor >= 0:
            self.table.setItem(self._task_cursor, 5, QTableWidgetItem("Failed"))
            self._update_manifest_task(status="failed")
        self.cancel_button.setEnabled(False)
        self.plan_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.log.append("Campaign stopped after failure: " + message)

    def choose_manifest(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open campaign manifest", "", "JSON (*.json)")
        if path:
            self.package_manifest.setText(path)

    def validate_completed_campaign(self) -> None:
        task = self.state.task_status
        if not task.begin(
            "Validating final benchmark campaign",
            detail="Reconstructing stored solutions independently",
            progress=0,
        ):
            return
        QApplication.processEvents()
        try:
            manifest = Path(self.package_manifest.text().strip())

            def progress(payload: dict) -> None:
                task.update(
                    payload.get("percent", 0),
                    f"{payload.get('completed', 0)}/{payload.get('total', 0)} runs · passed {payload.get('passed', 0)} · failed {payload.get('failed', 0)} · {payload.get('algorithm', '')}",
                )
                QApplication.processEvents()

            summary = validate_campaign(
                self.state.database,
                manifest,
                only_unverified=True,
                progress_callback=progress,
            )
            self.state.runs_changed.emit()
            self.log.append(
                "Campaign validation completed: "
                f"{summary['passed']} passed, {summary['failed']} failed, "
                f"{summary['validated']} newly validated."
            )
            task.finish("Campaign validation completed")
        except Exception as exc:
            task.fail(str(exc))
            QMessageBox.critical(self, "Campaign validation failed", str(exc))

    def generate_package(self) -> None:
        task = self.state.task_status
        if not task.begin(
            "Generating Transactions research package",
            detail="Collecting completed benchmark evidence",
        ):
            return
        QApplication.processEvents()
        try:
            manifest = Path(self.package_manifest.text().strip())
            output = manifest.parent / "transactions_research_package"
            archive = TransactionsPackageBuilder(self.state.database).build(
                campaign_manifest=manifest,
                output_directory=output,
                freeze_manifest=self.freeze_path.text().strip(),
            )
            self.log.append(f"Transactions research package created: {archive.resolve()}")
            task.finish("Transactions research package generated")
        except Exception as exc:
            task.fail(str(exc))
            QMessageBox.critical(self, "Evidence package failed", str(exc))
