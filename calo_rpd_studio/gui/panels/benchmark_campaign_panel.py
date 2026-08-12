"""Frozen benchmark campaign and comprehensive evidence workspace."""

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
    QWidget,
)

from calo_rpd_studio.algorithms.registry import primary_algorithm_names
from calo_rpd_studio.benchmarking.campaign import (
    BenchmarkCampaignConfig,
    build_campaign,
    verify_campaign_plan_design,
    write_campaign_plan,
)
from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.benchmarking.package import ScientificEvidencePackageBuilder
from calo_rpd_studio.benchmarking.validation import validate_campaign
from calo_rpd_studio.benchmarking.suite import standard_benchmark_suite
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage
from calo_rpd_studio.gui.widgets.workspace_tabs import WorkspaceTabs
from calo_rpd_studio.version import FREEZE_MANIFEST


class BenchmarkCampaignPanel(WorkspacePage):
    """Configure and execute a frozen, power-aware benchmark campaign."""

    def __init__(self, state, experiment_manager, parent=None) -> None:
        super().__init__(
            "Benchmark & Evidence",
            "Verify the CALO software freeze, execute a preregistered held-out campaign, and "
            "generate comprehensive statistical and reproducibility evidence.",
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

        freeze_page = QWidget()
        freeze_layout = QVBoxLayout(freeze_page)
        freeze_layout.setContentsMargins(18, 18, 18, 18)
        freeze_description = QLabel(
            "Held-out execution is blocked unless the software freeze matches the CALO equations, "
            "operators, state, archives, policy architecture, training semantics, hyperparameters, "
            "decoder, and feasibility rules. The governing policy is separately bound by explicit "
            "artifact SHA-256."
        )
        freeze_description.setWordWrap(True)
        freeze_layout.addWidget(freeze_description)
        freeze_row = QHBoxLayout()
        self.freeze_path = QLineEdit(
            str(Path(__file__).resolve().parents[2] / "data" / "frozen" / FREEZE_MANIFEST)
        )
        self.freeze_path.setProperty("fullWidthInput", True)
        self.freeze_path.setAccessibleName("Frozen CALO manifest path")
        self.freeze_path.setToolTip(self.freeze_path.text())
        self.freeze_status = QLabel("Not verified")
        verify = QPushButton("Verify frozen CALO")
        verify.setObjectName("PrimaryButton")
        verify.clicked.connect(self.verify_freeze)
        freeze_row.addWidget(self.freeze_path, 1)
        freeze_row.addWidget(verify)
        freeze_row.addWidget(self.freeze_status)
        freeze_layout.addLayout(freeze_row)
        freeze_layout.addStretch(1)

        design_page = QWidget()
        design_layout = QVBoxLayout(design_page)
        design_layout.setContentsMargins(18, 18, 18, 18)
        design_description = QLabel(
            "The campaign uses the frozen registered comparator set, equal objective-function "
            "evaluation budgets, paired run seeds, and a multiplicity-aware powered run plan."
        )
        design_description.setWordWrap(True)
        design_layout.addWidget(design_description)
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
        recommended_runs = int(BenchmarkCampaignConfig().runs)
        self.runs.setRange(2, 10_000)
        self.runs.setValue(recommended_runs)
        self.runs.setToolTip(
            "Default powered approximation for the current CALO-versus-comparator family. "
            "Replace it only with a preregistered pilot/simulation design."
        )
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
        self.output_directory = QLineEdit("benchmark_v600a4")
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
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 2)
        design_layout.addLayout(grid, 1)

        buttons = QHBoxLayout()
        self.plan_button = QPushButton("Build frozen campaign plan")
        self.plan_button.clicked.connect(self.build_plan)
        self.start_button = QPushButton("Start frozen held-out campaign")
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
        design_layout.addLayout(buttons)

        queue_page = QWidget()
        queue_layout = QVBoxLayout(queue_page)
        queue_layout.setContentsMargins(18, 18, 18, 18)
        queue_description = QLabel(
            "Each row is a complete repeated-run comparison. Held-out experiments are "
            "automatically locked out of historical learning."
        )
        queue_description.setWordWrap(True)
        queue_layout.addWidget(queue_description)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["#", "Task", "Case", "Evidence role", "Study", "Jobs", "Status", "Experiment ID"]
        )
        self.table.setMinimumHeight(260)
        queue_layout.addWidget(self.table, 1)

        package_page = QWidget()
        package_layout = QVBoxLayout(package_page)
        package_layout.setContentsMargins(18, 18, 18, 18)
        package_description = QLabel(
            "Generate verified tables, advanced publication figures, global nonparametric "
            "statistics, evidence-based interpretation, raw run records, experiment "
            "configurations, validation status, frozen CALO manifest, and a reproducibility archive."
        )
        package_description.setWordWrap(True)
        package_layout.addWidget(package_description)
        package_row = QHBoxLayout()
        self.package_manifest = QLineEdit("benchmark_v600a4/campaign_manifest.json")
        self.package_manifest.setProperty("fullWidthInput", True)
        self.package_manifest.setAccessibleName("Completed campaign manifest path")
        browse = QPushButton("Load campaign manifest")
        browse.clicked.connect(self.choose_manifest)
        validate_button = QPushButton("Validate completed campaign")
        validate_button.clicked.connect(self.validate_completed_campaign)
        build_package = QPushButton("Generate scientific evidence package")
        build_package.setObjectName("PrimaryButton")
        build_package.clicked.connect(self.generate_package)
        package_row.addWidget(self.package_manifest, 1)
        package_row.addWidget(browse)
        package_row.addWidget(validate_button)
        package_row.addWidget(build_package)
        package_layout.addLayout(package_row)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        package_layout.addWidget(self.log, 1)

        self.section_tabs = WorkspaceTabs("Benchmark and evidence sections")
        self.section_tabs.add_section(
            "Freeze gate",
            freeze_page,
            "Verify the frozen CALO manifest before held-out execution.",
        )
        self.section_tabs.add_section(
            "Campaign design",
            design_page,
            "Choose benchmark systems, studies, budgets, seeds, and execution controls.",
        )
        self.section_tabs.add_section(
            "Task queue",
            queue_page,
            "Monitor complete repeated-run comparison tasks.",
        )
        self.section_tabs.add_section(
            "Evidence package",
            package_page,
            "Validate a completed campaign and generate its reproducibility package.",
        )
        self.layout_root.addWidget(self.section_tabs, 1)

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
            output_directory=self.output_directory.text().strip() or "benchmark_v600a4",
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
                    "Protected test" if task.evidence_role == "test" else "Validation replay",
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
                "The confirmatory campaign cannot start until the frozen CALO manifest verifies successfully.",
            )
            return
        if self._manifest_path is None:
            QMessageBox.critical(self, "Campaign plan", "Build the campaign plan before starting.")
            return
        design_ok, design_message = verify_campaign_plan_design(self._manifest_path)
        if not design_ok:
            QMessageBox.critical(self, "Campaign design verification failed", design_message)
            return
        answer = QMessageBox.question(
            self,
            "Start frozen confirmatory campaign",
            "This starts the frozen benchmark. Previously used systems are labeled validation "
            "replays; unseen systems are labeled protected tests. All are locked out of historical "
            "learning. Continue?",
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
        freeze = verify_freeze_manifest(self.freeze_path.text().strip())
        design_ok, design_message = (
            verify_campaign_plan_design(self._manifest_path)
            if self._manifest_path is not None
            else (False, "Campaign plan is unavailable.")
        )
        if not freeze.passed or not design_ok:
            self._campaign_active = False
            self.cancel_button.setEnabled(False)
            self.plan_button.setEnabled(True)
            self.start_button.setEnabled(True)
            message = freeze.message if not freeze.passed else design_message
            self.log.append("Campaign stopped before the next task: " + message)
            QMessageBox.critical(self, "Frozen campaign verification failed", message)
            return
        self._task_cursor += 1
        if self._task_cursor >= len(self._tasks):
            self._campaign_active = False
            self.cancel_button.setEnabled(False)
            self.plan_button.setEnabled(True)
            self.log.append(
                "Frozen validation/test campaign completed. Generate the scientific evidence "
                "package after independent validation is complete."
            )
            return
        task = self._tasks[self._task_cursor]
        self.table.setItem(self._task_cursor, 6, QTableWidgetItem("Starting"))
        self.state.config = deepcopy(task.config)
        self.state.update_config()
        started = self.experiment_manager.start_comparison(task.config)
        if not started:
            self._campaign_active = False
            self.table.setItem(self._task_cursor, 6, QTableWidgetItem("Blocked"))
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
        evidence_role = self._tasks[self._task_cursor].evidence_role
        self.state.database.set_experiment_learning_role(
            experiment_id,
            evidence_role,
            eligible=False,
            locked=True,
        )
        role_label = "TEST" if evidence_role == "test" else "VALIDATION"
        self.table.setItem(self._task_cursor, 6, QTableWidgetItem(f"Running · {role_label} locked"))
        self.table.setItem(self._task_cursor, 7, QTableWidgetItem(experiment_id))
        self._update_manifest_task(experiment_id=experiment_id, status="running")

    def _on_experiment_completed(self, experiment_id: str) -> None:
        if not self._campaign_active or experiment_id != self._current_experiment_id:
            return
        failures = self.state.database.list_failures(experiment_id)
        status = "completed" if not failures else "completed_with_failures"
        label = "Completed" if not failures else f"Completed with {len(failures)} failure(s)"
        self.table.setItem(self._task_cursor, 6, QTableWidgetItem(label))
        self._update_manifest_task(experiment_id=experiment_id, status=status)
        self.log.append(
            f"Completed task {self._tasks[self._task_cursor].task_id} · {experiment_id} · {label}"
        )
        QTimer.singleShot(250, self._start_next_task)

    def _on_experiment_cancelled(self, experiment_id: str) -> None:
        if not self._campaign_active or experiment_id != self._current_experiment_id:
            return
        self._campaign_active = False
        self.table.setItem(self._task_cursor, 6, QTableWidgetItem("Cancelled"))
        self._update_manifest_task(experiment_id=experiment_id, status="cancelled")
        self.cancel_button.setEnabled(False)
        self.plan_button.setEnabled(True)
        self.start_button.setEnabled(True)

    def _on_experiment_failed(self, message: str) -> None:
        if not self._campaign_active:
            return
        self._campaign_active = False
        if self._task_cursor >= 0:
            self.table.setItem(self._task_cursor, 6, QTableWidgetItem("Failed"))
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
            "Generating scientific evidence package",
            detail="Collecting completed benchmark evidence",
        ):
            return
        QApplication.processEvents()
        try:
            manifest = Path(self.package_manifest.text().strip())
            output = manifest.parent / "scientific_evidence_package"
            archive = ScientificEvidencePackageBuilder(self.state.database).build(
                campaign_manifest=manifest,
                output_directory=output,
                freeze_manifest=self.freeze_path.text().strip(),
            )
            self.log.append(f"Scientific evidence package created: {archive.resolve()}")
            task.finish("Scientific evidence package generated")
        except Exception as exc:
            task.fail(str(exc))
            QMessageBox.critical(self, "Evidence package failed", str(exc))
