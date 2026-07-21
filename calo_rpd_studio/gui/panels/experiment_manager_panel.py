"""Experiment configuration, fairness audit, queue status, and execution."""

from __future__ import annotations

from copy import deepcopy
import json
import os

import psutil

from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
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

from calo_rpd_studio.compute.resource_scheduler import ResourceMonitor, build_weighted_lane_plan
from calo_rpd_studio.accelerated.parity_audit import run_configuration_parity_audit
from calo_rpd_studio.experiments.evaluation_budget import BudgetPolicy
from calo_rpd_studio.portfolio.planner import PortfolioPlanner
from calo_rpd_studio.portfolio.fingerprint import run_fingerprint
from calo_rpd_studio.experiments.seed_manager import SeedManager
from calo_rpd_studio.experiments.execution_plan import (
    ABLATION_MODE,
    COMPARISON_MODE,
    build_execution_plan,
    labels_for_mode,
    planned_item_count,
)
from calo_rpd_studio.experiments.fairness_validator import validate_fairness
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage
from calo_rpd_studio.results.database import ResultDatabase


class ScientificAuditWorker(QThread):
    """Run parity, fairness, and reuse checks away from the Qt GUI thread."""

    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str, int)

    def __init__(
        self, config, database_path: str, *, parity_only: bool = False, parent=None
    ) -> None:
        super().__init__(parent)
        self.config = deepcopy(config)
        self.database_path = str(database_path)
        self.parity_only = bool(parity_only)

    @staticmethod
    def preferred_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu:0"
        except Exception:
            pass
        return "cpu"

    def run(self) -> None:
        try:
            self.progress.emit("Validating experiment configuration", 5)
            self.config.validate()
            parity = None
            if self.parity_only or bool(self.config.require_backend_parity):
                device = self.preferred_device()
                # A CPU fallback parity audit can otherwise let dense Torch/LAPACK workers consume
                # every host core, starving the Qt event loop despite this QThread. Restrict only
                # the audit worker; numerical experiments keep their configured scheduler.
                torch_module = None
                previous_threads = None
                if device == "cpu":
                    try:
                        import torch as torch_module

                        previous_threads = int(torch_module.get_num_threads())
                        torch_module.set_num_threads(1)
                    except Exception:
                        torch_module = None
                        previous_threads = None
                candidates = 1 if str(self.config.case_name) == "case300" and device == "cpu" else 5
                self.progress.emit(
                    f"Auditing CPU/accelerator parity on {device} ({candidates} deterministic candidate{'s' if candidates != 1 else ''})",
                    15,
                )
                try:
                    parity = run_configuration_parity_audit(
                        self.config,
                        device=device,
                        candidates=candidates,
                    )
                finally:
                    if torch_module is not None and previous_threads is not None:
                        try:
                            torch_module.set_num_threads(previous_threads)
                        except Exception:
                            pass
                if bool(self.config.require_backend_parity) and not bool(parity.get("passed")):
                    raise RuntimeError("CPU/accelerator numerical parity gate did not pass")
            if self.parity_only:
                self.progress.emit("Parity audit complete", 100)
                self.completed.emit({"parity_only": True, "parity": parity})
                return

            self.progress.emit("Checking comparative fairness and portfolio dependencies", 70)
            fairness = validate_fairness(self.config)
            portfolio_plan = PortfolioPlanner.plan(
                self.config, self.config.portfolio, benchmark_blocks=1
            )
            self.progress.emit("Checking reusable verified runs", 82)
            seeds = SeedManager(self.config.master_seed).generate(self.config.runs)
            reusable = 0
            if self.config.reuse_compatible_results:
                database = ResultDatabase(self.database_path)
                for item in build_execution_plan(self.config, COMPARISON_MODE):
                    fingerprint = run_fingerprint(
                        self.config, item.label, item.run_index, seeds[item.run_index]
                    )
                    if database.find_reusable_run(
                        fingerprint,
                        verified_only=bool(self.config.portfolio.require_independent_validation),
                    ):
                        reusable += 1
            self.progress.emit("Scientific audit complete", 100)
            self.completed.emit(
                {
                    "parity_only": False,
                    "parity": parity,
                    "fairness": fairness,
                    "portfolio_plan": portfolio_plan,
                    "reusable": reusable,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


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
        self.resource_monitor = ResourceMonitor()
        self.completed_runs = 0
        self.failed_runs = 0
        self.expected_runs = 0
        self.fairness_passed = False
        self.backend_parity_passed = False
        self.backend_parity_report = None
        self.audit_worker: ScientificAuditWorker | None = None

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
        self.runs.setReadOnly(True)
        self.runs.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.runs.setToolTip(
            "Derived by Portfolio Manager from the selected evidence profile and output dependencies."
        )
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
            "Maximum number of independent optimizer processes admitted at the same time."
        )
        self.execution_backend = QComboBox()
        self.execution_backend.addItem(
            "GPU maximum resident — 100% CUDA when available (recommended)", "gpu_preferred"
        )
        self.execution_backend.addItem("CUDA-only resident — require 100% CUDA", "cuda_only")
        self.execution_backend.addItem(
            "CUDA-resident priority — 80% CUDA / 10% XPU / 10% CPU", "cuda_priority"
        )
        self.execution_backend.addItem("Auto-tuned Batched Throughput Engine", "throughput_auto")
        self.execution_backend.addItem("Custom weighted split", "weighted_split")
        self.execution_backend.addItem("Adaptive hybrid CPU + GPU", "adaptive_hybrid")
        self.execution_backend.addItem("CPU only", "cpu_only")
        self.scientific_backend = QComboBox()
        self.scientific_backend.addItem(
            "PyTorch FP64 batched AC Newton-Raphson (CPU/CUDA/XPU)", "torch_fp64"
        )
        self.scientific_backend.addItem("Trusted legacy CPU reference", "cpu_reference")
        self.tensor_batch_size = QSpinBox()
        self.tensor_batch_size.setRange(1, 4096)
        self.tensor_batch_size.setToolTip(
            "Candidates per accelerator power-flow batch. Larger values improve throughput but consume more device memory."
        )
        self.auto_batch_calibration = QCheckBox(
            "Calibrate evaluator microbatch size before the campaign"
        )
        self.auto_batch_calibration.setChecked(True)
        self.persistent_workers = QCheckBox("Keep one process/context alive per compute device")
        self.persistent_workers.setChecked(True)
        self.cross_run_batching = QCheckBox(
            "Combine compatible population requests across independent runs"
        )
        self.cross_run_batching.setChecked(True)
        self.batch_window = QDoubleSpinBox()
        self.batch_window.setRange(0.1, 100.0)
        self.batch_window.setDecimals(1)
        self.batch_window.setSuffix(" ms")
        self.batch_window.setToolTip(
            "Short collection window used to combine compatible run requests into one device batch."
        )
        self.max_cross_batch = QSpinBox()
        self.max_cross_batch.setRange(16, 1_000_000)
        self.max_cross_batch.setToolTip(
            "Maximum candidates combined in one cross-run device submission."
        )
        self.calibration_repetitions = QSpinBox()
        self.calibration_repetitions.setRange(1, 20)
        self.calibration_repetitions.setToolTip(
            "Repeated timing passes per evaluator candidate microbatch size; this does not benchmark optimizer-control overhead."
        )
        self.telemetry_interval = QSpinBox()
        self.telemetry_interval.setRange(1, 10_000)
        self.telemetry_interval.setSuffix(" iterations")
        self.buffered_traces = QCheckBox("Buffer convergence traces and write in blocks")
        self.buffered_traces.setChecked(True)
        self.compile_kernels = QCheckBox("Compile stable tensor kernels when supported")
        self.compile_kernels.setToolTip(
            "Optional torch.compile path. Disabled by default because parity must be re-audited after compiler/runtime changes."
        )
        self.device_resident_execution = QCheckBox(
            "Keep optimizer, decoder, power-flow and constraint tensors resident on the assigned device"
        )
        self.device_resident_execution.setChecked(True)
        self.device_resident_execution.setToolTip(
            "v3.4 minimizes host/device round trips. One packed population result is materialized on the host for common provenance, GUI and persistence; final independent validation remains CPU-reference based."
        )
        self.cuda_priority_work_stealing = QCheckBox(
            "Allow idle CUDA capacity to take unstarted XPU/CPU work"
        )
        self.cuda_priority_work_stealing.setChecked(True)
        self.parity_gate = QCheckBox(
            "Require CPU/accelerator numerical parity before final benchmark"
        )
        self.parity_gate.setChecked(True)
        self.gpu_target = QSpinBox()
        self.gpu_target.setRange(10, 100)
        self.gpu_target.setSuffix(" %")
        self.cpu_target = QSpinBox()
        self.cpu_target.setRange(10, 100)
        self.cpu_target.setSuffix(" %")
        self.gpu_memory_limit = QSpinBox()
        self.gpu_memory_limit.setRange(20, 100)
        self.gpu_memory_limit.setSuffix(" %")
        self.gpu_jobs = QSpinBox()
        self.gpu_jobs.setRange(1, 16)
        self.xpu_target = QSpinBox()
        self.xpu_target.setRange(10, 100)
        self.xpu_target.setSuffix(" %")
        self.xpu_memory_limit = QSpinBox()
        self.xpu_memory_limit.setRange(20, 100)
        self.xpu_memory_limit.setSuffix(" %")
        self.xpu_jobs = QSpinBox()
        self.xpu_jobs.setRange(1, 16)
        self.system_memory_limit = QSpinBox()
        self.system_memory_limit.setRange(20, 100)
        self.system_memory_limit.setSuffix(" %")
        self.cuda_share = QSpinBox()
        self.cuda_share.setRange(0, 100)
        self.cuda_share.setSuffix(" %")
        self.xpu_share = QSpinBox()
        self.xpu_share.setRange(0, 100)
        self.xpu_share.setSuffix(" %")
        self.cpu_share = QSpinBox()
        self.cpu_share.setRange(0, 100)
        self.cpu_share.setSuffix(" %")
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
        self.device_inventory = QLabel()
        self.device_inventory.setWordWrap(True)
        self.device_inventory.setObjectName("InfoText")

        fields = [
            ("Independent runs", self.runs),
            ("Population size", self.population),
            ("Budget policy", self.policy),
            ("Objective evaluations", self.budget),
            ("Wall-clock budget", self.wall),
            ("Iteration safety limit", self.maxit),
            ("Parallel workers", self.workers),
            ("Master seed", self.seed),
            ("Compute scheduler", self.execution_backend),
            ("Scientific evaluator", self.scientific_backend),
            ("Manual/fallback batch size", self.tensor_batch_size),
            ("Auto batch calibration", self.auto_batch_calibration),
            ("Persistent device workers", self.persistent_workers),
            ("Cross-run batching", self.cross_run_batching),
            ("Batch collection window", self.batch_window),
            ("Maximum cross-run batch", self.max_cross_batch),
            ("Calibration repetitions", self.calibration_repetitions),
            ("Telemetry interval", self.telemetry_interval),
            ("Buffered trace writes", self.buffered_traces),
            ("Stable-kernel compilation", self.compile_kernels),
            ("Device-resident execution", self.device_resident_execution),
            ("CUDA-priority work stealing", self.cuda_priority_work_stealing),
            ("Backend parity gate", self.parity_gate),
            ("NVIDIA CUDA target", self.gpu_target),
            ("CUDA VRAM limit", self.gpu_memory_limit),
            ("Max CUDA jobs", self.gpu_jobs),
            ("Intel XPU target", self.xpu_target),
            ("XPU memory limit", self.xpu_memory_limit),
            ("Max XPU jobs", self.xpu_jobs),
            ("CPU utilization target", self.cpu_target),
            ("System RAM safety limit", self.system_memory_limit),
            ("CUDA task share", self.cuda_share),
            ("XPU task share", self.xpu_share),
            ("CPU task share", self.cpu_share),
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
        use_recommended.setToolTip(
            "Set a conservative CPU-process count based on available physical cores."
        )
        use_recommended.clicked.connect(lambda: self.workers.setValue(self.recommended_workers))
        base_row = (len(fields) + 1) // 2
        grid.addWidget(QLabel("CPU execution"), base_row, 0)
        grid.addWidget(use_recommended, base_row, 1)
        grid.addWidget(QLabel("Result array directory"), base_row + 1, 0)
        grid.addWidget(output_widget, base_row + 1, 1, 1, 3)
        grid.addWidget(QLabel("Primary algorithms"), base_row + 2, 0)
        grid.addWidget(self.selected, base_row + 2, 1, 1, 3)
        grid.addWidget(self.plan_summary, base_row + 3, 0, 1, 4)
        grid.addWidget(self.device_inventory, base_row + 4, 0, 1, 4)
        grid.addWidget(self.execution_note, base_row + 5, 0, 1, 4)
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
        self.parity_button = QPushButton("Run CPU/accelerator parity audit")
        self.parity_button.setMinimumHeight(36)
        self.parity_button.clicked.connect(self.run_backend_parity_audit)
        self.audit_state = QLabel("Required before execution")
        self.audit_state.setObjectName("InfoText")
        self.audit_state.setWordWrap(True)
        audit_actions.addWidget(self.audit_button)
        audit_actions.addWidget(self.parity_button)
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
        self.compare.setToolTip(
            "Run exactly the primary algorithms selected on the Algorithms page."
        )
        self.calo.setToolTip(
            f"Run {len(labels_for_mode(self.state.config, ABLATION_MODE))} fixed CALO/TLBO ablation variants. Primary algorithm checkboxes are not used by this study."
        )
        self.pause = QPushButton("Pause safely")
        self.pause.setEnabled(False)
        self.cancel = QPushButton("Stop immediately")
        self.cancel.setEnabled(False)
        self.compare.setEnabled(False)
        self.calo.setEnabled(False)
        for button in (self.compare, self.calo, self.pause, self.cancel):
            button.setMinimumHeight(36)
        self.compare.clicked.connect(self.start_comparison)
        self.calo.clicked.connect(self.start_calo)
        self.pause.clicked.connect(self.pause_requested)
        self.cancel.clicked.connect(self.cancel_requested)
        buttons.addWidget(self.compare)
        buttons.addWidget(self.calo)
        buttons.addStretch(1)
        buttons.addWidget(self.pause)
        buttons.addWidget(self.cancel)
        self.execution_card.layout_root.addLayout(buttons)
        self.status = QLabel(
            "Complete the fairness audit above before starting an experiment. Global task progress is shown in the bottom status bar."
        )
        self.status.setWordWrap(True)
        self.status.setObjectName("InfoText")
        self.execution_card.layout_root.addWidget(self.status)
        self.body_layout.addWidget(self.execution_card)

        self.evolution_card = SectionCard(
            "4. Experiment evolution / continuation",
            "Extend an existing experiment without overwriting earlier evidence. Adding independent runs is publication-safe. Same-run FE continuation requires exact optimizer-state checkpoints; post-hoc selected extensions are marked exploratory.",
        )
        evolution_grid = QGridLayout()
        self.extension_experiment = QLineEdit()
        self.extension_experiment.setReadOnly(True)
        self.extension_experiment.setPlaceholderText("Open/restore an experiment first")
        self.extension_runs = QSpinBox()
        self.extension_runs.setRange(1, 100_000)
        self.extend_runs_button = QPushButton("Increase independent-run target")
        self.extend_runs_button.clicked.connect(self.extend_independent_runs)
        self.extension_evaluations = QSpinBox()
        self.extension_evaluations.setRange(1, 2_000_000_000)
        self.extension_protocol = QComboBox()
        self.extension_protocol.addItem(
            "All paired algorithms and runs — publication eligible", "all_paired"
        )
        self.extension_protocol.addItem(
            "Predeclared deterministic paired subset — publication eligible", "deterministic_subset"
        )
        self.extension_protocol.addItem(
            "Manual/post-hoc selected runs — exploratory only", "manual_exploratory"
        )
        self.extension_strategy = QComboBox()
        self.extension_source_horizon = QComboBox()
        self.extension_source_horizon.setToolTip(
            "For exact CALO continuation, choose the preserved FE horizon whose optimizer checkpoint should be resumed. "
            "Recompute-from-seed ignores this field and starts a new paired trajectory at FE=0."
        )
        self.extension_strategy.addItem(
            "Recompute from original paired seeds at new horizon — publication-safe for all algorithms",
            "recompute_from_seed",
        )
        self.extension_strategy.addItem(
            "Exact optimizer-state continuation — CALO checkpoint trajectories only",
            "exact_continue",
        )
        self.extension_strategy.currentIndexChanged.connect(
            lambda *_: self.extension_source_horizon.setEnabled(
                str(self.extension_strategy.currentData()) == "exact_continue"
            )
        )
        self.extension_source_horizon.setEnabled(False)
        self.extension_run_indices = QLineEdit()
        self.extension_run_indices.setPlaceholderText(
            "Run numbers, e.g. 1,6,11 (blank = all where protocol permits)"
        )
        self.extension_algorithms = QLineEdit()
        self.extension_algorithms.setPlaceholderText("Algorithms, e.g. CALO (blank = all)")
        self.extend_horizon_button = QPushButton("Extend evaluation horizon")
        self.extend_horizon_button.clicked.connect(self.extend_evaluation_horizon)
        evolution_grid.addWidget(QLabel("Experiment ID"), 0, 0)
        evolution_grid.addWidget(self.extension_experiment, 0, 1, 1, 3)
        evolution_grid.addWidget(QLabel("New total independent runs"), 1, 0)
        evolution_grid.addWidget(self.extension_runs, 1, 1)
        evolution_grid.addWidget(self.extend_runs_button, 1, 2, 1, 2)
        evolution_grid.addWidget(QLabel("New FE horizon"), 2, 0)
        evolution_grid.addWidget(self.extension_evaluations, 2, 1)
        evolution_grid.addWidget(QLabel("Extension protocol"), 2, 2)
        evolution_grid.addWidget(self.extension_protocol, 2, 3)
        evolution_grid.addWidget(QLabel("Execution strategy"), 3, 0)
        evolution_grid.addWidget(self.extension_strategy, 3, 1, 1, 3)
        evolution_grid.addWidget(QLabel("Exact-continuation source horizon"), 4, 0)
        evolution_grid.addWidget(self.extension_source_horizon, 4, 1, 1, 3)
        evolution_grid.addWidget(QLabel("Selected run numbers"), 5, 0)
        evolution_grid.addWidget(self.extension_run_indices, 5, 1)
        evolution_grid.addWidget(QLabel("Selected algorithms"), 5, 2)
        evolution_grid.addWidget(self.extension_algorithms, 5, 3)
        evolution_grid.addWidget(self.extend_horizon_button, 6, 2, 1, 2)
        self.evolution_card.layout_root.addLayout(evolution_grid)
        self.evolution_note = QLabel(
            "Two scientifically distinct horizon modes are available. Recompute-from-seed creates a new paired evidence horizon for all algorithms while preserving older evidence. Exact continuation resumes CALO's complete optimizer checkpoint from the explicitly selected preserved source horizon and creates a segmented/branched trajectory; it is never silently substituted for baseline algorithms."
        )
        self.evolution_note.setWordWrap(True)
        self.evolution_note.setObjectName("HelpText")
        self.evolution_card.layout_root.addWidget(self.evolution_note)
        self.revision_table = QTableWidget(0, 7)
        self.revision_table.setHorizontalHeaderLabels(
            ["Revision", "Mode", "Runs", "FE horizon", "Primary-stat eligible", "Status", "Created"]
        )
        self.revision_table.setMinimumHeight(150)
        self.revision_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.revision_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.evolution_card.layout_root.addWidget(self.revision_table)
        self.body_layout.addWidget(self.evolution_card)

        self.queue_card = SectionCard(
            "Run queue",
            "The exact algorithm/run jobs for the active study are listed here.",
        )
        self.queue = QTableWidget(0, 4)
        self.queue.setMinimumHeight(280)
        self.queue.setHorizontalHeaderLabels(
            ["Run", "Algorithm / CALO variant", "Planned lane", "Status"]
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
        self.queue.horizontalHeader().setSectionResizeMode(
            3,
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
        self.execution_backend.currentIndexChanged.connect(self._controls)
        self.scientific_backend.currentIndexChanged.connect(self._controls)
        self.auto_batch_calibration.stateChanged.connect(self._controls)
        for widget in (
            self.runs,
            self.population,
            self.policy,
            self.budget,
            self.wall,
            self.maxit,
            self.workers,
            self.seed,
            self.execution_backend,
            self.gpu_target,
            self.cpu_target,
            self.gpu_memory_limit,
            self.gpu_jobs,
            self.xpu_target,
            self.xpu_memory_limit,
            self.xpu_jobs,
            self.system_memory_limit,
            self.cuda_share,
            self.xpu_share,
            self.cpu_share,
            self.scientific_backend,
            self.tensor_batch_size,
            self.batch_window,
            self.max_cross_batch,
            self.calibration_repetitions,
            self.telemetry_interval,
        ):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._invalidate_fairness)
                widget.valueChanged.connect(self._update_plan_summary)
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._invalidate_fairness)
                widget.currentIndexChanged.connect(self._update_plan_summary)
        for checkbox in (
            self.parity_gate,
            self.auto_batch_calibration,
            self.persistent_workers,
            self.cross_run_batching,
            self.buffered_traces,
            self.compile_kernels,
            self.device_resident_execution,
            self.cuda_priority_work_stealing,
        ):
            checkbox.stateChanged.connect(self._invalidate_fairness)
            checkbox.stateChanged.connect(self._update_plan_summary)
        self.output.textChanged.connect(self._invalidate_fairness)
        state.config_changed.connect(lambda _: self.refresh())
        self.refresh()
        self.resource_timer = QTimer(self)
        self.resource_timer.setInterval(2000)
        self.resource_timer.timeout.connect(self._refresh_resource_status)
        self.resource_timer.start()
        self._set_running(manager.running)

    @staticmethod
    def _recommended_worker_count() -> int:
        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        # Leave one physical core responsive where possible and cap the default to avoid excessive
        # per-process memory use from NumPy/SciPy/PyTorch imports.
        return max(1, min(12, physical - 1 if physical > 2 else physical))

    def _refresh_resource_status(self) -> None:
        try:
            snapshot = self.resource_monitor.sample()
            parts = [f"CPU {snapshot.cpu_percent:.0f}% · RAM {snapshot.system_memory_percent:.0f}%"]
            for device in snapshot.devices:
                util = (
                    f"{device.utilization_percent:.0f}% compute"
                    if device.utilization_percent is not None
                    else "compute utilization unavailable"
                )
                runtime = (
                    "secondary XPU runtime" if device.runtime == "sidecar" else "primary runtime"
                )
                parts.append(
                    f"{device.device_id} — {device.name}: {util}, memory {device.memory_percent:.0f}% ({runtime})"
                )
            if not snapshot.devices:
                parts.append(
                    "No CUDA/XPU accelerator is currently available to a verified PyTorch runtime"
                )
            self.device_inventory.setText(
                "Detected compute priority: NVIDIA CUDA → Intel XPU → CPU. "
                + " | ".join(parts)
                + ". PyTorch backend IDs do not necessarily match Windows Task Manager GPU numbers."
            )
        except Exception as exc:
            self.device_inventory.setText(f"Compute resource sampling failed: {exc}")

    def _update_plan_summary(self, *_args) -> None:
        runs = int(self.runs.value())
        selected_count = len(self.state.config.algorithms)
        comparison_jobs = runs * selected_count
        ablation_jobs = runs * len(labels_for_mode(self.state.config, ABLATION_MODE))
        portfolio = getattr(self.state.config, "portfolio", None)
        portfolio_name = getattr(portfolio, "name", "Experiment portfolio")
        summary_text = (
            f"{portfolio_name}: {selected_count} selected algorithms × {runs} paired runs = {comparison_jobs} jobs. "
            f"CALO ablation study: {len(labels_for_mode(self.state.config, ABLATION_MODE))} fixed variants × {runs} runs = {ablation_jobs} jobs. "
            f"Run count is controlled by Portfolio Manager; this page configures how the required jobs execute."
        )
        current_backend = str(self.execution_backend.currentData() or "")
        if current_backend == "throughput_auto":
            summary_text += (
                " The evaluator calibration will benchmark candidate-scenario throughput only (not CALO control overhead) on each verified CUDA/XPU/CPU lane, "
                "select a stable microbatch, and allocate jobs in proportion to measured evaluations per second."
            )
        if current_backend in {"weighted_split", "cuda_priority", "cuda_only", "gpu_preferred"}:
            try:
                temp = deepcopy(self.state.config)
                temp.runs = runs
                snapshot = self.resource_monitor.sample()
                _lanes, allocation = build_weighted_lane_plan(
                    build_execution_plan(temp, COMPARISON_MODE),
                    COMPARISON_MODE,
                    cuda_available=bool(snapshot.by_backend("cuda")),
                    xpu_available=bool(snapshot.by_backend("xpu")),
                    cuda_share=self.cuda_share.value(),
                    xpu_share=self.xpu_share.value(),
                    cpu_share=self.cpu_share.value(),
                )
                summary_text += (
                    f" Current attainable primary allocation: {allocation.effective_text}; "
                    f"{allocation.accelerator_eligible_jobs}/{allocation.total_jobs} jobs are accelerator-compatible under the v3 torch FP64 backend."
                )
            except Exception:
                pass
        self.plan_summary.setText(summary_text)
        workers = int(self.workers.value())
        backend = str(self.execution_backend.currentData() or "adaptive_hybrid")
        if backend == "cpu_only":
            scheduler_text = "CPU-only scheduling is selected."
        elif backend == "throughput_auto":
            scheduler_text = (
                "The Batched Throughput Engine keeps one long-lived process per device and calibrates evaluator-only candidate-scenario throughput; per-run optimizer-control overhead is recorded separately, "
                "selects the fastest stable microbatch, combines compatible population requests across runs, and allocates whole jobs by measured capacity."
            )
        elif backend in {"weighted_split", "cuda_priority", "cuda_only", "gpu_preferred"}:
            share_total = self.cuda_share.value() + self.xpu_share.value() + self.cpu_share.value()
            scheduler_text = (
                f"Device-resident admission assigns numerical jobs as CUDA {self.cuda_share.value()}%, "
                f"XPU {self.xpu_share.value()}%, and CPU {self.cpu_share.value()}% (current total {share_total}%). "
                "Each persistent lane keeps optimizer state and the FP64 evaluator on-device; jobs are never migrated after starting."
            )
        else:
            scheduler_text = (
                f"Accelerator-first admission uses NVIDIA CUDA first below {self.gpu_target.value()}% compute and "
                f"{self.gpu_memory_limit.value()}% VRAM, then Intel XPU below {self.xpu_target.value()}% compute when telemetry is available "
                f"and {self.xpu_memory_limit.value()}% device memory, then CPU below {self.cpu_target.value()}% while system RAM stays below "
                f"{self.system_memory_limit.value()}%. Running jobs are never migrated mid-run."
            )
        timing_note = (
            " Use one worker and CPU-only mode for strict publication-quality runtime comparisons."
            if workers > 1 or backend != "cpu_only"
            else " Single-worker CPU mode is appropriate for strict runtime comparisons."
        )
        self.execution_note.setText(
            scheduler_text
            + " Under the v3.4 torch FP64 backend, all primary algorithms use tensor-native population kernels or the CALO policy path plus device-resident mixed-variable decoding, batched AC power flow, constraints, robust aggregation, ranking and L-index evaluation. "
            + "CPU is limited to mandatory orchestration, sparse telemetry, packed result materialization, persistence, checkpointing and independent reference validation. When XPU utilization telemetry is unavailable, the XPU memory threshold and explicit job cap are used instead of inventing a utilization value."
            + timing_note
        )
        self._refresh_resource_status()

    def _invalidate_fairness(self, *_args) -> None:
        if self.manager.running:
            return
        self.fairness_passed = False
        self.backend_parity_passed = False
        self.backend_parity_report = None
        self.compare.setEnabled(False)
        self.calo.setEnabled(False)
        self.audit_state.setText("Configuration changed — audit required")
        self.status.setText(
            "Configuration changed. Run the fairness audit before starting an experiment."
        )

    def refresh(self) -> None:
        self._refresh_experiment_evolution()
        config = self.state.config
        self.runs.setValue(config.runs)
        self.population.setValue(config.population_size)
        index = self.policy.findData(config.budget.policy.value)
        self.policy.setCurrentIndex(max(index, 0))
        self.budget.setValue(config.budget.max_evaluations)
        self.wall.setValue(config.budget.wall_clock_seconds or 60)
        self.maxit.setValue(config.max_iterations)
        self.workers.setValue(config.parallel_workers)
        backend_index = self.execution_backend.findData(config.execution_backend)
        self.execution_backend.setCurrentIndex(max(backend_index, 0))
        scientific_index = self.scientific_backend.findData(
            getattr(config, "scientific_backend", "torch_fp64")
        )
        self.scientific_backend.setCurrentIndex(max(scientific_index, 0))
        self.tensor_batch_size.setValue(int(getattr(config, "tensor_batch_size", 64)))
        self.auto_batch_calibration.setChecked(
            bool(getattr(config, "automatic_batch_calibration", True))
        )
        self.persistent_workers.setChecked(
            bool(getattr(config, "persistent_accelerator_workers", True))
        )
        self.cross_run_batching.setChecked(bool(getattr(config, "cross_run_batching", True)))
        self.batch_window.setValue(float(getattr(config, "cross_run_batch_window_ms", 4.0)))
        self.max_cross_batch.setValue(int(getattr(config, "max_cross_run_batch", 4096)))
        self.calibration_repetitions.setValue(int(getattr(config, "calibration_repetitions", 1)))
        self.telemetry_interval.setValue(int(getattr(config, "telemetry_iteration_interval", 10)))
        self.buffered_traces.setChecked(bool(getattr(config, "buffered_trace_writes", True)))
        self.compile_kernels.setChecked(bool(getattr(config, "compile_stable_kernels", False)))
        self.parity_gate.setChecked(bool(getattr(config, "require_backend_parity", True)))
        self.gpu_target.setValue(config.gpu_utilization_target)
        self.cpu_target.setValue(config.cpu_utilization_target)
        self.gpu_memory_limit.setValue(config.gpu_memory_limit)
        self.gpu_jobs.setValue(config.gpu_parallel_jobs)
        self.xpu_target.setValue(config.xpu_utilization_target)
        self.xpu_memory_limit.setValue(config.xpu_memory_limit)
        self.xpu_jobs.setValue(config.xpu_parallel_jobs)
        self.system_memory_limit.setValue(config.system_memory_limit)
        self.cuda_share.setValue(getattr(config, "cuda_task_share", 100))
        self.xpu_share.setValue(getattr(config, "xpu_task_share", 0))
        self.cpu_share.setValue(getattr(config, "cpu_task_share", 0))
        self.device_resident_execution.setChecked(
            bool(getattr(config, "device_resident_execution", True))
        )
        self.cuda_priority_work_stealing.setChecked(
            bool(getattr(config, "cuda_priority_work_stealing", True))
        )
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
        backend = str(self.execution_backend.currentData() or "")
        if backend == "cuda_priority":
            self.cuda_share.setValue(80)
            self.xpu_share.setValue(10)
            self.cpu_share.setValue(10)
        elif backend in {"cuda_only", "gpu_preferred"}:
            self.cuda_share.setValue(100)
            self.xpu_share.setValue(0)
            self.cpu_share.setValue(0)
        weighted = backend == "weighted_split"
        throughput = backend in {"throughput_auto", "cuda_priority", "cuda_only", "gpu_preferred"}
        for widget in (self.cuda_share, self.xpu_share, self.cpu_share):
            widget.setEnabled(weighted)
        for widget in (
            self.auto_batch_calibration,
            self.persistent_workers,
            self.cross_run_batching,
            self.batch_window,
            self.max_cross_batch,
            self.calibration_repetitions,
            self.telemetry_interval,
            self.buffered_traces,
            self.compile_kernels,
        ):
            widget.setEnabled(
                throughput and str(self.scientific_backend.currentData()) == "torch_fp64"
            )
        torch_backend = str(self.scientific_backend.currentData()) == "torch_fp64"
        self.device_resident_execution.setEnabled(torch_backend)
        self.cuda_priority_work_stealing.setEnabled(backend == "cuda_priority" and torch_backend)
        self.tensor_batch_size.setEnabled(
            not throughput or not self.auto_batch_calibration.isChecked()
        )

    def apply(self) -> None:
        config = self.state.config
        config.runs = int(self.runs.value())
        config.population_size = self.population.value()
        config.budget.policy = BudgetPolicy(self.policy.currentData())
        config.budget.max_evaluations = self.budget.value()
        config.budget.wall_clock_seconds = (
            self.wall.value() if config.budget.policy is BudgetPolicy.EQUAL_WALL_CLOCK else None
        )
        config.max_iterations = self.maxit.value()
        config.parallel_workers = self.workers.value()
        config.execution_backend = str(self.execution_backend.currentData())
        config.scientific_backend = str(self.scientific_backend.currentData())
        config.tensor_batch_size = self.tensor_batch_size.value()
        config.automatic_batch_calibration = self.auto_batch_calibration.isChecked()
        config.persistent_accelerator_workers = self.persistent_workers.isChecked()
        config.cross_run_batching = self.cross_run_batching.isChecked()
        config.cross_run_batch_window_ms = self.batch_window.value()
        config.max_cross_run_batch = self.max_cross_batch.value()
        config.calibration_repetitions = self.calibration_repetitions.value()
        config.telemetry_iteration_interval = self.telemetry_interval.value()
        config.buffered_trace_writes = self.buffered_traces.isChecked()
        config.compile_stable_kernels = self.compile_kernels.isChecked()
        config.device_resident_execution = self.device_resident_execution.isChecked()
        config.cuda_priority_work_stealing = self.cuda_priority_work_stealing.isChecked()
        config.require_backend_parity = self.parity_gate.isChecked()
        config.gpu_utilization_target = self.gpu_target.value()
        config.cpu_utilization_target = self.cpu_target.value()
        config.gpu_memory_limit = self.gpu_memory_limit.value()
        config.gpu_parallel_jobs = self.gpu_jobs.value()
        config.xpu_utilization_target = self.xpu_target.value()
        config.xpu_memory_limit = self.xpu_memory_limit.value()
        config.xpu_parallel_jobs = self.xpu_jobs.value()
        config.system_memory_limit = self.system_memory_limit.value()
        config.cuda_task_share = self.cuda_share.value()
        config.xpu_task_share = self.xpu_share.value()
        config.cpu_task_share = self.cpu_share.value()
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

    def _set_audit_running(self, running: bool) -> None:
        self.audit_button.setEnabled(not running and not self.manager.running)
        self.parity_button.setEnabled(not running and not self.manager.running)
        self.compare.setEnabled((not running) and self.fairness_passed and not self.manager.running)
        self.calo.setEnabled((not running) and self.fairness_passed and not self.manager.running)
        if running:
            self.audit_state.setText("Audit running in background — GUI remains responsive")

    @staticmethod
    def _format_parity(report: dict | None) -> str:
        if not report:
            return "No backend parity report was produced."
        tolerances = dict(report.get("tolerances", {}))
        status = "PASS" if report.get("passed") else "FAIL"
        return (
            f"{status}: CPU/accelerator scientific parity audit.\n"
            f"Device: {report.get('device')} — {report.get('device_name')}\n"
            f"Case: {report.get('case')} · scenarios: {report.get('scenario_count')} · candidates: {report.get('candidate_count')}\n"
            f"Maximum objective error: {report.get('max_objective_error'):.6g} (tol {tolerances.get('objective', float('nan')):.3g})\n"
            f"Maximum violation error: {report.get('max_violation_error'):.6g} (tol {tolerances.get('violation', float('nan')):.3g})\n"
            f"Maximum voltage error: {report.get('max_voltage_error'):.6g} p.u. (tol {tolerances.get('voltage_pu', float('nan')):.3g})\n"
            f"Feasibility mismatches: {report.get('feasibility_mismatches')}\n\n"
            + json.dumps(report.get("details", []), indent=2)
        )

    def _start_audit(self, *, parity_only: bool) -> bool:
        if self.manager.running or (
            self.audit_worker is not None and self.audit_worker.isRunning()
        ):
            self.audit.setPlainText("An experiment or scientific audit is already running.")
            return False
        try:
            self.apply()
        except Exception as exc:
            self.audit.setPlainText(str(exc))
            self.audit_state.setText("Audit could not be started")
            return False
        self.fairness_passed = False if not parity_only else self.fairness_passed
        self._set_audit_running(True)
        self.audit.setPlainText(
            "Running CPU/accelerator parity audit in a background worker…"
            if parity_only
            else "Running parity, fairness, portfolio, and reusable-result checks in a background worker…"
        )
        self.state.task_status.begin(
            "Auditing backend parity" if parity_only else "Auditing experiment fairness",
            detail="Scientific checks are executing outside the GUI thread",
        )
        self.audit_worker = ScientificAuditWorker(
            self.state.config,
            self.state.database.path,
            parity_only=parity_only,
            parent=self,
        )
        self.audit_worker.completed.connect(self._on_audit_completed)
        self.audit_worker.failed.connect(self._on_audit_failed)
        self.audit_worker.progress.connect(self._on_audit_progress)
        self.audit_worker.finished.connect(lambda: self._set_audit_running(False))
        self.audit_worker.start()
        return True

    def run_backend_parity_audit(self) -> bool:
        return self._start_audit(parity_only=True)

    def run_fairness_audit(self) -> bool:
        return self._start_audit(parity_only=False)

    def _on_audit_progress(self, message: str, percent: int) -> None:
        self.audit_state.setText(str(message))
        self.status.setText(str(message))
        self.state.task_status.update(int(percent), str(message))

    def _on_audit_failed(self, message: str) -> None:
        self.fairness_passed = False
        self.backend_parity_passed = False
        self.compare.setEnabled(False)
        self.calo.setEnabled(False)
        self.audit.setPlainText(f"Scientific audit failed to execute: {message}")
        self.audit_state.setText("Audit failed — correct the reported issue")
        self.status.setText("Fairness audit failed. Review the audit output before execution.")
        self.state.task_status.fail(message)

    def _on_audit_completed(self, payload: dict) -> None:
        parity = payload.get("parity")
        if parity is not None:
            self.backend_parity_report = parity
            self.backend_parity_passed = bool(parity.get("passed"))
        if payload.get("parity_only"):
            self.audit.setPlainText(self._format_parity(parity))
            self.audit_state.setText(
                "Backend parity passed — run fairness audit"
                if self.backend_parity_passed
                else "Backend parity failed"
            )
            if self.backend_parity_passed:
                self.state.task_status.finish("Backend parity audit passed")
            else:
                self.state.task_status.fail("Backend parity audit failed")
            return

        report = payload["fairness"]
        portfolio_plan = payload["portfolio_plan"]
        reusable = int(payload.get("reusable", 0))
        total_jobs = planned_item_count(self.state.config, COMPARISON_MODE)
        lines = [
            "PASS: comparative protocol is internally consistent."
            if report.fair
            else "FAIL: comparative protocol requires correction.",
            f"PORTFOLIO PLAN: {self.state.config.portfolio.kind.value} · {self.state.config.portfolio.evidence_profile.value} · {len(self.state.config.portfolio.requested_outputs)} requested outputs.",
            f"PRIMARY COMPARISON PLAN: {len(self.state.config.algorithms)} selected algorithms × {self.state.config.runs} runs = {total_jobs} jobs.",
            f"EXACT RESULT REUSE: {reusable} compatible job(s) can be reused; {total_jobs - reusable} new job(s) remain.",
            f"REQUIRED STORED EVIDENCE: {', '.join(portfolio_plan.required_fields)}.",
            f"CALO ABLATION PLAN: {len(labels_for_mode(self.state.config, ABLATION_MODE))} fixed variants × {self.state.config.runs} runs = {planned_item_count(self.state.config, ABLATION_MODE)} jobs.",
        ]
        if self.state.config.execution_backend in {"gpu_preferred", "cuda_only"}:
            lines.append(
                "GPU-MAXIMUM PLAN: 100% of optimizer, decoder, batched AC power-flow, constraints, robust aggregation, ranking, and CALO policy inference stay on CUDA when available; XPU then CPU are fallback lanes only for GPU-preferred mode."
            )
        if parity:
            lines.append(
                "BACKEND PARITY: "
                + ("PASS" if self.backend_parity_passed else "FAIL")
                + f" · max objective error {parity.get('max_objective_error', float('nan')):.3g}"
                + f" · max violation error {parity.get('max_violation_error', float('nan')):.3g}"
                + f" · max voltage error {parity.get('max_voltage_error', float('nan')):.3g} p.u."
            )
        lines.extend(f"ERROR: {message}" for message in report.errors)
        lines.extend(f"NOTICE: {message}" for message in report.warnings)
        self.audit.setPlainText("\n".join(lines))
        self.fairness_passed = bool(
            report.fair
            and (not self.state.config.require_backend_parity or self.backend_parity_passed)
        )
        self.compare.setEnabled(self.fairness_passed and not self.manager.running)
        self.calo.setEnabled(self.fairness_passed and not self.manager.running)
        if self.fairness_passed:
            self.audit_state.setText("Passed — study execution unlocked")
            self.status.setText(
                "Fairness audit passed. Primary comparison and CALO ablation execution are unlocked."
            )
            self.state.task_status.finish("Fairness audit passed")
        else:
            self.audit_state.setText("Failed — correct the reported issues")
            self.status.setText(
                "Fairness audit failed. Correct the reported issues before execution."
            )
            self.state.task_status.fail("Fairness audit failed")

    def _populate_queue(self, labels: list[str], mode: str) -> None:
        plan = build_execution_plan(self.state.config, mode)
        lane_by_job = {
            item.job_index: (
                "CPU"
                if self.state.config.execution_backend == "cpu_only"
                else (
                    "Auto-calibrated"
                    if self.state.config.execution_backend == "throughput_auto"
                    else "Dynamic"
                )
            )
            for item in plan
        }
        if self.state.config.execution_backend in {
            "weighted_split",
            "cuda_priority",
            "cuda_only",
            "gpu_preferred",
        }:
            snapshot = self.resource_monitor.sample()
            weighted, _summary = build_weighted_lane_plan(
                plan,
                mode,
                cuda_available=bool(snapshot.by_backend("cuda")),
                xpu_available=bool(snapshot.by_backend("xpu")),
                cuda_share=self.state.config.cuda_task_share,
                xpu_share=self.state.config.xpu_task_share,
                cpu_share=self.state.config.cpu_task_share,
            )
            lane_by_job = {job: lane.upper() for job, lane in weighted.items()}
        self.queue.setRowCount(len(plan))
        for row, item in enumerate(plan):
            self.queue.setItem(row, 0, QTableWidgetItem(str(item.run_index + 1)))
            self.queue.setItem(row, 1, QTableWidgetItem(item.label))
            self.queue.setItem(row, 2, QTableWidgetItem(lane_by_job.get(item.job_index, "Dynamic")))
            self.queue.setItem(row, 3, QTableWidgetItem("Queued"))

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
        if not self.fairness_passed:
            QMessageBox.information(
                self,
                "Fairness audit required",
                "Run the fairness audit and wait for its background checks to complete before starting the comparison.",
            )
            return
        labels = list(labels_for_mode(self.state.config, COMPARISON_MODE))
        self._populate_queue(labels, COMPARISON_MODE)
        self.expected_runs = self.state.config.runs * len(labels)
        self.completed_runs = 0
        self.failed_runs = 0
        self._set_running(True)
        if not self.manager.start_comparison(self.state.config):
            self._set_running(self.manager.running)

    def start_calo(self) -> None:
        if not self._manager_available():
            return
        if not self.fairness_passed:
            QMessageBox.information(
                self,
                "Fairness audit required",
                "Run the fairness audit and wait for its background checks to complete before starting CALO analysis.",
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
            f"This is not the 20-algorithm comparison. It runs {len(labels_for_mode(self.state.config, ABLATION_MODE))} fixed CALO/TLBO ablation variants and intentionally ignores the primary algorithm checkbox selection. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        labels = list(labels_for_mode(self.state.config, ABLATION_MODE))
        self._populate_queue(labels, ABLATION_MODE)
        self.expected_runs = self.state.config.runs * len(labels)
        self.completed_runs = 0
        self.failed_runs = 0
        self._set_running(True)
        if not self.manager.start_calo_analysis(self.state.config):
            self._set_running(self.manager.running)

    @staticmethod
    def _parse_run_numbers(text: str) -> tuple[int, ...]:
        values = []
        for token in str(text or "").replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            value = int(token)
            if value < 1:
                raise ValueError("Run numbers are 1-based and must be positive")
            values.append(value - 1)
        return tuple(sorted(set(values)))

    def _refresh_experiment_evolution(self) -> None:
        experiment_id = str(getattr(self.state, "current_experiment_id", "") or "")
        self.extension_experiment.setText(experiment_id)
        self.extension_runs.setValue(max(int(getattr(self.state.config, "runs", 1)), 1))
        self.extension_evaluations.setValue(
            max(int(getattr(self.state.config.budget, "max_evaluations", 1)), 1)
        )
        current_source = self.extension_source_horizon.currentData()
        self.extension_source_horizon.blockSignals(True)
        self.extension_source_horizon.clear()
        if experiment_id:
            for horizon in self.state.database.list_experiment_horizons(experiment_id):
                self.extension_source_horizon.addItem(f"{int(horizon):,} FE", int(horizon))
        source_index = self.extension_source_horizon.findData(current_source)
        self.extension_source_horizon.setCurrentIndex(
            source_index if source_index >= 0 else max(self.extension_source_horizon.count() - 1, 0)
        )
        self.extension_source_horizon.blockSignals(False)
        rows = self.state.database.list_experiment_revisions(experiment_id) if experiment_id else []
        self.revision_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            vals = (
                row.get("revision_number", ""),
                row.get("extension_mode", ""),
                row.get("run_target", ""),
                row.get("evaluation_target", ""),
                "yes" if bool(row.get("publication_eligible")) else "exploratory",
                row.get("status", ""),
                str(row.get("created_at", ""))[:19],
            )
            for c, value in enumerate(vals):
                self.revision_table.setItem(r, c, QTableWidgetItem(str(value)))

    def extend_independent_runs(self) -> None:
        experiment_id = str(getattr(self.state, "current_experiment_id", "") or "")
        if not experiment_id:
            QMessageBox.information(
                self, "Experiment extension", "Open or restore an existing experiment first."
            )
            return
        new_total = int(self.extension_runs.value())
        answer = QMessageBox.question(
            self,
            "Increase independent runs",
            f"Extend experiment {experiment_id[:12]}… to {new_total} total paired independent runs? Existing runs and evidence snapshots are preserved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.manager.extend_run_count(experiment_id, new_total):
            self.status.setText(
                f"Experiment revision started: extending independent-run target to {new_total}."
            )

    def extend_evaluation_horizon(self) -> None:
        experiment_id = str(getattr(self.state, "current_experiment_id", "") or "")
        if not experiment_id:
            QMessageBox.information(
                self, "Horizon extension", "Open or restore an existing experiment first."
            )
            return
        try:
            run_indices = self._parse_run_numbers(self.extension_run_indices.text())
        except Exception as exc:
            QMessageBox.critical(self, "Run selection", str(exc))
            return
        algorithms = tuple(
            a.strip() for a in self.extension_algorithms.text().split(",") if a.strip()
        )
        protocol = str(self.extension_protocol.currentData())
        strategy = str(self.extension_strategy.currentData() or "recompute_from_seed")
        source_horizon = (
            int(self.extension_source_horizon.currentData())
            if strategy == "exact_continue"
            and self.extension_source_horizon.currentData() is not None
            else None
        )
        new_target = int(self.extension_evaluations.value())
        if protocol == "manual_exploratory":
            warning = "This post-hoc selective extension is exploratory and will be excluded from unbiased primary statistics."
        elif strategy == "exact_continue":
            warning = (
                f"Publication eligibility requires every paired participant to have exact optimizer-state checkpoints; currently this is practical for CALO-only exact trajectories. "
                f"This branch will resume the preserved {source_horizon:,}-FE checkpoint."
                if source_horizon
                else "Select a preserved source horizon before exact continuation."
            )
        else:
            warning = "Selected paired runs will be recomputed from their original seeds under the new horizon. This is scientifically comparable but is a new horizon trajectory, not an exact continuation of the shorter run."
        answer = QMessageBox.question(
            self,
            "Extend evaluation horizon",
            f"Continue eligible historical runs to {new_target} requested objective evaluations.\n\n{warning}\n\nOriginal horizon evidence will be snapshotted before any run is updated. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.manager.extend_evaluation_horizon(
            experiment_id,
            new_target,
            protocol=protocol,
            run_indices=run_indices,
            algorithm_names=algorithms,
            execution_strategy=strategy,
            source_horizon=source_horizon,
        ):
            self.status.setText(f"Evaluation-horizon revision started toward {new_target} FE.")

    def _set_running(self, running: bool) -> None:
        self.compare.setEnabled((not running) and self.fairness_passed)
        self.calo.setEnabled((not running) and self.fairness_passed)
        self.audit_button.setEnabled(not running)
        self.parity_button.setEnabled(not running)
        self.pause.setEnabled(running)
        self.cancel.setEnabled(running)
        if running:
            self.audit_state.setText("Locked while experiment is running")
        elif self.fairness_passed:
            self.audit_state.setText("Passed — study execution unlocked")

    def _mark_job(self, run_index: int, algorithm: str, status: str) -> None:
        for row in range(self.queue.rowCount()):
            run_item = self.queue.item(row, 0)
            algorithm_item = self.queue.item(row, 1)
            status_item = self.queue.item(row, 3)
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
            device = str(data.get("compute_device", "")).strip()
            status = f"Active · {device}" if device else "Active"
            self._mark_job(run_index, algorithm, status)

    def on_started(self, experiment_id: str) -> None:
        self._set_running(True)
        workers = self.state.config.parallel_workers
        backend = self.state.config.execution_backend.replace("_", " ")
        self.status.setText(
            f"Experiment {experiment_id} is running with {backend} scheduling and up to {workers} concurrent job{'s' if workers != 1 else ''}. "
            f"Planned jobs: {self.expected_runs}."
        )

    def on_run_completed(self, run_id: str, algorithm: str, run_index: int) -> None:
        self.completed_runs += 1
        self._mark_job(run_index, algorithm, "Completed")
        self._update_status(f"Latest completed: {algorithm}.")

    def on_run_failed(self, failure_id: str, algorithm: str, run_index: int) -> None:
        self.failed_runs += 1
        self._mark_job(run_index, algorithm, "Failed")
        self._update_status(f"Latest failed: {algorithm}; failure record {failure_id[:8]}.")

    def _update_status(self, suffix: str) -> None:
        finished = self.completed_runs + self.failed_runs
        self.status.setText(
            f"Finished {finished} of {self.expected_runs} runs: "
            f"{self.completed_runs} completed, {self.failed_runs} failed. {suffix}"
        )

    def pause_requested(self) -> None:
        self.manager.pause()
        self.pause.setEnabled(False)
        for row in range(self.queue.rowCount()):
            item = self.queue.item(row, 3)
            if item is not None and item.text() == "Queued":
                item.setText("Paused after active jobs")
        self.status.setText(
            "Safe pause requested. No new jobs will start; active jobs will finish and commit before the campaign becomes resumable."
        )

    def cancel_requested(self) -> None:
        self.state.task_status.cancel()
        for row in range(self.queue.rowCount()):
            item = self.queue.item(row, 3)
            if item is not None and item.text() == "Queued":
                item.setText("Cancelled")
        self.status.setText(
            "Immediate stop requested. Completed jobs remain committed. Interrupted CALO jobs resume from exact checkpoints when available; other interrupted algorithms restart from their original paired seeds."
        )

    def on_completed(self, experiment_id: str) -> None:
        self._set_running(False)
        self._refresh_experiment_evolution()
        self.status.setText(
            f"Experiment {experiment_id} finished: {self.completed_runs} completed and "
            f"{self.failed_runs} failed runs. All outcomes are stored with provenance."
        )

    def on_cancelled(self, experiment_id: str) -> None:
        self._set_running(False)
        self._refresh_experiment_evolution()
        self.status.setText(
            f"Experiment {experiment_id} is paused. Completed runs remain stored; Resume Center will schedule only unfinished jobs."
        )

    def on_failed(self, message: str) -> None:
        self._set_running(False)
        self.status.setText(
            "Experiment stopped because an execution or configuration error occurred."
        )
        QMessageBox.critical(self, "Experiment execution failed", message)

    def on_busy(self, message: str) -> None:
        self.status.setText(message)
