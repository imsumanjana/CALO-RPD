"""CALO policy, cognitive architecture, training, and ablation controls."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import numpy as np
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QInputDialog,
    QAbstractItemView,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.ai.model_io import load_checkpoint
from calo_rpd_studio.algorithms.calo.training import (
    TrainingConfig,
    available_training_devices,
    recommended_rollout_workers,
    train_policy,
)
from calo_rpd_studio.algorithms.calo.heterogeneous_training import (
    HeterogeneousTrainingConfig,
    plan_training_lanes,
    train_policy_heterogeneous,
)
from calo_rpd_studio.gui.widgets.page_header import PageHeader
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage
from calo_rpd_studio.gui.widgets.historical_experience_widget import HistoricalExperienceWidget
from calo_rpd_studio.gui.plotting.scientific_plot import ScientificPlotWidget
from calo_rpd_studio.algorithms.calo.policy_qualification import (
    PolicyQualifier,
    PolicyQualificationConfig,
)
from calo_rpd_studio.algorithms.calo.v5_disputes import DISPUTES
from calo_rpd_studio.resume.models import ResumeStatus, ResumeTaskType


class TrainingWorker(QThread):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, config: TrainingConfig | HeterogeneousTrainingConfig, path: str) -> None:
        super().__init__()
        import threading

        self.config = config
        self.path = path
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            if (
                isinstance(self.config, HeterogeneousTrainingConfig)
                and self.config.heterogeneous_rollouts
            ):
                train_policy_heterogeneous(
                    self.config,
                    self.path,
                    progress_callback=self.progress.emit,
                    cancel_callback=self._cancel_event.is_set,
                )
            elif str(self.config.ppo_device).lower() == "xpu_sidecar":
                from calo_rpd_studio.compute.xpu_sidecar import train_policy_in_xpu_sidecar

                train_policy_in_xpu_sidecar(
                    self.config,
                    self.path,
                    progress_callback=self.progress.emit,
                    cancel_callback=self._cancel_event.is_set,
                )
            else:
                train_policy(
                    self.config,
                    self.path,
                    progress_callback=self.progress.emit,
                    cancel_callback=self._cancel_event.is_set,
                )
            self.completed.emit(self.path)
        except Exception as exc:
            from calo_rpd_studio.algorithms.calo.training import TrainingCancelled

            if isinstance(exc, TrainingCancelled):
                self.cancelled.emit(str(exc))
            else:
                self.failed.emit(f"{type(exc).__name__}: {exc}")


class PolicyQualificationWorker(QThread):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, qualifier, candidate_id: str, reference_id: str, config) -> None:
        super().__init__()
        import threading

        self.qualifier = qualifier
        self.candidate_id = candidate_id
        self.reference_id = reference_id
        self.config = config
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            result = self.qualifier.run(
                self.candidate_id,
                reference_policy_id=self.reference_id,
                config=self.config,
                progress_callback=self.progress.emit,
                cancel_callback=self._cancel.is_set,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CALOIntelligencePanel(ScrollablePage):
    stage_completed = pyqtSignal()
    experiment_manager_requested = pyqtSignal()

    def __init__(self, state, experiment_manager, parent=None) -> None:
        content = QWidget()
        super().__init__(content, parent)
        self.state = state
        self.experiment_manager = experiment_manager
        self.worker: TrainingWorker | None = None
        self.qualification_worker: PolicyQualificationWorker | None = None
        self.training_resume_task_id = ""
        self._policy_rows = []

        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        layout.addWidget(
            PageHeader(
                "CALO Intelligence",
                "Manage CALO v5.0 policies, qualify Candidate vs active/reference vs No-AI CALO under paired budgets, bind an immutable policy to runtime, and train the native 32-feature policy schema reproducibly.",
            )
        )

        policy_center = QGroupBox(
            "CALO Policy Center — library, qualification, comparison, and activation"
        )
        center_layout = QVBoxLayout(policy_center)
        splitter = QSplitter()
        library_host = QWidget()
        library_layout = QVBoxLayout(library_host)
        library_layout.setContentsMargins(0, 0, 0, 0)
        self.policy_table = QTableWidget(0, 10)
        self.policy_table.setHorizontalHeaderLabels(
            [
                "Active",
                "Policy",
                "Lineage",
                "Epoch",
                "Role",
                "Grade",
                "Scientific status",
                "Runtime architecture",
                "State schema",
                "SHA-256",
            ]
        )
        self.policy_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.policy_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.policy_table.setAlternatingRowColors(True)
        self.policy_table.verticalHeader().setVisible(False)
        self.policy_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.policy_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in range(2, 10):
            self.policy_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.policy_table.itemSelectionChanged.connect(self._policy_selection_changed)
        library_layout.addWidget(self.policy_table)
        library_buttons = QHBoxLayout()
        self.policy_import_button = QPushButton("Import policy")
        self.policy_activate_button = QPushButton("Set active")
        self.policy_archive_button = QPushButton("Archive")
        self.policy_delete_button = QPushButton("Delete safely")
        self.policy_continue_button = QPushButton("Continue/fine-tune")
        self.policy_fork_button = QPushButton("Fork lineage")
        self.policy_refresh_button = QPushButton("Refresh")
        self.show_archived_policies = QCheckBox("Show archived")
        self.show_archived_policies.toggled.connect(lambda _checked: self.refresh_policy_library())
        self.policy_import_button.clicked.connect(self.import_policy)
        self.policy_activate_button.clicked.connect(self.activate_selected_policy)
        self.policy_archive_button.clicked.connect(self.archive_selected_policy)
        self.policy_delete_button.clicked.connect(self.delete_selected_policy)
        self.policy_continue_button.clicked.connect(self.continue_selected_policy)
        self.policy_fork_button.clicked.connect(self.fork_selected_policy)
        self.policy_refresh_button.clicked.connect(self.refresh_policy_library)
        for button in (
            self.policy_import_button,
            self.policy_activate_button,
            self.policy_archive_button,
            self.policy_delete_button,
            self.policy_continue_button,
            self.policy_fork_button,
            self.policy_refresh_button,
        ):
            library_buttons.addWidget(button)
        library_buttons.addWidget(self.show_archived_policies)
        library_buttons.addStretch(1)
        library_layout.addLayout(library_buttons)

        comparison_host = QWidget()
        comparison_layout = QVBoxLayout(comparison_host)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        compare_controls = QHBoxLayout()
        self.policy_metric = QComboBox()
        self.policy_metric.addItem("Qualification grade index (ordinal)", "score")
        self.policy_metric.addItem("Median final feasible objective", "median_objective")
        self.policy_metric.addItem("Convergence AUC", "median_auc")
        self.policy_metric.addItem("Feasible-run probability", "feasible_probability")
        self.policy_metric.addItem("Evaluations to first feasibility", "median_eval_to_feasible")
        self.policy_metric.addItem("Runtime", "mean_runtime_seconds")
        self.policy_metric.currentIndexChanged.connect(self._draw_policy_comparison)
        compare_controls.addWidget(QLabel("Comparison metric"))
        compare_controls.addWidget(self.policy_metric, 1)
        comparison_layout.addLayout(compare_controls)
        self.policy_plot = ScientificPlotWidget(
            title="Policy qualification comparison",
            xlabel="Policy",
            ylabel="Score",
            square_preview=False,
        )
        self.policy_plot.setMinimumHeight(360)
        comparison_layout.addWidget(self.policy_plot, 1)
        splitter.addWidget(library_host)
        splitter.addWidget(comparison_host)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        center_layout.addWidget(splitter)

        qualification = QGroupBox("Policy qualification gate")
        qualification_form = QFormLayout(qualification)
        self.qualification_reference = QComboBox()
        self.qualification_cases = QLineEdit("case30, case57")
        self.qualification_runs = QSpinBox()
        self.qualification_runs.setRange(2, 100)
        self.qualification_runs.setValue(5)
        self.qualification_budget = QSpinBox()
        self.qualification_budget.setRange(100, 10_000_000)
        self.qualification_budget.setValue(1000)
        self.qualification_population = QSpinBox()
        self.qualification_population.setRange(4, 1000)
        self.qualification_population.setValue(40)
        self.qualify_button = QPushButton(
            "Evaluate selected policy vs active policy and No-AI CALO"
        )
        self.qualify_button.setObjectName("PrimaryButton")
        self.qualify_button.clicked.connect(self.qualify_selected_policy)
        qualification_form.addRow("Reference/frozen policy", self.qualification_reference)
        qualification_form.addRow("Development/qualification cases", self.qualification_cases)
        qualification_form.addRow("Paired runs per case", self.qualification_runs)
        qualification_form.addRow("FE budget per run", self.qualification_budget)
        qualification_form.addRow("Population", self.qualification_population)
        qualification_form.addRow("", self.qualify_button)
        self.qualification_status = QLabel(
            "A policy is graded only from paired optimization outcomes. PPO loss/episode return alone never qualifies a policy. IEEE 118/300 are protected holdouts by default."
        )
        self.qualification_status.setWordWrap(True)
        qualification_form.addRow("Status", self.qualification_status)
        center_layout.addWidget(qualification)
        layout.addWidget(policy_center)

        policy = QGroupBox("Policy controller")
        form = QFormLayout(policy)
        self.path = QLineEdit(
            str(
                Path(__file__).resolve().parents[2]
                / "data"
                / "trained_models"
                / "calo_policy_v2.pt"
            )
        )
        choose = QPushButton("Choose policy")
        choose.clicked.connect(self.choose_policy)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(7)
        row_layout.addWidget(self.path, 1)
        row_layout.addWidget(choose)
        self.no_ai_mode = QCheckBox(
            "No-AI CALO — use rule/contextual online cognition without a neural policy"
        )
        self.no_ai_mode.toggled.connect(lambda checked: self.path.setEnabled(not checked))
        self.deterministic = QCheckBox("Deterministic operator selection during evaluation")
        self.allow_unqualified = QCheckBox(
            "Allow unqualified/legacy policy for research-only runs (strict publication qualification not implied)"
        )
        form.addRow("Policy checkpoint", row)
        form.addRow("", self.no_ai_mode)
        form.addRow("", self.deterministic)
        form.addRow("", self.allow_unqualified)

        inspect = QPushButton("Inspect policy metadata")
        inspect.clicked.connect(self.inspect_policy)
        form.addRow("", inspect)
        self.metadata = QTextEdit()
        self.metadata.setReadOnly(True)
        self.metadata.setMinimumHeight(150)
        self.metadata.setMaximumHeight(210)
        form.addRow("Metadata", self.metadata)
        self.apply_policy_button = QPushButton("Validate and apply CALO configuration")
        self.apply_policy_button.setObjectName("PrimaryButton")
        self.apply_policy_button.clicked.connect(self.apply_policy_configuration)
        form.addRow("", self.apply_policy_button)
        layout.addWidget(policy)

        architecture = QGroupBox("Cognitive adaptive architecture")
        architecture_layout = QVBoxLayout(architecture)
        description = QLabel(
            "Cognitive state — exact and epsilon-feasible ratios, total and component-wise "
            "constraint violation, objective and constraint progress, population and elite "
            "diversity, separate stagnation states, archive occupancy, remaining evaluation "
            "budget, and online operator credit.\n\n"
            "CALO v5.0 operators — feasible-elite learning, constraint-boundary differential "
            "learning, cognitive teacher learning, success-distribution memory, mixed-variable "
            "neighbourhood learning, and diversity recovery. Operators are allocated per learner.\n\n"
            "The hierarchical policy selects a search regime, operator probabilities, and bounded "
            "continuous controls. Online operator credit is blended with the policy so current-run "
            "evidence can correct the learned prior. Training uses PPO with clipped updates and GAE."
        )
        description.setWordWrap(True)
        architecture_layout.addWidget(description)
        layout.addWidget(architecture)

        self.historical_experience = HistoricalExperienceWidget(self.state, self.experiment_manager)
        self.historical_experience.repository_changed.connect(self._historical_repository_changed)
        layout.addWidget(self.historical_experience)

        training = QGroupBox("Reproducible policy training")
        training_form = QFormLayout(training)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 2_000_000_000)
        self.epochs.setValue(24)
        self.training_mode = QComboBox()
        self.training_mode.addItem("Cumulative target epoch", "cumulative")
        self.training_mode.addItem("Additional epochs from saved state", "additional")
        self.training_mode.addItem("Indefinite until Safe Stop", "indefinite")
        self.training_mode.currentIndexChanged.connect(self._update_training_mode_controls)
        self.policy_lineage_name = QLineEdit("CALO-policy-lineage")
        self.policy_lineage_name.setToolTip(
            "Stable policy family name. Continued sessions add immutable checkpoints to this lineage."
        )
        self.checkpoint_interval = QSpinBox()
        self.checkpoint_interval.setRange(1, 10_000_000)
        self.checkpoint_interval.setValue(1)
        self.deployable_interval = QSpinBox()
        self.deployable_interval.setRange(1, 100_000_000)
        self.deployable_interval.setValue(1000)
        self.qualification_interval = QSpinBox()
        self.qualification_interval.setRange(1, 100_000_000)
        self.qualification_interval.setValue(10000)
        self.episodes = QSpinBox()
        self.episodes.setRange(1, 10_000)
        self.episodes.setValue(12)
        self.horizon = QSpinBox()
        self.horizon.setRange(2, 10_000)
        self.horizon.setValue(28)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(2026)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(7)
        self.lr.setRange(1e-7, 1)
        self.lr.setValue(3e-4)
        self.gamma = QDoubleSpinBox()
        self.gamma.setDecimals(4)
        self.gamma.setRange(0.0, 0.9999)
        self.gamma.setValue(0.98)
        self.gae_lambda = QDoubleSpinBox()
        self.gae_lambda.setDecimals(4)
        self.gae_lambda.setRange(0.0, 1.0)
        self.gae_lambda.setValue(0.95)
        self.clip_ratio = QDoubleSpinBox()
        self.clip_ratio.setDecimals(3)
        self.clip_ratio.setRange(0.01, 0.8)
        self.clip_ratio.setValue(0.20)
        self.ppo_epochs = QSpinBox()
        self.ppo_epochs.setRange(1, 50)
        self.ppo_epochs.setValue(4)
        self.minibatch = QSpinBox()
        self.minibatch.setRange(8, 8192)
        self.minibatch.setValue(128)
        self.training_population = QSpinBox()
        self.training_population.setRange(4, 1000)
        self.training_population.setValue(20)

        self.training_device = QComboBox()
        self.training_device.addItem("Automatic (CUDA → XPU → CPU)", "auto")
        self.training_device.addItem("NVIDIA CUDA GPU", "cuda")
        self.training_device.addItem("Intel XPU — current runtime", "xpu")
        self.training_device.addItem("Intel XPU — secondary runtime", "xpu_sidecar")
        self.training_device.addItem("CPU", "cpu")
        device_info = available_training_devices()
        if not device_info["cuda_available"]:
            cuda_index = self.training_device.findData("cuda")
            self.training_device.model().item(cuda_index).setEnabled(False)
        if not device_info["xpu_available"]:
            xpu_index = self.training_device.findData("xpu")
            self.training_device.model().item(xpu_index).setEnabled(False)
        if not device_info["xpu_sidecar_available"]:
            sidecar_index = self.training_device.findData("xpu_sidecar")
            self.training_device.model().item(sidecar_index).setEnabled(False)
        device_parts = []
        if device_info["cuda_available"]:
            device_parts.append(f"CUDA: {device_info['cuda_name']}")
        if device_info["xpu_available"]:
            device_parts.append(f"XPU: {device_info['xpu_name']}")
        if device_info["xpu_sidecar_available"]:
            device_parts.append("secondary Intel XPU runtime: ready")
        device_text = (
            "; ".join(device_parts) or "No verified GPU backend; CPU fallback is available"
        )
        self.training_device.setToolTip(
            "The selected primary device performs centralized PPO updates. In weighted mode, "
            "CUDA, XPU, and CPU actors simultaneously collect fixed shares of fresh rollouts from "
            "one synchronized policy snapshot. Automatic learner priority is NVIDIA CUDA, then "
            "direct Intel XPU, then CPU. " + device_text
        )

        self.rollout_mode = QComboBox()
        self.rollout_mode.addItem(
            "GPU-maximum device-resident actors — CUDA 100% when available",
            "weighted",
        )
        self.rollout_mode.addItem(
            "Legacy CPU actors + one PPO learner device",
            "legacy_cpu",
        )
        self.rollout_mode.currentIndexChanged.connect(self._update_training_plan)

        self.cuda_rollout_share = QSpinBox()
        self.cuda_rollout_share.setRange(0, 100)
        self.cuda_rollout_share.setValue(100)
        self.cuda_rollout_share.setSuffix(" % CUDA")
        self.xpu_rollout_share = QSpinBox()
        self.xpu_rollout_share.setRange(0, 100)
        self.xpu_rollout_share.setValue(0)
        self.xpu_rollout_share.setSuffix(" % XPU")
        self.cpu_rollout_share = QSpinBox()
        self.cpu_rollout_share.setRange(0, 100)
        self.cpu_rollout_share.setValue(0)
        self.cpu_rollout_share.setSuffix(" % CPU")
        split_row = QWidget()
        split_layout = QHBoxLayout(split_row)
        split_layout.setContentsMargins(0, 0, 0, 0)
        split_layout.setSpacing(7)
        split_layout.addWidget(self.cuda_rollout_share)
        split_layout.addWidget(self.xpu_rollout_share)
        split_layout.addWidget(self.cpu_rollout_share)
        self.training_cuda_priority = QPushButton("100/0/0 GPU max")
        self.training_cuda_priority.setToolTip("CUDA-priority device-resident rollout allocation")
        self.training_cuda_only = QPushButton("100% CUDA")
        self.training_cuda_only.setToolTip(
            "Route all compatible rollout episodes and the PPO learner to NVIDIA CUDA"
        )
        self.training_cuda_priority.clicked.connect(lambda: self._set_training_split(100, 0, 0))
        self.training_cuda_only.clicked.connect(lambda: self._set_training_split(100, 0, 0))
        split_layout.addWidget(self.training_cuda_priority)
        split_layout.addWidget(self.training_cuda_only)
        for control in (
            self.cuda_rollout_share,
            self.xpu_rollout_share,
            self.cpu_rollout_share,
        ):
            control.valueChanged.connect(self._update_training_plan)

        self.auto_tuned_training = QCheckBox(
            "Auto-tune CUDA/XPU/CPU rollout shares from measured complete actor throughput"
        )
        self.auto_tuned_training.setChecked(False)
        self.auto_tuned_training.setToolTip(
            "Runs short discarded calibration episodes on every verified device, then allocates "
            "fresh on-policy episodes by measured transitions per second. The 100/0/0 values "
            "remain the deterministic fallback when calibration is unavailable."
        )
        self.persistent_training_actors = QCheckBox(
            "Keep CUDA/XPU actors and CPU rollout pool persistent for the full training session"
        )
        self.persistent_training_actors.setChecked(True)
        self.accelerated_training_orpd = QCheckBox(
            "Use FP64 accelerator-native ORPD evaluation in development-case rollouts"
        )
        self.accelerated_training_orpd.setChecked(True)
        self.accelerated_training_orpd.setToolTip(
            "CALO v5.0 uses the native 32-feature policy state/action schema and accelerator-native FP64 ORPD evaluation for configured development-case rollouts; synthetic curriculum stages remain lightweight host environments."
        )
        self.cross_episode_training_batch = QCheckBox(
            "Batch compatible ORPD populations across simultaneous rollout episodes"
        )
        self.cross_episode_training_batch.setChecked(True)

        self.training_calibration_episodes = QSpinBox()
        self.training_calibration_episodes.setRange(1, 20)
        self.training_calibration_episodes.setValue(1)
        self.training_calibration_episodes.setToolTip(
            "Calibration trajectories are timed and discarded; they never enter the PPO buffer."
        )
        self.training_tensor_batch = QSpinBox()
        self.training_tensor_batch.setRange(1, 65536)
        self.training_tensor_batch.setValue(64)
        self.training_tensor_batch.setToolTip(
            "Candidate microbatch used by the FP64 ORPD evaluator inside policy-training actors."
        )
        self.training_cross_batch = QSpinBox()
        self.training_cross_batch.setRange(16, 262144)
        self.training_cross_batch.setValue(2048)
        self.training_cross_batch.setToolTip(
            "Maximum number of compatible candidate evaluations merged across rollout episodes."
        )
        self.training_batch_window = QDoubleSpinBox()
        self.training_batch_window.setDecimals(1)
        self.training_batch_window.setRange(0.1, 100.0)
        self.training_batch_window.setValue(4.0)
        self.training_batch_window.setSuffix(" ms")

        for control in (
            self.auto_tuned_training,
            self.persistent_training_actors,
            self.accelerated_training_orpd,
            self.cross_episode_training_batch,
        ):
            control.toggled.connect(self._update_training_plan)
        for control in (
            self.training_calibration_episodes,
            self.training_tensor_batch,
            self.training_cross_batch,
            self.training_batch_window,
        ):
            control.valueChanged.connect(self._update_training_plan)

        self.rollout_workers = QSpinBox()
        self.rollout_workers.setRange(1, max(1, recommended_rollout_workers() + 8))
        self.rollout_workers.setValue(recommended_rollout_workers(self.episodes.value()))
        self.recommended_workers_button = QPushButton("Use recommended")
        self.recommended_workers_button.clicked.connect(self._use_recommended_workers)
        worker_row = QWidget()
        worker_layout = QHBoxLayout(worker_row)
        worker_layout.setContentsMargins(0, 0, 0, 0)
        worker_layout.addWidget(self.rollout_workers, 1)
        worker_layout.addWidget(self.recommended_workers_button)
        self.episodes.valueChanged.connect(self._use_recommended_workers)
        self.episodes.valueChanged.connect(self._update_training_plan)

        self.accelerator_status = QLabel()
        self.accelerator_status.setWordWrap(True)
        self._device_text = device_text
        self._update_training_plan()

        self.development_cases = QLineEdit()
        self.development_cases.setPlaceholderText(
            "Optional custom ORPD development case paths, comma-separated"
        )
        self.development_cases.setToolTip(
            "Optional explicit development systems for the final curriculum stage. Keep final publication benchmark systems separate from policy training."
        )
        self.train_button = QPushButton("Start / continue CALO policy lineage")
        self.train_button.setObjectName("PrimaryButton")
        self.train_button.clicked.connect(self.train_policy)
        self.resume_training_button = QPushButton("Resume exact saved training")
        self.resume_training_button.clicked.connect(self.resume_saved_training)
        training_button_row = QWidget()
        training_button_layout = QHBoxLayout(training_button_row)
        training_button_layout.setContentsMargins(0, 0, 0, 0)
        training_button_layout.addWidget(self.train_button, 1)
        training_button_layout.addWidget(self.resume_training_button)
        training_form.addRow("Training continuation mode", self.training_mode)
        training_form.addRow("Target / additional epochs", self.epochs)
        training_form.addRow("Policy lineage", self.policy_lineage_name)
        training_form.addRow("Exact resume checkpoint every", self.checkpoint_interval)
        training_form.addRow("Usable policy snapshot every", self.deployable_interval)
        training_form.addRow("Suggested qualification interval", self.qualification_interval)
        training_form.addRow("Episodes per epoch", self.episodes)
        training_form.addRow("Episode horizon", self.horizon)
        training_form.addRow("Training seed", self.seed)
        training_form.addRow("Learning rate", self.lr)
        training_form.addRow("Discount factor γ", self.gamma)
        training_form.addRow("GAE λ", self.gae_lambda)
        training_form.addRow("PPO clip ratio", self.clip_ratio)
        training_form.addRow("PPO update epochs", self.ppo_epochs)
        training_form.addRow("Minibatch size", self.minibatch)
        training_form.addRow("Training population", self.training_population)
        training_form.addRow("PPO learner device", self.training_device)
        training_form.addRow("Rollout execution", self.rollout_mode)
        training_form.addRow("Rollout transition split", split_row)
        training_form.addRow("Throughput allocation", self.auto_tuned_training)
        training_form.addRow("Persistent training actors", self.persistent_training_actors)
        training_form.addRow("Accelerated ORPD rollouts", self.accelerated_training_orpd)
        training_form.addRow("Cross-episode batching", self.cross_episode_training_batch)
        training_form.addRow("Calibration episodes/device", self.training_calibration_episodes)
        training_form.addRow("ORPD tensor microbatch", self.training_tensor_batch)
        training_form.addRow("Maximum merged candidates", self.training_cross_batch)
        training_form.addRow("Cross-episode batch window", self.training_batch_window)
        training_form.addRow("CPU actor workers", worker_row)
        training_form.addRow("Compute status", self.accelerator_status)
        training_form.addRow("ORPD development cases", self.development_cases)
        training_form.addRow("", training_button_row)
        self._update_training_mode_controls()
        layout.addWidget(training)

        ablation = QGroupBox("Ablation analysis")
        ablation_layout = QVBoxLayout(ablation)
        self.ablation_button = QPushButton("Open Experiment Manager for CALO Analysis")
        self.ablation_button.setEnabled(False)
        self.ablation_button.clicked.connect(self.experiment_manager_requested.emit)
        ablation_layout.addWidget(self.ablation_button)
        layout.addWidget(ablation)

        dispute_box = QGroupBox("v5.0 continuation and audited scientific/performance register")
        dispute_layout = QVBoxLayout(dispute_box)
        dispute_help = QLabel(
            "RESOLVED items are closed in v5.0; PARTIAL/OPEN/DEFERRED items remain explicit future work and must not be described as solved. "
            "This register separates scientific correctness from performance engineering."
        )
        dispute_help.setWordWrap(True)
        dispute_help.setObjectName("HelpText")
        dispute_layout.addWidget(dispute_help)
        self.dispute_table = QTableWidget(len(DISPUTES), 4)
        self.dispute_table.setHorizontalHeaderLabels(
            ["ID", "Status", "Severity", "Audited finding / action"]
        )
        self.dispute_table.verticalHeader().setVisible(False)
        self.dispute_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.dispute_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.dispute_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.dispute_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.dispute_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.dispute_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        for row, item in enumerate(DISPUTES):
            text = f"{item.finding} — {item.evidence_or_action}"
            for col, value in enumerate((item.id, item.status, item.severity, text)):
                self.dispute_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.dispute_table.setMinimumHeight(300)
        dispute_layout.addWidget(self.dispute_table)
        layout.addWidget(dispute_box)
        layout.addStretch(1)
        self.refresh_policy_library()
        self.state.config_changed.connect(lambda config: self.load_from_config(config))
        self.load_from_config(self.state.config)

    def refresh_policy_library(self) -> None:
        bundled = Path(__file__).resolve().parents[2] / "data" / "trained_models"
        self.state.policy_registry.discover_bundled(bundled)
        self._policy_rows = self.state.policy_registry.list(
            include_archived=self.show_archived_policies.isChecked()
        )
        self.policy_table.setRowCount(len(self._policy_rows))
        for row, policy in enumerate(self._policy_rows):
            checkpoint = self.state.database.get_policy_checkpoint_by_sha256(policy.sha256)
            lineage_name = ""
            epoch = ""
            role = ""
            if checkpoint is not None:
                lineage = self.state.database.get_policy_lineage(
                    str(checkpoint.get("lineage_id", ""))
                )
                lineage_name = str((lineage or {}).get("name", ""))
                epoch = str(int(checkpoint.get("cumulative_epoch", 0) or 0))
                role = (
                    "best"
                    if bool(checkpoint.get("is_best"))
                    else ("latest" if bool(checkpoint.get("is_latest")) else "checkpoint")
                )
            values = [
                "●" if policy.active else "",
                policy.name,
                lineage_name,
                epoch,
                role,
                policy.grade,
                ("archived · " if policy.archived else "") + policy.qualification_status,
                policy.architecture_version,
                policy.state_schema_version,
                policy.sha256[:12],
            ]
            for col, value in enumerate(values):
                self.policy_table.setItem(row, col, QTableWidgetItem(str(value)))
        current_ref = self.qualification_reference.currentData()
        self.qualification_reference.blockSignals(True)
        self.qualification_reference.clear()
        self.qualification_reference.addItem("No separate reference", "")
        for policy in self._policy_rows:
            label = f"{policy.name} · {policy.grade} · {policy.qualification_status}"
            self.qualification_reference.addItem(label, policy.id)
        idx = self.qualification_reference.findData(current_ref)
        if idx < 0 or not current_ref:
            active = next((item for item in self._policy_rows if item.active), None)
            idx = self.qualification_reference.findData(active.id) if active is not None else -1
        if idx >= 0:
            self.qualification_reference.setCurrentIndex(idx)
        self.qualification_reference.blockSignals(False)
        self._draw_policy_comparison()

    def _selected_policy(self):
        row = self.policy_table.currentRow()
        return self._policy_rows[row] if 0 <= row < len(self._policy_rows) else None

    def _select_policy_id(self, policy_id: str) -> None:
        for row, policy in enumerate(self._policy_rows):
            if policy.id == policy_id:
                self.policy_table.selectRow(row)
                return

    def _policy_selection_changed(self) -> None:
        policy = self._selected_policy()
        if policy is None:
            return
        self.path.setText(policy.checkpoint_path)
        self.policy_archive_button.setText("Restore archived" if policy.archived else "Archive")
        self.inspect_policy()

    def _selected_policy_checkpoint_record(self):
        policy = self._selected_policy()
        if policy is None:
            return None, None
        return policy, self.state.database.get_policy_checkpoint_by_sha256(policy.sha256)

    def continue_selected_policy(self) -> None:
        """Prepare a weights-only continuation phase from any usable checkpoint.

        Exact optimizer/RNG continuation of the latest training session remains the separate
        Resume exact saved training action. This operation intentionally creates a documented new
        phase while retaining the same lineage.
        """
        policy, checkpoint = self._selected_policy_checkpoint_record()
        if policy is None:
            return
        self._pending_initial_policy_checkpoint = str(policy.checkpoint_path)
        if checkpoint is not None:
            lineage = self.state.database.get_policy_lineage(str(checkpoint.get("lineage_id", "")))
            self._pending_policy_lineage_id = str(checkpoint.get("lineage_id", ""))
            self._pending_policy_phase_index = int(checkpoint.get("phase_index", 1) or 1) + 1
            self.policy_lineage_name.setText(str((lineage or {}).get("name", policy.name)))
        else:
            self._pending_policy_lineage_id = ""
            self._pending_policy_phase_index = 1
            self.policy_lineage_name.setText(policy.name)
        self.qualification_status.setText(
            f"Prepared fine-tuning from immutable checkpoint {policy.name} ({policy.sha256[:12]}…). "
            "Choose training duration and click Start / continue lineage. Existing experiments remain bound to the old SHA."
        )

    def fork_selected_policy(self) -> None:
        policy, checkpoint = self._selected_policy_checkpoint_record()
        if policy is None:
            return
        name, ok = QInputDialog.getText(
            self, "Fork policy lineage", "New lineage name:", text=f"{policy.name}-fork"
        )
        if not ok or not name.strip():
            return
        try:
            if checkpoint is not None:
                lineage_id = self.state.policy_registry.lineages.fork(
                    str(checkpoint["id"]),
                    name.strip(),
                    notes=f"Forked from {policy.name} {policy.sha256}",
                )
            else:
                lineage_id = self.state.policy_registry.create_lineage(name.strip())
            self._pending_initial_policy_checkpoint = str(policy.checkpoint_path)
            self._pending_policy_lineage_id = str(lineage_id)
            self._pending_policy_phase_index = 1
            self.policy_lineage_name.setText(name.strip())
            self.qualification_status.setText(
                f"Fork {name.strip()} prepared from {policy.name}. Start training to create the first child checkpoint; the parent artifact is unchanged."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Policy fork failed", str(exc))

    def import_policy(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import CALO policy", "", "PyTorch checkpoint (*.pt)"
        )
        if not path:
            return
        try:
            policy = self.state.policy_registry.register(path)
            self.refresh_policy_library()
            self._select_policy_id(policy.id)
        except Exception as exc:
            QMessageBox.critical(self, "Policy import failed", str(exc))

    def activate_selected_policy(self) -> None:
        policy = self._selected_policy()
        if policy is None:
            return
        try:
            self.state.policy_registry.activate(policy.id)
            self.refresh_policy_library()
            self._select_policy_id(policy.id)
            self.qualification_status.setText(
                f"Active default policy: {policy.name}. Existing experiments remain bound to their original immutable checkpoint SHA."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Policy activation failed", str(exc))

    def archive_selected_policy(self) -> None:
        policy = self._selected_policy()
        if policy is None:
            return
        try:
            if policy.archived:
                self.state.policy_registry.unarchive(policy.id)
            else:
                self.state.policy_registry.archive(policy.id)
            self.refresh_policy_library()
        except Exception as exc:
            QMessageBox.critical(self, "Policy archive failed", str(exc))

    def delete_selected_policy(self) -> None:
        policy = self._selected_policy()
        if policy is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete policy",
            "Delete the policy library record and checkpoint file only if no experiment references it? "
            "Referenced policies cannot be deleted and must be archived instead.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.state.policy_registry.delete(policy.id, delete_artifact=True)
            self.refresh_policy_library()
        except Exception as exc:
            QMessageBox.critical(self, "Policy deletion blocked", str(exc))

    def qualify_selected_policy(self) -> None:
        policy = self._selected_policy()
        if policy is None:
            QMessageBox.information(self, "Policy qualification", "Select a policy first.")
            return
        if self.state.task_status.busy:
            QMessageBox.information(self, "Task busy", "Wait for the active task to finish first.")
            return
        cases = tuple(
            item.strip() for item in self.qualification_cases.text().split(",") if item.strip()
        )
        config = PolicyQualificationConfig(
            cases=cases,
            runs=self.qualification_runs.value(),
            max_evaluations=self.qualification_budget.value(),
            population_size=self.qualification_population.value(),
            master_seed=int(self.state.config.master_seed),
        )
        try:
            config.validate()
        except Exception as exc:
            QMessageBox.critical(self, "Policy qualification", str(exc))
            return
        reference_id = str(self.qualification_reference.currentData() or "")
        if reference_id == policy.id:
            reference_id = ""
        qualifier = PolicyQualifier(self.state.config, self.state.policy_registry)
        self.qualification_worker = PolicyQualificationWorker(
            qualifier, policy.id, reference_id, config
        )
        self.qualification_worker.progress.connect(self._qualification_progress)
        self.qualification_worker.completed.connect(self._qualification_done)
        self.qualification_worker.failed.connect(self._qualification_failed)
        self.state.task_status.cancel_requested.connect(self.qualification_worker.cancel)
        self.state.task_status.begin(
            "Qualifying CALO policy", detail=policy.name, progress=0, cancellable=True
        )
        self.qualify_button.setEnabled(False)
        self.qualification_worker.start()

    def _qualification_progress(self, percent: int, detail: str) -> None:
        self.state.task_status.update(percent, detail)
        self.qualification_status.setText(detail)

    def _qualification_done(self, result: dict) -> None:
        try:
            self.state.task_status.cancel_requested.disconnect(self.qualification_worker.cancel)
        except (TypeError, AttributeError):
            pass
        self.qualify_button.setEnabled(True)
        qualification_status = (
            "qualified"
            if result.get("passed") and result.get("native_v41")
            else "legacy_qualified"
            if result.get("passed")
            else "failed"
        )
        self.state.database.add_policy_qualification(
            qualification_id=result["qualification_id"],
            policy_id=result["candidate_policy_id"],
            reference_policy_id=result.get("reference_policy_id", ""),
            config=result.get("config", {}),
            metrics=result,
            passed=bool(result.get("passed")),
            grade=str(result.get("grade", "U")),
            score=float(result.get("score", 0.0)),
            qualification_status=qualification_status,
        )
        # Keep policy lineage history immutable: qualification updates the matching checkpoint
        # record, while "latest" and "best-qualified" remain separate concepts.
        checkpoint = self.state.database.get_policy_checkpoint_by_sha256(
            str(result.get("candidate_policy_sha256", ""))
        )
        if checkpoint is not None:
            candidate_metrics = dict(result.get("participants", {}).get("candidate", {}))
            self.state.database.update_policy_checkpoint_qualification(
                checkpoint["id"],
                qualification_status=qualification_status,
                grade=str(result.get("grade", "U")),
                metadata_updates={
                    "latest_qualification_id": result["qualification_id"],
                    "qualification_metrics": candidate_metrics,
                },
            )
            if bool(result.get("passed")):
                lineage_id = str(checkpoint["lineage_id"])
                best = next(
                    (
                        row
                        for row in self.state.database.list_policy_checkpoints(lineage_id)
                        if bool(row.get("is_best"))
                    ),
                    None,
                )
                rank = {
                    "A+": 5,
                    "A": 4,
                    "A-": 3.5,
                    "B+": 3,
                    "B": 2,
                    "B-": 1.5,
                    "C": 1,
                    "U": 0,
                    "F": -1,
                }
                promote = best is None or rank.get(str(result.get("grade", "U")), 0) > rank.get(
                    str(best.get("grade", "U")), 0
                )
                if best is not None and rank.get(str(result.get("grade", "U")), 0) == rank.get(
                    str(best.get("grade", "U")), 0
                ):
                    try:
                        old_obj = float(
                            dict(best.get("metadata", {}))
                            .get("qualification_metrics", {})
                            .get("median_objective", float("inf"))
                        )
                        new_obj = float(candidate_metrics.get("median_objective", float("inf")))
                        promote = np.isfinite(new_obj) and new_obj < old_obj
                    except Exception:
                        promote = False
                if promote:
                    self.state.database.mark_best_policy_checkpoint(lineage_id, checkpoint["id"])
        self.state.task_status.finish(
            f"Policy qualification {'passed' if result.get('passed') else 'failed'} · grade {result.get('grade', 'U')}"
        )
        paired = result.get("paired_evidence", {}).get("vs_no_ai", {})
        pvalue = paired.get("wilcoxon_p_two_sided")
        ptext = (
            f" · paired p={float(pvalue):.3g}"
            if isinstance(pvalue, (int, float)) and np.isfinite(float(pvalue))
            else ""
        )
        native_text = "native v5.0" if result.get("native_v41") else "legacy-compatible"
        self.qualification_status.setText(
            f"Qualification {'PASS' if result.get('passed') else 'FAIL'} · {native_text} · grade {result.get('grade')} · "
            f"grade index {float(result.get('score', 0.0)):.0f}{ptext}. "
            + " ".join(result.get("reasons", []))
        )
        self.refresh_policy_library()
        self._select_policy_id(str(result["candidate_policy_id"]))

    def _qualification_failed(self, message: str) -> None:
        try:
            self.state.task_status.cancel_requested.disconnect(self.qualification_worker.cancel)
        except (TypeError, AttributeError):
            pass
        self.qualify_button.setEnabled(True)
        self.state.task_status.fail(message)
        self.qualification_status.setText(message)
        QMessageBox.critical(self, "Policy qualification failed", message)

    def _draw_policy_comparison(self, *_args) -> None:
        key = str(self.policy_metric.currentData() or "score")
        labels, values = [], []
        for policy in self._policy_rows:
            qualifications = self.state.database.list_policy_qualifications(policy.id)
            if not qualifications:
                continue
            import json as _json

            latest = qualifications[0]
            payload = _json.loads(latest.get("metrics_json") or "{}")
            if key == "score":
                value = float(latest.get("score", 0.0))
            else:
                value = float(
                    payload.get("participants", {}).get("candidate", {}).get(key, float("nan"))
                )
            if not np.isfinite(value):
                continue
            labels.append(policy.name)
            values.append(value)
        axis = self.policy_plot.axis
        axis.clear()
        if not labels:
            self.policy_plot.show_message(
                "No qualified policy evidence yet. Run Policy Qualification to compare Candidate vs reference policy vs No-AI CALO.",
                title="Policy qualification comparison",
                xlabel="Policy",
                ylabel="Metric",
            )
            return
        selected = self._selected_policy()
        if key != "score" and selected is not None:
            qualifications = self.state.database.list_policy_qualifications(selected.id)
            if qualifications:
                payload = json.loads(qualifications[0].get("metrics_json") or "{}")
                participants = payload.get("participants", {})
                for participant_key, display_name in (
                    ("reference", "Reference policy"),
                    ("no_ai", "No-AI CALO"),
                ):
                    value = participants.get(participant_key, {}).get(key)
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        labels.append(display_name)
                        values.append(value)
        x = np.arange(len(labels))
        axis.bar(x, values)
        axis.set_xticks(x, labels, rotation=30, ha="right")
        titles = {
            "score": ("Policy qualification grade index (ordinal)", "Grade index"),
            "median_objective": ("Median final feasible objective", "Objective"),
            "median_auc": ("Median convergence AUC", "AUC"),
            "feasible_probability": ("Feasible-run probability", "Probability"),
            "median_eval_to_feasible": ("Evaluations to first feasibility", "Evaluations"),
            "mean_runtime_seconds": ("Mean runtime", "Seconds"),
        }
        title, ylabel = titles.get(key, (key, key))
        meta = self.policy_plot.manager.records[self.policy_plot.plot_id].metadata
        meta.update({"title": title, "xlabel": "Policy", "ylabel": ylabel})
        self.policy_plot.manager.apply(self.policy_plot.plot_id, self.policy_plot.style)

    def load_from_config(self, config) -> None:
        """Rehydrate the exact experiment-bound CALO runtime/intelligence selections."""
        parameters = dict(config.algorithm_parameters.get("CALO", {}))
        checkpoint = str(parameters.get("policy_checkpoint", "") or "")
        self.path.setText(checkpoint)
        self.no_ai_mode.setChecked(not bool(parameters.get("use_ai", True)))
        self.deterministic.setChecked(bool(parameters.get("deterministic_policy", False)))
        self.allow_unqualified.setChecked(bool(parameters.get("allow_unqualified_policy", False)))
        policy_id = str(parameters.get("policy_id", "") or "")
        self.policy_table.clearSelection()
        if policy_id:
            self._select_policy_id(policy_id)

        historical = self.historical_experience
        has_historical_runtime = bool(
            parameters.get("historical_repository")
            or parameters.get("use_historical_parameter_priors", False)
            or parameters.get("use_cross_algorithm_warm_start", False)
        )
        mode = str(parameters.get("historical_learning_mode", "") or "")
        if not mode:
            mode = "historical_warm_start" if has_historical_runtime else "cold_start"
        idx = historical.learning_mode.findData(mode)
        if idx >= 0:
            historical.learning_mode.setCurrentIndex(idx)
        repository = str(parameters.get("historical_repository", "") or "")
        historical.repository_path.setText(repository)
        historical.use_parameter_priors.setChecked(
            bool(parameters.get("use_historical_parameter_priors", False))
        )
        warm_start = bool(parameters.get("use_cross_algorithm_warm_start", False))
        historical.use_cross_algorithm_knowledge.setChecked(warm_start)
        historical.allow_population_warm_start.setChecked(warm_start)
        historical.warm_start_percent.setValue(
            int(round(100 * float(parameters.get("historical_warm_start_fraction", 0.15))))
        )

    def _historical_repository_changed(self, path: str) -> None:
        self.metadata.setPlainText(
            "Historical experience repository selected:\n"
            f"{path}\n\n"
            "Eligible historical CALO trajectories can be used for offline pretraining before fresh on-policy PPO. "
            "Cross-algorithm solutions and CALO parameter priors are applied only when explicitly enabled."
        )

    def _use_recommended_workers(self, *_args) -> None:
        self.rollout_workers.setValue(recommended_rollout_workers(self.episodes.value()))

    def _set_training_split(self, cuda: int, xpu: int, cpu: int) -> None:
        self.cuda_rollout_share.setValue(int(cuda))
        self.xpu_rollout_share.setValue(int(xpu))
        self.cpu_rollout_share.setValue(int(cpu))
        if cuda == 100:
            cuda_index = self.training_device.findData("cuda")
            if cuda_index >= 0:
                self.training_device.setCurrentIndex(cuda_index)
        self._update_training_plan()

    def _update_training_plan(self, *_args) -> None:
        weighted = str(self.rollout_mode.currentData()) == "weighted"
        advanced_controls = (
            self.auto_tuned_training,
            self.persistent_training_actors,
            self.accelerated_training_orpd,
            self.cross_episode_training_batch,
            self.training_calibration_episodes,
            self.training_tensor_batch,
            self.training_cross_batch,
            self.training_batch_window,
        )
        for control in (
            self.cuda_rollout_share,
            self.xpu_rollout_share,
            self.cpu_rollout_share,
            *advanced_controls,
        ):
            control.setEnabled(weighted)
        self.training_calibration_episodes.setEnabled(
            weighted and self.auto_tuned_training.isChecked()
        )
        batching_enabled = (
            weighted
            and self.accelerated_training_orpd.isChecked()
            and self.cross_episode_training_batch.isChecked()
        )
        self.training_cross_batch.setEnabled(batching_enabled)
        self.training_batch_window.setEnabled(batching_enabled)
        self.training_tensor_batch.setEnabled(
            weighted and self.accelerated_training_orpd.isChecked()
        )
        if not weighted:
            self.accelerator_status.setText(
                "Legacy architecture: CPU rollout processes collect all episodes; one selected "
                "device performs centralized PPO updates. " + self._device_text
            )
            return
        total = (
            self.cuda_rollout_share.value()
            + self.xpu_rollout_share.value()
            + self.cpu_rollout_share.value()
        )
        if total != 100:
            self.accelerator_status.setText(
                f"Invalid fallback split: {total}%. CUDA, XPU, and CPU shares must total 100%."
            )
            return
        try:
            plan = plan_training_lanes(
                self.episodes.value(),
                cuda_share=self.cuda_rollout_share.value(),
                xpu_share=self.xpu_rollout_share.value(),
                cpu_share=self.cpu_rollout_share.value(),
            )
            warning = (" " + " ".join(plan.warnings)) if plan.warnings else ""
            allocation_text = (
                "The displayed 100/0/0 split is a fallback only; short discarded calibration "
                "episodes measure complete transitions per second and subsequent epochs are "
                "allocated by measured throughput."
                if self.auto_tuned_training.isChecked()
                else "The fixed episode split shown below is used for every epoch."
            )
            persistence_text = (
                " CUDA/XPU contexts, actor networks and CPU worker processes remain resident."
                if self.persistent_training_actors.isChecked()
                else " Actor runtimes may be recreated between collection calls."
            )
            orpd_text = (
                " Development-case ORPD stages use the same FP64 accelerator evaluator as the "
                "comparative engine"
                + (
                    f" and merge compatible episode populations for up to "
                    f"{self.training_cross_batch.value()} candidates."
                    if batching_enabled
                    else "."
                )
                if self.accelerated_training_orpd.isChecked()
                else " Development-case ORPD stages use the reference host evaluator."
            )
            self.accelerator_status.setText(
                "Initial synchronous actor plan: "
                + plan.summary()
                + ". "
                + allocation_text
                + persistence_text
                + orpd_text
                + " Shares refer to rollout episodes/transitions, not exact hardware utilization. "
                "All accepted trajectories use one policy snapshot and PPO starts only after "
                "all matching lanes return. " + self._device_text + warning
            )
        except Exception as exc:
            self.accelerator_status.setText(str(exc))

    def apply_policy_configuration(self) -> None:
        if self.no_ai_mode.isChecked():
            parameters = dict(self.state.config.algorithm_parameters.get("CALO", {}))
            for key in (
                "policy_id",
                "policy_checkpoint",
                "policy_sha256",
                "policy_architecture_version",
                "policy_state_schema_version",
                "policy_action_schema_version",
                "policy_training_environment_version",
                "policy_qualification_status",
                "policy_grade",
            ):
                parameters.pop(key, None)
            parameters.update(
                {
                    "use_ai": False,
                    "strict_policy_binding": False,
                    "deterministic_policy": bool(self.deterministic.isChecked()),
                    "allow_unqualified_policy": False,
                }
            )
            self.state.config.algorithm_parameters["CALO"] = parameters
            self.metadata.setPlainText(
                "No-AI CALO selected. Runtime uses rule-based cognitive priors plus current-run "
                "contextual credit/memory only; no neural policy checkpoint is loaded."
            )
            self.state.update_config()
            self.stage_completed.emit()
            return
        path = Path(self.path.text().strip())
        if not path.exists():
            QMessageBox.critical(
                self, "CALO policy configuration", "Select a valid CALO policy checkpoint first."
            )
            return
        try:
            policy = self._selected_policy()
            if policy is None or Path(policy.checkpoint_path).resolve() != path.resolve():
                policy = self.state.policy_registry.register(path)
                self.refresh_policy_library()
                self._select_policy_id(policy.id)
            self.state.config.algorithm_parameters.setdefault("CALO", {})["use_ai"] = True
            binding = self.state.policy_registry.bind_to_experiment_config(
                policy.id,
                self.state.config,
                deterministic=self.deterministic.isChecked(),
                allow_unqualified=self.allow_unqualified.isChecked(),
            )
            binding["policy_name"] = policy.name
            self.metadata.setPlainText(json.dumps({**policy.metadata, **binding}, indent=2))
        except Exception as exc:
            QMessageBox.critical(self, "CALO policy configuration", str(exc))
            return
        self.state.update_config()
        self.stage_completed.emit()

    def choose_policy(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CALO policy",
            "",
            "PyTorch checkpoint (*.pt)",
        )
        if path:
            self.path.setText(path)

    def inspect_policy(self) -> None:
        path = Path(self.path.text())
        if not path.exists():
            self.metadata.setPlainText("Policy file was not found.")
            return
        try:
            payload = load_checkpoint(path, map_location="cpu")
            metadata = payload.get("metadata", {})
            metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.metadata.setPlainText(json.dumps(metadata, indent=2))
        except Exception as exc:
            self.metadata.setPlainText(str(exc))

    def _update_training_mode_controls(self) -> None:
        mode = (
            str(self.training_mode.currentData() or "cumulative")
            if hasattr(self, "training_mode")
            else "cumulative"
        )
        if hasattr(self, "epochs"):
            self.epochs.setEnabled(mode != "indefinite")
            self.epochs.setToolTip(
                "Absolute cumulative epoch target"
                if mode == "cumulative"
                else "Number of additional epochs to run from the saved optimizer/RNG state"
                if mode == "additional"
                else "Ignored in indefinite mode; training continues until Safe Stop and remains resumable"
            )

    def train_policy(self) -> None:
        weighted = str(self.rollout_mode.currentData()) == "weighted"
        default_path = self.path.text()
        frozen_policy = (
            Path(__file__).resolve().parents[2] / "data" / "trained_models" / "calo_policy_v2.pt"
        )
        if weighted:
            default_path = str(frozen_policy.with_name("calo_policy_v4_1_candidate.pt"))
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save trained CALO policy candidate" if weighted else "Save trained CALO policy",
            default_path,
            "PyTorch checkpoint (*.pt)",
        )
        if not path:
            return
        if weighted and Path(path).resolve() == frozen_policy.resolve():
            QMessageBox.critical(
                self,
                "Frozen CALO protection",
                "The frozen CALO v2 policy cannot be overwritten by candidate training. Save the "
                "candidate under a new filename, validate it, and create a new freeze manifest "
                "before using it in a TEST campaign.",
            )
            return
        selected_training_device = str(self.training_device.currentData())
        device_info = available_training_devices()
        if (
            not weighted
            and selected_training_device == "auto"
            and device_info["recommended_device"] == "xpu_sidecar"
        ):
            selected_training_device = "xpu_sidecar"
        if weighted and selected_training_device == "xpu_sidecar":
            QMessageBox.critical(
                self,
                "Policy training configuration",
                "Weighted multi-device training requires the PPO learner in the primary runtime. "
                "Choose Automatic, NVIDIA CUDA, direct Intel XPU, or CPU. The secondary XPU "
                "runtime is still used automatically as the XPU actor lane.",
            )
            return
        share_total = (
            self.cuda_rollout_share.value()
            + self.xpu_rollout_share.value()
            + self.cpu_rollout_share.value()
        )
        if weighted and share_total != 100:
            QMessageBox.critical(
                self,
                "Policy training configuration",
                f"The CUDA/XPU/CPU rollout shares total {share_total}%. They must total 100%.",
            )
            return
        historical_options = self.historical_experience.policy_training_options()
        common = dict(
            epochs=self.epochs.value(),
            episodes_per_epoch=self.episodes.value(),
            horizon=self.horizon.value(),
            seed=self.seed.value(),
            learning_rate=self.lr.value(),
            gamma=self.gamma.value(),
            gae_lambda=self.gae_lambda.value(),
            clip_ratio=self.clip_ratio.value(),
            ppo_epochs=self.ppo_epochs.value(),
            minibatch_size=self.minibatch.value(),
            population_size=self.training_population.value(),
            rollout_workers=self.rollout_workers.value(),
            ppo_device=selected_training_device,
            development_cases=tuple(
                item.strip() for item in self.development_cases.text().split(",") if item.strip()
            ),
            historical_repository=historical_options["historical_repository"],
            use_historical_trajectories=historical_options["use_historical_trajectories"],
            historical_pretraining_epochs=historical_options["historical_pretraining_epochs"],
            training_mode=str(self.training_mode.currentData() or "cumulative"),
            checkpoint_interval_epochs=self.checkpoint_interval.value(),
            deployable_checkpoint_interval_epochs=self.deployable_interval.value(),
            qualification_interval_epochs=self.qualification_interval.value(),
            policy_lineage_name=self.policy_lineage_name.text().strip() or Path(path).stem,
            policy_phase_index=int(getattr(self, "_pending_policy_phase_index", 1) or 1),
            initial_policy_checkpoint=str(
                getattr(self, "_pending_initial_policy_checkpoint", "") or ""
            ),
            keep_resume_after_completion=True,
        )
        if weighted:
            config = HeterogeneousTrainingConfig(
                **common,
                heterogeneous_rollouts=True,
                cuda_rollout_share=self.cuda_rollout_share.value(),
                xpu_rollout_share=self.xpu_rollout_share.value(),
                cpu_rollout_share=self.cpu_rollout_share.value(),
                throughput_adaptive_rollouts=self.auto_tuned_training.isChecked(),
                persistent_actor_workers=self.persistent_training_actors.isChecked(),
                actor_calibration_episodes=self.training_calibration_episodes.value(),
                use_accelerated_orpd_rollouts=self.accelerated_training_orpd.isChecked(),
                training_cross_episode_batching=self.cross_episode_training_batch.isChecked(),
                training_batch_window_ms=self.training_batch_window.value(),
                training_max_cross_batch=self.training_cross_batch.value(),
                training_tensor_batch_size=self.training_tensor_batch.value(),
            )
        else:
            config = TrainingConfig(**common)
        try:
            lineage_name = str(config.policy_lineage_name or Path(path).stem)
            pending_lineage = str(getattr(self, "_pending_policy_lineage_id", "") or "")
            existing = next(
                (
                    row
                    for row in self.state.database.list_policy_lineages(include_archived=True)
                    if str(row["name"]) == lineage_name
                ),
                None,
            )
            config.policy_lineage_id = pending_lineage or (
                str(existing["id"])
                if existing
                else self.state.policy_registry.create_lineage(lineage_name)
            )
            self._pending_initial_policy_checkpoint = ""
            self._pending_policy_lineage_id = ""
            self._pending_policy_phase_index = 1
        except Exception as exc:
            QMessageBox.critical(self, "Policy lineage", str(exc))
            return
        self._launch_training(config, path)

    def _launch_training(self, config, path: str, *, resume_task_id: str = "") -> None:
        if self.state.task_status.busy:
            QMessageBox.information(self, "Task busy", "Wait for the active task to finish first.")
            return
        configured_resume = str(getattr(config, "resume_checkpoint", "") or "").strip()
        resume_path = (
            Path(configured_resume) if configured_resume else Path(path).with_suffix(".resume.pt")
        )
        config.resume_checkpoint = str(resume_path)
        state_payload = {
            "output_path": str(path),
            "resume_checkpoint": str(resume_path),
            "config": asdict(config),
            "heterogeneous": isinstance(config, HeterogeneousTrainingConfig),
        }
        if resume_task_id:
            task_id = resume_task_id
            self.state.resume_service.update(
                task_id,
                status=ResumeStatus.RUNNING,
                state=state_payload,
                resumable=True,
            )
        else:
            task_id = self.state.resume_service.register(
                ResumeTaskType.POLICY_TRAINING,
                f"CALO policy training → {Path(path).name}",
                state_payload,
                total=(
                    0
                    if str(getattr(config, "training_mode", "cumulative")) == "indefinite"
                    else int(config.epochs)
                ),
                status=ResumeStatus.RUNNING,
            )
        config.resume_task_id = task_id
        self.training_resume_task_id = task_id
        self.train_button.setEnabled(False)
        self.resume_training_button.setEnabled(False)
        self.metadata.setPlainText(
            "CALO v5 policy-lineage training is running with crash-safe exact-resume checkpoints and immutable usable policy snapshots. "
            "A completed checkpoint remains a usable policy; later sessions continue the same lineage without changing experiments already bound to an older SHA."
        )
        self.worker = TrainingWorker(config, path)
        self.worker.progress.connect(self._training_progress)
        self.worker.completed.connect(self._training_done)
        self.worker.cancelled.connect(self._training_cancelled)
        self.worker.failed.connect(self._training_failed)
        self.state.task_status.cancel_requested.connect(self._cancel_training)
        self.state.task_status.begin(
            "Training CALO policy",
            detail=(
                "Resuming from the last completed PPO epoch"
                if Path(config.resume_checkpoint).is_file()
                else "Initializing reproducible training"
            ),
            progress=0,
            cancellable=True,
        )
        self.worker.start()

    def resume_saved_training(self) -> None:
        items = self.state.resume_service.list_all(
            task_type=ResumeTaskType.POLICY_TRAINING, resumable_only=True
        )
        if not items:
            QMessageBox.information(
                self,
                "Policy training resume",
                "No resumable CALO policy-training checkpoint was found.",
            )
            return
        # Exact resume is a training-session operation, not merely a model-weight operation. When
        # several resumable sessions exist the user must select the intended lineage/session rather
        # than silently resuming whichever task was updated most recently.
        if len(items) > 1:
            labels = [
                f"{entry.title} · {entry.status} · updated {entry.updated_at}" for entry in items
            ]
            choice, accepted = QInputDialog.getItem(
                self, "Resume exact policy training", "Saved training session", labels, 0, False
            )
            if not accepted:
                return
            item = items[labels.index(choice)]
        else:
            item = items[0]
        payload = dict(item.state)
        config_data = dict(payload.get("config", {}))
        cls = HeterogeneousTrainingConfig if payload.get("heterogeneous") else TrainingConfig
        valid = {
            key: value for key, value in config_data.items() if key in cls.__dataclass_fields__
        }
        if "development_cases" in valid:
            valid["development_cases"] = tuple(valid["development_cases"])
        config = cls(**valid)
        if str(item.status) == ResumeStatus.COMPLETED.value:
            config.training_mode = str(self.training_mode.currentData() or "additional")
            config.epochs = int(self.epochs.value())
            config.checkpoint_interval_epochs = int(self.checkpoint_interval.value())
            config.deployable_checkpoint_interval_epochs = int(self.deployable_interval.value())
            config.qualification_interval_epochs = int(self.qualification_interval.value())
        path = str(payload.get("output_path", ""))
        if not path:
            QMessageBox.critical(
                self,
                "Policy training resume",
                "The saved task does not contain an output checkpoint path.",
            )
            return
        if str(item.status) == ResumeStatus.COMPLETED.value:
            # Never overwrite a previously completed/deployable policy artifact. Continue the exact
            # optimizer/RNG state into a fresh working alias; immutable lineage checkpoints remain stable.
            source = Path(path)
            candidate = source.with_name(source.stem + "_continued" + source.suffix)
            counter = 2
            while candidate.exists():
                candidate = source.with_name(f"{source.stem}_continued_{counter}{source.suffix}")
                counter += 1
            path = str(candidate)
        self._launch_training(config, path, resume_task_id=item.id)

    def _training_progress(self, percent: int, detail: str) -> None:
        self.state.task_status.update(percent, detail)
        if self.training_resume_task_id:
            self.state.resume_service.update(
                self.training_resume_task_id,
                current=max(0, int(percent)),
                total=100,
                status=ResumeStatus.RUNNING,
            )

    def _cancel_training(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()

    def _disconnect_training_cancel(self) -> None:
        try:
            self.state.task_status.cancel_requested.disconnect(self._cancel_training)
        except TypeError:
            pass

    def _register_training_snapshots(self, output_path: str, config=None) -> list[str]:
        """Register deployable lineage snapshots without changing the active/default policy."""
        output = Path(output_path)
        lineage_dir = output.parent / f"{output.stem}_lineage"
        if not lineage_dir.is_dir():
            return []
        lineage_id = (
            str(getattr(config, "policy_lineage_id", "") or "") if config is not None else ""
        )
        registered: list[str] = []
        known_sha = (
            {
                str(row.get("sha256", "")).lower()
                for row in self.state.database.list_policy_checkpoints(lineage_id)
            }
            if lineage_id
            else set()
        )
        for snapshot in sorted(lineage_dir.glob("epoch_*.pt")):
            try:
                payload = load_checkpoint(snapshot, map_location="cpu")
                metadata = dict(payload.get("metadata", {}) or {})
                snapshot_lineage = str(metadata.get("policy_lineage_id", "") or lineage_id)
                cumulative_epoch = int(metadata.get("cumulative_epoch", 0) or 0)
                lineage_name = str(metadata.get("policy_lineage_name", "") or "CALO-policy")
                policy = self.state.policy_registry.register(
                    str(snapshot), name=f"{lineage_name}@{cumulative_epoch}", status="candidate"
                )
                if snapshot_lineage and policy.sha256.lower() not in known_sha:
                    self.state.policy_registry.lineages.register_checkpoint(
                        snapshot_lineage,
                        snapshot,
                        cumulative_epoch=cumulative_epoch,
                        phase_index=int(metadata.get("policy_phase_index", 1) or 1),
                        resume_path=output.with_suffix(".resume.pt"),
                        metadata={
                            "policy_id": policy.id,
                            "policy_name": policy.name,
                            "terminal": False,
                        },
                    )
                    known_sha.add(policy.sha256.lower())
                registered.append(policy.id)
            except Exception:
                # A partial/crash-written artifact is intentionally ignored; CheckpointManager's
                # atomic write means valid snapshots remain self-contained and discoverable.
                continue
        return registered

    def _training_done(self, path: str) -> None:
        self._disconnect_training_cancel()
        self.train_button.setEnabled(True)
        self.resume_training_button.setEnabled(True)
        if self.training_resume_task_id:
            self.state.resume_service.update(
                self.training_resume_task_id,
                status=ResumeStatus.COMPLETED,
                current=100,
                total=100,
                resumable=True,
            )
        selected_policy = None
        try:
            config = getattr(self.worker, "config", None)
            # Read the mutable working alias only to locate its immutable terminal snapshot.
            alias_payload = load_checkpoint(path, map_location="cpu")
            alias_metadata = dict(alias_payload.get("metadata", {}) or {})
            immutable_path = str(alias_metadata.get("immutable_terminal_checkpoint", "") or "")
            self._register_training_snapshots(path, config)
            if immutable_path and Path(immutable_path).is_file():
                selected_policy = self.state.policy_registry.register(
                    immutable_path,
                    name=f"{alias_metadata.get('policy_lineage_name') or Path(path).stem}@{int(alias_metadata.get('cumulative_epoch', 0) or 0)}",
                    status="candidate",
                )
                self.path.setText(immutable_path)
            else:
                # Compatibility fallback for a legacy trainer that did not emit immutable snapshots.
                selected_policy = self.state.policy_registry.register(path, status="candidate")
                self.path.setText(path)
            self.refresh_policy_library()
            self._select_policy_id(selected_policy.id)
        except Exception:
            self.path.setText(path)
        self.inspect_policy()
        self.state.task_status.finish("CALO policy training completed")
        QMessageBox.information(
            self,
            "CALO policy",
            "Training session completed. An immutable usable checkpoint was saved and the exact training state remains resumable for future sessions.",
        )

    def _training_cancelled(self, message: str) -> None:
        self._disconnect_training_cancel()
        self.train_button.setEnabled(True)
        self.resume_training_button.setEnabled(True)
        try:
            if self.worker is not None:
                self._register_training_snapshots(
                    str(self.worker.path), getattr(self.worker, "config", None)
                )
                self.refresh_policy_library()
        except Exception:
            pass
        if self.training_resume_task_id:
            self.state.resume_service.update(
                self.training_resume_task_id, status=ResumeStatus.PAUSED, resumable=True
            )
        self.state.task_status.cancelled(message)

    def _training_failed(self, message: str) -> None:
        self._disconnect_training_cancel()
        self.train_button.setEnabled(True)
        self.resume_training_button.setEnabled(True)
        if self.training_resume_task_id:
            self.state.resume_service.update(
                self.training_resume_task_id, status=ResumeStatus.INTERRUPTED, resumable=True
            )
        self.state.task_status.fail(message)
        QMessageBox.critical(self, "Policy training failed", message)

    def set_experiment_navigation_enabled(self, enabled: bool) -> None:
        self.ablation_button.setEnabled(bool(enabled))
