"""CALO policy, cognitive architecture, and training controls."""

from __future__ import annotations

import logging
from types import SimpleNamespace

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
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.ai.model_io import checkpoint_sha256, load_checkpoint
from calo_rpd_studio.algorithms.calo.training import (
    TrainingCancelled,
    TrainingConfig,
    available_training_devices,
    recommended_rollout_workers,
    recommended_worker_distribution,
    train_policy,
    train_policy_parallel,
)
from calo_rpd_studio.algorithms.calo.competitive_training import (
    discard_recovery_session,
    list_recoverable_sessions,
    recover_competitive_session,
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
from calo_rpd_studio.resume.models import ResumeStatus, ResumeTaskType
from calo_rpd_studio.compute.training_resources import build_training_resource_plan, protected_rollout_shares


_LOG = logging.getLogger(__name__)

class TrainingWorker(QThread):
    progress = pyqtSignal(int, str)
    session_state = pyqtSignal(object)
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
            pr = int(getattr(self.config, "parallel_runs", 1))
            if pr >= 1 and str(self.config.ppo_device).lower() != "xpu_sidecar":
                # v5.9 uses the competitive branch coordinator even for one branch so Cumulative,
                # Infinite, Safe Stop, exact branch resume, champion tracking and Base-Guided Fork
                # all share one scientifically consistent persistence contract.
                result = train_policy_parallel(
                    self.config,
                    self.path,
                    parallel_runs=pr,
                    progress_callback=self.progress.emit,
                    cancel_callback=self._cancel_event.is_set,
                    session_state_callback=self.session_state.emit,
                )
                status = str(getattr(getattr(result, "status", ""), "value", getattr(result, "status", "")))
                if status.startswith("SAFE_STOPPED"):
                    epoch = int(getattr(result, "common_resume_epoch", 0) or 0)
                    degraded = tuple(getattr(result, "degraded_branches", ()) or ())
                    detail = f"CALO policy training safe-stopped at common exact epoch {epoch}."
                    if degraded:
                        detail += " Forced termination after the grace deadline: " + ", ".join(degraded)
                    self.cancelled.emit(detail)
                    return
                selected = str(getattr(result, "selected_artifact_path", "") or "")
                self.completed.emit(selected or str(getattr(result, "output_path", self.path)))
                return
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
                "Governing CALO intelligence control plane. Train/provision first, qualify candidates, and explicitly activate one integrity-verified compatible policy before Power System can unlock. Safe-80 compute protection governs protected parallel training and queued branch admission.",
            )
        )

        training = QGroupBox("Policy provisioning and reproducible training")
        self.training_group = training
        training_form = QFormLayout(training)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 2_000_000_000)
        self.epochs.setValue(24)
        self.training_mode = QComboBox()
        self.training_mode.addItem("Cumulative fixed-length session", "cumulative")
        self.training_mode.addItem("Infinite until Safe Stop", "indefinite")
        self.training_mode.currentIndexChanged.connect(self._update_training_mode_controls)
        self.policy_lineage_name = QLineEdit("CALO-policy-lineage")
        self.policy_lineage_name.setToolTip(
            "Stable policy family name. Continued sessions add immutable checkpoints to this lineage."
        )
        self.checkpoint_interval = QSpinBox()
        self.checkpoint_interval.setRange(10, 10)
        self.checkpoint_interval.setValue(10)
        self.checkpoint_interval.setToolTip(
            "v5.9 keeps bounded rolling temporary exact states every 10 epochs on disk; the exact session-start state is also a valid Safe Stop point. No permanent intermediate snapshots are created before transactional commit."
        )
        self.qualification_interval = QSpinBox()
        self.qualification_interval.setRange(0, 0)
        self.qualification_interval.setValue(0)
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
        self.training_device.currentIndexChanged.connect(self._update_training_plan)

        self.rollout_mode = QComboBox()
        self.rollout_mode.addItem(
            "Accelerator-routed policy actors — CUDA 100% episode routing when available",
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
        self.training_cuda_priority.setToolTip("CUDA-priority rollout episode/transition routing; this does not mean every CALO environment operation is GPU-resident")
        self.training_cuda_only = QPushButton("100% CUDA")
        self.training_cuda_only.setToolTip(
            "Route all compatible rollout episodes and the PPO learner to NVIDIA CUDA. v6.4 Stage B also enables the device-resident synthetic curriculum evaluation kernel when its parity gate passes."
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
            control.valueChanged.connect(self._on_rollout_shares_changed)

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
        self.device_resident_synthetic = QCheckBox(
            "Use v6.4 device-resident FP64 synthetic curriculum evaluation on CUDA/XPU"
        )
        self.device_resident_synthetic.setChecked(True)
        self.device_resident_synthetic.setToolTip(
            "Stage B keeps generated curriculum tasks and their objective/constraint tensors resident on the admitted accelerator, cross-episode microbatches population evaluations, and performs a fail-closed NumPy-reference parity check before trusting each generated task. CALO controller/archive logic remains scientifically identical to the reference transition kernel."
        )
        self.accelerated_training_orpd = QCheckBox(
            "Use FP64 accelerator-native ORPD evaluation in development-case rollouts"
        )
        self.accelerated_training_orpd.setChecked(True)
        self.accelerated_training_orpd.setToolTip(
            "Uses the exact declared development ExperimentConfig objective/controls/PF/scenarios with accelerator-native FP64 ORPD evaluation."
        )
        self.cross_episode_training_batch = QCheckBox(
            "Batch compatible synthetic/ORPD populations across simultaneous rollout episodes"
        )
        self.cross_episode_training_batch.setChecked(True)

        self.enable_development_suite = QCheckBox(
            "Enable real ORPD policy-development suite after the synthetic curriculum"
        )
        self.enable_development_suite.setChecked(True)
        self.development_cases = QLineEdit("case30, case57")
        self.development_cases.setToolTip(
            "Comma-separated TRAIN/DEVELOPMENT cases only. case118 and case300 are protected holdouts and cannot be used here by default."
        )
        default_development_config = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "examples"
            / "policy_development_active_loss.yaml"
        )
        self.development_experiment_config = QLineEdit(str(default_development_config))
        self.development_experiment_config.setToolTip(
            "Exact ExperimentConfig template used for every development case. The case_name is replaced by each selected development case; objective, controls, PF options, robust scenarios and tolerances are preserved."
        )
        self.development_config_browse = QPushButton("Browse…")
        self.development_config_browse.clicked.connect(self._browse_development_config)
        development_config_row = QWidget()
        development_config_layout = QHBoxLayout(development_config_row)
        development_config_layout.setContentsMargins(0, 0, 0, 0)
        development_config_layout.addWidget(self.development_experiment_config, 1)
        development_config_layout.addWidget(self.development_config_browse)

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
            self.device_resident_synthetic,
            self.accelerated_training_orpd,
            self.cross_episode_training_batch,
            self.enable_development_suite,
        ):
            control.toggled.connect(self._update_training_plan)
        for control in (
            self.training_calibration_episodes,
            self.training_tensor_batch,
            self.training_cross_batch,
            self.training_batch_window,
        ):
            control.valueChanged.connect(self._update_training_plan)
        self.development_cases.textChanged.connect(self._update_training_plan)
        self.development_experiment_config.textChanged.connect(self._update_training_plan)

        self.rollout_workers = QSpinBox()
        self.rollout_workers.setRange(1, max(1, recommended_rollout_workers() + 8))
        self.rollout_workers.setValue(recommended_rollout_workers(self.episodes.value()))
        self.rollout_workers.setToolTip(
            "Maximum host CPU rollout processes used only when the CPU rollout lane has assigned episodes. "
            "This is not the number of CUDA/XPU actor processes and is further clamped by the global Safe-80 per-branch CPU budget."
        )
        self.recommended_workers_button = QPushButton("Use recommended")
        self.recommended_workers_button.clicked.connect(self._use_recommended_workers)
        worker_row = QWidget()
        worker_layout = QHBoxLayout(worker_row)
        worker_layout.setContentsMargins(0, 0, 0, 0)
        worker_layout.addWidget(self.rollout_workers, 1)
        worker_layout.addWidget(self.recommended_workers_button)
        self.episodes.valueChanged.connect(self._use_recommended_workers)
        self.episodes.valueChanged.connect(self._update_training_plan)

        self.cuda_workers = QSpinBox()
        self.cuda_workers.setRange(0, 999)
        self.cuda_workers.setValue(0)
        self.cuda_workers.setSuffix(" CUDA units")
        self.cuda_workers.setToolTip("Nominal share-planner units used to derive the CUDA rollout percentage. This is not a count of CUDA processes.")
        self.xpu_workers = QSpinBox()
        self.xpu_workers.setRange(0, 999)
        self.xpu_workers.setValue(0)
        self.xpu_workers.setSuffix(" XPU units")
        self.xpu_workers.setToolTip("Nominal share-planner units used to derive the XPU rollout percentage. This is not a count of XPU processes.")
        self.cpu_workers = QSpinBox()
        self.cpu_workers.setRange(0, 999)
        self.cpu_workers.setValue(0)
        self.cpu_workers.setSuffix(" CPU units")
        self.cpu_workers.setToolTip("Nominal share-planner units used to derive the CPU rollout percentage. Actual CPU process count is governed by the CPU rollout process cap and Safe-80 budget.")
        self.recommend_workers_button = QPushButton("Apply recommendation")
        self.recommend_workers_button.setToolTip(
            "Apply the advisory hardware-based rollout routing. The displayed equivalent units only derive percentages; they are not literal CUDA/XPU process counts. The recommendation never replaces the selected routing unless you click this button."
        )
        self.recommend_workers_button.clicked.connect(self._apply_recommended_worker_split)
        task_share_row = QWidget()
        task_share_layout = QHBoxLayout(task_share_row)
        task_share_layout.setContentsMargins(0, 0, 0, 0)
        task_share_layout.setSpacing(7)
        task_share_layout.addWidget(self.cuda_workers, 1)
        task_share_layout.addWidget(self.xpu_workers, 1)
        task_share_layout.addWidget(self.cpu_workers, 1)
        task_share_layout.addWidget(self.recommend_workers_button)
        self.task_share_status = QLabel()
        self.task_share_status.setWordWrap(True)
        self.task_share_status.setObjectName("HelpText")
        for ctrl in (self.cuda_workers, self.xpu_workers, self.cpu_workers):
            ctrl.valueChanged.connect(self._sync_shares_from_workers)
        # v6.4 preserves the Stage-A selected-routing contract while adding the Stage-B accelerator kernels.
        # Recommendations are advisory and are applied only by the explicit Apply recommendation button.
        self.rollout_workers.valueChanged.connect(self._sync_workers_from_shares)

        self.recommended_share_status = QLabel()
        self.recommended_share_status.setWordWrap(True)
        self.recommended_share_status.setObjectName("HelpText")
        self.protected_allocation_status = QLabel()
        self.protected_allocation_status.setWordWrap(True)
        self.protected_allocation_status.setObjectName("HelpText")
        self.runtime_assignment_status = QLabel()
        self.runtime_assignment_status.setWordWrap(True)
        self.runtime_assignment_status.setObjectName("HelpText")
        self.accelerator_status = QLabel()
        self.accelerator_status.setWordWrap(True)
        self._device_text = device_text
        self._sync_workers_from_shares()
        self._update_training_plan()

        self.train_button = QPushButton("Start / continue CALO policy lineage")
        self.train_button.setObjectName("PrimaryButton")
        self.train_button.clicked.connect(self.train_policy)
        self.resume_training_button = QPushButton("Resume exact saved training")
        self.resume_training_button.clicked.connect(self.resume_saved_training)
        self.recover_training_button = QPushButton("Recover interrupted session")
        self.recover_training_button.clicked.connect(self.recover_interrupted_training)
        self.discard_recovery_button = QPushButton("Discard recovery")
        self.discard_recovery_button.clicked.connect(self.discard_interrupted_training_recovery)
        self.training_import_button = QPushButton("Import existing policy")
        self.training_import_button.clicked.connect(self.import_policy)
        training_button_row = QWidget()
        training_button_layout = QHBoxLayout(training_button_row)
        training_button_layout.setContentsMargins(0, 0, 0, 0)
        training_button_layout.addWidget(self.train_button, 1)
        training_button_layout.addWidget(self.resume_training_button)
        training_button_layout.addWidget(self.recover_training_button)
        training_button_layout.addWidget(self.discard_recovery_button)
        training_button_layout.addWidget(self.training_import_button)
        training_form.addRow("Training continuation mode", self.training_mode)
        training_form.addRow("Cumulative session epochs", self.epochs)
        training_form.addRow("Policy lineage", self.policy_lineage_name)
        training_form.addRow("Safe rollback interval", self.checkpoint_interval)
        training_form.addRow("Periodic formal qualification", QLabel("Disabled by design — qualify only saved Base artifacts"))
        self.qualification_interval.hide()
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
        training_form.addRow("Stage-B synthetic accelerator kernel", self.device_resident_synthetic)
        training_form.addRow("Accelerated ORPD rollouts", self.accelerated_training_orpd)
        training_form.addRow("Cross-episode batching", self.cross_episode_training_batch)
        training_form.addRow("Real ORPD development suite", self.enable_development_suite)
        training_form.addRow("Development cases", self.development_cases)
        training_form.addRow("Development formulation", development_config_row)
        training_form.addRow("Calibration episodes/device", self.training_calibration_episodes)
        training_form.addRow("ORPD tensor microbatch", self.training_tensor_batch)
        training_form.addRow("Maximum merged candidates", self.training_cross_batch)
        training_form.addRow("Cross-episode batch window", self.training_batch_window)
        training_form.addRow("CPU rollout process cap", worker_row)
        training_form.addRow("Share planner (equivalent units)", task_share_row)
        training_form.addRow("Selected rollout routing", self.task_share_status)
        training_form.addRow("Recommended routing", self.recommended_share_status)
        training_form.addRow("Safe-80 branch admission", self.protected_allocation_status)
        training_form.addRow("Runtime device mapping", self.runtime_assignment_status)
        training_form.addRow("Execution scope", self.accelerator_status)
        self.parallel_runs = QSpinBox()
        self.parallel_runs.setRange(1, 64)
        self.parallel_runs.setValue(1)
        self.parallel_runs.setSuffix(" runs")
        self.parallel_runs.valueChanged.connect(self._update_training_plan)
        self.parallel_runs.setToolTip(
            "Total competitive branches. Branches are independent exact PPO trajectories; their "
            "neural weights are never averaged. The best scientifically evaluated champion becomes the base model."
        )
        self.parallel_concurrency = QSpinBox()
        self.parallel_concurrency.setRange(1, 64)
        self.parallel_concurrency.setValue(1)
        self.parallel_concurrency.setToolTip(
            "Maximum branches allowed to execute simultaneously. This may be lower than total scientific branches. "
            "The Dashboard Safe-80 ceiling is a hard maximum; remaining branches are queued and exact-resume rotated."
        )
        self.parallel_concurrency.valueChanged.connect(self._update_queue_plan_label)
        self.parallel_concurrency.valueChanged.connect(self._update_training_plan)
        self.queue_plan_status = QLabel()
        self.queue_plan_status.setWordWrap(True)
        self.queue_plan_status.setObjectName("HelpText")
        self.parallel_start_mode = QComboBox()
        self.parallel_start_mode.addItem("New independent branches", "new")
        self.parallel_start_mode.addItem("Exact resume existing branches", "exact_resume")
        self.parallel_start_mode.addItem("Base-Guided Fork from current base", "base_guided_fork")
        self.same_seed_branches = QSpinBox()
        self.same_seed_branches.setRange(0, 64)
        self.same_seed_branches.setValue(1)
        self.incremental_seed_branches = QSpinBox()
        self.incremental_seed_branches.setRange(0, 64)
        self.incremental_seed_branches.setValue(0)
        self.decremental_seed_branches = QSpinBox()
        self.decremental_seed_branches.setRange(0, 64)
        self.decremental_seed_branches.setValue(0)
        self.custom_branch_seeds = QLineEdit()
        self.custom_branch_seeds.setPlaceholderText("Optional comma-separated seeds, e.g. 575,901")
        self.training_scratch_dir = QLineEdit()
        self.training_scratch_dir.setPlaceholderText("Blank = fast local OS temp directory")
        seed_plan_host = QWidget()
        seed_plan_layout = QHBoxLayout(seed_plan_host)
        seed_plan_layout.setContentsMargins(0, 0, 0, 0)
        seed_plan_layout.addWidget(QLabel("same"))
        seed_plan_layout.addWidget(self.same_seed_branches)
        seed_plan_layout.addWidget(QLabel("+seed"))
        seed_plan_layout.addWidget(self.incremental_seed_branches)
        seed_plan_layout.addWidget(QLabel("-seed"))
        seed_plan_layout.addWidget(self.decremental_seed_branches)
        for control in (self.same_seed_branches, self.incremental_seed_branches, self.decremental_seed_branches):
            control.valueChanged.connect(self._sync_parallel_branch_count)
        self.custom_branch_seeds.textChanged.connect(self._sync_parallel_branch_count)
        self.parallel_runs.setReadOnly(True)
        training_form.addRow("Parallel branch start", self.parallel_start_mode)
        training_form.addRow("Branch seed plan", seed_plan_host)
        training_form.addRow("Custom branch seeds", self.custom_branch_seeds)
        training_form.addRow("Total scientific branches", self.parallel_runs)
        training_form.addRow("Maximum simultaneous branches", self.parallel_concurrency)
        training_form.addRow("Protected queue plan", self.queue_plan_status)
        self.safe_parallel_limit = QLabel("Safe parallel limit: waiting for Dashboard system scan")
        self.safe_parallel_limit.setWordWrap(True)
        self.safe_parallel_limit.setObjectName("HelpText")
        training_form.addRow("Safe-80 branch protection", self.safe_parallel_limit)
        training_form.addRow("Training scratch storage", self.training_scratch_dir)
        self._sync_parallel_branch_count()
        self.policy_gate_status = QLabel()
        self.policy_gate_status.setWordWrap(True)
        self.policy_gate_status.setObjectName("HelpText")
        training_form.addRow("Policy availability", self.policy_gate_status)
        training_form.addRow("", training_button_row)
        self._update_training_mode_controls()
        layout.addWidget(training)

        policy_center = QGroupBox(
            "CALO Policy Center — library, qualification, comparison, and activation"
        )
        self.policy_center_group = policy_center
        center_layout = QVBoxLayout(policy_center)
        library_host = QWidget()
        library_layout = QVBoxLayout(library_host)
        library_layout.setContentsMargins(0, 0, 0, 0)
        self.policy_table = QTableWidget(0, 12)
        self.policy_table.setHorizontalHeaderLabels(
            [
                "Active",
                "Policy",
                "Lineage",
                "Epoch",
                "Branches",
                "Seed plan",
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
        for column in range(2, 12):
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
        self.policy_continue_button = QPushButton("Base-guided continue")
        self.policy_exact_resume_button = QPushButton("Exact resume branches")
        self.policy_fork_button = QPushButton("Fork lineage")
        self.policy_refresh_button = QPushButton("Refresh")
        self.show_archived_policies = QCheckBox("Show archived")
        self.show_archived_policies.toggled.connect(lambda _checked: self.refresh_policy_library())
        self.policy_import_button.clicked.connect(self.import_policy)
        self.policy_activate_button.clicked.connect(self.activate_selected_policy)
        self.policy_archive_button.clicked.connect(self.archive_selected_policy)
        self.policy_delete_button.clicked.connect(self.delete_selected_policy)
        self.policy_continue_button.clicked.connect(self.continue_selected_policy)
        self.policy_exact_resume_button.clicked.connect(self.exact_resume_selected_policy)
        self.policy_fork_button.clicked.connect(self.fork_selected_policy)
        self.policy_refresh_button.clicked.connect(self.refresh_policy_library)
        for button in (
            self.policy_import_button,
            self.policy_activate_button,
            self.policy_archive_button,
            self.policy_delete_button,
            self.policy_continue_button,
            self.policy_exact_resume_button,
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
        center_layout.addWidget(library_host)

        qualification = QGroupBox("Policy qualification gate")
        qualification_form = QFormLayout(qualification)
        self.qualification_reference = QComboBox()
        self.qualification_cases = QLineEdit("case30, case57")
        self.qualification_runs = QSpinBox()
        self.qualification_runs.setRange(2, 100)
        self.qualification_runs.setValue(30)
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
        center_layout.addWidget(comparison_host)
        layout.addWidget(policy_center)

        policy = QGroupBox("Policy controller")
        self.policy_controller_group = policy
        form = QFormLayout(policy)
        self.path = QLineEdit("")
        self.path.setReadOnly(True)
        self.path.setPlaceholderText("Activate a compatible policy in the Policy Center")
        choose = QPushButton("Import policy")
        choose.clicked.connect(self.import_policy)
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
        self.architecture_group = architecture
        architecture_layout = QVBoxLayout(architecture)
        description = QLabel(
            "Cognitive state — exact and epsilon-feasible ratios, total and component-wise "
            "constraint violation, objective and constraint progress, population and elite "
            "diversity, separate stagnation states, archive occupancy, remaining evaluation "
            "budget, and online operator credit.\n\n"
            "CALO v6.2 operators — feasible-elite learning, constraint-boundary differential "
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

        self.refresh_policy_library()
        self.state.config_changed.connect(lambda config: self.load_from_config(config))
        self.state.compute_profile_changed.connect(lambda _profile: self._update_compute_protection())
        self.load_from_config(self.state.config)
        self._update_compute_protection()

    def refresh_policy_library(self) -> None:
        bundled = Path(__file__).resolve().parents[2] / "data" / "trained_models"
        self.state.policy_registry.discover_bundled(bundled)
        self._policy_rows = [
            p for p in self.state.policy_registry.list(
                include_archived=self.show_archived_policies.isChecked()
            )
            if not p.checkpoint_path.endswith(".resume.pt")
            and "_lineage" not in str(p.checkpoint_path)
        ]
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
            metadata = dict(policy.metadata or {})
            branch_count = metadata.get("parallel_branches", "")
            seed_plan = metadata.get("parallel_seed_plan", []) or []
            seed_text = ""
            if seed_plan:
                counts = {}
                for item in seed_plan:
                    strategy = str(item.get("strategy", "?")) if isinstance(item, dict) else "?"
                    counts[strategy] = counts.get(strategy, 0) + 1
                seed_text = "/".join(f"{key}:{value}" for key, value in sorted(counts.items()))
            values = [
                "●" if policy.active else "",
                policy.name,
                lineage_name,
                epoch or str(metadata.get("champion_epoch", metadata.get("cumulative_epoch", ""))),
                branch_count,
                seed_text,
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
        self._update_policy_gate_state()
        self._draw_policy_comparison()
        self.state.notify_policy_state_changed()

    def _update_policy_gate_state(self) -> None:
        """Fail closed until a real policy exists and a compatible policy is explicitly active."""
        all_records = self.state.policy_registry.list(include_archived=True)
        usable_records = [p for p in all_records if not p.archived and p.usable]
        active = next(
            (p for p in usable_records if p.active and p.runtime_compatible),
            None,
        )
        has_registered_policy = bool(all_records)

        # Training/import provisioning is always available. With no policy record at all, every
        # other CALO-intelligence block is deliberately disabled so a missing model can never be
        # mistaken for a usable AI controller. Existing archived/incompatible records keep the
        # Policy Center reachable for inspection/restoration/deletion, but never unlock runtime.
        self.training_group.setEnabled(True)
        self.policy_center_group.setEnabled(has_registered_policy)
        # v6.1: when no policy exists, only Training & Provisioning is enabled. Once any
        # policy record exists, the remaining intelligence blocks become inspectable/configurable;
        # scientific runtime application still fails closed unless a qualified compatible policy is active.
        self.policy_controller_group.setEnabled(has_registered_policy)
        self.architecture_group.setEnabled(has_registered_policy)
        self.historical_experience.setEnabled(has_registered_policy)

        if not has_registered_policy:
            self.policy_gate_status.setText(
                "NO CALO POLICY AVAILABLE. Policy-assisted CALO is locked. Train a policy or "
                "import an existing compatible policy first; no random/untrained/default fallback model will be created."
            )
            self.path.clear()
            return
        if active is None:
            compatible = sum(1 for p in usable_records if p.runtime_compatible)
            self.policy_gate_status.setText(
                f"{len(all_records)} policy record(s) registered ({compatible} usable runtime-compatible), "
                "but no compatible policy is active. Qualify/select the intended policy and click Set active before runtime configuration is enabled."
            )
            self.path.clear()
            return

        self.policy_gate_status.setText(
            f"ACTIVE POLICY: {active.name} · {active.grade} · {active.qualification_status} · "
            f"SHA-256 {active.sha256[:12]}… Runtime configuration is gated to this explicit active policy."
        )
        parameters = dict(self.state.config.algorithm_parameters.get("CALO", {}))
        bound_policy_id = str(parameters.get("policy_id", "") or "")
        if not bound_policy_id:
            self.path.setText(active.checkpoint_path)
            self._select_policy_id(active.id)

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
        """Start a same-lineage Base-Guided Fork from the selected deployable policy.

        This is deliberately not Exact Resume: branch optimizer/RNG trajectories are fresh while
        the selected base weights provide the starting knowledge and promotion threshold. Exact
        Resume is launched from a saved resumable training session and restores every branch state.
        """
        policy, checkpoint = self._selected_policy_checkpoint_record()
        if policy is None:
            return
        self._pending_initial_policy_checkpoint = ""
        self.parallel_start_mode.setCurrentIndex(
            max(0, self.parallel_start_mode.findData("base_guided_fork"))
        )
        if checkpoint is not None:
            lineage = self.state.database.get_policy_lineage(str(checkpoint.get("lineage_id", "")))
            self._pending_policy_lineage_id = str(checkpoint.get("lineage_id", ""))
            self._pending_policy_phase_index = int(checkpoint.get("phase_index", 1) or 1) + 1
            self.policy_lineage_name.setText(str((lineage or {}).get("name", policy.name)))
        else:
            self._pending_policy_lineage_id = ""
            self._pending_policy_phase_index = 1
            self.policy_lineage_name.setText(policy.name)
        tc = {}
        if checkpoint is not None:
            tc = (checkpoint.get("metadata") or {}).get("training_config") or {}
        if not tc:
            ckpt_path = Path(policy.checkpoint_path)
            if ckpt_path.is_file():
                try:
                    payload = load_checkpoint(ckpt_path, map_location="cpu")
                    tc = (payload.get("metadata") or {}).get("training_config") or {}
                except Exception:
                    tc = {}
        if tc:
            reply = QMessageBox.question(
                self,
                "Continue training",
                f"Saved training parameters were found in {policy.name}.\n\n"
                "Use the saved parameters from the checkpoint, or keep your current "
                "GUI values?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.epochs.setValue(int(tc.get("epochs", self.epochs.value())))
                self.episodes.setValue(int(tc.get("episodes_per_epoch", self.episodes.value())))
                self.horizon.setValue(int(tc.get("horizon", self.horizon.value())))
                self.seed.setValue(int(tc.get("seed", self.seed.value())))
                self.lr.setValue(float(tc.get("learning_rate", self.lr.value())))
                self.gamma.setValue(float(tc.get("gamma", self.gamma.value())))
                self.gae_lambda.setValue(float(tc.get("gae_lambda", self.gae_lambda.value())))
                self.clip_ratio.setValue(float(tc.get("clip_ratio", self.clip_ratio.value())))
                self.ppo_epochs.setValue(int(tc.get("ppo_epochs", self.ppo_epochs.value())))
                self.minibatch.setValue(int(tc.get("minibatch_size", self.minibatch.value())))
                self.training_population.setValue(
                    int(tc.get("population_size", self.training_population.value()))
                )
                self.rollout_workers.setValue(
                    int(tc.get("rollout_workers", self.rollout_workers.value()))
                )
                self.checkpoint_interval.setValue(
                    int(tc.get("checkpoint_interval_epochs", self.checkpoint_interval.value()))
                )
                self.qualification_interval.setValue(
                    int(
                        tc.get(
                            "qualification_interval_epochs",
                            self.qualification_interval.value(),
                        )
                    )
                )
                device_val = str(tc.get("ppo_device", ""))
                if device_val:
                    idx = self.training_device.findData(device_val)
                    if idx >= 0:
                        self.training_device.setCurrentIndex(idx)
                mode = str(tc.get("training_mode", ""))
                if mode:
                    idx = self.training_mode.findData(mode)
                    if idx >= 0:
                        self.training_mode.setCurrentIndex(idx)
                self.cuda_rollout_share.setValue(
                    int(tc.get("cuda_rollout_share", self.cuda_rollout_share.value()))
                )
                self.xpu_rollout_share.setValue(
                    int(tc.get("xpu_rollout_share", self.xpu_rollout_share.value()))
                )
                self.cpu_rollout_share.setValue(
                    int(tc.get("cpu_rollout_share", self.cpu_rollout_share.value()))
                )
                self.auto_tuned_training.setChecked(
                    bool(tc.get("throughput_adaptive_rollouts", False))
                )
                self.persistent_training_actors.setChecked(
                    bool(tc.get("persistent_actor_workers", True))
                )
                self.device_resident_synthetic.setChecked(
                    bool(tc.get("device_resident_synthetic_rollouts", True))
                )
                saved_development_cases = tuple(tc.get("development_cases", ()) or ())
                self.enable_development_suite.setChecked(bool(saved_development_cases))
                if saved_development_cases:
                    self.development_cases.setText(", ".join(str(item) for item in saved_development_cases))
                saved_development_config = str(tc.get("development_experiment_config_path", "") or "")
                if saved_development_config:
                    self.development_experiment_config.setText(saved_development_config)
                self.accelerated_training_orpd.setChecked(
                    bool(tc.get("use_accelerated_orpd_rollouts", True))
                )
                self.cross_episode_training_batch.setChecked(
                    bool(tc.get("training_cross_episode_batching", True))
                )
                self.training_calibration_episodes.setValue(
                    int(tc.get("actor_calibration_episodes", self.training_calibration_episodes.value()))
                )
                self.training_tensor_batch.setValue(
                    int(tc.get("training_tensor_batch_size", self.training_tensor_batch.value()))
                )
                self.training_cross_batch.setValue(
                    int(tc.get("training_max_cross_batch", self.training_cross_batch.value()))
                )
                self.training_batch_window.setValue(
                    float(tc.get("training_batch_window_ms", self.training_batch_window.value()))
                )
                self.qualification_status.setText(
                    f"Loaded saved training parameters from {policy.name} ({policy.sha256[:12]}…). "
                    "Starting training automatically."
                )
            else:
                self.qualification_status.setText(
                    f"Using current GUI values for {policy.name} ({policy.sha256[:12]}…). "
                    "Starting training automatically."
                )
        else:
            self.qualification_status.setText(
                f"Prepared fine-tuning from {policy.name} ({policy.sha256[:12]}…). "
                "No stored training config found; using current widget values."
            )
        policy_source = Path(policy.checkpoint_path)
        if policy_source.parent.name.endswith("_artifacts"):
            alias_stem = policy_source.parent.name[: -len("_artifacts")]
            output_path = str(policy_source.parent.parent / f"{alias_stem}.pt")
        else:
            output_path = str(policy_source)
        try:
            self._validate_safe_training_capacity()
        except Exception as exc:
            QMessageBox.critical(self, "Safe-80 compute protection", str(exc))
            return
        weighted = str(self.rollout_mode.currentData()) == "weighted"
        selected_training_device = str(self.training_device.currentData())
        device_info = available_training_devices()
        if (
            not weighted
            and selected_training_device == "auto"
            and device_info["recommended_device"] == "xpu_sidecar"
        ):
            selected_training_device = "xpu_sidecar"
        if selected_training_device == "xpu_sidecar" and self.parallel_runs.value() > 1:
            QMessageBox.critical(
                self,
                "Policy training configuration",
                "The secondary XPU sidecar runtime currently supports one branch per training job. "
                "Choose Automatic/direct XPU/CUDA/CPU for competitive multi-branch training, or "
                "reduce the branch count to one.",
            )
            return
        historical_options = self.historical_experience.policy_training_options()
        try:
            custom_seeds = self._custom_seed_values()
            development_cases, development_config_path = self._development_suite_values()
        except Exception as exc:
            QMessageBox.critical(self, "Policy training configuration", str(exc))
            return
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
            development_cases=development_cases,
            development_experiment_config_path=development_config_path,
            historical_repository=historical_options["historical_repository"],
            use_historical_trajectories=historical_options["use_historical_trajectories"],
            historical_pretraining_epochs=historical_options["historical_pretraining_epochs"],
            training_mode=str(self.training_mode.currentData() or "cumulative"),
            checkpoint_interval_epochs=self.checkpoint_interval.value(),
            qualification_interval_epochs=self.qualification_interval.value(),
            policy_lineage_name=self.policy_lineage_name.text().strip() or Path(output_path).stem,
            policy_phase_index=int(getattr(self, "_pending_policy_phase_index", 1) or 1),
            initial_policy_checkpoint=str(
                getattr(self, "_pending_initial_policy_checkpoint", "") or ""
            ),
            keep_resume_after_completion=True,
            parallel_runs=self.parallel_runs.value(),
            parallel_concurrency=self.parallel_concurrency.value(),
            branch_queue_quantum_epochs=10,
            parallel_same_seed_branches=self.same_seed_branches.value(),
            parallel_incremental_branches=self.incremental_seed_branches.value(),
            parallel_decremental_branches=self.decremental_seed_branches.value(),
            parallel_custom_seeds=custom_seeds,
            parallel_start_mode=str(self.parallel_start_mode.currentData() or "new"),
            base_model_checkpoint=(
                str(policy.checkpoint_path)
                if str(self.parallel_start_mode.currentData() or "new") == "base_guided_fork"
                else ""
            ),
            training_scratch_dir=self.training_scratch_dir.text().strip(),
            safe_snapshot_interval_epochs=10,
            safe_parallel_branches=int(self.state.compute_protection_profile.safe_parallel_branches),
            safe_global_cpu_workers=int(self.state.compute_protection_profile.safe_cpu_worker_budget),
            compute_profile_fingerprint=str(self.state.compute_protection_profile.profile_fingerprint),
            compute_topology_fingerprint=str(self.state.compute_topology.fingerprint if self.state.compute_topology is not None else ""),
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
                device_resident_synthetic_rollouts=self.device_resident_synthetic.isChecked(),
                synthetic_cross_episode_batching=self.cross_episode_training_batch.isChecked(),
                synthetic_batch_window_ms=self.training_batch_window.value(),
                synthetic_max_cross_batch=self.training_cross_batch.value(),
            )
        else:
            config = TrainingConfig(**common)
        try:
            lineage_name = str(config.policy_lineage_name or Path(output_path).stem)
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
            self._pending_base_model_checkpoint = ""
            self._pending_policy_lineage_id = ""
            self._pending_policy_phase_index = 1
        except Exception as exc:
            QMessageBox.critical(self, "Policy lineage", str(exc))
            return
        self._launch_training(config, output_path)

    def exact_resume_selected_policy(self) -> None:
        """Resume the exact independent branch states behind the selected current Base Model."""
        policy = self._selected_policy()
        if policy is None:
            return
        source = Path(policy.checkpoint_path).expanduser().resolve()
        if source.parent.name.endswith("_artifacts"):
            alias_stem = source.parent.name[: -len("_artifacts")]
            output_path = source.parent.parent / f"{alias_stem}.pt"
        else:
            output_path = source
        manifest_path = output_path.with_suffix(".branches.json")
        if not manifest_path.is_file():
            QMessageBox.critical(
                self,
                "Exact branch resume",
                "The selected policy has no competitive-branch manifest. Use Base-guided continue "
                "to start fresh branches from this policy instead.",
            )
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if str(manifest.get("base_sha256", "")).lower() != str(policy.sha256).lower():
                raise ValueError(
                    "The selected artifact is not the current Base Model represented by this branch "
                    "manifest. Select the current Base or use Base-guided continue/fork."
                )
            payload = load_checkpoint(source, map_location="cpu")
            metadata = dict(payload.get("metadata", {}) or {})
            config_data = dict(metadata.get("training_config", {}) or {})
            use_hetero = bool(config_data.get("heterogeneous_rollouts", False))
            cls = HeterogeneousTrainingConfig if use_hetero else TrainingConfig
            valid = {
                key: value for key, value in config_data.items() if key in cls.__dataclass_fields__
            }
            if "development_cases" in valid:
                valid["development_cases"] = tuple(valid["development_cases"] or ())
            if "parallel_custom_seeds" in valid:
                valid["parallel_custom_seeds"] = tuple(valid["parallel_custom_seeds"] or ())
            config = cls(**valid)
            config.parallel_start_mode = "exact_resume"
            config.parallel_runs = max(1, len(manifest.get("branches", []) or []))
            config.base_model_checkpoint = str(manifest.get("base_artifact_path", "") or source)

            previous_mode = str(manifest.get("previous_training_mode", "cumulative") or "cumulative")
            mode_labels = ["Cumulative fixed-length session", "Infinite until Safe Stop"]
            default_index = 1 if previous_mode == "indefinite" else 0
            mode_choice, accepted = QInputDialog.getItem(
                self,
                "Exact resume duration",
                "Continue the exact saved branches as:",
                mode_labels,
                default_index,
                False,
            )
            if not accepted:
                return
            if mode_choice == mode_labels[1]:
                config.training_mode = "indefinite"
            else:
                previous_epochs = int(manifest.get("previous_session_epochs", 0) or self.epochs.value())
                session_epochs, accepted = QInputDialog.getInt(
                    self,
                    "Cumulative session epochs",
                    "Epochs to add to every exact-resumed branch in this session:",
                    max(1, previous_epochs),
                    1,
                    2_000_000_000,
                    1,
                )
                if not accepted:
                    return
                config.training_mode = "cumulative"
                config.epochs = int(session_epochs)
            scratch = self.training_scratch_dir.text().strip()
            if scratch:
                config.training_scratch_dir = scratch
            self._launch_training(config, str(output_path))
        except Exception as exc:
            QMessageBox.critical(self, "Exact branch resume", str(exc))

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
            self._pending_initial_policy_checkpoint = ""
            self._pending_base_model_checkpoint = str(policy.checkpoint_path)
            idx = self.parallel_start_mode.findData("base_guided_fork")
            if idx >= 0:
                self.parallel_start_mode.setCurrentIndex(idx)
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
            activated = self.state.policy_registry.activate(policy.id)
            self.refresh_policy_library()
            self._select_policy_id(activated.id)
            self.path.setText(activated.checkpoint_path)
            self._update_policy_gate_state()
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
        except (TypeError, RuntimeError, AttributeError):
            pass
        self.qualify_button.setEnabled(True)
        qualification_status = (
            "qualified"
            if result.get("passed") and result.get("native_v59", result.get("native_v41"))
            else "legacy_qualified"
            if result.get("passed")
            else "failed"
        )
        self.state.database.add_policy_qualification(
            qualification_id=result.get("qualification_id", ""),
            policy_id=result.get("candidate_policy_id", ""),
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
                    "latest_qualification_id": result.get("qualification_id", ""),
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
        native_text = "native v5.9-compatible" if result.get("native_v59", result.get("native_v41")) else "legacy-compatible"
        self.qualification_status.setText(
            f"Qualification {'PASS' if result.get('passed') else 'FAIL'} · {native_text} · grade {result.get('grade')} · "
            f"grade index {float(result.get('score', 0.0)):.0f}{ptext}. "
            + " ".join(result.get("reasons", []))
        )
        self.refresh_policy_library()
        self._select_policy_id(str(result.get("candidate_policy_id", "")))

    def _qualification_failed(self, message: str) -> None:
        try:
            self.state.task_status.cancel_requested.disconnect(self.qualification_worker.cancel)
        except (TypeError, RuntimeError, AttributeError):
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
            latest = qualifications[0]
            payload = json.loads(latest.get("metrics_json") or "{}")
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
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=30, ha="right")
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

    @staticmethod
    def _integer_worker_allocation(total: int, cuda_share: int, xpu_share: int, cpu_share: int) -> tuple[int, int, int]:
        """Convert percentage shares to an exact integer worker split without changing the total."""
        total = max(0, int(total))
        shares = [max(0, int(cuda_share)), max(0, int(xpu_share)), max(0, int(cpu_share))]
        share_sum = sum(shares)
        if total == 0:
            return 0, 0, 0
        if share_sum <= 0:
            return 0, 0, total
        exact = [total * value / share_sum for value in shares]
        base = [int(value) for value in exact]
        remainder = total - sum(base)
        order = sorted(range(3), key=lambda i: (-(exact[i] - base[i]), i))
        for index in order[:remainder]:
            base[index] += 1
        return int(base[0]), int(base[1]), int(base[2])

    def _on_rollout_shares_changed(self, *_args) -> None:
        self._sync_workers_from_shares()

    def _sync_workers_from_shares(self, *_args) -> None:
        c, x, p = self._integer_worker_allocation(
            self.rollout_workers.value(),
            self.cuda_rollout_share.value(),
            self.xpu_rollout_share.value(),
            self.cpu_rollout_share.value(),
        )
        self.cuda_workers.blockSignals(True)
        self.xpu_workers.blockSignals(True)
        self.cpu_workers.blockSignals(True)
        self.cuda_workers.setValue(c)
        self.xpu_workers.setValue(x)
        self.cpu_workers.setValue(p)
        self.cuda_workers.blockSignals(False)
        self.xpu_workers.blockSignals(False)
        self.cpu_workers.blockSignals(False)
        self._update_task_share_status()
        self._update_training_plan()

    def _apply_recommended_worker_split(self, *_args) -> None:
        total = self.rollout_workers.value()
        dist = recommended_worker_distribution(total)
        self.cuda_workers.blockSignals(True)
        self.xpu_workers.blockSignals(True)
        self.cpu_workers.blockSignals(True)
        self.cuda_workers.setValue(dist.get("cuda", 0))
        self.xpu_workers.setValue(dist.get("xpu", 0))
        self.cpu_workers.setValue(dist.get("cpu", total))
        self.cuda_workers.blockSignals(False)
        self.xpu_workers.blockSignals(False)
        self.cpu_workers.blockSignals(False)
        self._sync_shares_from_workers()

    def _sync_shares_from_workers(self, *_args) -> None:
        c, x, p = self.cuda_workers.value(), self.xpu_workers.value(), self.cpu_workers.value()
        total = c + x + p
        if total <= 0:
            self._update_task_share_status()
            return
        self.cuda_rollout_share.blockSignals(True)
        self.xpu_rollout_share.blockSignals(True)
        self.cpu_rollout_share.blockSignals(True)
        cuda_pct = round(100 * c / total)
        xpu_pct = round(100 * x / total)
        self.cuda_rollout_share.setValue(cuda_pct)
        self.xpu_rollout_share.setValue(xpu_pct)
        self.cpu_rollout_share.setValue(100 - cuda_pct - xpu_pct)
        self.cuda_rollout_share.blockSignals(False)
        self.xpu_rollout_share.blockSignals(False)
        self.cpu_rollout_share.blockSignals(False)
        self._update_task_share_status()
        self._update_training_plan()

    def _update_task_share_status(self) -> None:
        if not hasattr(self, "task_share_status"):
            return
        c, x, p = self.cuda_workers.value(), self.xpu_workers.value(), self.cpu_workers.value()
        total = c + x + p
        workers_total = self.rollout_workers.value()
        cuda_pct = int(self.cuda_rollout_share.value())
        xpu_pct = int(self.xpu_rollout_share.value())
        cpu_pct = int(self.cpu_rollout_share.value())
        match_str = "✓" if total == workers_total else f"WARNING: planner-unit total {total} ≠ reference scale {workers_total}"
        self.task_share_status.setText(
            f"Selected rollout routing: CUDA {cuda_pct}% · XPU {xpu_pct}% · CPU {cpu_pct}%. "
            f"Equivalent planner units: CUDA {c} · XPU {x} · CPU {p} · {match_str}. "
            "Planner units derive the percentages only; they are NOT counts of CUDA/XPU processes."
        )
        if hasattr(self, "recommended_share_status"):
            dist = recommended_worker_distribution(workers_total)
            rc, rx, rp = int(dist.get('cuda', 0)), int(dist.get('xpu', 0)), int(dist.get('cpu', 0))
            denom = max(1, rc + rx + rp)
            rc_pct = round(100 * rc / denom)
            rx_pct = round(100 * rx / denom)
            rp_pct = 100 - rc_pct - rx_pct
            self.recommended_share_status.setText(
                "Advisory only — not selected automatically: "
                f"CUDA {rc_pct}% · XPU {rx_pct}% · CPU {rp_pct}% "
                f"(equivalent planner units {rc}/{rx}/{rp}). Click Apply recommendation to replace the selected routing."
            )

    def _set_training_split(self, cuda: int, xpu: int, cpu: int) -> None:
        self.cuda_rollout_share.blockSignals(True)
        self.xpu_rollout_share.blockSignals(True)
        self.cpu_rollout_share.blockSignals(True)
        self.cuda_rollout_share.setValue(int(cuda))
        self.xpu_rollout_share.setValue(int(xpu))
        self.cpu_rollout_share.setValue(int(cpu))
        self.cuda_rollout_share.blockSignals(False)
        self.xpu_rollout_share.blockSignals(False)
        self.cpu_rollout_share.blockSignals(False)
        if cuda == 100:
            weighted_idx = self.rollout_mode.findData("weighted")
            if weighted_idx >= 0:
                self.rollout_mode.setCurrentIndex(weighted_idx)
            cuda_idx = self.training_device.findData("cuda")
            if cuda_idx >= 0 and self.training_device.model().item(cuda_idx).isEnabled():
                self.training_device.setCurrentIndex(cuda_idx)
        self._sync_workers_from_shares()
        self._update_task_share_status()
        self._update_training_plan()

    def _runtime_device_request_text(self) -> str:
        requested = str(self.training_device.currentData() or "auto").lower()
        topology = getattr(self.state, "compute_topology", None)
        if topology is None:
            return f"Requested primary learner device: {requested}. Dashboard system map is not available yet."
        devices = list(getattr(topology, "devices", ()) or ())
        if requested == "cpu":
            return "Requested primary learner device: CPU. No accelerator branch is requested."
        if requested == "xpu_sidecar":
            matches = [d for d in devices if d.backend == "xpu" and d.runtime != "primary"]
            if matches:
                d = matches[0]
                return f"Requested secondary XPU runtime: {d.mapping_text} · {d.name}. This is an auxiliary actor/evaluator runtime, not a full competitive branch."
            return "Requested secondary XPU runtime, but no mapped sidecar is currently available."
        if requested.startswith("cuda"):
            matches = [d for d in devices if d.backend == "cuda" and d.full_training_branch]
        elif requested.startswith("xpu"):
            matches = [d for d in devices if d.backend == "xpu" and d.runtime == "primary" and d.full_training_branch]
        else:
            matches = [d for d in devices if d.full_training_branch and d.backend in {"cuda", "xpu"}]
        if matches:
            d = matches[0]
            prefix = "Automatic protected primary preview" if requested == "auto" else "Requested primary learner device"
            return f"{prefix}: {d.mapping_text} · {d.name} · backend {d.backend}. Final branch assignment is frozen by the Safe-80 scheduler at launch."
        return f"Requested primary learner device: {requested}. No validated matching full-branch accelerator is currently mapped; launch will fail closed rather than silently spill to CPU."

    def _protected_assignment_preview_text(self) -> str:
        profile = getattr(self.state, "compute_protection_profile", None)
        topology = getattr(self.state, "compute_topology", None)
        if profile is None or topology is None:
            return "Safe-80 preview unavailable until Dashboard system mapping is complete."
        if not hasattr(self, "parallel_runs") or not hasattr(self, "parallel_concurrency"):
            return (
                f"Safe-80 profile {profile.profile_name}: global CPU worker budget {profile.safe_cpu_worker_budget}; "
                f"hard simultaneous branch ceiling {profile.safe_parallel_branches}."
            )
        total = max(1, int(self.parallel_runs.value()))
        concurrency = max(1, min(total, int(self.parallel_concurrency.value())))
        preview = SimpleNamespace(
            ppo_device=str(self.training_device.currentData() or "auto"),
            parallel_concurrency=concurrency,
            safe_global_cpu_workers=int(profile.safe_cpu_worker_budget),
            compute_topology_fingerprint=str(topology.fingerprint),
        )
        try:
            plan = build_training_resource_plan(preview, total, topology=topology, profile=profile)
        except Exception as exc:
            _LOG.debug("Protected assignment preview is not admissible", exc_info=True)
            return f"Protected assignment preview: NOT ADMISSIBLE — {type(exc).__name__}: {exc}"
        slot_texts = []
        for slot in plan.slots:
            effective = protected_rollout_shares(
                cuda_share=int(self.cuda_rollout_share.value()),
                xpu_share=int(self.xpu_rollout_share.value()),
                cpu_share=int(self.cpu_rollout_share.value()),
                primary_device=str(slot.primary_device),
                auxiliary_xpu_runtime=str(slot.auxiliary_xpu_runtime or ""),
            )
            route = f"CUDA {effective['cuda']}% / XPU {effective['xpu']}% / CPU {effective['cpu']}%"
            slot_texts.append(
                f"slot {slot.slot_index + 1} → {slot.primary_device} ({slot.device_name}) · effective routing {route}"
            )
        slots = "; ".join(slot_texts) or "no active slot"
        return (
            f"Protected preview: {plan.simultaneous_branches} simultaneous / {plan.total_branches} total; "
            f"{plan.queued_branches} queued; {slots}. Global CPU budget {plan.global_cpu_worker_budget}. "
            "The scheduler does not silently create extra CPU branches."
        )

    def _actual_runtime_assignment_text(self, payload: dict | None = None) -> str:
        payload = dict(payload or {})
        plan = dict(payload.get("resource_plan", {}) or {})
        slots = list(plan.get("slots", []) or [])
        if not slots:
            return self._runtime_device_request_text()
        parts = []
        for slot in slots:
            primary = str(slot.get("primary_device", "") or "")
            name = str(slot.get("device_name", "") or "")
            aux = str(slot.get("auxiliary_xpu_runtime", "") or "")
            effective = protected_rollout_shares(
                cuda_share=int(self.cuda_rollout_share.value()),
                xpu_share=int(self.xpu_rollout_share.value()),
                cpu_share=int(self.cpu_rollout_share.value()),
                primary_device=primary,
                auxiliary_xpu_runtime=aux,
            )
            text = f"slot {int(slot.get('slot_index', 0)) + 1} → {primary}"
            if name:
                text += f" ({name})"
            if aux:
                text += f" + auxiliary {aux}"
            text += (
                f" · effective episode routing CUDA {effective['cuda']}% / "
                f"XPU {effective['xpu']}% / CPU {effective['cpu']}%"
            )
            parts.append(text)
        return "Actual protected runtime assignment: " + " · ".join(parts)

    def _browse_development_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select policy-development ExperimentConfig",
            self.development_experiment_config.text().strip(),
            "Experiment configuration (*.yaml *.yml *.json);;All files (*)",
        )
        if path:
            self.development_experiment_config.setText(path)

    def _development_suite_values(self) -> tuple[tuple[str, ...], str]:
        if not self.enable_development_suite.isChecked():
            return (), ""
        cases = tuple(
            item.strip()
            for item in self.development_cases.text().split(",")
            if item.strip()
        )
        if not cases:
            raise ValueError("Enable at least one real ORPD development case or disable the development suite.")
        from calo_rpd_studio.power_system.case_identity import protected_holdout_matches

        forbidden = list(protected_holdout_matches(cases))
        if forbidden:
            raise ValueError(
                "Protected held-out final benchmark cases cannot be used for policy development: "
                + ", ".join(forbidden)
            )
        config_path = self.development_experiment_config.text().strip()
        if not config_path:
            raise ValueError("Select an exact ExperimentConfig for the real ORPD development suite.")
        candidate = Path(config_path).expanduser()
        if not candidate.is_file():
            raise ValueError(f"Development ExperimentConfig does not exist: {candidate}")
        from calo_rpd_studio.experiments.experiment_config import ExperimentConfig

        experiment = ExperimentConfig.load(candidate)
        experiment.validate()
        return cases, str(candidate.resolve())

    def _update_training_plan(self, *_args) -> None:
        weighted = str(self.rollout_mode.currentData()) == "weighted"
        advanced_controls = (
            self.auto_tuned_training,
            self.persistent_training_actors,
            self.device_resident_synthetic,
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
            self.cuda_workers,
            self.xpu_workers,
            self.cpu_workers,
            self.recommend_workers_button,
            *advanced_controls,
        ):
            control.setEnabled(weighted)
        self.training_calibration_episodes.setEnabled(
            weighted and self.auto_tuned_training.isChecked()
        )
        development_enabled = self.enable_development_suite.isChecked()
        self.development_cases.setEnabled(development_enabled)
        self.development_experiment_config.setEnabled(development_enabled)
        self.development_config_browse.setEnabled(development_enabled)
        batching_enabled = (
            weighted
            and self.cross_episode_training_batch.isChecked()
            and (
                self.device_resident_synthetic.isChecked()
                or (self.accelerated_training_orpd.isChecked() and self.enable_development_suite.isChecked())
            )
        )
        self.training_cross_batch.setEnabled(batching_enabled)
        self.training_batch_window.setEnabled(batching_enabled)
        self.training_tensor_batch.setEnabled(
            weighted and self.accelerated_training_orpd.isChecked()
        )

        self._update_task_share_status()
        if hasattr(self, "protected_allocation_status"):
            self.protected_allocation_status.setText(
                f"Selected rollout routing remains CUDA {self.cuda_rollout_share.value()}% · "
                f"XPU {self.xpu_rollout_share.value()}% · CPU {self.cpu_rollout_share.value()}%. "
                + self._protected_assignment_preview_text()
                + " Host-support CPU budget and the CPU rollout process cap are separate from the selected CPU rollout percentage; they do not mean hidden CPU rollout spillover."
            )
        if hasattr(self, "runtime_assignment_status") and not bool(getattr(self.state, "policy_training_active", False)):
            self.runtime_assignment_status.setText(self._runtime_device_request_text())

        if not weighted:
            self.accelerator_status.setText(
                "Legacy execution scope: rollout environments run in CPU processes; the selected learner device "
                "performs centralized PPO updates only. This mode is not GPU-resident CALO training. "
                + self._device_text
            )
            return

        total = (
            self.cuda_rollout_share.value()
            + self.xpu_rollout_share.value()
            + self.cpu_rollout_share.value()
        )
        if total != 100:
            self.accelerator_status.setText(
                f"Invalid selected rollout split: {total}%. CUDA, XPU, and CPU shares must total 100%."
            )
            return
        try:
            requested_primary = str(self.training_device.currentData() or "auto").lower()
            plan = plan_training_lanes(
                self.episodes.value(),
                cuda_share=self.cuda_rollout_share.value(),
                xpu_share=self.xpu_rollout_share.value(),
                cpu_share=self.cpu_rollout_share.value(),
                strict_unavailable=requested_primary in {"cuda", "xpu"},
            )
            warning = (" " + " ".join(plan.warnings)) if plan.warnings else ""
            routing_text = (
                f"Per-epoch episode/transition routing ({self.episodes.value()} episode(s)): {plan.summary()}. "
                "This count is EPISODES, not rollout-worker count. "
            )
            allocation_text = (
                "Auto-tuning is enabled: the displayed share is the deterministic fallback; short discarded "
                "calibration episodes may change later episode routing according to measured complete-transition throughput. "
                if self.auto_tuned_training.isChecked()
                else "The selected episode/transition split is fixed for each epoch unless the protected branch scheduler must fail closed or rebind an unavailable auxiliary accelerator to the already-admitted primary. "
            )
            persistence_text = (
                "Actor runtimes are persistent for the training session. "
                if self.persistent_training_actors.isChecked()
                else "Actor runtimes may be recreated between collection calls. "
            )
            try:
                development_cases, development_config = self._development_suite_values()
                development_text = (
                    "Real ORPD development suite: " + ", ".join(development_cases)
                    + f" · formulation {Path(development_config).name}. "
                    if development_cases
                    else "Real ORPD development suite: disabled. "
                )
            except Exception as exc:
                development_text = f"Real ORPD development suite configuration error: {exc}. "
            synthetic_text = (
                "Stage-B synthetic curriculum evaluation: device-resident FP64 objective/constraint tensors with fail-closed startup parity and cross-episode microbatching on admitted CUDA/XPU actors. "
                if self.device_resident_synthetic.isChecked()
                else "Stage-B synthetic accelerator kernel: disabled; synthetic curriculum evaluation remains on the CPU reference path. "
            )
            compute_text = (
                "Neural policy inference and PPO tensors use the admitted learner/actor device where supported. "
                "The stochastic CALO controller/archive/memory semantics remain the trusted trajectory authority; Stage B accelerates deterministic population evaluation and batches compatible episode work without redefining the policy ABI. "
                "CUDA/XPU/CPU percentages describe eligible rollout episode routing, not exact Task Manager utilization percentages. "
            )
            batching_text = (
                f"Cross-episode batching is enabled for up to {self.training_cross_batch.value()} compatible candidates with a {self.training_batch_window.value():.1f} ms merge window. "
                if batching_enabled
                else "Cross-episode batching is not active for the current configuration. "
            )
            self.accelerator_status.setText(
                routing_text
                + allocation_text
                + persistence_text
                + compute_text
                + synthetic_text
                + development_text
                + batching_text
                + self._device_text
                + warning
            )
        except Exception as exc:
            self.accelerator_status.setText(f"Training execution-plan validation failed: {type(exc).__name__}: {exc}")

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
        try:
            active = next(
                (p for p in self.state.policy_registry.list() if p.active and p.usable and p.runtime_compatible),
                None,
            )
            if active is None:
                raise RuntimeError(
                    "No compatible active CALO policy exists. Train/import, qualify, and explicitly activate a policy first."
                )
            policy = self._selected_policy()
            if policy is None or policy.id != active.id:
                policy = active
                self.refresh_policy_library()
                self._select_policy_id(policy.id)
            self.path.setText(policy.checkpoint_path)
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
            metadata["sha256"] = checkpoint_sha256(path)
            self.metadata.setPlainText(json.dumps(metadata, indent=2))
        except Exception as exc:
            self.metadata.setPlainText(str(exc))

    def _update_compute_protection(self) -> None:
        profile = getattr(self.state, "compute_protection_profile", None)
        if profile is None:
            if hasattr(self, "safe_parallel_limit"):
                self.safe_parallel_limit.setText("Dashboard system scan pending; training is protected until a Safe-80 profile exists.")
            return
        if hasattr(self, "safe_parallel_limit"):
            self.safe_parallel_limit.setText(
                f"{profile.profile_name}: maximum {profile.safe_parallel_branches} simultaneous competitive branch(es); "
                f"global CPU worker budget {profile.safe_cpu_worker_budget}; {profile.reserve_percent}% reserve retained."
            )
        if hasattr(self, "rollout_workers"):
            self.rollout_workers.setMaximum(max(1, int(profile.safe_cpu_worker_budget)))
            if self.rollout_workers.value() > int(profile.safe_cpu_worker_budget):
                self.rollout_workers.setValue(max(1, int(profile.safe_cpu_worker_budget)))
        if hasattr(self, "parallel_concurrency"):
            safe_limit = max(1, int(profile.safe_parallel_branches)) if profile.ready else 1
            self.parallel_concurrency.setMaximum(safe_limit)
            self.parallel_concurrency.setValue(min(self.parallel_concurrency.value(), safe_limit))
        if hasattr(self, "parallel_runs"):
            self._sync_parallel_branch_count()
            self._update_queue_plan_label()
        if hasattr(self, "accelerator_status"):
            self._update_training_plan()

    def _safe_parallel_branch_limit(self) -> int:
        profile = getattr(self.state, "compute_protection_profile", None)
        if profile is None or not bool(getattr(profile, "ready", False)):
            return 0
        return max(0, int(getattr(profile, "safe_parallel_branches", 0) or 0))

    def _validate_safe_training_capacity(self) -> None:
        profile = getattr(self.state, "compute_protection_profile", None)
        if profile is None:
            raise RuntimeError("Dashboard system readiness scan has not completed. Refresh the system map before training.")
        if not profile.ready or int(profile.safe_parallel_branches) < 1:
            raise RuntimeError(
                "Safe-80 compute protection reports no safe training-branch capacity. "
                + " ".join(profile.reasons)
            )
        total = int(self.parallel_runs.value())
        concurrency = int(self.parallel_concurrency.value())
        if concurrency > int(profile.safe_parallel_branches):
            raise RuntimeError(
                f"Requested simultaneous concurrency ({concurrency}) exceeds the Dashboard Safe-80 hard limit "
                f"({profile.safe_parallel_branches}). Lower concurrency; total scientific branches ({total}) may remain unchanged."
            )
        if concurrency > total:
            raise RuntimeError("Simultaneous branch concurrency cannot exceed total scientific branch count.")
        if int(self.rollout_workers.value()) > int(profile.safe_cpu_worker_budget):
            raise RuntimeError(
                f"CPU rollout process cap ({self.rollout_workers.value()}) exceeds the global Safe-80 worker budget "
                f"({profile.safe_cpu_worker_budget})."
            )

    def _custom_seed_values(self) -> tuple[int, ...]:
        values = []
        if not hasattr(self, "custom_branch_seeds"):
            return ()
        for token in self.custom_branch_seeds.text().split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Invalid custom branch seed: {token!r}") from exc
        return tuple(values)

    def _sync_parallel_branch_count(self) -> None:
        if not hasattr(self, "parallel_runs"):
            return
        custom = []
        if hasattr(self, "custom_branch_seeds"):
            for token in self.custom_branch_seeds.text().split(","):
                token = token.strip()
                if token:
                    try:
                        custom.append(int(token))
                    except ValueError:
                        continue
        total = (
            int(getattr(self, "same_seed_branches", self.parallel_runs).value())
            + int(getattr(self, "incremental_seed_branches", self.parallel_runs).value())
            + int(getattr(self, "decremental_seed_branches", self.parallel_runs).value())
            + len(custom)
        )
        total = max(1, total)
        self.parallel_runs.setValue(total)
        limit = self._safe_parallel_branch_limit() if hasattr(self, "state") else 0
        if hasattr(self, "parallel_concurrency"):
            self.parallel_concurrency.setMaximum(max(1, limit if limit > 0 else total))
            desired = min(total, max(1, limit if limit > 0 else 1))
            if self.parallel_concurrency.value() > total or self.parallel_concurrency.value() < 1:
                self.parallel_concurrency.setValue(desired)
            elif total == 1:
                self.parallel_concurrency.setValue(1)
        if hasattr(self, "safe_parallel_limit") and limit > 0:
            profile = self.state.compute_protection_profile
            self.safe_parallel_limit.setText(
                f"{profile.profile_name}: hard simultaneous ceiling {limit}; total scientific branches may exceed this and are queued. "
                f"Global CPU worker budget {profile.safe_cpu_worker_budget}."
            )
        self._update_queue_plan_label()

    def _update_queue_plan_label(self, *_args) -> None:
        if not hasattr(self, "queue_plan_status") or not hasattr(self, "parallel_runs"):
            return
        total = max(1, int(self.parallel_runs.value()))
        concurrency = max(1, min(total, int(self.parallel_concurrency.value())))
        queued = max(0, total - concurrency)
        self.queue_plan_status.setText(
            f"{total} independent scientific branch(es): up to {concurrency} active simultaneously, {queued} initially queued. "
            "Queued branches use exact-resume time slices; branch weights are never averaged."
        )

    def _update_training_mode_controls(self) -> None:
        mode = (
            str(self.training_mode.currentData() or "cumulative")
            if hasattr(self, "training_mode")
            else "cumulative"
        )
        if hasattr(self, "epochs"):
            self.epochs.setEnabled(mode != "indefinite")
            self.epochs.setToolTip(
                "Fixed number of epochs to add during this cumulative session"
                if mode == "cumulative"
                else "Ignored in infinite mode; training continues until Safe Stop and remains exactly resumable"
            )

    def train_policy(self) -> None:
        try:
            self._validate_safe_training_capacity()
        except Exception as exc:
            QMessageBox.critical(self, "Safe-80 compute protection", str(exc))
            return
        weighted = str(self.rollout_mode.currentData()) == "weighted"
        models_dir = Path(__file__).resolve().parents[2] / "data" / "trained_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        default_path = str(models_dir / "calo_policy_candidate.pt")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save trained CALO policy candidate" if weighted else "Save trained CALO policy",
            default_path,
            "PyTorch checkpoint (*.pt)",
        )
        if not path:
            return
        target = Path(path).expanduser().resolve()
        protected_paths = {
            Path(policy.checkpoint_path).expanduser().resolve()
            for policy in self.state.policy_registry.list(include_archived=True)
            if Path(policy.checkpoint_path).is_file()
        }
        if target in protected_paths:
            QMessageBox.critical(
                self,
                "Policy artifact protection",
                "Training cannot overwrite a registered policy artifact. Save to a new candidate "
                "filename so existing experiment and policy SHA-256 provenance remains immutable.",
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
        if selected_training_device == "xpu_sidecar" and self.parallel_runs.value() > 1:
            QMessageBox.critical(
                self,
                "Policy training configuration",
                "The secondary XPU sidecar runtime currently supports one policy-training branch per "
                "training job. Choose Automatic/direct XPU/CUDA/CPU for competitive multi-branch "
                "training, or reduce the branch count to one.",
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
        try:
            development_cases, development_config_path = self._development_suite_values()
        except Exception as exc:
            QMessageBox.critical(self, "Policy training configuration", str(exc))
            return
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
            development_cases=development_cases,
            development_experiment_config_path=development_config_path,
            historical_repository=historical_options["historical_repository"],
            use_historical_trajectories=historical_options["use_historical_trajectories"],
            historical_pretraining_epochs=historical_options["historical_pretraining_epochs"],
            training_mode=str(self.training_mode.currentData() or "cumulative"),
            checkpoint_interval_epochs=self.checkpoint_interval.value(),
            qualification_interval_epochs=self.qualification_interval.value(),
            policy_lineage_name=self.policy_lineage_name.text().strip() or Path(path).stem,
            policy_phase_index=int(getattr(self, "_pending_policy_phase_index", 1) or 1),
            initial_policy_checkpoint=str(
                getattr(self, "_pending_initial_policy_checkpoint", "") or ""
            ),
            keep_resume_after_completion=True,
            parallel_runs=self.parallel_runs.value(),
            parallel_concurrency=self.parallel_concurrency.value(),
            branch_queue_quantum_epochs=10,
            parallel_same_seed_branches=self.same_seed_branches.value(),
            parallel_incremental_branches=self.incremental_seed_branches.value(),
            parallel_decremental_branches=self.decremental_seed_branches.value(),
            parallel_custom_seeds=self._custom_seed_values(),
            parallel_start_mode=str(self.parallel_start_mode.currentData() or "new"),
            base_model_checkpoint=str(getattr(self, "_pending_base_model_checkpoint", "") or ""),
            training_scratch_dir=self.training_scratch_dir.text().strip(),
            safe_snapshot_interval_epochs=10,
            safe_parallel_branches=int(self.state.compute_protection_profile.safe_parallel_branches),
            safe_global_cpu_workers=int(self.state.compute_protection_profile.safe_cpu_worker_budget),
            compute_profile_fingerprint=str(self.state.compute_protection_profile.profile_fingerprint),
            compute_topology_fingerprint=str(self.state.compute_topology.fingerprint if self.state.compute_topology is not None else ""),
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
                device_resident_synthetic_rollouts=self.device_resident_synthetic.isChecked(),
                synthetic_cross_episode_batching=self.cross_episode_training_batch.isChecked(),
                synthetic_batch_window_ms=self.training_batch_window.value(),
                synthetic_max_cross_batch=self.training_cross_batch.value(),
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
            self._pending_base_model_checkpoint = ""
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
        profile = getattr(self.state, "compute_protection_profile", None)
        if profile is None or not profile.ready:
            QMessageBox.critical(
                self,
                "Safe-80 compute protection",
                "Training is blocked until Dashboard completes a READY Safe-80 system scan.",
            )
            return
        requested = max(1, int(getattr(config, "parallel_runs", 1) or 1))
        concurrency = int(getattr(config, "parallel_concurrency", 0) or 0)
        if concurrency <= 0:
            concurrency = min(requested, int(profile.safe_parallel_branches))
        config.parallel_concurrency = max(1, min(requested, concurrency))
        config.safe_parallel_branches = int(profile.safe_parallel_branches)
        config.safe_global_cpu_workers = int(profile.safe_cpu_worker_budget)
        config.compute_profile_fingerprint = str(profile.profile_fingerprint)
        config.compute_topology_fingerprint = str(self.state.compute_topology.fingerprint if self.state.compute_topology is not None else "")
        config.rollout_workers = max(1, min(int(getattr(config, "rollout_workers", 1) or 1), int(profile.safe_cpu_worker_budget)))
        # Preflight the exact selected device/capability plan before registering a resumable task or
        # entering the Global Training Exclusive Lock. This catches cases such as requesting two
        # CUDA branches on one CUDA device even when a different XPU contributed another aggregate
        # Dashboard slot. The coordinator recalculates the same plan at process launch and fails
        # closed if live headroom/topology has changed.
        try:
            preflight_plan = build_training_resource_plan(
                config,
                requested,
                topology=self.state.compute_topology,
                profile=profile,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Protected training resource plan", str(exc))
            return
        config.parallel_concurrency = int(preflight_plan.simultaneous_branches)
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
        self.recover_training_button.setEnabled(False)
        self.discard_recovery_button.setEnabled(False)
        self.metadata.setPlainText(
            "CALO v6.4 Stage-B protected policy-lineage training is running under the Global Training Exclusive Lock. "
            "Total scientific branch count is separated from Safe-80 simultaneous concurrency; excess branches are exact-resume queued/rotated. "
            "One global CPU worker budget is shared across active slots, and XPU roles are capability-aware. Only a committed generation manifest is authoritative."
        )
        self.worker = TrainingWorker(config, path)
        self.worker.progress.connect(self._training_progress)
        self.worker.session_state.connect(self._training_session_state)
        self.worker.completed.connect(self._training_done)
        self.worker.cancelled.connect(self._training_cancelled)
        self.worker.failed.connect(self._training_failed)
        self.state.task_status.cancel_requested.connect(self._cancel_training)
        initial_plan = {
            "status": "STARTING",
            "total_branches": requested,
            "simultaneous_limit": int(config.parallel_concurrency),
            "active_branches": 0,
            "queued_branches": requested,
            "completed_branches": 0,
            "safe_parallel_ceiling": int(profile.safe_parallel_branches),
            "global_cpu_worker_budget": int(preflight_plan.global_cpu_worker_budget),
            "compute_profile_fingerprint": str(profile.profile_fingerprint),
            "resource_plan": preflight_plan.to_dict(),
        }
        self.state.update_policy_training_plan(initial_plan)
        if hasattr(self, "runtime_assignment_status"):
            self.runtime_assignment_status.setText(self._actual_runtime_assignment_text(initial_plan))
        if hasattr(self, "protected_allocation_status"):
            self.protected_allocation_status.setText(
                f"Selected rollout routing: CUDA {self.cuda_rollout_share.value()}% · XPU {self.xpu_rollout_share.value()}% · CPU {self.cpu_rollout_share.value()}%. "
                + self._protected_assignment_preview_text()
                + " Training launch accepted by Safe-80; actual slot assignments are shown in Runtime device mapping."
            )
        self.state.begin_policy_training(
            f"CALO policy training · {requested} total branch(es) · max {config.parallel_concurrency} simultaneous"
        )
        fixed_target_text = (
            f" · {int(config.epochs)} epoch(s) per branch"
            if str(getattr(config, "training_mode", "cumulative")) != "indefinite"
            else " · indefinite until Safe Stop"
        )
        self.state.task_status.begin(
            "Training CALO policy",
            detail=(
                "Resuming protected exact training" + fixed_target_text
                if Path(config.resume_checkpoint).is_file()
                else f"Initializing protected queue · {requested} total / {config.parallel_concurrency} simultaneous" + fixed_target_text
            ),
            progress=(0 if str(getattr(config, "training_mode", "cumulative")) != "indefinite" else -1),
            cancellable=True,
        )
        try:
            self.worker.start()
        except Exception as exc:
            self.state.task_status.fail(f"Policy training could not start: {type(exc).__name__}: {exc}")
            self.state.end_policy_training("Policy training launch failed")
            self.state.resume_service.update(
                task_id, status=ResumeStatus.FAILED, state=state_payload, resumable=True
            )
            self.train_button.setEnabled(True)
            self.resume_training_button.setEnabled(True)
            QMessageBox.critical(self, "Policy training launch failed", f"{type(exc).__name__}: {exc}")


    def _choose_recovery_session(self):
        output, _ = QFileDialog.getOpenFileName(
            self,
            "Select CALO logical policy output",
            "",
            "CALO policy (*.pt);;All files (*)",
        )
        if not output:
            return "", None
        sessions = list_recoverable_sessions(output)
        if not sessions:
            QMessageBox.information(
                self, "Competitive training recovery", "No interrupted/recoverable branch session was found for this output."
            )
            return "", None
        labels = [
            f"{row.get('session_id')} · {row.get('status')} · common safe {row.get('latest_common_safe_epoch', '?')}"
            for row in sessions
        ]
        choice, accepted = QInputDialog.getItem(
            self, "Competitive training recovery", "Interrupted session", labels, 0, False
        )
        if not accepted:
            return "", None
        return output, sessions[labels.index(choice)]

    def recover_interrupted_training(self) -> None:
        output, session = self._choose_recovery_session()
        if not output or not session:
            return
        try:
            manifest = recover_competitive_session(output, str(session["session_id"]))
        except Exception as exc:
            QMessageBox.critical(self, "Competitive training recovery", str(exc))
            return
        QMessageBox.information(
            self,
            "Competitive training recovery",
            f"Recovered one coherent exact-resume generation at common epoch {manifest.get('common_resume_epoch')}. "
            "The previously committed Base was retained; unfinalized branch champions were not promoted.",
        )
        self.refresh_policy_library()

    def discard_interrupted_training_recovery(self) -> None:
        output, session = self._choose_recovery_session()
        if not output or not session:
            return
        answer = QMessageBox.question(
            self,
            "Discard competitive recovery",
            "Permanently discard this interrupted session's scratch recovery data? Previously committed generations are not changed.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            discard_recovery_session(output, str(session["session_id"]))
        except Exception as exc:
            QMessageBox.critical(self, "Discard competitive recovery", str(exc))
            return
        QMessageBox.information(self, "Discard competitive recovery", "Interrupted-session recovery data was discarded.")

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
            config.training_mode = str(self.training_mode.currentData() or "cumulative")
            config.epochs = int(self.epochs.value())
            config.checkpoint_interval_epochs = int(self.checkpoint_interval.value())
            config.qualification_interval_epochs = int(self.qualification_interval.value())
        path = str(payload.get("output_path", ""))
        if not path:
            QMessageBox.critical(
                self,
                "Policy training resume",
                "The saved task does not contain an output checkpoint path.",
            )
            return
        # v5.9 keeps one logical base alias while immutable artifacts preserve every experiment-bound SHA.
        # Exact multi-branch resume reopens the branch manifest and resumes each branch's own optimizer/RNG state.
        if Path(path).with_suffix(".branches.json").is_file():
            config.parallel_start_mode = "exact_resume"
        self._launch_training(config, path, resume_task_id=item.id)

    def resume_task_by_id(self, task_id: str) -> None:
        """Resume a specific policy training task by its resume service ID.

        Used by the Resume Center to directly resume training without requiring
        the user to navigate panels or select from a list.
        """
        if self.state.task_status.busy:
            QMessageBox.information(self, "Task busy", "Wait for the active task to finish first.")
            return
        items = self.state.resume_service.list_all(
            task_type=ResumeTaskType.POLICY_TRAINING, resumable_only=True
        )
        item = next((i for i in items if i.id == task_id), None)
        if item is None:
            QMessageBox.critical(
                self, "Policy training resume", f"Resumable training task not found: {task_id}"
            )
            return
        payload = dict(item.state)
        config_data = dict(payload.get("config", {}))
        cls = HeterogeneousTrainingConfig if payload.get("heterogeneous") else TrainingConfig
        valid = {
            key: value for key, value in config_data.items() if key in cls.__dataclass_fields__
        }
        if "development_cases" in valid:
            valid["development_cases"] = tuple(valid["development_cases"])
        config = cls(**valid)
        path = str(payload.get("output_path", ""))
        if not path:
            QMessageBox.critical(
                self,
                "Policy training resume",
                "The saved task does not contain an output checkpoint path.",
            )
            return
        if Path(path).with_suffix(".branches.json").is_file():
            config.parallel_start_mode = "exact_resume"
        self._launch_training(config, path, resume_task_id=item.id)

    def _training_session_state(self, payload: dict) -> None:
        payload = dict(payload or {})
        self.state.update_policy_training_plan(payload)
        if hasattr(self, "runtime_assignment_status"):
            self.runtime_assignment_status.setText(self._actual_runtime_assignment_text(payload))
        if hasattr(self, "protected_allocation_status"):
            overall_raw = payload.get("overall_percent", -1)
            overall = int(overall_raw) if overall_raw is not None else -1
            branch_progress = list(payload.get("branch_progress", []) or [])
            selected = (
                f"Selected rollout routing: CUDA {self.cuda_rollout_share.value()}% · "
                f"XPU {self.xpu_rollout_share.value()}% · CPU {self.cpu_rollout_share.value()}%. "
            )
            progress_text = ""
            if branch_progress:
                parts = []
                for row in branch_progress[:6]:
                    target = int(row.get("session_target", 0) or 0)
                    done = int(row.get("session_done", 0) or 0)
                    state = str(row.get("state", ""))
                    parts.append(f"{row.get('branch_id')} {state} {done}/{target if target > 0 else '∞'}")
                progress_text = " Runtime: " + " · ".join(parts)
                if overall >= 0:
                    progress_text += f" · overall {overall}%"
            self.protected_allocation_status.setText(selected + self._protected_assignment_preview_text() + progress_text)

    def request_training_safe_stop(self) -> None:
        """Public application-level Safe Stop hook used by the global exclusive-lock close path."""
        self._cancel_training()

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
        except (TypeError, RuntimeError, AttributeError):
            pass

    def _register_training_snapshots(self, output_path: str, config=None) -> list[str]:
        """Register legacy epoch_* lineage snapshots when importing older trainer outputs."""
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
            except Exception as exc:
                # Legacy snapshot import is non-authoritative, but failures are never silent because
                # missing lineage evidence must remain diagnosable.
                _LOG.warning("Could not register legacy training snapshot %s: %s", snapshot, exc, exc_info=True)
                continue
        return registered

    def _training_done(self, path: str) -> None:
        self._disconnect_training_cancel()
        self.train_button.setEnabled(True)
        self.resume_training_button.setEnabled(True)
        self.recover_training_button.setEnabled(True)
        self.discard_recovery_button.setEnabled(True)
        if self.training_resume_task_id:
            self.state.resume_service.update(
                self.training_resume_task_id,
                status=ResumeStatus.COMPLETED,
                current=100,
                total=100,
                resumable=True,
            )
        selected_policy = None
        deployable_eligible: bool | None = None
        try:
            config = getattr(self.worker, "config", None)
            # Read the mutable working alias only to locate its immutable terminal snapshot.
            alias_payload = load_checkpoint(path, map_location="cpu")
            alias_metadata = dict(alias_payload.get("metadata", {}) or {})
            deployable_eligible = bool(alias_metadata.get("base_eligible", True))
            immutable_path = str(alias_metadata.get("immutable_terminal_checkpoint", "") or "")
            self._register_training_snapshots(path, config)
            if immutable_path and Path(immutable_path).is_file():
                selected_policy = self.state.policy_registry.register(
                    immutable_path,
                    name=f"{alias_metadata.get('policy_lineage_name') or Path(path).stem}@{int(alias_metadata.get('champion_epoch', alias_metadata.get('cumulative_epoch', 0)) or 0)}",
                    status="candidate",
                )
                lineage_id = str(getattr(config, "policy_lineage_id", "") or "") if config is not None else ""
                if lineage_id and self.state.database.get_policy_checkpoint_by_sha256(selected_policy.sha256) is None:
                    resume_reference = (
                        str(Path(path).with_suffix(".branches.json"))
                        if Path(path).with_suffix(".branches.json").is_file()
                        else str(Path(path).with_suffix(".resume.pt"))
                    )
                    self.state.policy_registry.lineages.register_checkpoint(
                        lineage_id,
                        immutable_path,
                        cumulative_epoch=int(alias_metadata.get("champion_epoch", alias_metadata.get("cumulative_epoch", 0)) or 0),
                        phase_index=int(alias_metadata.get("policy_phase_index", 1) or 1),
                        resume_path=resume_reference,
                        metadata={
                            "policy_id": selected_policy.id,
                            "policy_name": selected_policy.name,
                            "competitive_base": bool(alias_metadata.get("checkpoint_role") == "competitive_base_model"),
                            "parallel_branches": alias_metadata.get("parallel_branches", 1),
                            "champion_metrics": alias_metadata.get("champion_metrics", {}),
                        },
                    )
                self.path.setText(immutable_path)
            else:
                # Compatibility fallback for a legacy trainer that did not emit immutable snapshots.
                selected_policy = self.state.policy_registry.register(path, status="candidate")
                self.path.setText(path)
            self.refresh_policy_library()
            self._select_policy_id(selected_policy.id)
        except Exception as exc:
            _LOG.error("Training completed but policy registration/lineage finalization failed: %s", exc, exc_info=True)
            self.path.setText(path)
            QMessageBox.warning(
                self,
                "CALO policy registration",
                "Training completed, but the resulting policy could not be fully registered in the lineage database. "
                "The checkpoint remains on disk. Review the application log before using it for scientific evidence.\n\n" + str(exc),
            )
        self.inspect_policy()
        self.state.task_status.finish("CALO policy training completed")
        self.state.update_policy_training_plan({
            **dict(getattr(self.state, "policy_training_plan", {}) or {}),
            "status": "COMPLETED",
            "active_branches": 0,
            "queued_branches": 0,
        })
        self.state.end_policy_training("CALO policy training completed")
        self._update_training_plan()
        if deployable_eligible is False:
            message = (
                "Training completed and an immutable Training Champion candidate was saved. "
                "It is explicitly provisional and cannot be activated as a deployable scientific Base "
                "until it passes the required exact real-ORPD development/qualification evidence gates. "
                "Exact branch training state remains resumable."
            )
        elif deployable_eligible is True:
            message = (
                "Training completed. An immutable deployable-eligible Base/candidate artifact was saved, "
                "and exact branch training state remains resumable for future sessions."
            )
        else:
            message = (
                "Training completed, but deployable eligibility could not be confirmed during final "
                "registration. The checkpoint remains recoverable; review the registration warning/log "
                "before activation or scientific use."
            )
        QMessageBox.information(self, "CALO policy", message)

    def _training_cancelled(self, message: str) -> None:
        self._disconnect_training_cancel()
        self.train_button.setEnabled(True)
        self.resume_training_button.setEnabled(True)
        self.recover_training_button.setEnabled(True)
        self.discard_recovery_button.setEnabled(True)
        try:
            if self.worker is not None:
                self._register_training_snapshots(
                    str(self.worker.path), getattr(self.worker, "config", None)
                )
                self.refresh_policy_library()
        except Exception as exc:
            _LOG.warning("Safe-stop postprocessing could not register all legacy snapshots: %s", exc, exc_info=True)
        if self.training_resume_task_id:
            self.state.resume_service.update(
                self.training_resume_task_id, status=ResumeStatus.PAUSED, resumable=True
            )
        self.state.task_status.cancelled(message)
        self.state.update_policy_training_plan({
            **dict(getattr(self.state, "policy_training_plan", {}) or {}),
            "status": "SAFE_STOPPED",
            "active_branches": 0,
        })
        self.state.end_policy_training(message)
        self._update_training_plan()

    def _training_failed(self, message: str) -> None:
        self._disconnect_training_cancel()
        self.train_button.setEnabled(True)
        self.resume_training_button.setEnabled(True)
        self.recover_training_button.setEnabled(True)
        self.discard_recovery_button.setEnabled(True)
        if self.training_resume_task_id:
            self.state.resume_service.update(
                self.training_resume_task_id, status=ResumeStatus.INTERRUPTED, resumable=True
            )
        self.state.task_status.fail(message)
        self.state.update_policy_training_plan({
            **dict(getattr(self.state, "policy_training_plan", {}) or {}),
            "status": "FAILED",
            "active_branches": 0,
        })
        self.state.end_policy_training(message)
        self._update_training_plan()
        QMessageBox.critical(self, "Policy training failed", message)

    def set_experiment_navigation_enabled(self, enabled: bool) -> None:  # noqa: ARG002
        pass
