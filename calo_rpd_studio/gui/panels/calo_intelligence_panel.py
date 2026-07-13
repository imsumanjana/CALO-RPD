"""CALO policy, cognitive architecture, training, and ablation controls."""
from __future__ import annotations

import hashlib
import json
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
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.algorithms.calo.training import (
    TrainingConfig,
    available_training_devices,
    recommended_rollout_workers,
    train_policy,
)
from calo_rpd_studio.gui.widgets.page_header import PageHeader
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage
from calo_rpd_studio.gui.widgets.historical_experience_widget import HistoricalExperienceWidget


class TrainingWorker(QThread):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, config: TrainingConfig, path: str) -> None:
        super().__init__()
        import threading

        self.config = config
        self.path = path
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            if str(self.config.ppo_device).lower() == "xpu_sidecar":
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


class CALOIntelligencePanel(ScrollablePage):
    stage_completed = pyqtSignal()
    experiment_manager_requested = pyqtSignal()

    def __init__(self, state, experiment_manager, parent=None) -> None:
        content = QWidget()
        super().__init__(content, parent)
        self.state = state
        self.experiment_manager = experiment_manager
        self.worker: TrainingWorker | None = None

        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        layout.addWidget(
            PageHeader(
                "CALO Intelligence",
                "Inspect CALO Core v2, its hierarchical policy controller, constraint-aware cognitive state, dual archives, online operator credit, and reproducible PPO training.",
            )
        )

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
        self.deterministic = QCheckBox(
            "Deterministic operator selection during evaluation"
        )
        form.addRow("Policy checkpoint", row)
        form.addRow("", self.deterministic)

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
            "CALO Core v2 operators — feasible-elite learning, constraint-boundary differential "
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
        self.epochs.setRange(1, 10_000)
        self.epochs.setValue(24)
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
        device_text = "; ".join(device_parts) or "No verified GPU backend; CPU fallback is available"
        self.training_device.setToolTip(
            "CPU processes collect CALO rollout episodes in parallel. The selected device performs centralized PPO neural-network updates. "
            "Automatic priority is NVIDIA CUDA, then Intel XPU, then CPU. " + device_text
        )

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

        self.accelerator_status = QLabel(
            "Training architecture: parallel CPU rollout collection + centralized PPO update. "
            "Device priority: NVIDIA CUDA → Intel XPU → CPU. " + device_text
        )
        self.accelerator_status.setWordWrap(True)

        self.development_cases = QLineEdit()
        self.development_cases.setPlaceholderText("Optional custom ORPD development case paths, comma-separated")
        self.development_cases.setToolTip(
            "Optional explicit development systems for the final curriculum stage. Keep final publication benchmark systems separate from policy training."
        )
        self.train_button = QPushButton("Train and save CALO policy")
        self.train_button.setObjectName("PrimaryButton")
        self.train_button.clicked.connect(self.train_policy)
        training_form.addRow("Epochs", self.epochs)
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
        training_form.addRow("PPO compute device", self.training_device)
        training_form.addRow("Parallel rollout workers", worker_row)
        training_form.addRow("Compute status", self.accelerator_status)
        training_form.addRow("ORPD development cases", self.development_cases)
        training_form.addRow("", self.train_button)
        layout.addWidget(training)

        ablation = QGroupBox("Ablation analysis")
        ablation_layout = QVBoxLayout(ablation)
        self.ablation_button = QPushButton("Open Experiment Manager for CALO Analysis")
        self.ablation_button.setEnabled(False)
        self.ablation_button.clicked.connect(self.experiment_manager_requested.emit)
        ablation_layout.addWidget(self.ablation_button)
        layout.addWidget(ablation)
        layout.addStretch(1)


    def _historical_repository_changed(self, path: str) -> None:
        self.metadata.setPlainText(
            "Historical experience repository selected:\n"
            f"{path}\n\n"
            "Eligible historical CALO trajectories can be used for offline pretraining before fresh on-policy PPO. "
            "Cross-algorithm solutions and CALO parameter priors are applied only when explicitly enabled."
        )

    def _use_recommended_workers(self, *_args) -> None:
        self.rollout_workers.setValue(recommended_rollout_workers(self.episodes.value()))

    def apply_policy_configuration(self) -> None:
        path = Path(self.path.text().strip())
        if not path.exists():
            QMessageBox.critical(self, "CALO policy configuration", "Select a valid CALO policy checkpoint first.")
            return
        try:
            import torch
            from calo_rpd_studio.algorithms.calo.ai_controller import AIController

            AIController(path, deterministic=True)  # validates CALO Core v2 architecture
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {})
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception as exc:
            QMessageBox.critical(self, "CALO policy configuration", str(exc))
            return
        parameters = dict(self.state.config.algorithm_parameters.get("CALO", {}))
        parameters["policy_checkpoint"] = str(path.resolve())
        parameters["deterministic_policy"] = self.deterministic.isChecked()
        self.state.config.algorithm_parameters["CALO"] = parameters
        metadata = dict(metadata)
        metadata["sha256"] = checksum
        self.metadata.setPlainText(json.dumps(metadata, indent=2))
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
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {})
            metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.metadata.setPlainText(json.dumps(metadata, indent=2))
        except Exception as exc:
            self.metadata.setPlainText(str(exc))

    def train_policy(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save trained CALO policy",
            self.path.text(),
            "PyTorch checkpoint (*.pt)",
        )
        if not path:
            return
        selected_training_device = str(self.training_device.currentData())
        device_info = available_training_devices()
        if selected_training_device == "auto" and device_info["recommended_device"] == "xpu_sidecar":
            selected_training_device = "xpu_sidecar"
        historical_options = self.historical_experience.policy_training_options()
        config = TrainingConfig(
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
                item.strip()
                for item in self.development_cases.text().split(",")
                if item.strip()
            ),
            historical_repository=historical_options["historical_repository"],
            use_historical_trajectories=historical_options["use_historical_trajectories"],
            historical_pretraining_epochs=historical_options["historical_pretraining_epochs"],
        )
        if self.state.task_status.busy:
            QMessageBox.information(self, "Task busy", "Wait for the active task to finish first.")
            return
        self.train_button.setEnabled(False)
        self.metadata.setPlainText(
            "Policy training is running with the displayed reproducibility configuration."
        )
        self.worker = TrainingWorker(config, path)
        self.worker.progress.connect(self._training_progress)
        self.worker.completed.connect(self._training_done)
        self.worker.cancelled.connect(self._training_cancelled)
        self.worker.failed.connect(self._training_failed)
        self.state.task_status.cancel_requested.connect(self._cancel_training)
        self.state.task_status.begin(
            "Training CALO policy",
            detail="Initializing reproducible training",
            progress=0,
            cancellable=True,
        )
        self.worker.start()

    def _training_progress(self, percent: int, detail: str) -> None:
        self.state.task_status.update(percent, detail)

    def _cancel_training(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()

    def _disconnect_training_cancel(self) -> None:
        try:
            self.state.task_status.cancel_requested.disconnect(self._cancel_training)
        except TypeError:
            pass

    def _training_done(self, path: str) -> None:
        self._disconnect_training_cancel()
        self.train_button.setEnabled(True)
        self.path.setText(path)
        self.inspect_policy()
        self.state.task_status.finish("CALO policy training completed")
        QMessageBox.information(
            self,
            "CALO policy",
            "Policy training completed and the checkpoint was saved.",
        )

    def _training_cancelled(self, message: str) -> None:
        self._disconnect_training_cancel()
        self.train_button.setEnabled(True)
        self.state.task_status.cancelled(message)

    def _training_failed(self, message: str) -> None:
        self._disconnect_training_cancel()
        self.train_button.setEnabled(True)
        self.state.task_status.fail(message)
        QMessageBox.critical(self, "Policy training failed", message)

    def set_experiment_navigation_enabled(self, enabled: bool) -> None:
        self.ablation_button.setEnabled(bool(enabled))

