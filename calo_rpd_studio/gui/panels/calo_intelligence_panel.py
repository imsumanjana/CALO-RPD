"""CALO policy, cognitive architecture, training, and ablation controls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
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

from calo_rpd_studio.algorithms.calo.training import TrainingConfig, train_policy
from calo_rpd_studio.gui.widgets.page_header import PageHeader
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage


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
                "Inspect the frozen policy controller, cognitive state features, adaptive learning modes, bounded success memory, and reproducible policy training.",
            )
        )

        policy = QGroupBox("Policy controller")
        form = QFormLayout(policy)
        self.path = QLineEdit(
            str(
                Path(__file__).resolve().parents[2]
                / "data"
                / "trained_models"
                / "calo_policy_v1.pt"
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
            "Cognitive state — diversity, recent best and median improvement, stagnation, "
            "feasible-solution ratio, normalized constraint violation, elite spread, remaining "
            "evaluation budget, and operator-success memory.\n\n"
            "Learning modes — teacher-guided exploitation, contrastive peer learning, "
            "self-reflective memory learning, adaptive exploration, feasibility recovery, "
            "and stagnation escape.\n\n"
            "The AI policy controls operator probabilities and bounded search coefficients. "
            "Success memory is capacity-limited and recency weighted, while recovery preserves elite solutions."
        )
        description.setWordWrap(True)
        architecture_layout.addWidget(description)
        layout.addWidget(architecture)

        training = QGroupBox("Reproducible policy training")
        training_form = QFormLayout(training)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10_000)
        self.epochs.setValue(20)
        self.episodes = QSpinBox()
        self.episodes.setRange(1, 10_000)
        self.episodes.setValue(16)
        self.horizon = QSpinBox()
        self.horizon.setRange(2, 10_000)
        self.horizon.setValue(24)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(2026)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(7)
        self.lr.setRange(1e-7, 1)
        self.lr.setValue(3e-4)
        self.train_button = QPushButton("Train and save CALO policy")
        self.train_button.setObjectName("PrimaryButton")
        self.train_button.clicked.connect(self.train_policy)
        training_form.addRow("Epochs", self.epochs)
        training_form.addRow("Episodes per epoch", self.episodes)
        training_form.addRow("Episode horizon", self.horizon)
        training_form.addRow("Training seed", self.seed)
        training_form.addRow("Learning rate", self.lr)
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


    def apply_policy_configuration(self) -> None:
        path = Path(self.path.text().strip())
        if not path.exists():
            QMessageBox.critical(self, "CALO policy configuration", "Select a valid CALO policy checkpoint first.")
            return
        try:
            import torch

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
        config = TrainingConfig(
            self.epochs.value(),
            self.episodes.value(),
            self.horizon.value(),
            self.seed.value(),
            self.lr.value(),
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

