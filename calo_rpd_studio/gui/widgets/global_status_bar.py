"""Persistent task/progress widget embedded in the QStatusBar."""

from __future__ import annotations

from PyQt6.QtCore import QElapsedTimer, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
)

from calo_rpd_studio.algorithms.calo.policy_readiness import governing_policy_user_message
from calo_rpd_studio.gui.user_feedback import log_technical_error
from calo_rpd_studio.version import PRODUCT_VERSION


def truthful_runtime_assignment(config) -> str:
    """Return an actual assignment only after the runtime resolver recorded one."""
    resolution = dict(getattr(config, "runtime_device_resolution", {}) or {})
    if not resolution:
        return "not assigned"
    return str(
        resolution.get("runtime_compute_device")
        or getattr(config, "runtime_compute_device", "")
        or "not assigned"
    )


class GlobalStatusBarWidget(QWidget):
    cancel_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(10)

        self.state_label = QLabel("Ready")
        self.state_label.setObjectName("GlobalTaskState")
        self.task_label = QLabel("")
        self.task_label.setObjectName("GlobalTaskDetail")
        self.task_label.setMinimumWidth(220)

        self.progress = QProgressBar()
        self.progress.setObjectName("GlobalTaskProgress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFixedWidth(230)
        self.progress.hide()

        self.elapsed_label = QLabel("")
        self.elapsed_label.setObjectName("GlobalTaskElapsed")
        self.elapsed_label.setMinimumWidth(58)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("StatusCancelButton")
        self.cancel_button.clicked.connect(self.cancel_clicked)
        self.cancel_button.hide()

        self.compute_label = QLabel("Mode: CUDA-preferred")
        self.compute_label.setObjectName("StatusCompute")
        self.device_label = QLabel("Device: not assigned")
        self.device_label.setObjectName("StatusDevice")
        self.memory_label = QLabel("Memory: not scanned")
        self.memory_label.setObjectName("StatusMemory")
        self.policy_label = QLabel("Policy: not ready")
        self.policy_label.setObjectName("StatusPolicy")
        self.version_label = QLabel(f"v{PRODUCT_VERSION}")
        self.version_label.setObjectName("StatusVersion")

        layout.addWidget(self.state_label)
        layout.addWidget(self.task_label, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.elapsed_label)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.compute_label)
        layout.addWidget(self.device_label)
        layout.addWidget(self.memory_label)
        layout.addWidget(self.policy_label)
        layout.addWidget(self.version_label)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_elapsed)
        self._elapsed = QElapsedTimer()

    def apply_snapshot(self, snapshot: dict) -> None:
        state = str(snapshot.get("state", "Ready"))
        busy = bool(snapshot.get("busy", False))
        title = str(snapshot.get("title", ""))
        detail = str(snapshot.get("detail", ""))
        progress = int(snapshot.get("progress", 0))
        cancellable = bool(snapshot.get("cancellable", False))

        self.state_label.setText(state)
        self.state_label.setProperty("taskState", state.lower())
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)

        combined = title
        if detail:
            combined = f"{title} · {detail}" if title else detail
        self.task_label.setText(combined)
        self.task_label.setToolTip(combined)

        if busy:
            if not self._timer.isActive():
                self._elapsed.start()
                self._timer.start()
            self.progress.show()
            if progress < 0:
                self.progress.setRange(0, 0)
            else:
                self.progress.setRange(0, 100)
                self.progress.setValue(progress)
                self.progress.setFormat(f"{progress}%")
            safe_pause_task = any(
                token in title.casefold()
                for token in ("policy training", "policy qualification")
            )
            self.cancel_button.setText("Pause safely" if safe_pause_task else "Cancel")
            self.cancel_button.setToolTip(
                "Stop after the current verified checkpoint is committed."
                if safe_pause_task
                else "Request cancellation of the active task."
            )
            self.cancel_button.setVisible(cancellable)
        else:
            self._timer.stop()
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, progress)))
            self.progress.setVisible(state != "Ready")
            self.cancel_button.hide()
            self.elapsed_label.setText("")

    def _update_elapsed(self) -> None:
        if not self._elapsed.isValid():
            return
        seconds = max(0, self._elapsed.elapsed() // 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            self.elapsed_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            self.elapsed_label.setText(f"{minutes:02d}:{seconds:02d}")

    def apply_context(self, state) -> None:
        """Show configured intent separately from actual runtime and admission ceilings."""
        config = state.config
        configured = str(getattr(config, "execution_backend", "cuda_preferred"))
        requested = str(getattr(config, "requested_compute_device", "auto"))
        actual = truthful_runtime_assignment(config)
        self.compute_label.setText(f"Mode: {configured}")
        self.compute_label.setToolTip(
            f"Configured intent: {configured}; requested device: {requested}. This is not proof of actual execution."
        )
        self.device_label.setText(f"Device: {actual}")
        self.device_label.setToolTip(
            f"Last recorded actual runtime assignment: {actual}; requested: {requested}"
        )
        profile = getattr(state, "compute_protection_profile", None)
        if profile is None:
            memory = "not scanned"
        else:
            fraction = float(getattr(profile, "allocation_limit_fraction", 0.80))
            memory = f"ceiling {fraction:.0%}"
        self.memory_label.setText(f"Memory: {memory}")
        self.memory_label.setToolTip(
            "Computations begin only when they fit within the available-memory safety limit."
        )
        try:
            policy = state.governing_policy_status()
            ready = bool(getattr(policy, "ready", False))
            text = "ready" if ready else "not ready"
            reason = governing_policy_user_message(policy)
        except Exception as exc:
            text = "unavailable"
            reason = "Policy status could not be refreshed. Review Activity > Logs for details."
            log_technical_error("governing policy status", exc)
        self.policy_label.setText(f"Policy: {text}")
        self.policy_label.setToolTip(reason)
