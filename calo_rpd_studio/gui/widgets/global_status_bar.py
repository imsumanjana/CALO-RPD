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

        layout.addWidget(self.state_label)
        layout.addWidget(self.task_label, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.elapsed_label)
        layout.addWidget(self.cancel_button)

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
