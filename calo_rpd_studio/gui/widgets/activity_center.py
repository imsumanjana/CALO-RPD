"""Shared jobs, logs, warnings, device, and provenance presentation."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import logging

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.widgets.global_status_bar import truthful_runtime_assignment
from calo_rpd_studio.version import DISPLAY_VERSION, VERSION_STAGE


class _LogBridge(QObject):
    received = pyqtSignal(str, str, str)


class QtActivityLogHandler(logging.Handler):
    """Bridge standard logging into Qt through a queued cross-thread signal."""

    def __init__(self, bridge: _LogBridge) -> None:
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.bridge.received.emit(record.levelname, record.name, self.format(record))
        except Exception:
            self.handleError(record)


class ActivityCenter(QTabWidget):
    """Presentation adapter; ``TaskStatus`` remains the authoritative foreground task state."""

    def __init__(self, state, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setObjectName("ActivityCenter")
        self.setDocumentMode(True)
        self.tabBar().setDrawBase(False)
        self.setAccessibleName("Application activity")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.setMinimumHeight(96)
        self._entries: deque[tuple[str, str, str, str]] = deque(maxlen=5000)
        self._job_sequence = 0
        self._last_busy = False

        self.jobs = QTableWidget(0, 6)
        self.jobs.setHorizontalHeaderLabels(("Time", "State", "Task", "Stage", "Progress", "Mode"))
        self.jobs.setAlternatingRowColors(True)
        self.jobs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.jobs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.jobs.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.jobs.setShowGrid(False)
        self.jobs.verticalHeader().setVisible(False)
        header = self.jobs.horizontalHeader()
        for column in (0, 1, 4, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        for column in (2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.addTab(self.jobs, "Jobs")

        logs_page = QWidget()
        logs_layout = QVBoxLayout(logs_page)
        logs_layout.setContentsMargins(6, 6, 6, 6)
        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search visible logs")
        self.search.setAccessibleName("Search activity logs")
        self.severity = QComboBox()
        self.severity.addItems(("All severities", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
        self.pause = QCheckBox("Pause autoscroll")
        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(self._copy_logs)
        clear_button = QPushButton("Clear display")
        clear_button.setToolTip(
            "Clear only this display; durable application evidence is not deleted."
        )
        clear_button.clicked.connect(self._clear_display)
        filters.addWidget(self.search, 1)
        filters.addWidget(self.severity)
        filters.addWidget(self.pause)
        filters.addWidget(copy_button)
        filters.addWidget(clear_button)
        logs_layout.addLayout(filters)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setAccessibleName("Searchable application log display")
        logs_layout.addWidget(self.logs)
        self.search.textChanged.connect(lambda _text: self._render_logs())
        self.severity.currentTextChanged.connect(lambda _text: self._render_logs())
        self.addTab(logs_page, "Logs")

        self.warnings = QListWidget()
        self.warnings.setAccessibleName("Warnings and failures")
        self.addTab(self.warnings, "Warnings")
        self.device = QLabel("Device telemetry has not been scanned.")
        self.device.setWordWrap(True)
        self.device.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.device.setMargin(12)
        self.addTab(self.device, "Device")
        self.provenance = QLabel("No active task provenance.")
        self.provenance.setWordWrap(True)
        self.provenance.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.provenance.setMargin(12)
        self.addTab(self.provenance, "Provenance")

        self._bridge = _LogBridge(self)
        self._bridge.received.connect(self.append_log)
        self.log_handler = QtActivityLogHandler(self._bridge)
        self.log_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(self.log_handler)
        self.state.task_status.changed.connect(self.apply_task_snapshot)
        self.state.compute_profile_changed.connect(lambda _profile: self.refresh_context())
        self.state.compute_governor_changed.connect(lambda _decision: self.refresh_context())
        self.state.policy_state_changed.connect(lambda _status: self.refresh_context())
        self.apply_task_snapshot(self.state.task_status.snapshot())
        self.refresh_context()

    def detach_logging(self) -> None:
        logging.getLogger().removeHandler(self.log_handler)

    def append_external(self, severity: str, message: str) -> None:
        self.append_log(str(severity), "independent-training", str(message))

    def append_log(self, severity: str, source: str, message: str) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        level = str(severity).upper()
        self._entries.append((timestamp, level, str(source), str(message)))
        if level in {"WARNING", "ERROR", "CRITICAL"}:
            marker = "ERROR" if level in {"ERROR", "CRITICAL"} else "WARNING"
            self.warnings.addItem(f"[{marker}] {timestamp} · {source} · {message}")
        self._render_logs()

    def _render_logs(self) -> None:
        query = self.search.text().strip().lower()
        selected = self.severity.currentText()
        lines = []
        for timestamp, severity, source, message in self._entries:
            line = f"{timestamp} [{severity}] {source}: {message}"
            if selected != "All severities" and severity != selected:
                continue
            if query and query not in line.lower():
                continue
            lines.append(line)
        self.logs.setPlainText("\n".join(lines))
        if not self.pause.isChecked():
            scrollbar = self.logs.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _clear_display(self) -> None:
        self._entries.clear()
        self.logs.clear()

    def _copy_logs(self) -> None:
        self.logs.selectAll()
        self.logs.copy()

    def apply_task_snapshot(self, snapshot: dict) -> None:
        busy = bool(snapshot.get("busy", False))
        state = str(snapshot.get("state", "Ready"))
        title = str(snapshot.get("title", "") or "Foreground task")
        detail = str(snapshot.get("detail", ""))
        progress = int(snapshot.get("progress", 0))
        should_add = busy != self._last_busy or state in {"Completed", "Failed", "Cancelled"}
        self._last_busy = busy
        if should_add:
            self._job_sequence += 1
            row = self.jobs.rowCount()
            self.jobs.insertRow(row)
            mode = str(getattr(self.state.config, "execution_backend", ""))
            progress_text = "indeterminate" if progress < 0 else f"{progress}%"
            values = (
                datetime.now().astimezone().strftime("%H:%M:%S"),
                state,
                title,
                detail,
                progress_text,
                mode,
            )
            for column, value in enumerate(values):
                self.jobs.setItem(row, column, QTableWidgetItem(value))
        if state in {"Failed", "Cancelled"}:
            self.append_log("ERROR" if state == "Failed" else "WARNING", "task", detail)
        self.refresh_context()

    def refresh_context(self) -> None:
        config = self.state.config
        profile = getattr(self.state, "compute_protection_profile", None)
        decision = getattr(self.state, "compute_governor_decision", None)
        requested = f"{config.execution_backend} / {config.requested_compute_device}"
        actual = truthful_runtime_assignment(config)
        profile_text = "Safe-80 profile not scanned"
        if profile is not None:
            profile_text = (
                f"Safe-80 ceiling fraction: {float(getattr(profile, 'allocation_limit_fraction', 0.80)):.0%}. "
                "This is an admission ceiling, not measured consumption."
            )
        decision_text = ""
        if decision is not None:
            decision_text = (
                f"\nLatest governor state: {getattr(decision, 'state', type(decision).__name__)}"
            )
        self.device.setText(
            f"Configured intent: {requested}\nLast recorded actual assignment: {actual}\n"
            f"{profile_text}{decision_text}\nIntel XPU: not executable"
        )
        try:
            policy = self.state.governing_policy_status()
            policy_text = (
                f"Policy ready: {bool(getattr(policy, 'ready', False))}; "
                f"status: {getattr(policy, 'reason', '') or getattr(policy, 'qualification_status', 'not ready')}"
            )
        except Exception as exc:
            policy_text = f"Policy status unavailable: {type(exc).__name__}"
        task = self.state.task_status.snapshot()
        self.provenance.setText(
            f"Task: {task.get('state', 'Ready')} · {task.get('title', '')}\n"
            f"Technical build: {DISPLAY_VERSION} · stage: {VERSION_STAGE}\n"
            f"Configured execution: {requested}\nActual execution: {actual}\n{policy_text}\n"
            "Scientific counters and durable evidence remain authoritative in their originating workflows."
        )
