"""v6.2 System Readiness, Adaptive Compute Protection, and scientific-context dashboard."""

from __future__ import annotations

from copy import deepcopy
import logging

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.widgets.section_card import MetricCard, SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage
from calo_rpd_studio.experiments.study_strength import (
    StudyStrength,
    apply_study_strength,
    study_strength_plan,
    summarize_study_protocol_change,
)
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.power_system.case_validation import validate_case
from calo_rpd_studio.power_system.network_metrics import summarize_case

_LOG = logging.getLogger(__name__)


def _bytes_text(value: int) -> str:
    amount = float(max(0, int(value)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def _scrollable_tab(content: QWidget) -> QScrollArea:
    """Wrap one dashboard tab in its own width-safe vertical scroll area."""
    scroll = QScrollArea()
    scroll.setObjectName("DashboardTabScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    content.setObjectName("DashboardTabContent")
    content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    scroll.setWidget(content)
    return scroll


class DashboardPanel(WorkspacePage):
    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Dashboard",
            "System readiness, protected compute selection, governing-policy status, and current scientific context.",
            parent,
        )
        self.state = state

        # The dashboard body is vertically scrollable so summary cards and the
        # active tab keep their natural size instead of being compressed when
        # the application window is shorter than the preferred dashboard height.
        self.dashboard_body = QWidget()
        self.dashboard_body.setObjectName("DashboardScrollableBody")
        self.dashboard_body.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.dashboard_body_layout = QVBoxLayout(self.dashboard_body)
        self.dashboard_body_layout.setContentsMargins(0, 0, 0, 0)
        self.dashboard_body_layout.setSpacing(16)

        self.dashboard_scroll = QScrollArea()
        self.dashboard_scroll.setObjectName("DashboardPageScroll")
        self.dashboard_scroll.setWidgetResizable(True)
        self.dashboard_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.dashboard_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.dashboard_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.dashboard_scroll.setWidget(self.dashboard_body)
        self.layout_root.addWidget(self.dashboard_scroll, 1)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(12)
        self.system_metric = MetricCard(
            "System protection", "Scanning", "Automatic memory protection"
        )
        self.branch_metric = MetricCard(
            "Simultaneous tasks", "—", "Calculated from protected hardware capacity"
        )
        self.policy_metric = MetricCard(
            "CALO governing intelligence", "Not ready", "Qualified active policy required"
        )
        self.verified_metric = MetricCard(
            "Verified results", "0", "Independent validation required for export"
        )
        self.training_metric = MetricCard(
            "Policy training queue", "Idle", "Requested branches are queued when necessary"
        )
        metric_cards = (
            self.system_metric,
            self.branch_metric,
            self.policy_metric,
            self.verified_metric,
            self.training_metric,
        )
        for index, card in enumerate(metric_cards):
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row, column = divmod(index, 3)
            metrics.addWidget(card, row, column)
        for column in range(3):
            metrics.setColumnStretch(column, 1)
        self.dashboard_body_layout.addLayout(metrics)

        self.dashboard_tabs = QTabWidget()
        self.dashboard_tabs.setObjectName("DashboardTabs")
        self.dashboard_tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.dashboard_tabs.setMinimumHeight(500)

        study_setup = SectionCard(
            "Guided study setup",
            "Choose an evidence strength and reference case once. Applying the protocol updates the case, algorithms, paired runs, evaluation budget, scenarios, required outputs, validation, storage, resume, and export prerequisites throughout the workspace.",
        )
        study_form = QFormLayout()
        self.study_strength = QComboBox()
        self.study_strength.addItem(
            "Strong — comprehensive confirmatory", StudyStrength.STRONG.value
        )
        self.study_strength.addItem("Good — rigorous robust", StudyStrength.GOOD.value)
        self.study_strength.addItem("Moderate — comparative", StudyStrength.MODERATE.value)
        self.study_strength.addItem("Low — screening", StudyStrength.LOW.value)
        self.study_strength.setCurrentIndex(1)
        self.study_case = QComboBox()
        self.study_case.addItems(CaseLoader.available_cases())
        case_index = self.study_case.findText(str(self.state.config.case_name))
        if case_index >= 0:
            self.study_case.setCurrentIndex(case_index)
        initial_plan = study_strength_plan(str(self.study_strength.currentData()))
        self.study_effect = QDoubleSpinBox()
        self.study_effect.setDecimals(2)
        self.study_effect.setRange(0.10, 3.00)
        self.study_effect.setSingleStep(0.05)
        self.study_effect.setValue(float(initial_plan.default_standardized_effect or 0.50))
        self.study_effect.setToolTip(
            "Smallest scientifically meaningful paired difference divided by the pilot standard "
            "deviation of paired differences. Replace the default with preregistered pilot evidence."
        )
        self.study_power = QDoubleSpinBox()
        self.study_power.setDecimals(2)
        self.study_power.setRange(0.50, 0.99)
        self.study_power.setSingleStep(0.05)
        self.study_power.setValue(float(initial_plan.default_power or 0.80))
        self.study_power.setToolTip(
            "Probability targeted for detecting the declared smallest meaningful paired effect."
        )
        self.apply_study_button = QPushButton("Apply study protocol throughout")
        self.apply_study_button.setObjectName("PrimaryButton")
        self.study_guidance = QLabel()
        self.study_guidance.setWordWrap(True)
        self.study_guidance.setObjectName("InfoText")
        study_form.addRow("Evidence strength", self.study_strength)
        study_form.addRow("Primary reference case", self.study_case)
        study_form.addRow("Smallest standardized paired effect", self.study_effect)
        study_form.addRow("Target detection power", self.study_power)
        study_form.addRow("", self.apply_study_button)
        study_setup.layout_root.addLayout(study_form)
        study_setup.layout_root.addWidget(self.study_guidance)
        self.study_strength.currentIndexChanged.connect(self._study_strength_changed)
        self.study_case.currentIndexChanged.connect(self._refresh_study_guidance)
        self.study_effect.valueChanged.connect(self._refresh_study_guidance)
        self.study_power.valueChanged.connect(self._refresh_study_guidance)
        self.apply_study_button.clicked.connect(self._apply_study_protocol)
        self._study_strength_changed()

        study_tab = QWidget()
        study_tab_layout = QVBoxLayout(study_tab)
        study_tab_layout.setContentsMargins(10, 10, 10, 10)
        study_tab_layout.addWidget(study_setup)
        study_tab_layout.addStretch(1)
        self.dashboard_tabs.addTab(_scrollable_tab(study_tab), "Study Setup")

        readiness = SectionCard(
            "System Readiness",
            "The application checks available compute and memory before scientific work. It "
            "keeps operating headroom and queues work that cannot be admitted safely.",
        )
        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(22)
        status_grid.setVerticalSpacing(8)
        self.compute_labels: dict[str, QLabel] = {}
        fields = (
            "Protection status",
            "Processor",
            "Available system memory",
            "NVIDIA acceleration",
            "Available accelerator memory",
            "Simultaneous task limit",
            "Current action",
            "Protection note",
        )
        for index, name in enumerate(fields):
            row = index % 4
            col = (index // 4) * 2
            key = QLabel(name)
            key.setObjectName("MetricLabel")
            value = QLabel("—")
            value.setObjectName("ContextValue")
            value.setWordWrap(True)
            self.compute_labels[name] = value
            status_grid.addWidget(key, row, col)
            status_grid.addWidget(value, row, col + 1)
        status_grid.setColumnStretch(1, 1)
        status_grid.setColumnStretch(3, 1)
        readiness.layout_root.addLayout(status_grid)

        refresh_row = QHBoxLayout()
        self.refresh_system_button = QPushButton("Refresh system map")
        self.refresh_system_button.clicked.connect(self._request_compute_refresh)
        self.compute_note = QLabel(
            "Available memory is sampled again whenever work starts. A task may use at most "
            "80% of memory that is free at that admission boundary."
        )
        self.compute_note.setWordWrap(True)
        self.compute_note.setObjectName("HelpText")
        refresh_row.addWidget(self.refresh_system_button)
        refresh_row.addWidget(self.compute_note, 1)
        readiness.layout_root.addLayout(refresh_row)

        self.device_table = QTableWidget(0, 10)
        self.device_table.setHorizontalHeaderLabels(
            [
                "Hardware",
                "Availability",
                "Compute choice",
                "Name",
                "Available memory",
                "Temperature",
                "Power",
                "Supported work",
                "Status",
                "Measurement source",
            ]
        )
        self.device_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.device_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self.device_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for column in (4, 5, 6, 7, 8, 9):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.device_table.setMinimumHeight(280)
        self.device_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        readiness.layout_root.addWidget(self.device_table, 1)

        readiness_tab = QWidget()
        readiness_tab_layout = QVBoxLayout(readiness_tab)
        readiness_tab_layout.setContentsMargins(10, 10, 10, 10)
        readiness_tab_layout.setSpacing(0)
        readiness_tab_layout.addWidget(readiness)
        readiness_tab_layout.addStretch(1)
        self.dashboard_tabs.addTab(_scrollable_tab(readiness_tab), "System Readiness")

        training_queue = SectionCard(
            "Policy Training Queue",
            "The requested scientific branches are preserved. Work that cannot run safely at "
            "the same time waits in the queue and resumes from a recoverable checkpoint.",
        )
        queue_grid = QGridLayout()
        queue_grid.setHorizontalSpacing(22)
        queue_grid.setVerticalSpacing(8)
        self.training_labels: dict[str, QLabel] = {}
        queue_fields = (
            "Policy training status",
            "Scientific branches",
            "Active now",
            "Waiting",
            "Completed",
            "Epoch progress",
            "Recoverable checkpoint",
            "Compute assignment",
        )
        for index, name in enumerate(queue_fields):
            row = index % 5
            col = (index // 5) * 2
            key = QLabel(name)
            key.setObjectName("MetricLabel")
            value = QLabel("—")
            value.setWordWrap(True)
            value.setObjectName("ContextValue")
            self.training_labels[name] = value
            queue_grid.addWidget(key, row, col)
            queue_grid.addWidget(value, row, col + 1)
        queue_grid.setColumnStretch(1, 1)
        queue_grid.setColumnStretch(3, 1)
        training_queue.layout_root.addLayout(queue_grid)

        training_tab = QWidget()
        training_tab_layout = QVBoxLayout(training_tab)
        training_tab_layout.setContentsMargins(10, 10, 10, 10)
        training_tab_layout.setSpacing(0)
        training_tab_layout.addWidget(training_queue)
        training_tab_layout.addStretch(1)
        self.dashboard_tabs.addTab(_scrollable_tab(training_tab), "Training Queue")

        context = SectionCard(
            "Scientific context",
            "Power-system and experiment context remains visible here, but Power System is workflow-locked until CALO governing intelligence is qualified and active.",
        )
        grid = QGridLayout()
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(9)
        context.layout_root.addLayout(grid)
        self.labels: dict[str, QLabel] = {}
        names = [
            "Power-system case",
            "Buses",
            "Generators",
            "Branches",
            "Transformers",
            "Shunt buses",
            "ORPD objective",
            "Primary algorithms",
            "Scenario mode",
            "Completed experiments",
            "Verified results",
            "Governing policy",
        ]
        for index, name in enumerate(names):
            row = index % 6
            col = (index // 6) * 2
            key = QLabel(name)
            key.setObjectName("MetricLabel")
            value = QLabel("—")
            value.setWordWrap(True)
            value.setObjectName("ContextValue")
            self.labels[name] = value
            grid.addWidget(key, row, col)
            grid.addWidget(value, row, col + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        context_tab = QWidget()
        context_tab_layout = QVBoxLayout(context_tab)
        context_tab_layout.setContentsMargins(10, 10, 10, 10)
        context_tab_layout.setSpacing(0)
        context_tab_layout.addWidget(context)
        context_tab_layout.addStretch(1)
        self.dashboard_tabs.addTab(_scrollable_tab(context_tab), "Scientific Context")

        self.dashboard_body_layout.addWidget(self.dashboard_tabs, 1)

        state.case_changed.connect(lambda _: self.refresh())
        state.config_changed.connect(lambda _: self.refresh())
        state.runs_changed.connect(self.refresh)
        state.compute_profile_changed.connect(lambda _profile: self.refresh_compute())
        state.compute_governor_changed.connect(lambda _decision: self.refresh_governor())
        state.policy_state_changed.connect(lambda _status: self.refresh_policy())
        state.policy_training_plan_changed.connect(lambda _payload: self.refresh_training_plan())
        state.policy_training_changed.connect(lambda _active, _detail: self.refresh_training_plan())
        self.refresh()
        self.refresh_compute()
        self.refresh_policy()
        self.refresh_training_plan()
        self.refresh_governor()
        self._protection_timer = QTimer(self)
        self._protection_timer.setInterval(2000)
        self._protection_timer.timeout.connect(self._sample_live_protection)
        self._protection_timer.start()

    def _refresh_study_guidance(self) -> None:
        plan = study_strength_plan(str(self.study_strength.currentData()))
        screening = plan.strength is StudyStrength.LOW
        self.study_guidance.setText(
            plan.guidance(
                self.study_case.currentText(),
                standardized_effect=None if screening else self.study_effect.value(),
                target_power=None if screening else self.study_power.value(),
            )
        )

    def _study_strength_changed(self) -> None:
        plan = study_strength_plan(str(self.study_strength.currentData()))
        screening = plan.strength is StudyStrength.LOW
        self.study_effect.setEnabled(not screening)
        self.study_power.setEnabled(not screening)
        if not screening:
            self.study_effect.blockSignals(True)
            self.study_power.blockSignals(True)
            self.study_effect.setValue(float(plan.default_standardized_effect))
            self.study_power.setValue(float(plan.default_power))
            self.study_effect.blockSignals(False)
            self.study_power.blockSignals(False)
        self._refresh_study_guidance()

    def _apply_study_protocol(self) -> None:
        if bool(getattr(self.state, "policy_training_active", False)):
            QMessageBox.information(
                self,
                "Policy training active",
                "Request Safe Stop before changing the experiment protocol.",
            )
            return
        case_name = self.study_case.currentText()
        try:
            current = self.state.config
            candidate = deepcopy(current)
            plan = apply_study_strength(
                candidate,
                str(self.study_strength.currentData()),
                case_name=case_name,
                standardized_effect=(
                    None
                    if str(self.study_strength.currentData()) == StudyStrength.LOW.value
                    else self.study_effect.value()
                ),
                target_power=(
                    None
                    if str(self.study_strength.currentData()) == StudyStrength.LOW.value
                    else self.study_power.value()
                ),
            )
            case = CaseLoader.load(case_name)
            report = validate_case(case)
            if not report.valid:
                raise ValueError("\n".join(report.errors))
            candidate.validate()
            changes = summarize_study_protocol_change(current, candidate)
            self.state.config = candidate
            self.state.set_case(case)
            self.state.update_config()
            self.state.task_status.finish(
                f"{plan.label} applied throughout; continue with base power-flow validation"
            )
            self._refresh_study_guidance()
            QMessageBox.information(
                self,
                "Study protocol applied",
                f"{plan.label} is now applied to {case_name}. The case is loaded; Power System "
                "still requires the base power flow and independent cross-check before execution."
                + ("\n\nApplied throughout:\n• " + "\n• ".join(changes) if changes else ""),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Study setup", str(exc))

    def _sample_live_protection(self) -> None:
        if getattr(self.state, "compute_protection_profile", None) is None:
            return
        try:
            self.state.sample_compute_governor()
        except Exception:
            # Live telemetry is advisory to the GUI. Training/experiment governors enforce their
            # own fail-closed protection and provenance; a GUI telemetry failure must not fabricate data.
            _LOG.warning("Dashboard live-protection telemetry sampling failed", exc_info=True)
            return

    def refresh_governor(self) -> None:
        decision = getattr(self.state, "compute_governor_decision", None)
        if decision is None:
            for name in ("Protection status", "Current action", "Protection note"):
                if name in self.compute_labels:
                    self.compute_labels[name].setText("—")
            return
        state_text = str(
            getattr(
                getattr(decision, "state", None), "value", getattr(decision, "state", "UNKNOWN")
            )
        )
        self.compute_labels["Protection status"].setText(state_text.replace("_", " ").title())
        if bool(getattr(decision, "request_safe_stop", False)):
            action = "Stop safely and preserve a recoverable checkpoint"
        elif bool(getattr(decision, "allow_new_admission", False)):
            action = "New work may start"
        else:
            action = "New work is waiting; active work is protected"
        self.compute_labels["Current action"].setText(action)
        reasons = tuple(getattr(decision, "reasons", ()) or ())
        self.compute_labels["Protection note"].setText(
            "; ".join(reasons) if reasons else "System headroom is available"
        )

    def _request_compute_refresh(self) -> None:
        self.refresh_system_button.setEnabled(False)
        self.system_metric.set_metric("Scanning", "Mapping CPU/CUDA resources")
        try:
            self.state.refresh_compute_profile()
            self.state.task_status.finish("System readiness and memory protection refreshed")
        except Exception as exc:
            QMessageBox.critical(
                self, "System readiness scan failed", f"{type(exc).__name__}: {exc}"
            )
            self.state.task_status.fail(
                f"System readiness scan failed: {type(exc).__name__}: {exc}"
            )
        finally:
            self.refresh_system_button.setEnabled(
                not bool(getattr(self.state, "policy_training_active", False))
            )

    def refresh_compute(self) -> None:
        topology = getattr(self.state, "compute_topology", None)
        profile = getattr(self.state, "compute_protection_profile", None)
        if topology is None or profile is None:
            self.system_metric.set_metric("Scanning", "Startup readiness scan pending")
            self.branch_metric.set_metric("—", "Safe limit not calculated yet")
            self.device_table.setRowCount(0)
            return

        self.system_metric.set_metric(profile.status, "Protected automatic resource selection")
        self.branch_metric.set_metric(
            str(profile.safe_parallel_branches),
            "Safe simultaneous ceiling; excess scientific branches remain queued",
        )
        self.compute_labels["Protection status"].setText(profile.status)
        self.compute_labels["Processor"].setText(topology.cpu_name)
        free_system_memory = max(
            0,
            int(topology.ram_total_bytes * (1.0 - topology.ram_used_percent / 100.0)),
        )
        self.compute_labels["Available system memory"].setText(
            f"Approximately {_bytes_text(free_system_memory)} now"
        )
        cuda_devices = [device for device in topology.devices if device.backend == "cuda"]
        self.compute_labels["NVIDIA acceleration"].setText(
            "Available" if cuda_devices else "Not available; CPU mode remains usable"
        )
        free_accelerator_memory = sum(
            max(
                0,
                int(device.memory_total_bytes * (1.0 - device.memory_used_percent / 100.0)),
            )
            for device in cuda_devices
        )
        self.compute_labels["Available accelerator memory"].setText(
            f"Approximately {_bytes_text(free_accelerator_memory)} now"
            if cuda_devices
            else "Not available"
        )
        self.compute_labels["Simultaneous task limit"].setText(str(profile.safe_parallel_branches))
        reason_text = " ".join(profile.reasons)
        self.compute_note.setText(
            "Memory availability is sampled again at task admission and each task is capped at "
            "80% of what is free at that moment. "
            + (
                f"Protection note: {reason_text}"
                if reason_text
                else "Protected resource headroom is available."
            )
        )

        self.device_table.setRowCount(len(topology.devices) + 1)
        cpu_roles = "CPU-only experiments; protected fallback; independent validation"
        decision = getattr(self.state, "compute_governor_decision", None)
        live_snapshot = (
            dict(getattr(decision, "snapshot", {}) or {}) if decision is not None else {}
        )
        cpu_temp = live_snapshot.get("cpu_temperature_c")
        cpu_values = [
            "System processor",
            "Available",
            "CPU only",
            topology.cpu_name,
            (
                f"~{_bytes_text(int(topology.ram_total_bytes * (1.0 - topology.ram_used_percent / 100.0)))} "
                f"of {_bytes_text(topology.ram_total_bytes)}"
            ),
            ("unavailable" if cpu_temp is None else f"{float(cpu_temp):.1f} °C"),
            "—",
            cpu_roles,
            "Ready",
            "System monitor",
        ]
        for column, value in enumerate(cpu_values):
            self.device_table.setItem(0, column, QTableWidgetItem(str(value)))
        for row, device in enumerate(topology.devices, start=1):
            roles = []
            if device.ppo_learner or device.policy_actor:
                roles.append("Policy training")
            if device.orpd_evaluator:
                roles.append("Power-system experiments")
            if device.full_training_branch:
                roles.append("Complete training branch")
            live_device = next(
                (
                    row
                    for row in list(live_snapshot.get("devices", []) or [])
                    if str(row.get("device_id", "")) == device.runtime_id
                ),
                {},
            )
            device_temp = live_device.get("temperature_c")
            power_w = live_device.get("power_w")
            power_limit = live_device.get("power_limit_w")
            power_text = "unavailable"
            if power_w is not None:
                power_text = f"{float(power_w):.1f} W" + (
                    f" / {float(power_limit):.1f} W" if power_limit is not None else ""
                )
            values = [
                device.os_label,
                "Available" if device.capability_status == "validated" else "Detected",
                "NVIDIA acceleration",
                device.name,
                (
                    f"~{_bytes_text(int(device.memory_total_bytes * (1.0 - device.memory_used_percent / 100.0)))} "
                    f"of {_bytes_text(device.memory_total_bytes)}"
                    if device.memory_total_bytes
                    else "unavailable"
                ),
                ("unavailable" if device_temp is None else f"{float(device_temp):.1f} °C"),
                power_text,
                ", ".join(roles) or "Availability check only",
                (
                    device.capability_status
                    + (f" — {device.capability_detail}" if device.capability_detail else "")
                ),
                device.telemetry or "Runtime capability probe",
            ]
            for column, value in enumerate(values):
                self.device_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.refresh_governor()

    def set_training_exclusive_mode(self, active: bool, detail: str = "") -> None:
        # Dashboard stays readable during policy training; only actions that could alter the
        # authoritative hardware profile are frozen by the Global Training Exclusive Lock.
        self.refresh_system_button.setEnabled(not bool(active))
        if active and detail:
            self.training_metric.set_metric("ACTIVE / LOCKED", detail)
        self.refresh_training_plan()

    def refresh_training_plan(self) -> None:
        plan = dict(getattr(self.state, "policy_training_plan", {}) or {})
        active_lock = bool(getattr(self.state, "policy_training_active", False))
        if not plan:
            self.training_metric.set_metric("Idle", "No policy-training queue is active")
            for label in self.training_labels.values():
                label.setText("—")
            self.training_labels["Policy training status"].setText("IDLE")
            return
        total = int(plan.get("total_branches", 0) or 0)
        simultaneous = int(plan.get("simultaneous_limit", 0) or 0)
        active = int(plan.get("active_branches", 0) or 0)
        queued = int(plan.get("queued_branches", 0) or 0)
        completed = int(plan.get("completed_branches", 0) or 0)
        status = str(plan.get("status", "RUNNING" if active_lock else "IDLE") or "")
        resource_plan = dict(plan.get("resource_plan", {}) or {})
        slots = list(resource_plan.get("slots", []) or [])
        assignments = []
        for slot in slots:
            primary = str(slot.get("primary_device", "") or "")
            text = f"slot {slot.get('slot_index', '?')}: {primary}"
            assignments.append(text)
        self.training_metric.set_metric(
            "ACTIVE" if active_lock else status,
            f"{active} active · {queued} queued · {completed}/{total} completed",
        )
        self.training_labels["Policy training status"].setText(
            status or ("RUNNING" if active_lock else "IDLE")
        )
        self.training_labels["Scientific branches"].setText(
            f"{total} requested; up to {simultaneous} at the same time"
        )
        self.training_labels["Active now"].setText(str(active))
        self.training_labels["Waiting"].setText(str(queued))
        self.training_labels["Completed"].setText(str(completed))
        overall_raw = plan.get("overall_percent", -1)
        overall = int(overall_raw) if overall_raw is not None else -1
        completed_branch_epochs = int(plan.get("completed_branch_epochs", 0) or 0)
        total_branch_epochs = int(plan.get("total_branch_epochs", 0) or 0)
        if overall >= 0 and total_branch_epochs > 0:
            epoch_progress = (
                f"{overall}% · {completed_branch_epochs}/{total_branch_epochs} branch-epochs"
            )
        elif active_lock:
            branch_rows = list(plan.get("branch_progress", []) or [])
            epoch_progress = (
                " · ".join(
                    f"{row.get('branch_id')} e{int(row.get('current_epoch', 0) or 0)}"
                    for row in branch_rows[:6]
                )
                or "Indefinite / initializing"
            )
        else:
            epoch_progress = "—"
        self.training_labels["Epoch progress"].setText(epoch_progress)
        safe_epoch = plan.get("common_safe_epoch", None)
        self.training_labels["Recoverable checkpoint"].setText(
            f"Last common exact epoch {int(safe_epoch)}"
            if safe_epoch is not None and int(safe_epoch) >= 0
            else "Not yet materialized"
        )
        self.training_labels["Compute assignment"].setText(
            " · ".join(assignments) if assignments else "Planning / initialization"
        )

    def refresh_policy(self) -> None:
        status = self.state.governing_policy_status()
        if status.ready:
            self.policy_metric.set_metric("READY", f"{status.policy_name} · {status.grade}")
            self.labels["Governing policy"].setText(
                f"{status.policy_name} · {status.grade} · SHA {status.policy_sha256[:12]}…"
            )
        else:
            self.policy_metric.set_metric("NOT READY", status.state.replace("_", " ").title())
            self.labels["Governing policy"].setText(status.reason)

    def refresh(self) -> None:
        case = self.state.current_case
        if case:
            metrics = summarize_case(case)
            self.labels["Power-system case"].setText(case.name)
            self.labels["Buses"].setText(str(metrics["buses"]))
            self.labels["Generators"].setText(str(metrics["generators"]))
            self.labels["Branches"].setText(str(metrics["branches"]))
            self.labels["Transformers"].setText(str(metrics["transformers"]))
            self.labels["Shunt buses"].setText(str(metrics["shunt_buses"]))
        else:
            for name in (
                "Power-system case",
                "Buses",
                "Generators",
                "Branches",
                "Transformers",
                "Shunt buses",
            ):
                self.labels[name].setText("—")

        objective = self.state.config.objective.kind.value
        algorithms = list(self.state.config.algorithms)
        self.labels["ORPD objective"].setText(objective)
        self.labels["Primary algorithms"].setText(", ".join(algorithms))
        self.labels["Scenario mode"].setText(self.state.config.scenarios.mode)

        experiments = self.state.database.list_experiments()
        verified = sum(
            1
            for experiment in experiments
            for run in self.state.database.list_runs(experiment["id"])
            if run["validation_status"] == "verified"
        )
        self.labels["Completed experiments"].setText(str(len(experiments)))
        self.labels["Verified results"].setText(str(verified))
        self.verified_metric.set_metric(str(verified), f"{len(experiments)} experiment record(s)")
        self.refresh_policy()
