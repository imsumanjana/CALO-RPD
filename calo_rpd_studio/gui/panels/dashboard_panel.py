"""v6.2 System Readiness, Adaptive Compute Protection, and scientific-context dashboard."""

from __future__ import annotations

import logging

_LOG = logging.getLogger(__name__)

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
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
from calo_rpd_studio.power_system.network_metrics import summarize_case


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
            "System readiness, CPU/XPU/GPU runtime mapping, Safe-80 compute protection, governing-policy status, and current scientific context.",
            parent,
        )
        self.state = state

        # The dashboard body is vertically scrollable so summary cards and the
        # active tab keep their natural size instead of being compressed when
        # the application window is shorter than the preferred dashboard height.
        self.dashboard_body = QWidget()
        self.dashboard_body.setObjectName("DashboardScrollableBody")
        self.dashboard_body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
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
        self.system_metric = MetricCard("System protection", "Scanning", "Safe-80 compute envelope")
        self.branch_metric = MetricCard("Safe parallel branches", "—", "Calculated from protected hardware capacity")
        self.policy_metric = MetricCard("CALO governing intelligence", "Not ready", "Qualified active policy required")
        self.verified_metric = MetricCard("Verified results", "0", "Independent validation required for export")
        self.training_metric = MetricCard("Policy training queue", "Idle", "Total branches and Safe-80 concurrency are separate")
        metric_cards = (self.system_metric, self.branch_metric, self.policy_metric, self.verified_metric, self.training_metric)
        for index, card in enumerate(metric_cards):
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row, column = divmod(index, 3)
            metrics.addWidget(card, row, column)
        for column in range(3):
            metrics.setColumnStretch(column, 1)
        self.dashboard_body_layout.addLayout(metrics)

        self.dashboard_tabs = QTabWidget()
        self.dashboard_tabs.setObjectName("DashboardTabs")
        self.dashboard_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.dashboard_tabs.setMinimumHeight(500)

        readiness = SectionCard(
            "System Readiness & Compute Protection",
            "CALO-RPD maps physical CPU/XPU/GPU resources to runtime identifiers before scientific work. The default Safe-80 profile reserves 20% operating headroom and calculates the hard simultaneous-branch ceiling used by the protected queue scheduler.",
        )
        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(22)
        status_grid.setVerticalSpacing(8)
        self.compute_labels: dict[str, QLabel] = {}
        fields = (
            "Protection profile",
            "System status",
            "CPU topology",
            "Safe CPU worker budget",
            "System RAM",
            "Safe RAM ceiling",
            "Accelerator branch slots",
            "Maximum safe simultaneous branches",
            "Live protection state",
            "CPU live load / temperature",
            "Protection action",
            "Last protection reason",
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
            "OS GPU numbering and PyTorch runtime numbering are separate. The table below explicitly links the detected physical/OS adapter to CALO runtime IDs such as cuda:0 or xpu:0."
        )
        self.compute_note.setWordWrap(True)
        self.compute_note.setObjectName("HelpText")
        refresh_row.addWidget(self.refresh_system_button)
        refresh_row.addWidget(self.compute_note, 1)
        readiness.layout_root.addLayout(refresh_row)

        self.device_table = QTableWidget(0, 10)
        self.device_table.setHorizontalHeaderLabels(
            ["OS / physical adapter", "CALO runtime", "Backend", "Device", "Memory", "Temperature", "Power", "Validated roles", "Capability status", "Telemetry"]
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
            "Protected Policy Training Queue",
            "v6.2 separates scientific branch diversity from simultaneous execution. Dashboard Safe-80 sets the hard concurrency ceiling; excess branches remain queued and rotate through exact-resume leases without silent CPU spillover.",
        )
        queue_grid = QGridLayout()
        queue_grid.setHorizontalSpacing(22)
        queue_grid.setVerticalSpacing(8)
        self.training_labels: dict[str, QLabel] = {}
        queue_fields = (
            "Training status",
            "Total scientific branches",
            "Safe simultaneous limit",
            "Active branches",
            "Queued branches",
            "Completed branches",
            "Global CPU worker budget",
            "Resource assignment",
        )
        for index, name in enumerate(queue_fields):
            row = index % 4
            col = (index // 4) * 2
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
            for name in ("Live protection state", "CPU live load / temperature", "Protection action", "Last protection reason"):
                if name in self.compute_labels:
                    self.compute_labels[name].setText("—")
            return
        state_text = str(getattr(getattr(decision, "state", None), "value", getattr(decision, "state", "UNKNOWN")))
        snapshot = dict(getattr(decision, "snapshot", {}) or {})
        cpu = float(snapshot.get("cpu_percent", 0.0) or 0.0)
        temp = snapshot.get("cpu_temperature_c")
        temp_text = "temperature unavailable" if temp is None else f"{float(temp):.1f} °C"
        self.compute_labels["Live protection state"].setText(state_text)
        self.compute_labels["CPU live load / temperature"].setText(f"{cpu:.1f}% · {temp_text}")
        if bool(getattr(decision, "request_safe_stop", False)):
            action = "RED · exact Safe Stop required"
        elif bool(getattr(decision, "allow_new_admission", False)):
            action = "GREEN · staged admission allowed"
        else:
            action = "AMBER · no new admission / active workload throttled"
        self.compute_labels["Protection action"].setText(action)
        reasons = tuple(getattr(decision, "reasons", ()) or ())
        self.compute_labels["Last protection reason"].setText("; ".join(reasons) if reasons else "No protection threshold currently exceeded")

    def _request_compute_refresh(self) -> None:
        self.refresh_system_button.setEnabled(False)
        self.system_metric.set_metric("Scanning", "Mapping CPU/XPU/GPU resources")
        try:
            self.state.refresh_compute_profile()
            self.state.task_status.finish("System compute map and Safe-80 protection profile refreshed")
        except Exception as exc:
            QMessageBox.critical(self, "System readiness scan failed", f"{type(exc).__name__}: {exc}")
            self.state.task_status.fail(f"System readiness scan failed: {type(exc).__name__}: {exc}")
        finally:
            self.refresh_system_button.setEnabled(not bool(getattr(self.state, "policy_training_active", False)))

    def refresh_compute(self) -> None:
        topology = getattr(self.state, "compute_topology", None)
        profile = getattr(self.state, "compute_protection_profile", None)
        if topology is None or profile is None:
            self.system_metric.set_metric("Scanning", "Startup readiness scan pending")
            self.branch_metric.set_metric("—", "Safe limit not calculated yet")
            self.device_table.setRowCount(0)
            return

        self.system_metric.set_metric(profile.status, f"{profile.profile_name} · {profile.reserve_percent}% reserve")
        self.branch_metric.set_metric(
            str(profile.safe_parallel_branches),
            "Safe simultaneous ceiling; excess scientific branches remain queued",
        )
        self.compute_labels["Protection profile"].setText(
            f"{profile.profile_name} · {profile.reserve_percent}% reserved"
        )
        self.compute_labels["System status"].setText(profile.status)
        self.compute_labels["CPU topology"].setText(
            f"{topology.cpu_name} · {topology.physical_cores} physical / {topology.logical_threads} logical"
        )
        self.compute_labels["Safe CPU worker budget"].setText(
            f"{profile.safe_cpu_worker_budget} global worker-equivalents (shared across active work)"
        )
        self.compute_labels["System RAM"].setText(
            f"{_bytes_text(topology.ram_total_bytes)} · {topology.ram_used_percent:.1f}% currently used"
        )
        self.compute_labels["Safe RAM ceiling"].setText(_bytes_text(profile.safe_ram_ceiling_bytes))
        self.compute_labels["Accelerator branch slots"].setText(str(profile.accelerator_branch_slots))
        self.compute_labels["Maximum safe simultaneous branches"].setText(
            str(profile.safe_parallel_branches)
        )
        reason_text = " ".join(profile.reasons)
        self.compute_note.setText(
            "OS GPU numbering and PyTorch runtime numbering are separate. "
            "The mapping below is authoritative for CALO scheduling. "
            + (f"Protection note: {reason_text}" if reason_text else "Safe-80 resource headroom is available.")
        )

        self.device_table.setRowCount(len(topology.devices) + 1)
        cpu_roles = "Host orchestration / protected CPU fallback"
        decision = getattr(self.state, "compute_governor_decision", None)
        live_snapshot = dict(getattr(decision, "snapshot", {}) or {}) if decision is not None else {}
        cpu_temp = live_snapshot.get("cpu_temperature_c")
        cpu_values = [
            "CPU",
            "cpu",
            "CPU",
            topology.cpu_name,
            _bytes_text(topology.ram_total_bytes),
            ("unavailable" if cpu_temp is None else f"{float(cpu_temp):.1f} °C"),
            "—",
            cpu_roles,
            "Safe-80 protected host runtime",
            "psutil host telemetry",
        ]
        for column, value in enumerate(cpu_values):
            self.device_table.setItem(0, column, QTableWidgetItem(str(value)))
        for row, device in enumerate(topology.devices, start=1):
            roles = []
            if device.ppo_learner:
                roles.append("PPO")
            if device.policy_actor:
                roles.append("Actor")
            if device.orpd_evaluator:
                roles.append("ORPD")
            if device.full_training_branch:
                roles.append("Full branch")
            live_device = next(
                (row for row in list(live_snapshot.get("devices", []) or []) if str(row.get("device_id", "")) == device.runtime_id),
                {},
            )
            device_temp = live_device.get("temperature_c")
            power_w = live_device.get("power_w")
            power_limit = live_device.get("power_limit_w")
            power_text = "unavailable"
            if power_w is not None:
                power_text = f"{float(power_w):.1f} W" + (f" / {float(power_limit):.1f} W" if power_limit is not None else "")
            values = [
                device.os_label,
                device.runtime_id,
                f"{device.backend.upper()} / {device.runtime}",
                device.name,
                (_bytes_text(device.memory_total_bytes) if device.memory_total_bytes else f"{device.memory_used_percent:.1f}% used"),
                ("unavailable" if device_temp is None else f"{float(device_temp):.1f} °C"),
                power_text,
                ", ".join(roles) or "Detected only",
                (device.capability_status + (f" — {device.capability_detail}" if device.capability_detail else "")),
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
            self.training_labels["Training status"].setText("IDLE")
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
            aux = str(slot.get("auxiliary_xpu_runtime", "") or "")
            text = f"slot {slot.get('slot_index', '?')}: {primary}"
            if aux:
                text += f" + XPU {aux} actor/evaluator"
            assignments.append(text)
        self.training_metric.set_metric(
            "ACTIVE" if active_lock else status,
            f"{active} active · {queued} queued · {completed}/{total} completed",
        )
        self.training_labels["Training status"].setText(status or ("RUNNING" if active_lock else "IDLE"))
        self.training_labels["Total scientific branches"].setText(str(total))
        self.training_labels["Safe simultaneous limit"].setText(str(simultaneous))
        self.training_labels["Active branches"].setText(str(active))
        self.training_labels["Queued branches"].setText(str(queued))
        self.training_labels["Completed branches"].setText(str(completed))
        self.training_labels["Global CPU worker budget"].setText(
            str(plan.get("global_cpu_worker_budget", resource_plan.get("global_cpu_worker_budget", "—")))
        )
        self.training_labels["Resource assignment"].setText(" · ".join(assignments) if assignments else "Planning / initialization")

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
            for name in ("Power-system case", "Buses", "Generators", "Branches", "Transformers", "Shunt buses"):
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
