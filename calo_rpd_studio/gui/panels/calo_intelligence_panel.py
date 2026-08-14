"""Policy inventory and explicit TSH-CALO lifecycle controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment, QStandardPaths, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.ai.model_io import checkpoint_sha256
from calo_rpd_studio.algorithms.calo.policy_readiness import policy_record_user_status
from calo_rpd_studio.algorithms.calo.tsh_calo_automatic_qualification import (
    AutomaticQualificationRejected,
    AutomaticQualificationSourceSnapshot,
    AutomaticQualificationWorkspace,
    AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST,
    automatic_qualification_workload,
    automatic_qualification_workflow_payload,
    build_automatic_formal_qualification_plan,
    freeze_plan,
    prepare_automatic_source_snapshot,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification_campaign import (
    QUALIFICATION_CONTROL_FILE,
    QUALIFICATION_STATUS_FILE,
    TSH_CALO_QUALIFICATION_CONTROL_SCHEMA,
    TSH_CALO_QUALIFICATION_EVENT_SCHEMA,
    TSH_CALO_QUALIFICATION_PAUSE_EXIT_CODE,
    TSH_CALO_QUALIFICATION_STATUS_SCHEMA,
    TSHCALOQualificationPlan,
    qualification_candidate_contract,
    request_tsh_calo_qualification_pause,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import TSH_CALO_ALGORITHM_ID
from calo_rpd_studio.gui.user_feedback import show_error
from calo_rpd_studio.gui.widgets.page_header import PageHeader
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.scripts.qualify_tsh_calo import (
    QUALIFICATION_EVENT_PREFIX,
    ROOT as QUALIFICATION_SOURCE_ROOT,
    load_plan as load_qualification_plan,
    validate_repository_for_plan,
)


class CALOIntelligencePanel(ScrollablePage):
    """Manage policies without embedding or retaining a second training implementation."""

    stage_completed = pyqtSignal()
    independent_training_requested = pyqtSignal()
    activity_message = pyqtSignal(str, str)

    def __init__(self, state, experiment_manager, model_library=None, parent=None) -> None:
        del experiment_manager
        content = QWidget()
        super().__init__(content, parent)
        self.state = state
        self.model_library = model_library
        self._policy_rows = []
        self._qualification_process: QProcess | None = None
        self._qualification_process_output = ""
        self._qualification_stdout_buffer = ""
        self._qualification_policy_id = ""
        self._qualification_process_stage = ""
        self._qualification_workspace: AutomaticQualificationWorkspace | None = None
        self._qualification_source_snapshot: AutomaticQualificationSourceSnapshot | None = None
        self._qualification_expected_cells = 0
        self._qualification_reported_cells = -1
        self._qualification_pause_requested = False
        self._qualification_live_event: dict = {}
        self._qualification_last_event: dict = {}
        self._qualification_progress_timer = QTimer(self)
        self._qualification_progress_timer.setInterval(500)
        self._qualification_progress_timer.timeout.connect(self._update_qualification_progress)
        self.state.task_status.cancel_requested.connect(
            self.request_safe_qualification_pause
        )

        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 72)
        layout.setSpacing(16)
        layout.addWidget(PageHeader("CALO Intelligence", "Manage the governing TSH-CALO policy."))

        actions = SectionCard("Policy actions")
        action_row = QHBoxLayout()
        self.new_training_button = QPushButton("Train policy")
        self.new_training_button.setObjectName("PrimaryButton")
        self.new_training_button.setToolTip("Open the independent policy-training inputs.")
        self.new_training_button.clicked.connect(self.independent_training_requested)
        action_row.addWidget(self.new_training_button)
        action_row.addStretch(1)
        actions.layout_root.addLayout(action_row)
        layout.addWidget(actions)

        library = QGroupBox("Policy library")
        self.policy_center_group = library
        library.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        library_layout = QVBoxLayout(library)
        self.policy_table = QTableWidget(0, 7)
        self.policy_table.setHorizontalHeaderLabels(
            (
                "Active",
                "Policy",
                "Training evaluations",
                "Grade",
                "Scientific status",
                "Evidence",
                "Compatibility",
            )
        )
        self.policy_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.policy_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.policy_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.policy_table.setAlternatingRowColors(True)
        self.policy_table.setWordWrap(True)
        self.policy_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.policy_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.policy_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.policy_table.verticalHeader().setVisible(False)
        for column in (0, 2, 3, 5, 6):
            self.policy_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        for column in (1, 4):
            self.policy_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.Stretch
            )
        self.policy_table.itemSelectionChanged.connect(self._policy_selection_changed)
        library_layout.addWidget(self.policy_table)

        qualification_buttons = QHBoxLayout()
        self.policy_import_button = QPushButton("Import policy")
        self.qualification_button = QPushButton("Qualify policy")
        self.qualification_compare_button = QPushButton("Compare qualified policies")
        self.policy_activate_button = QPushButton("Activate for experiments")
        self.policy_archive_button = QPushButton("Archive")
        self.policy_delete_button = QPushButton("Delete model files")
        self.policy_refresh_button = QPushButton("Refresh")
        self.show_archived_policies = QCheckBox("Show archived")
        self.policy_import_button.clicked.connect(self.import_policy)
        self.qualification_button.clicked.connect(self.qualify_selected_policy)
        self.qualification_compare_button.clicked.connect(self.compare_qualified_policies)
        self.policy_activate_button.clicked.connect(self.activate_selected_policy)
        self.policy_archive_button.clicked.connect(self.archive_selected_policy)
        self.policy_delete_button.clicked.connect(self.delete_selected_model_files)
        self.policy_refresh_button.clicked.connect(self.refresh_policy_library)
        self.show_archived_policies.toggled.connect(lambda _checked: self.refresh_policy_library())
        for button in (
            self.policy_import_button,
            self.qualification_button,
            self.qualification_compare_button,
            self.policy_activate_button,
        ):
            qualification_buttons.addWidget(button)
        qualification_buttons.addStretch(1)
        library_layout.addLayout(qualification_buttons)
        lifecycle_buttons = QHBoxLayout()
        for button in (
            self.policy_archive_button,
            self.policy_delete_button,
            self.policy_refresh_button,
        ):
            lifecycle_buttons.addWidget(button)
        lifecycle_buttons.addWidget(self.show_archived_policies)
        lifecycle_buttons.addStretch(1)
        library_layout.addLayout(lifecycle_buttons)
        self.qualification_workflow_status = QLabel(
            "Workflow: import -> qualify (freeze, check, run/resume, verify, admit or reject) -> "
            "compare -> activate explicitly."
        )
        self.qualification_workflow_status.setWordWrap(True)
        self.qualification_workflow_status.setObjectName("ContextValue")
        library_layout.addWidget(self.qualification_workflow_status)
        layout.addWidget(library)

        controller = QGroupBox("Governing policy")
        self.policy_controller_group = controller
        controller.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        form = QFormLayout(controller)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.path = QLineEdit()
        self.path.setMinimumWidth(0)
        self.path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.path.setReadOnly(True)
        self.path.setPlaceholderText("Select a compatible policy in the Policy library")
        self.deterministic = QCheckBox("Use deterministic policy decisions during evaluation")
        self.deterministic.setChecked(True)
        self.apply_policy_button = QPushButton(
            "Apply governing policy and continue to Power System"
        )
        self.apply_policy_button.setObjectName("PrimaryButton")
        self.apply_policy_button.clicked.connect(self.apply_policy_configuration)
        self.policy_gate_status = QLabel()
        self.policy_gate_status.setWordWrap(True)
        self.policy_gate_status.setObjectName("ContextValue")
        self.policy_gate_status.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        form.addRow("Policy", self.path)
        form.addRow("Evaluation", self.deterministic)
        form.addRow("Status", self.policy_gate_status)
        form.addRow("", self.apply_policy_button)
        layout.addWidget(controller)
        layout.addStretch(1)

        self.state.policy_state_changed.connect(lambda _status: self._update_policy_gate_state())
        if self.model_library is not None:
            self.model_library.changed.connect(self.refresh_policy_library)
        self.refresh_policy_library()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if hasattr(self, "policy_table"):
            self._resize_policy_table_to_entries()

    def refresh_policy_library(self) -> None:
        selected_key = self._row_key(self._selected_row())
        governing = self.state.governing_policy_status()
        all_registered = [
            policy
            for policy in self.state.policy_registry.list(include_archived=True)
            if not policy.checkpoint_path.endswith(".resume.pt")
            and "_lineage" not in str(policy.checkpoint_path)
        ]
        registered = [
            policy
            for policy in all_registered
            if self.show_archived_policies.isChecked() or not policy.archived
        ]
        registered_by_path = {
            str(Path(policy.checkpoint_path).expanduser().resolve()).casefold(): policy
            for policy in all_registered
        }
        discovered: list[dict] = []
        matched_policy_ids: set[str] = set()
        if self.model_library is not None:
            for campaign in self.model_library.completed_campaigns():
                candidate_path = str(campaign.get("policy_candidate", ""))
                registered_policy = registered_by_path.get(candidate_path.casefold())
                if registered_policy is not None:
                    matched_policy_ids.add(registered_policy.id)
                discovered.append(
                    {
                        **campaign,
                        "row_kind": "completed_training",
                        "registered_policy": registered_policy,
                    }
                )
        self._policy_rows = [
            policy for policy in registered if policy.id not in matched_policy_ids
        ] + discovered
        self.policy_table.setRowCount(len(self._policy_rows))
        for row, policy in enumerate(self._policy_rows):
            if isinstance(policy, dict):
                registered_policy = policy.get("registered_policy")
                candidate_error = str(policy.get("candidate_error", ""))
                if registered_policy is None:
                    values = (
                        "",
                        str(policy.get("campaign_id", "Saved policy")),
                        self._training_evaluation_text(
                            policy.get("training_evaluations")
                        ),
                        "U",
                        (
                            "Training complete · import required"
                            if policy.get("policy_candidate")
                            else "Training complete · candidate unavailable"
                        ),
                        "Not admitted",
                        "Verified saved candidate" if not candidate_error else "Needs attention",
                    )
                else:
                    scientific_status = policy_record_user_status(registered_policy)
                    if (
                        registered_policy.active
                        and governing.ready
                        and governing.policy_id == registered_policy.id
                    ):
                        scientific_status = "Ready and active"
                    values = (
                        "Active" if registered_policy.active else "",
                        str(policy.get("campaign_id", registered_policy.name)),
                        self._training_evaluation_text(
                            self.state.policy_registry.training_evaluation_count(
                                registered_policy.id
                            )
                        ),
                        registered_policy.grade,
                        scientific_status,
                        (
                            "Passed evidence admitted"
                            if registered_policy.qualification_status == "qualified"
                            else "Formal qualification required"
                        ),
                        (
                            "Compatible"
                            if registered_policy.compatible_with(TSH_CALO_ALGORITHM_ID)
                            else "Not compatible"
                        ),
                    )
            else:
                scientific_status = policy_record_user_status(policy)
                if policy.active and governing.ready and governing.policy_id == policy.id:
                    scientific_status = "Ready and selected"
                if policy.active and not governing.ready:
                    compatibility = "Verification required"
                else:
                    compatibility = (
                        "Compatible"
                        if policy.compatible_with(TSH_CALO_ALGORITHM_ID)
                        else "Not compatible"
                    )
                values = (
                    "Active" if policy.active else "",
                    policy.name,
                    self._training_evaluation_text(
                        self.state.policy_registry.training_evaluation_count(policy.id)
                    ),
                    policy.grade,
                    scientific_status,
                    (
                        "Passed evidence admitted"
                        if policy.qualification_status == "qualified"
                        else "Formal qualification required"
                    ),
                    compatibility,
                )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 2:
                    item.setToolTip(
                        "Cumulative exact candidate evaluations used to train this model. "
                        "Completed extension segments are included; qualification and experiment "
                        "evaluations are excluded."
                    )
                self.policy_table.setItem(row, column, item)
        self._resize_policy_table_to_entries()
        if selected_key:
            self._select_row_key(selected_key)
        elif self._policy_rows:
            self.policy_table.selectRow(0)
        self._update_policy_gate_state()
        self.state.notify_policy_state_changed()

    def _resize_policy_table_to_entries(self) -> None:
        """Give the library one visible row per entry and no nested scrollbar."""

        self.policy_table.resizeRowsToContents()
        header = self.policy_table.horizontalHeader()
        header_height = max(header.height(), header.sizeHint().height())
        body_height = sum(
            self.policy_table.rowHeight(row) for row in range(self.policy_table.rowCount())
        )
        frame_height = self.policy_table.frameWidth() * 2
        self.policy_table.setFixedHeight(header_height + body_height + frame_height)
        self.policy_table.updateGeometry()
        self.policy_center_group.updateGeometry()
        if self.widget() is not None:
            self.widget().updateGeometry()
        self.updateGeometry()
        self._queue_external_height_sync()

    @staticmethod
    def _training_evaluation_text(value) -> str:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return f"{value:,}"
        return "Not available"

    def _selected_row(self):
        row = self.policy_table.currentRow()
        return self._policy_rows[row] if 0 <= row < len(self._policy_rows) else None

    def _selected_policy(self):
        selected = self._selected_row()
        if isinstance(selected, dict):
            return selected.get("registered_policy")
        return selected

    def _selected_completed_training(self) -> dict | None:
        selected = self._selected_row()
        return selected if isinstance(selected, dict) else None

    @staticmethod
    def _row_key(row) -> str:
        if isinstance(row, dict):
            return f"training:{row.get('directory', '')}"
        return f"policy:{getattr(row, 'id', '')}" if row is not None else ""

    def _select_row_key(self, row_key: str) -> None:
        for row, item in enumerate(self._policy_rows):
            if self._row_key(item) == row_key:
                self.policy_table.selectRow(row)
                return

    def _select_policy_id(self, policy_id: str) -> None:
        for row, policy in enumerate(self._policy_rows):
            registered_policy = (
                policy.get("registered_policy") if isinstance(policy, dict) else policy
            )
            if getattr(registered_policy, "id", "") == policy_id:
                self.policy_table.selectRow(row)
                return

    def _policy_selection_changed(self) -> None:
        completed_training = self._selected_completed_training()
        if completed_training is not None:
            policy = self._selected_policy()
            if policy is None:
                self.policy_import_button.setText("Import trained policy")
                self.policy_import_button.setEnabled(
                    bool(completed_training.get("policy_candidate", ""))
                )
                self.policy_activate_button.setText("Import before activation")
                self.policy_activate_button.setEnabled(False)
                self.policy_archive_button.setEnabled(False)
                self.policy_archive_button.setText("Archive")
                self.policy_delete_button.setToolTip(
                    "Permanently delete this unregistered completed campaign directory after "
                    "an exact-path confirmation."
                )
                self.policy_delete_button.setEnabled(
                    bool(completed_training.get("directory", ""))
                )
            else:
                self.policy_import_button.setText("Imported")
                self.policy_import_button.setEnabled(False)
                if policy.active:
                    self.policy_activate_button.setText("Active governing policy")
                    self.policy_activate_button.setEnabled(False)
                else:
                    eligible = policy_record_user_status(policy) == "Eligible to select"
                    self.policy_activate_button.setText(
                        "Activate for experiments" if eligible else "Qualification required"
                    )
                    self.policy_activate_button.setEnabled(eligible)
                self.policy_archive_button.setEnabled(not policy.active)
                self.policy_archive_button.setText(
                    "Restore archived" if policy.archived else "Archive"
                )
                removal_blocker = self._registered_completed_removal_blocker(
                    completed_training,
                    policy,
                )
                self.policy_delete_button.setEnabled(not removal_blocker)
                self.policy_delete_button.setToolTip(
                    removal_blocker
                    or "Permanently remove this inactive unqualified registration and its exact "
                    "completed campaign directory after confirmation."
                )
            self.path.setText(str(completed_training.get("campaign_id", "")))
            self._update_qualification_controls()
            return
        policy = self._selected_policy()
        self.policy_import_button.setText(
            "Imported" if policy is not None else "Import policy"
        )
        self.policy_import_button.setEnabled(policy is None)
        eligible = bool(policy and policy_record_user_status(policy) == "Eligible to select")
        self.policy_activate_button.setText(
            "Active governing policy"
            if policy is not None and policy.active
            else ("Activate for experiments" if eligible else "Qualification required")
        )
        self.policy_activate_button.setEnabled(eligible)
        self.policy_archive_button.setEnabled(policy is not None and not policy.active)
        removal_blocker = "Select a model to delete."
        if policy is not None:
            try:
                removal_blocker = (
                    self.state.policy_registry.unqualified_candidate_removal_blocker(policy.id)
                )
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                removal_blocker = str(exc) or "The selected policy cannot be removed."
        self.policy_delete_button.setEnabled(policy is not None and not removal_blocker)
        self.policy_delete_button.setToolTip(
            removal_blocker
            or "Permanently remove this exact inactive, unqualified, unreferenced model file "
            "and its registry entry after confirmation."
        )
        self.policy_archive_button.setText(
            "Restore archived" if policy is not None and policy.archived else "Archive"
        )
        if policy is not None:
            self.path.setText(policy.name)
        self._update_qualification_controls()

    def _qualification_candidate_blocker(self, policy) -> str:
        if policy is None:
            return "Import and select a policy first."
        if policy.active:
            return "The active governing policy cannot enter a new qualification workflow."
        if policy.archived:
            return "Restore the archived policy first."
        if policy.qualification_status == "qualified":
            return "Passed formal evidence is already admitted for this policy."
        if not policy.usable:
            return "The selected policy model file is unavailable."
        if not policy.compatible_with(TSH_CALO_ALGORITHM_ID):
            return "The selected policy is not compatible with the frozen TSH-CALO architecture."
        if int(policy.metadata.get("ensemble_size", 1)) < 2:
            return "Formal qualification requires an epistemic ensemble policy."
        return ""

    def _update_qualification_controls(self) -> None:
        policy = self._selected_policy()
        blocker = self._qualification_candidate_blocker(policy)
        process_running = self._qualification_process is not None
        self.qualification_button.setEnabled(policy is not None and not process_running)
        self.qualification_button.setToolTip(
            (f"Qualification unavailable: {blocker}" if blocker else "")
            or "Freeze the candidate architecture and training-parameter contract, start or "
            "exactly resume retained quality cells, then automatically admit verified passing "
            "evidence or retain the rejection. Activation always remains explicit."
        )
        has_qualified_policy = any(
            item.qualification_status == "qualified"
            for item in self.state.policy_registry.list(include_archived=False)
        )
        self.qualification_compare_button.setEnabled(has_qualified_policy and not process_running)
        self.qualification_compare_button.setToolTip(
            "Compare only policies whose retained formal evidence can be integrity-verified."
        )
        if policy is not None and blocker and not process_running:
            self.qualification_workflow_status.setText(
                f"Formal workflow blocked for {policy.name}: {blocker}"
            )

    def _automatic_qualification_base_directory(self) -> Path:
        if self.model_library is not None:
            return Path(self.model_library.default_directory).resolve().parent / (
                "policy-qualification"
            )
        local_data = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppLocalDataLocation
        )
        return Path(local_data).expanduser().resolve() / "policy-qualification"

    def _retained_qualification_resume(
        self, qualification_base: Path, policy, candidate_artifact
    ) -> tuple[
        TSHCALOQualificationPlan,
        AutomaticQualificationWorkspace,
        AutomaticQualificationSourceSnapshot,
    ] | None:
        """Find the newest valid retained plan or result for this exact candidate checksum."""

        campaigns = qualification_base / "campaigns"
        if not campaigns.is_dir():
            return None
        prefix = f"architecture-v2-{policy.sha256[:16].lower()}-"
        retained: list[tuple[int, Path, TSHCALOQualificationPlan]] = []
        for root in campaigns.glob(f"{prefix}*"):
            plan_path = root / "formal_qualification_plan.json"
            output = root / "formal-qualification-evidence"
            if (
                not root.is_dir()
                or not plan_path.is_file()
                or not output.is_dir()
                or (output / "campaign_integrity_failure.json").is_file()
            ):
                continue
            try:
                plan = load_qualification_plan(plan_path)
            except (OSError, RuntimeError, ValueError):
                continue
            if (
                plan.candidate_sha256.lower() != policy.sha256.lower()
                or Path(plan.candidate_path).expanduser().resolve()
                != Path(policy.checkpoint_path).expanduser().resolve()
            ):
                continue
            status_rank = 0
            if (output / "qualification_evidence.json").is_file():
                status_rank = 3
            status_path = output / QUALIFICATION_STATUS_FILE
            if status_path.is_file():
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    if status.get("state") == "paused":
                        status_rank = 4
                    elif status_rank < 3:
                        status_rank = 2 if status.get("state") == "running" else 1
                except (OSError, json.JSONDecodeError):
                    status_rank = max(status_rank, 0)
            retained.append((status_rank, root, plan))
        if not retained:
            return None
        _rank, root, stored_plan = max(
            retained,
            key=lambda item: (item[0], item[1].stat().st_mtime_ns),
        )
        if stored_plan.candidate_contract != qualification_candidate_contract(
            candidate_artifact
        ):
            raise ValueError(
                "The retained qualification plan no longer matches the candidate architecture "
                "and parameter contract"
            )
        snapshot_root = (
            qualification_base / "source-snapshots" / stored_plan.source_commit
        ).resolve()
        validate_repository_for_plan(stored_plan, root=snapshot_root)
        manifest_path = snapshot_root / AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Retained qualification source manifest is unreadable") from exc
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
            raise ValueError("Retained qualification source manifest is incompatible")
        workspace = AutomaticQualificationWorkspace.create(
            campaigns,
            candidate_sha256=policy.sha256,
            source_commit=stored_plan.source_commit,
        )
        if workspace.root != root.resolve():
            raise ValueError("Retained qualification workspace identity is inconsistent")
        snapshot = AutomaticQualificationSourceSnapshot(
            root=snapshot_root,
            source_commit=stored_plan.source_commit,
            worktree_sha256=str(manifest.get("worktree_sha256", "")),
            manifest_sha256=checkpoint_sha256(manifest_path),
            file_count=len(manifest["files"]),
        )
        return stored_plan, workspace, snapshot

    def qualify_selected_policy(self) -> None:
        """Run the one-action frozen qualification workflow; never activate the policy."""

        policy = self._selected_policy()
        if policy is None:
            return
        blocker = self._qualification_candidate_blocker(policy)
        if blocker:
            message = (
                f"The selected policy cannot enter formal qualification:\n\n{blocker}\n\n"
                "No plan, qualification evidence, registry state, or model file was changed."
            )
            self.qualification_workflow_status.setText(
                f"Formal workflow blocked for {policy.name}: {blocker}"
            )
            self.activity_message.emit("WARNING", self.qualification_workflow_status.text())
            QMessageBox.information(self, "Qualification unavailable", message)
            return
        if self._qualification_process is not None:
            return
        if self.state.task_status.busy:
            QMessageBox.information(
                self,
                "Another task is active",
                "Finish or safely pause the active task before starting formal qualification.",
            )
            return
        try:
            candidate_artifact = self.state.policy_registry.inspect_qualification_candidate(
                policy.id
            )
            qualification_base = self._automatic_qualification_base_directory()
            retained = self._retained_qualification_resume(
                qualification_base, policy, candidate_artifact
            )
            if retained is not None:
                qualification_plan, workspace, source_snapshot = retained
            else:
                source_snapshot = prepare_automatic_source_snapshot(
                    QUALIFICATION_SOURCE_ROOT,
                    qualification_base / "source-snapshots",
                )
                qualification_plan = build_automatic_formal_qualification_plan(
                    candidate_path=policy.checkpoint_path,
                    candidate_sha256=policy.sha256,
                    source_commit=source_snapshot.source_commit,
                    candidate_artifact=candidate_artifact,
                )
                workspace = AutomaticQualificationWorkspace.create(
                    qualification_base / "campaigns",
                    candidate_sha256=policy.sha256,
                    source_commit=source_snapshot.source_commit,
                )
            if workspace.qualification_plan.is_file():
                stored = load_qualification_plan(workspace.qualification_plan)
                if stored.execution_plan_sha256() != qualification_plan.execution_plan_sha256():
                    raise ValueError(
                        "The frozen architecture and quality plan changed; reuse is forbidden"
                    )
        except Exception as exc:
            show_error(
                self,
                "Policy was not admitted",
                "Automatic preflight rejected the candidate before any evaluation started. "
                "Correct the reported candidate, source, or architecture condition and try again.",
                exc,
                source="automatic qualification preflight",
            )
            return

        workload = automatic_qualification_workload()
        if (workspace.qualification_output / "qualification_evidence.json").is_file():
            action = "Verify and admit the retained formal result"
        elif workspace.qualification_output.exists():
            action = "Resume the retained formal qualification cells"
        else:
            action = "Start the frozen architecture and model-quality qualification"
        answer = QMessageBox.warning(
            self,
            "Qualify policy",
            f"{action} for {policy.name!r}?\n\n"
            f"Frozen design: {workload['cases']} qualification cases (case30 and case57), "
            f"{workload['runs_per_case']} paired runs per case, "
            f"{workload['qualification_cells']} paired optimizer cells. Every optimizer cell has "
            f"exactly {workload['evaluations_per_cell']} evaluations.\n\n"
            "The frozen candidate contract verifies the policy architecture, state/action and "
            "training schemas, ensemble membership, feature contract, training-design identity, "
            "and exact model checksum. Product version labels and project lifecycle labels are "
            "not qualification gates.\n\n"
            f"Source snapshot: {source_snapshot.source_commit[:12]} from "
            f"{source_snapshot.file_count} non-ignored files. The working source tree is not "
            "modified.\n\n"
            "Activity records a micro step every 500 evaluations. Pause safely commits the current "
            "optimizer state, so even a partial cell can continue later. Pause/resume has no count "
            "limit and never changes this finite budget. A failed frozen gate rejects the policy. "
            "A verified pass is admitted automatically and enables the separate Activate for "
            "experiments button; this action never activates or binds the policy.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            workflow_payload = automatic_qualification_workflow_payload(
                qualification_plan=qualification_plan
            )
            workflow_payload["source_snapshot"] = {
                "source_commit": source_snapshot.source_commit,
                "worktree_sha256": source_snapshot.worktree_sha256,
                "manifest_sha256": source_snapshot.manifest_sha256,
                "file_count": source_snapshot.file_count,
                "manifest_path": str(source_snapshot.root / AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST),
            }
            freeze_plan(
                workspace.workflow_plan,
                workflow_payload,
            )
            freeze_plan(workspace.qualification_plan, qualification_plan.to_dict())
            self._qualification_policy_id = policy.id
            self._qualification_workspace = workspace
            self._qualification_source_snapshot = source_snapshot
            self._continue_automatic_qualification(policy, qualification_plan, workspace)
        except AutomaticQualificationRejected as exc:
            self._record_qualification_rejection(policy.name, str(exc))
        except Exception as exc:
            show_error(
                self,
                "Policy was not admitted",
                "The automatic workflow stopped without changing policy activation or experiment "
                "settings. Retained exact cells remain available when safe to resume.",
                exc,
                source="automatic qualification",
            )
            self._update_qualification_controls()

    def _continue_automatic_qualification(
        self, policy, qualification_plan, workspace: AutomaticQualificationWorkspace
    ) -> None:
        if (workspace.qualification_output / "campaign_integrity_failure.json").is_file():
            raise AutomaticQualificationRejected(
                "Policy rejected: the retained formal campaign has a terminal integrity failure"
            )
        freeze_plan(workspace.qualification_plan, qualification_plan.to_dict())
        source_snapshot = self._qualification_source_snapshot
        if source_snapshot is None:
            raise RuntimeError("Automatic qualification source snapshot identity was lost")
        validate_repository_for_plan(qualification_plan, root=source_snapshot.root)
        resume = workspace.qualification_output.exists()
        if resume:
            stored = load_qualification_plan(
                workspace.qualification_output / "qualification_plan.json"
            )
            if stored.execution_plan_sha256() != qualification_plan.execution_plan_sha256():
                raise ValueError("Retained qualification cells belong to another frozen plan")
        if (workspace.qualification_output / "qualification_evidence.json").is_file():
            self._admit_automatic_qualification(policy, workspace)
            return
        self._start_qualification_process(
            stage="formal",
            module="calo_rpd_studio.scripts.qualify_tsh_calo",
            plan_path=workspace.qualification_plan,
            output=workspace.qualification_output,
            resume=resume,
        )

    def _start_qualification_process(
        self,
        *,
        stage: str,
        module: str,
        plan_path: Path,
        output: Path,
        resume: bool,
    ) -> None:
        source_snapshot = self._qualification_source_snapshot
        if source_snapshot is None:
            raise RuntimeError("Automatic qualification source snapshot identity was lost")
        arguments = ["-m", module, str(plan_path), "--output", str(output)]
        if resume:
            arguments.append("--resume")
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setWorkingDirectory(str(source_snapshot.root))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONDONTWRITEBYTECODE", "1")
        environment.insert("MPLCONFIGDIR", str(output.parent / "runtime-cache" / "matplotlib"))
        process.setProcessEnvironment(environment)
        process.setArguments(arguments)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_qualification_output)
        process.errorOccurred.connect(self._qualification_process_error)
        process.finished.connect(self._qualification_process_finished)
        self._qualification_process = process
        self._qualification_process_output = ""
        self._qualification_stdout_buffer = ""
        self._qualification_process_stage = stage
        self._qualification_expected_cells = automatic_qualification_workload()[
            "qualification_cells"
        ]
        self._qualification_reported_cells = -1
        self._qualification_pause_requested = False
        self._qualification_live_event = {}
        self._qualification_last_event = {}
        self.state.task_status.begin(
            "Independent policy qualification",
            detail="Preparing frozen checks and resumable optimizer checkpoints",
            progress=0,
            cancellable=True,
        )
        self.qualification_workflow_status.setText(
            f"Qualification {'resuming' if resume else 'running'}. Live micro-progress and durable "
            "cell counts are shown in the bottom bar and Activity. Use Pause safely to stop at "
            "an authenticated optimizer checkpoint."
        )
        self.activity_message.emit("INFO", self.qualification_workflow_status.text())
        self._update_qualification_controls()
        process.start()
        if self._qualification_process is process:
            self._update_qualification_progress()
            self._qualification_progress_timer.start()

    def _update_qualification_progress(self) -> None:
        """Publish live work separately from authoritative committed-cell progress."""

        if self._qualification_process is None or self._qualification_workspace is None:
            return
        output = self._qualification_workspace.qualification_output
        completed = sum(
            1
            for directory in (output / "records", output / "failures")
            if directory.is_dir()
            for _path in directory.glob("*.json")
        )
        expected = max(1, int(self._qualification_expected_cells))
        completed = min(completed, expected)
        evaluations_per_cell = int(
            automatic_qualification_workload()["evaluations_per_cell"]
        )
        live = dict(self._qualification_live_event or {})
        live_evaluations = 0
        live_base = int(live.get("committed_cells", -1))
        if live.get("event") == "cell_progress" and completed == live_base:
            live_evaluations = max(
                0,
                min(evaluations_per_cell, int(live.get("live_evaluations", 0))),
            )
        total_evaluations = expected * evaluations_per_cell
        observed_evaluations = completed * evaluations_per_cell + live_evaluations
        overall_percentage = 100.0 * observed_evaluations / max(total_evaluations, 1)
        percentage = min(99, int(overall_percentage))
        if live_evaluations:
            detail = (
                f"Live {overall_percentage:.1f}% | cell {int(live.get('cell_index', 0))}/"
                f"{expected} {str(live.get('label', ''))} "
                f"{live_evaluations}/{evaluations_per_cell} | {completed} cells durable"
            )
        else:
            detail = f"Model-quality checks | {completed}/{expected} cells durable"
        self.state.task_status.update(progress=percentage, detail=detail)
        if completed != self._qualification_reported_cells:
            self._qualification_reported_cells = completed
            self.activity_message.emit(
                "INFO",
                f"Qualification durable progress {percentage}% | {completed}/{expected} cells",
            )

    def request_safe_qualification_pause(self) -> None:
        """Request a cooperative pause; the child acknowledges only durable state."""

        if self._qualification_process is None or self._qualification_pause_requested:
            return
        workspace = self._qualification_workspace
        if workspace is None:
            self.state.task_status.rearm_cancel(
                "Safe pause was not recorded | qualification is still running"
            )
            return
        try:
            request = request_tsh_calo_qualification_pause(
                workspace.qualification_output
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self.activity_message.emit(
                "ERROR",
                f"Safe qualification pause could not be requested: {type(exc).__name__}: {exc}",
            )
            self.state.task_status.rearm_cancel(
                "Safe pause was not recorded | qualification is still running"
            )
            return
        self._qualification_pause_requested = True
        detail = "Pause requested | committing and authenticating the current optimizer state"
        self.state.task_status.update(detail=detail)
        self.qualification_workflow_status.setText(detail)
        self.activity_message.emit(
            "INFO",
            "Safe qualification pause request recorded; the current population transition will "
            f"finish before checkpoint acknowledgement ({str(request.get('request_id', ''))[:12]}).",
        )

    def _read_qualification_output(self) -> None:
        process = self._qualification_process
        if process is None:
            return
        output = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not output:
            return
        self._qualification_process_output += output
        self._qualification_stdout_buffer += output
        while "\n" in self._qualification_stdout_buffer:
            line, self._qualification_stdout_buffer = self._qualification_stdout_buffer.split(
                "\n", 1
            )
            self._consume_qualification_output_line(line.rstrip("\r"))

    def _flush_qualification_output(self) -> None:
        if self._qualification_stdout_buffer:
            self._consume_qualification_output_line(
                self._qualification_stdout_buffer.rstrip("\r")
            )
            self._qualification_stdout_buffer = ""

    def _consume_qualification_output_line(self, line: str) -> None:
        if not line:
            return
        if not line.startswith(QUALIFICATION_EVENT_PREFIX):
            self.activity_message.emit("DEBUG", line)
            return
        try:
            event = json.loads(line[len(QUALIFICATION_EVENT_PREFIX) :])
        except json.JSONDecodeError:
            self.activity_message.emit("WARNING", "A qualification progress event was unreadable.")
            return
        if (
            not isinstance(event, dict)
            or event.get("schema_version") != TSH_CALO_QUALIFICATION_EVENT_SCHEMA
        ):
            self.activity_message.emit("WARNING", "A qualification progress event was incompatible.")
            return
        self._qualification_last_event = dict(event)
        self._apply_qualification_event(event)

    def _apply_qualification_event(self, event: dict) -> None:
        name = str(event.get("event", ""))
        if name in {"campaign_started", "campaign_resumed"}:
            detail = (
                f"Qualification {'resumed' if name == 'campaign_resumed' else 'started'} | "
                f"{int(event.get('completed_cells', 0))}/{int(event.get('total_cells', 0))} "
                "cells durable"
            )
            self.state.task_status.update(detail=detail)
            self.activity_message.emit("INFO", detail)
            return
        if name == "calibration_started":
            detail = "OOD calibration running | optimizer cells have not started"
            self.state.task_status.update(detail=detail)
            self.activity_message.emit("INFO", detail)
            return
        if name == "calibration_progress":
            detail = (
                f"OOD calibration | sample {int(event.get('completed_samples', 0))}/"
                f"{int(event.get('total_samples', 0))} | {event.get('case')}"
            )
            self.state.task_status.update(detail=detail)
            self.activity_message.emit("INFO", detail)
            return
        if name == "calibration_completed":
            self.activity_message.emit("INFO", "OOD calibration committed and checksum verified.")
            return
        if name == "cell_started":
            self._qualification_live_event = {}
            detail = (
                f"Cell {int(event.get('cell_index', 0))}/{int(event.get('total_cells', 0))} | "
                f"{event.get('case')} run {int(event.get('run_number', 0))} "
                f"{event.get('label')} | "
                f"{'restoring exact checkpoint' if event.get('resumed_checkpoint') else 'starting'}"
            )
            self.state.task_status.update(detail=detail)
            self.activity_message.emit("INFO", detail)
            return
        if name == "cell_progress":
            self._qualification_live_event = dict(event)
            self._update_qualification_progress()
            feasible = event.get("best_feasible_objective")
            feasible_text = "none yet" if feasible is None else f"{float(feasible):.6g}"
            violation = event.get("best_constraint_violation")
            violation_text = "n/a" if violation is None else f"{float(violation):.6g}"
            eta = event.get("cell_eta_seconds")
            eta_text = "n/a" if eta is None else f"{float(eta):.0f}s"
            rate = event.get("evaluations_per_second")
            rate_text = "n/a" if rate is None else f"{float(rate):.2f} eval/s"
            first_feasible = event.get("first_feasible_evaluation")
            first_feasible_text = (
                "not reached" if first_feasible is None else str(int(first_feasible))
            )
            self.activity_message.emit(
                "INFO",
                f"Micro step | cell {int(event.get('cell_index', 0))}/"
                f"{int(event.get('total_cells', 0))} {event.get('case')} "
                f"run {int(event.get('run_number', 0))} {event.get('label')} | "
                f"{int(event.get('live_evaluations', 0))}/"
                f"{int(event.get('max_evaluations', 0))} evaluations "
                f"({float(event.get('cell_percent', 0.0)):.1f}%) | best feasible "
                f"{feasible_text} | violation {violation_text} | first feasible FE "
                f"{first_feasible_text} | {rate_text} | cell ETA {eta_text} | live, "
                "not yet a committed cell",
            )
            return
        if name in {"cell_completed", "cell_failed"}:
            self._qualification_live_event = {}
            self._update_qualification_progress()
            severity = "INFO" if name == "cell_completed" else "ERROR"
            self.activity_message.emit(
                severity,
                f"Cell {int(event.get('cell_index', 0))}/"
                f"{int(event.get('total_cells', 0))} "
                f"{'committed' if name == 'cell_completed' else 'retained as failed'} | "
                f"{int(event.get('committed_cells', 0))} cells durable",
            )
            return
        if name == "campaign_paused":
            self._qualification_live_event = {}
            detail = (
                f"Safe pause acknowledged | {int(event.get('completed_cells', 0))}/"
                f"{int(event.get('total_cells', 0))} cells durable | "
                f"boundary {event.get('boundary')}"
            )
            self.state.task_status.update(detail=detail)
            self.activity_message.emit("INFO", detail)
            return
        if name == "campaign_completed":
            self._qualification_live_event = {}
            self.state.task_status.update(progress=100, detail="Qualification evidence committed")
            self.activity_message.emit("INFO", "Qualification evidence and final decision committed.")

    def _qualification_process_error(self, error) -> None:
        process = self._qualification_process
        if process is None or error != QProcess.ProcessError.FailedToStart:
            return
        message = process.errorString()
        self._qualification_progress_timer.stop()
        self._qualification_process = None
        self._qualification_pause_requested = False
        stage = self._qualification_process_stage or "automatic"
        self.state.task_status.fail(f"Qualification {stage} process could not start")
        self.qualification_workflow_status.setText(
            f"Qualification {stage} process could not start; no evidence was admitted or activated."
        )
        self.activity_message.emit("ERROR", message)
        self._update_qualification_controls()
        show_error(
            self,
            "Qualification process could not start",
            "No evidence was admitted and no policy state changed.",
            message,
            source="formal qualification process",
        )
        process.deleteLater()

    def _qualification_process_finished(self, exit_code: int, _exit_status) -> None:
        process = self._qualification_process
        if process is None:
            return
        self._read_qualification_output()
        self._flush_qualification_output()
        self._qualification_progress_timer.stop()
        self._update_qualification_progress()
        process.deleteLater()
        self._qualification_process = None
        stage = self._qualification_process_stage
        policy_id = self._qualification_policy_id
        workspace = self._qualification_workspace
        paused = (
            int(exit_code) == TSH_CALO_QUALIFICATION_PAUSE_EXIT_CODE
            and self._confirmed_safe_qualification_pause()
        )
        if paused:
            self._qualification_pause_requested = False
            completed = int(self._qualification_last_event.get("completed_cells", 0))
            expected = max(1, int(self._qualification_expected_cells))
            detail = (
                f"Qualification paused safely | {completed}/{expected} cells durable | "
                "click Qualify policy to resume the exact frozen plan"
            )
            self.state.task_status.paused(detail)
            self.qualification_workflow_status.setText(detail)
            self.activity_message.emit(
                "INFO",
                "Qualification paused at an authenticated boundary. No policy evidence was "
                "admitted or activated; the same finite plan remains resumable.",
            )
            self._update_qualification_controls()
            return
        if exit_code == 0:
            try:
                if not policy_id or workspace is None:
                    raise RuntimeError("Automatic qualification process identity was lost")
                policy = self.state.policy_registry.get(policy_id)
                self._admit_automatic_qualification(policy, workspace)
                return
            except AutomaticQualificationRejected as exc:
                self._record_qualification_rejection(
                    getattr(locals().get("policy"), "name", policy_id), str(exc)
                )
            except Exception as exc:
                self.state.task_status.fail("Qualification evidence verification failed")
                show_error(
                    self,
                    "Policy was not admitted",
                    "The completed process did not provide admissible evidence. The policy remains "
                    "inactive and experiment settings were not changed.",
                    exc,
                    source="automatic qualification verification",
                )
        else:
            self.state.task_status.fail(
                f"Qualification {stage} stopped; retained completed cells can resume on next click"
            )
            self.qualification_workflow_status.setText(
                f"Qualification {stage} stopped with code {exit_code}. No evidence was admitted or "
                "activated. Click Qualify policy again to resume the exact retained plan."
            )
            self.activity_message.emit("ERROR", self.qualification_workflow_status.text())
        self._qualification_pause_requested = False
        self._update_qualification_controls()

    def _confirmed_safe_qualification_pause(self) -> bool:
        workspace = self._qualification_workspace
        if workspace is None:
            return False
        output = workspace.qualification_output
        try:
            status = json.loads((output / QUALIFICATION_STATUS_FILE).read_text("utf-8"))
            control = json.loads((output / QUALIFICATION_CONTROL_FILE).read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(status, dict) or not isinstance(control, dict):
            return False
        pause = status.get("pause", {})
        last_event = status.get("last_event", {})
        if not isinstance(pause, dict) or not isinstance(last_event, dict):
            return False
        durable_path = Path(str(control.get("durable_path", ""))).expanduser()
        try:
            durable_digest = (
                checkpoint_sha256(durable_path.resolve())
                if durable_path.resolve().is_file()
                else ""
            )
        except OSError:
            durable_digest = ""
        return bool(
            status.get("schema_version") == TSH_CALO_QUALIFICATION_STATUS_SCHEMA
            and control.get("schema_version") == TSH_CALO_QUALIFICATION_CONTROL_SCHEMA
            and last_event.get("schema_version") == TSH_CALO_QUALIFICATION_EVENT_SCHEMA
            and status.get("state") == "paused"
            and pause.get("reason") == "user_requested_safe_pause"
            and pause.get("resumable") is True
            and control.get("action") == "pause"
            and control.get("state") == "acknowledged"
            and last_event.get("event") == "campaign_paused"
            and pause.get("request_id") == control.get("request_id")
            and last_event.get("request_id") == control.get("request_id")
            and control.get("qualification_plan_sha256")
            == status.get("qualification_plan_sha256")
            and last_event.get("qualification_plan_sha256")
            == status.get("qualification_plan_sha256")
            and pause.get("durable_path") == control.get("durable_path")
            and pause.get("durable_sha256") == control.get("durable_sha256")
            and last_event.get("durable_sha256") == control.get("durable_sha256")
            and durable_digest == control.get("durable_sha256")
        )

    def _admit_automatic_qualification(
        self, policy, workspace: AutomaticQualificationWorkspace
    ) -> None:
        try:
            verified = self.state.policy_registry.inspect_qualification_evidence(
                policy.id, workspace.qualification_output
            )
        except Exception as exc:
            raise AutomaticQualificationRejected(
                f"Policy rejected: completed formal evidence did not pass every frozen gate ({exc})"
            ) from exc
        try:
            admitted = self.state.policy_registry.admit_qualification_evidence(
                policy.id, workspace.qualification_output
            )
        except Exception as exc:
            raise RuntimeError("Verified qualification evidence could not be admitted") from exc
        summary = verified.metrics["summary"]
        self.state.task_status.finish("Policy qualification passed and evidence was admitted")
        qualified_status = (
            f"Qualified: {admitted.name} earned grade {verified.grade}. Worst-case feasible "
            f"probability {summary['minimum_candidate_feasible_probability']:.1%}, median "
            f"objective improvement "
            f"{summary['minimum_relative_objective_improvement']:.2%}, and objective win rate "
            f"{summary['minimum_objective_win_rate']:.1%}. Activation remains explicit."
        )
        self.refresh_policy_library()
        self._select_policy_id(admitted.id)
        self.qualification_workflow_status.setText(qualified_status)
        self.activity_message.emit(
            "INFO",
            f"Passed formal evidence admitted for {admitted.name}; Activate for experiments is now "
            "available and the policy remains inactive.",
        )
        QMessageBox.information(
            self,
            "Policy qualified",
            f"{admitted.name!r} passed the frozen formal plan and its verified evidence was "
            "admitted. The policy is still inactive. Review the comparison if needed, then use "
            "Activate for experiments explicitly.",
        )

    def _record_qualification_rejection(self, policy_name: str, reason: str) -> None:
        self.state.task_status.finish("Policy rejected by the frozen qualification workflow")
        self.qualification_workflow_status.setText(f"{reason}. Activation remains disabled.")
        self.activity_message.emit("WARNING", self.qualification_workflow_status.text())
        self._update_qualification_controls()
        QMessageBox.information(
            self,
            "Policy not qualified",
            f"{policy_name!r} was not admitted.\n\n{reason}\n\nThe retained evidence remains "
            "immutable, and activation stays disabled.",
        )

    def compare_qualified_policies(self) -> None:
        summaries = self.state.policy_registry.qualification_evidence_summaries()
        if not summaries:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Compare qualified policy evidence")
        dialog.resize(1420, 540)
        layout = QVBoxLayout(dialog)
        guidance = QLabel(
            "Compare policies only when the Evidence design values match. Prefer a policy only when "
            "it has stronger conservative feasibility, objective improvement, win-rate, effect-size, "
            "and anytime-safety evidence. A trade-off requires scientist judgment; training duration "
            "or software version alone is not evidence of a better policy."
        )
        guidance.setWordWrap(True)
        layout.addWidget(guidance)
        table = QTableWidget(len(summaries), 11)
        table.setHorizontalHeaderLabels(
            (
                "Policy",
                "Evidence design",
                "Feasible floor",
                "Feasibility CI floor",
                "Objective improvement",
                "Win rate",
                "Effect size",
                "Anytime feasibility",
                "Anytime objective",
                "Holm p (max)",
                "Selection guidance",
            )
        )
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setWordWrap(True)
        for row, item in enumerate(summaries):
            summary = dict(item.get("summary", {}) or {})
            protocol = str(item.get("comparison_protocol_sha256", ""))
            evidence_design = (
                f"{protocol[:8]} | {len(item.get('development_cases', []))} cases | "
                f"{item.get('runs_per_case', 0)} runs | {item.get('max_evaluations', 0)} FE"
                if protocol
                else "Not verifiable"
            )
            values = (
                f"{'Active - ' if item.get('active') else ''}{item['policy_name']}",
                evidence_design,
                self._percent_or_dash(summary.get("minimum_candidate_feasible_probability")),
                self._percent_or_dash(summary.get("minimum_feasibility_ci_lower")),
                self._percent_or_dash(summary.get("minimum_relative_objective_improvement")),
                self._percent_or_dash(summary.get("minimum_objective_win_rate")),
                self._number_or_dash(summary.get("minimum_rank_biserial")),
                self._percent_or_dash(
                    summary.get("minimum_anytime_feasibility_difference")
                ),
                self._percent_or_dash(
                    summary.get("minimum_anytime_objective_improvement")
                ),
                self._number_or_dash(summary.get("maximum_holm_p")),
                str(item.get("recommendation", "Scientist review required")),
            )
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))
        table.resizeRowsToContents()
        for column in range(10):
            table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        table.horizontalHeader().setSectionResizeMode(10, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)
        note = QLabel(
            "'Strongest comparable evidence' is shown only when one policy is no worse on every "
            "listed conservative measure and better on at least one, within the same frozen evidence "
            "design. Activation is still a separate explicit action in the Policy library."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        close_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_buttons.rejected.connect(dialog.reject)
        layout.addWidget(close_buttons)
        dialog.exec()

    @staticmethod
    def _percent_or_dash(value) -> str:
        try:
            return f"{float(value):.2%}"
        except (TypeError, ValueError):
            return "Not available"

    @staticmethod
    def _number_or_dash(value) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "Not available"

    def _registered_completed_removal_blocker(self, completed_training: dict, policy) -> str:
        if self.model_library is None:
            return "The completed-model library is unavailable."
        try:
            directory = Path(str(completed_training.get("directory", ""))).expanduser().resolve()
            checkpoint = Path(policy.checkpoint_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return "The selected campaign or registered checkpoint path is invalid."
        if directory not in checkpoint.parents:
            return "The registered checkpoint is not contained by this completed campaign."
        try:
            return self.state.policy_registry.unqualified_candidate_removal_blocker(policy.id)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            return str(exc) or "The selected policy cannot be removed."

    def _update_policy_gate_state(self) -> None:
        records = self.state.policy_registry.list(include_archived=True)
        status = self.state.governing_policy_status()
        active = next(
            (policy for policy in records if status.ready and policy.id == status.policy_id),
            None,
        )
        self.new_training_button.setEnabled(
            not bool(getattr(self.state, "policy_training_active", False))
        )
        if active is None:
            self.policy_gate_status.setText(
                "No governing TSH-CALO policy is active. Rule-only CALO remains available."
            )
            self.apply_policy_button.setEnabled(False)
            if self._selected_row() is None:
                self.path.clear()
            return
        self.policy_gate_status.setText(
            f"Selected: {active.name} · grade {active.grade} · Ready for experiments"
        )
        self.path.setText(active.name)
        self.apply_policy_button.setEnabled(True)

    def import_policy(self) -> None:
        completed_training = self._selected_completed_training()
        if completed_training is not None:
            path = str(completed_training.get("policy_candidate", ""))
            policy_name = str(completed_training.get("campaign_id", "")) or None
        else:
            path, _filter = QFileDialog.getOpenFileName(
                self, "Import policy", "", "Policy checkpoint (*.pt)"
            )
            policy_name = None
            if not path:
                return
        try:
            policy = self.state.policy_registry.register(path, name=policy_name)
        except Exception as exc:
            show_error(
                self,
                "Policy could not be imported",
                "The selected file is not a compatible policy checkpoint.",
                exc,
                source="policy import",
            )
            return
        self.refresh_policy_library()
        self._select_policy_id(policy.id)

    def activate_selected_policy(self) -> None:
        policy = self._selected_policy()
        if policy is None:
            return
        try:
            activated = self.state.policy_registry.activate(
                policy.id, algorithm_id=TSH_CALO_ALGORITHM_ID
            )
        except Exception as exc:
            show_error(
                self,
                "Policy could not be selected",
                "Select a verified, compatible TSH-CALO policy.",
                exc,
                source="policy activation",
            )
            return
        self.refresh_policy_library()
        self._select_policy_id(activated.id)

    def archive_selected_policy(self) -> None:
        policy = self._selected_policy()
        if policy is None:
            return
        try:
            if policy.archived:
                self.state.policy_registry.unarchive(policy.id)
            else:
                self.state.policy_registry.archive(policy.id)
        except Exception as exc:
            show_error(
                self,
                "Policy status could not be changed",
                "The selected policy could not be archived or restored.",
                exc,
                source="policy archive",
            )
            return
        self.refresh_policy_library()

    def delete_selected_model_files(self) -> None:
        completed_training = self._selected_completed_training()
        if completed_training is not None:
            self._delete_completed_training(
                completed_training,
                registered_policy=self._selected_policy(),
            )
            return
        policy = self._selected_policy()
        if policy is not None:
            self._delete_standalone_policy_file(policy)

    def _delete_standalone_policy_file(self, policy) -> None:
        source = Path(policy.checkpoint_path).expanduser()
        try:
            if source.is_symlink():
                raise ValueError("Symbolic-link model targets cannot be deleted from the library.")
            checkpoint = source.resolve(strict=True)
            if not checkpoint.is_file():
                raise ValueError("The selected model target is not a regular file.")
            blocker = self.state.policy_registry.unqualified_candidate_removal_blocker(policy.id)
            if blocker:
                raise PermissionError(blocker)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            show_error(
                self,
                "Model could not be deleted",
                "The exact standalone model target could not be verified.",
                exc,
                source="standalone policy deletion preflight",
            )
            return
        answer = QMessageBox.warning(
            self,
            "Permanently delete model file",
            f"Delete the inactive, unqualified, unreferenced model {policy.name!r}?\n\n"
            f"{checkpoint}\n\n"
            "This permanently removes both the exact model file and its registry entry and "
            "cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        registration_removed = False
        try:
            self.state.policy_registry.remove_unqualified_candidate(
                policy.id,
                reason="user_deleted_standalone_model",
            )
            registration_removed = True
            if (
                self.state.policy_registry.inspect_checkpoint(checkpoint)["sha256"]
                != policy.sha256
            ):
                raise RuntimeError("Model checksum changed immediately before file deletion")
            checkpoint.unlink()
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            self.refresh_policy_library()
            show_error(
                self,
                "Model could not be deleted",
                (
                    "The registry entry was removed, but the exact model file remains at the "
                    "shown path and must be reviewed manually."
                    if registration_removed
                    else "No verified standalone-model deletion was accepted."
                ),
                exc,
                source="standalone policy deletion",
            )
            return
        self.refresh_policy_library()
        QMessageBox.information(
            self,
            "Model file deleted",
            f"The unqualified registry entry and exact model file were permanently deleted:\n"
            f"{checkpoint}",
        )

    def _delete_completed_training(
        self,
        completed_training: dict,
        *,
        registered_policy=None,
    ) -> None:
        if self.model_library is None:
            return
        directory_text = str(completed_training.get("directory", "")).strip()
        campaign_id = str(completed_training.get("campaign_id", "Completed model")).strip()
        if not directory_text:
            return
        try:
            directory = self.model_library.validate_completed_campaign_deletion(directory_text)
            for policy in self.state.policy_registry.list(include_archived=True):
                checkpoint = Path(policy.checkpoint_path).expanduser().resolve()
                if (
                    (checkpoint == directory or directory in checkpoint.parents)
                    and (registered_policy is None or policy.id != registered_policy.id)
                ):
                    raise ValueError(
                        "This completed campaign contains another registered policy. Model files "
                        "were not deleted."
                    )
            if registered_policy is not None:
                checkpoint = Path(registered_policy.checkpoint_path).expanduser().resolve()
                if directory not in checkpoint.parents:
                    raise ValueError(
                        "The selected registered policy is not contained by this campaign."
                    )
                blocker = self.state.policy_registry.unqualified_candidate_removal_blocker(
                    registered_policy.id
                )
                if blocker:
                    raise PermissionError(blocker)
        except (OSError, RuntimeError, ValueError) as exc:
            show_error(
                self,
                "Completed model could not be deleted",
                "The exact completed-campaign target could not be verified.",
                exc,
                source="completed training deletion preflight",
            )
            return
        confirmation_scope = (
            "This cannot be undone. Its exact inactive, unqualified, unreferenced registry "
            "entry will also be removed."
            if registered_policy is not None
            else "This cannot be undone. No registered or active policy will be changed."
        )
        answer = QMessageBox.warning(
            self,
            "Permanently delete completed model files",
            f"Delete completed campaign {campaign_id!r} and every checkpoint, log, extension, "
            f"and model file inside this exact directory?\n\n{directory}\n\n"
            + confirmation_scope,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        registration_removed = False
        try:
            if registered_policy is not None:
                self.state.policy_registry.remove_unqualified_candidate(registered_policy.id)
                registration_removed = True
            deleted = self.model_library.delete_completed_campaign(directory)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            self.refresh_policy_library()
            show_error(
                self,
                "Completed model could not be deleted",
                (
                    "The registry entry was removed, but the campaign files remain and can be "
                    "retried from the refreshed unregistered row."
                    if registration_removed
                    else "No verified completed-campaign deletion was accepted."
                ),
                exc,
                source="completed training deletion",
            )
            return
        QMessageBox.information(
            self,
            "Completed model files deleted",
            (
                "The unqualified registry entry and completed campaign directory were permanently "
                f"deleted:\n{deleted}"
                if registration_removed
                else f"The completed campaign directory was permanently deleted:\n{deleted}"
            ),
        )

    def apply_policy_configuration(self) -> None:
        status = self.state.governing_policy_status()
        active = next(
            (
                policy
                for policy in self.state.policy_registry.list()
                if status.ready and policy.id == status.policy_id
            ),
            None,
        )
        if active is None:
            return
        try:
            self.state.policy_registry.bind_to_experiment_config(
                active.id,
                self.state.config,
                deterministic=self.deterministic.isChecked(),
                algorithm_id=TSH_CALO_ALGORITHM_ID,
            )
        except Exception as exc:
            show_error(
                self,
                "Policy could not be applied",
                "The active policy is not compatible with the current experiment settings.",
                exc,
                source="governing policy application",
            )
            return
        if TSH_CALO_ALGORITHM_ID not in self.state.config.algorithms:
            self.state.config.algorithms.append(TSH_CALO_ALGORITHM_ID)
        self.state.update_config()
        self.stage_completed.emit()

    def load_from_config(self, _config) -> None:
        self._update_policy_gate_state()
