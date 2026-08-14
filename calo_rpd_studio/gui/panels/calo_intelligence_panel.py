"""Policy inventory and explicit TSH-CALO lifecycle controls."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, Qt, pyqtSignal
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

from calo_rpd_studio.algorithms.calo.policy_readiness import policy_record_user_status
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification_campaign import (
    TSHCALOQualificationPlan,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import TSH_CALO_ALGORITHM_ID
from calo_rpd_studio.gui.user_feedback import show_error
from calo_rpd_studio.gui.widgets.page_header import PageHeader
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.scripts.qualify_tsh_calo import load_plan, validate_repository_for_plan


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
        self._checked_qualification_plans: dict[str, tuple[str, TSHCALOQualificationPlan]] = {}
        self._qualification_process: QProcess | None = None
        self._qualification_process_output = ""
        self._qualification_output_directory = ""

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
        self.qualification_check_button = QPushButton("Check formal plan")
        self.qualification_run_button = QPushButton("Run / resume qualification")
        self.qualification_admit_button = QPushButton("Admit passed evidence")
        self.qualification_compare_button = QPushButton("Compare qualified policies")
        self.policy_activate_button = QPushButton("Activate for experiments")
        self.policy_archive_button = QPushButton("Archive")
        self.policy_delete_button = QPushButton("Delete model files")
        self.policy_refresh_button = QPushButton("Refresh")
        self.show_archived_policies = QCheckBox("Show archived")
        self.policy_import_button.clicked.connect(self.import_policy)
        self.qualification_check_button.clicked.connect(self.check_qualification_plan)
        self.qualification_run_button.clicked.connect(self.run_or_resume_qualification)
        self.qualification_admit_button.clicked.connect(self.admit_qualification_evidence)
        self.qualification_compare_button.clicked.connect(self.compare_qualified_policies)
        self.policy_activate_button.clicked.connect(self.activate_selected_policy)
        self.policy_archive_button.clicked.connect(self.archive_selected_policy)
        self.policy_delete_button.clicked.connect(self.delete_selected_model_files)
        self.policy_refresh_button.clicked.connect(self.refresh_policy_library)
        self.show_archived_policies.toggled.connect(lambda _checked: self.refresh_policy_library())
        for button in (
            self.policy_import_button,
            self.qualification_check_button,
            self.qualification_run_button,
            self.qualification_admit_button,
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
            "Workflow: import -> check formal plan -> run or resume -> admit passed evidence -> "
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
        if not policy.post_development_eligible:
            return "Only a new policy trained against the accepted development freeze is eligible."
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
        check_enabled = policy is not None and not blocker and not process_running
        checked = bool(policy and policy.id in self._checked_qualification_plans)
        self.qualification_check_button.setEnabled(check_enabled)
        self.qualification_check_button.setToolTip(
            blocker
            or "Select and validate a frozen formal qualification plan; no evaluation starts."
        )
        self.qualification_run_button.setEnabled(check_enabled and checked)
        self.qualification_run_button.setToolTip(
            blocker
            or (
                "Run or exactly resume the checked finite formal qualification plan."
                if checked
                else "Check a formal plan for this exact policy first."
            )
        )
        self.qualification_admit_button.setEnabled(check_enabled)
        self.qualification_admit_button.setToolTip(
            blocker
            or "Verify and explicitly admit a completed passed evidence directory; this does not "
            "activate the policy."
        )
        has_qualified_policy = any(
            item.qualification_status == "qualified"
            for item in self.state.policy_registry.list(include_archived=False)
        )
        self.qualification_compare_button.setEnabled(has_qualified_policy and not process_running)
        self.qualification_compare_button.setToolTip(
            "Compare only policies whose retained formal evidence can be integrity-verified."
        )

    def check_qualification_plan(self) -> None:
        policy = self._selected_policy()
        blocker = self._qualification_candidate_blocker(policy)
        if blocker:
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select frozen formal qualification plan",
            "",
            "Qualification plan (qualification_plan.json *.json)",
        )
        if not path:
            return
        try:
            plan = load_plan(path)
            if plan.mode != "formal":
                raise ValueError("A screening plan can never make a policy ready for experiments")
            if plan.candidate_sha256.lower() != policy.sha256.lower():
                raise ValueError("The selected plan belongs to a different policy checkpoint")
            validate_repository_for_plan(plan)
            inspected = self.state.policy_registry.inspect_checkpoint(policy.checkpoint_path)
            if inspected["sha256"] != policy.sha256:
                raise RuntimeError("The selected policy checkpoint changed after import")
        except Exception as exc:
            self._checked_qualification_plans.pop(policy.id, None)
            self._update_qualification_controls()
            show_error(
                self,
                "Formal plan is not ready",
                "No qualification was started. Correct the frozen plan or source identity and "
                "check again.",
                exc,
                source="qualification plan check",
            )
            return
        self._checked_qualification_plans[policy.id] = (str(Path(path).resolve()), plan)
        total_cells = len(plan.development_cases) * plan.runs * 2
        self.qualification_workflow_status.setText(
            f"Plan checked for {policy.name}: {len(plan.development_cases)} cases, {plan.runs} paired "
            f"runs per case, {total_cells} optimizer cells, {plan.max_evaluations} evaluations per cell. "
            "No qualification has started."
        )
        self.activity_message.emit("INFO", self.qualification_workflow_status.text())
        self._update_qualification_controls()

    def run_or_resume_qualification(self) -> None:
        policy = self._selected_policy()
        checked = self._checked_qualification_plans.get(getattr(policy, "id", ""))
        if policy is None or checked is None or self._qualification_process is not None:
            return
        if self.state.task_status.busy:
            QMessageBox.information(
                self,
                "Another task is active",
                "Finish or safely pause the active task before starting formal qualification.",
            )
            return
        plan_path, plan = checked
        try:
            current_plan = load_plan(plan_path)
            if current_plan.execution_plan_sha256() != plan.execution_plan_sha256():
                raise ValueError("The checked qualification plan changed; check it again")
        except Exception as exc:
            self._checked_qualification_plans.pop(policy.id, None)
            self._update_qualification_controls()
            show_error(
                self,
                "Formal plan changed",
                "No qualification was started. Check the exact frozen plan again.",
                exc,
                source="qualification start",
            )
            return
        selected_directory = QFileDialog.getExistingDirectory(
            self,
            "Select qualification evidence location",
            str(Path(policy.checkpoint_path).parent),
        )
        if not selected_directory:
            return
        selected = Path(selected_directory).resolve()
        resume = (selected / "qualification_plan.json").is_file()
        output = selected if resume else selected / f"qualification-{plan.qualification_run_id}"
        if not resume and output.exists():
            show_error(
                self,
                "Qualification location already exists",
                "Select that exact evidence directory to resume it, or choose a different parent "
                "directory.",
                FileExistsError(str(output)),
                source="qualification start",
            )
            return
        if resume and (output / "qualification_evidence.json").is_file():
            QMessageBox.information(
                self,
                "Qualification already completed",
                "Use Admit passed evidence to verify and admit this completed directory.",
            )
            return
        total_cells = len(plan.development_cases) * plan.runs * 2
        answer = QMessageBox.warning(
            self,
            "Run formal qualification",
            f"{'Resume' if resume else 'Start'} the checked formal plan for {policy.name!r}?\n\n"
            f"Evidence directory: {output}\n"
            f"Finite design: {len(plan.development_cases)} cases x {plan.runs} paired runs x 2 "
            f"algorithms = {total_cells} optimizer cells. Each cell has exactly "
            f"{plan.max_evaluations} evaluations.\n\n"
            "The process produces evidence only. It will not admit, activate, or bind the policy.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        arguments = [
            "-m",
            "calo_rpd_studio.scripts.qualify_tsh_calo",
            plan_path,
            "--output",
            str(output),
        ]
        if resume:
            arguments.append("--resume")
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(arguments)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_qualification_output)
        process.errorOccurred.connect(self._qualification_process_error)
        process.finished.connect(self._qualification_process_finished)
        self._qualification_process = process
        self._qualification_process_output = ""
        self._qualification_output_directory = str(output)
        self.state.task_status.begin(
            "Independent policy qualification",
            detail="Formal finite plan running; completed cells remain available for exact resume",
            progress=-1,
            cancellable=False,
        )
        self.qualification_workflow_status.setText(
            "Formal qualification running. Process output is being retained in Activity -> Logs."
        )
        self.activity_message.emit("INFO", self.qualification_workflow_status.text())
        self._update_qualification_controls()
        process.start()

    def _read_qualification_output(self) -> None:
        process = self._qualification_process
        if process is None:
            return
        output = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._qualification_process_output += output
        for line in output.splitlines():
            if line.strip():
                self.activity_message.emit("DEBUG", line.strip())

    def _qualification_process_error(self, error) -> None:
        process = self._qualification_process
        if process is None or error != QProcess.ProcessError.FailedToStart:
            return
        message = process.errorString()
        self._qualification_process = None
        self.state.task_status.fail("Formal qualification process could not start")
        self.qualification_workflow_status.setText(
            "Formal qualification process could not start; no evidence was admitted or activated."
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
        process.deleteLater()
        self._qualification_process = None
        if exit_code == 0:
            passed = '"passed": true' in self._qualification_process_output.lower()
            state = "passed" if passed else "completed without qualification"
            self.state.task_status.finish(f"Formal qualification {state}")
            self.qualification_workflow_status.setText(
                f"Formal qualification {state}. Evidence was retained at "
                f"{self._qualification_output_directory}. Use Admit passed evidence; activation "
                "remains separate."
            )
            self.activity_message.emit("INFO", self.qualification_workflow_status.text())
        else:
            self.state.task_status.fail(
                "Formal qualification stopped; retained completed cells may be resumed explicitly"
            )
            self.qualification_workflow_status.setText(
                f"Formal qualification stopped with code {exit_code}. No evidence was admitted or "
                "activated. Select the retained evidence directory to resume."
            )
            self.activity_message.emit("ERROR", self.qualification_workflow_status.text())
        self._update_qualification_controls()

    def admit_qualification_evidence(self) -> None:
        policy = self._selected_policy()
        blocker = self._qualification_candidate_blocker(policy)
        if blocker:
            return
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select completed qualification evidence directory",
            self._qualification_output_directory or str(Path(policy.checkpoint_path).parent),
        )
        if not directory:
            return
        try:
            verified = self.state.policy_registry.inspect_qualification_evidence(
                policy.id, directory
            )
        except Exception as exc:
            show_error(
                self,
                "Qualification evidence was not admitted",
                "The directory did not prove a complete passed formal qualification for this exact "
                "policy.",
                exc,
                source="qualification evidence verification",
            )
            return
        summary = verified.metrics["summary"]
        answer = QMessageBox.question(
            self,
            "Admit passed qualification evidence",
            f"Admit the verified grade {verified.grade} evidence for {policy.name!r}?\n\n"
            f"Cases: {verified.metrics['development_cases']}\n"
            f"Worst-case feasible probability: {summary['minimum_candidate_feasible_probability']:.1%}\n"
            f"Worst-case median objective improvement: "
            f"{summary['minimum_relative_objective_improvement']:.2%}\n"
            f"Worst-case objective win rate: {summary['minimum_objective_win_rate']:.1%}\n\n"
            "Admission makes the policy eligible for comparison and explicit activation. It does "
            "not activate or bind the policy.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            admitted = self.state.policy_registry.admit_qualification_evidence(
                policy.id, directory
            )
        except Exception as exc:
            show_error(
                self,
                "Qualification evidence was not admitted",
                "The registry remained unchanged.",
                exc,
                source="qualification evidence admission",
            )
            return
        self.activity_message.emit(
            "INFO",
            f"Passed formal evidence admitted for {admitted.name}; the policy remains inactive.",
        )
        self.refresh_policy_library()
        self._select_policy_id(admitted.id)

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
