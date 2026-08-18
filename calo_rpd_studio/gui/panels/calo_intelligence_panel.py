"""Policy inventory and explicit TSH-CALO lifecycle controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QStandardPaths, QThread, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.ai.model_io import checkpoint_sha256
from calo_rpd_studio.assistant import LocalAssistantConfig, OllamaParameterAdvisor
from calo_rpd_studio.algorithms.calo.policy_readiness import policy_record_user_status
from calo_rpd_studio.algorithms.calo.tsh_calo_automatic_qualification import (
    AutomaticQualificationRejected,
    AutomaticQualificationSourceSnapshot,
    AutomaticQualificationWorkspace,
    AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST,
    automatic_qualification_workload,
    automatic_qualification_workflow_payload,
    build_automatic_formal_qualification_plan,
    discard_incomplete_automatic_qualification_workspace,
    freeze_plan,
    frozen_qualification_restart_design_sha256,
    inspect_incomplete_automatic_qualification_workspace,
    inspect_verified_paused_automatic_qualification_workspace,
    prepare_automatic_source_snapshot,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification_campaign import (
    TSH_CALO_QUALIFICATION_EVENT_SCHEMA,
    TSH_CALO_QUALIFICATION_PAUSE_EXIT_CODE,
    TSHCALOQualificationPlan,
    inspect_tsh_calo_qualification_resume_state,
    qualification_candidate_contract,
    request_tsh_calo_qualification_pause,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
    parse_tsh_calo_extension_plan,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_influence import (
    build_training_parameter_influence,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_parameter_evidence import (
    verify_training_influence_campaign,
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


class _ParameterExplanationWorker(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config: LocalAssistantConfig, evidence: dict, question: str) -> None:
        super().__init__()
        self.config = config
        self.evidence = evidence
        self.question = question

    def run(self) -> None:
        try:
            response = OllamaParameterAdvisor(self.config).explain(
                evidence=self.evidence, question=self.question
            )
        except Exception as exc:  # network/model failures are surfaced without changing CALO state
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        self.completed.emit(response.to_dict())


class CALOIntelligencePanel(ScrollablePage):
    """Manage policies without embedding or retaining a second training implementation."""

    stage_completed = pyqtSignal()
    independent_training_requested = pyqtSignal()
    activity_message = pyqtSignal(str, str)

    def __init__(
        self, state, experiment_manager, model_library=None, parent=None, *, settings_manager=None
    ) -> None:
        del experiment_manager
        content = QWidget()
        super().__init__(content, parent)
        self.state = state
        self.model_library = model_library
        self.settings_manager = settings_manager
        self._current_influence_evidence: dict = {}
        self._assistant_thread: QThread | None = None
        self._assistant_worker: _ParameterExplanationWorker | None = None
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
        self._qualification_prior_incidents: list[dict] = []
        self._qualification_discardable_workspaces: list[
            tuple[TSHCALOQualificationPlan, AutomaticQualificationWorkspace]
        ] = []
        self._qualification_progress_timer = QTimer(self)
        self._qualification_progress_timer.setInterval(500)
        self._qualification_progress_timer.timeout.connect(self._update_qualification_progress)
        self.state.task_status.cancel_requested.connect(self.request_safe_qualification_pause)

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
                "Feasibility",
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
        self.policy_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.policy_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.policy_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
        self.qualification_button = QPushButton("Start fresh assessment")
        self.qualification_resume_button = QPushButton("Resume assessment")
        self.qualification_resume_button.setVisible(False)
        self.policy_select_button = QPushButton("Select for use")
        self.policy_activate_button = QPushButton("Activate for experiments")
        self.policy_delete_button = QPushButton("Delete model files")
        self.policy_refresh_button = QPushButton("Refresh")
        self.policy_import_button.clicked.connect(self.import_policy)
        self.qualification_button.clicked.connect(self.qualify_selected_policy)
        self.qualification_resume_button.clicked.connect(self.resume_selected_assessment)
        self.policy_select_button.clicked.connect(self.select_policy_for_use)
        self.policy_activate_button.clicked.connect(self.activate_selected_policy)
        self.policy_delete_button.clicked.connect(self.delete_selected_model_files)
        self.policy_refresh_button.clicked.connect(self._request_policy_library_refresh)
        for button in (
            self.policy_import_button,
            self.qualification_button,
            self.qualification_resume_button,
            self.policy_select_button,
            self.policy_activate_button,
            self.policy_delete_button,
            self.policy_refresh_button,
        ):
            qualification_buttons.addWidget(button)
        qualification_buttons.addStretch(1)
        library_layout.addLayout(qualification_buttons)
        self.qualification_workflow_status = QLabel(
            "Workflow: import -> start fresh, or resume one verified safe pause -> inspect ratings "
            "and training influence -> scientist selects -> activate explicitly. The software "
            "makes no suitability decision."
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

        feasibility = QGroupBox("Feasibility assessment")
        self.feasibility_group = feasibility
        feasibility_layout = QVBoxLayout(feasibility)
        self.feasibility_status = QLabel(
            "Select an assessed model to view its integrity-bound feasibility measurements."
        )
        self.feasibility_status.setWordWrap(True)
        self.feasibility_status.setObjectName("ContextValue")
        feasibility_layout.addWidget(self.feasibility_status)
        self.feasibility_table = QTableWidget(0, 4)
        self.feasibility_table.setHorizontalHeaderLabels(
            ("Rating", "Score", "Observation", "Authority")
        )
        self.feasibility_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.feasibility_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.feasibility_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.feasibility_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.feasibility_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.feasibility_table.verticalHeader().setVisible(False)
        self.feasibility_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.feasibility_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.feasibility_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.feasibility_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        feasibility_layout.addWidget(self.feasibility_table)
        layout.addWidget(feasibility)

        influence = QGroupBox("Training parameter influence")
        self.influence_group = influence
        influence_layout = QVBoxLayout(influence)
        self.influence_status = QLabel(
            "Select a model to inspect its immutable training values and comparable-campaign evidence."
        )
        self.influence_status.setWordWrap(True)
        self.influence_status.setObjectName("ContextValue")
        influence_layout.addWidget(self.influence_status)
        self.influence_table = QTableWidget(0, 5)
        self.influence_table.setHorizontalHeaderLabels(
            (
                "Training parameter",
                "Selected value",
                "Association",
                "Direction",
                "Most associated rating / evidence",
            )
        )
        self.influence_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.influence_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.influence_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.influence_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.influence_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.influence_table.verticalHeader().setVisible(False)
        for column in (0, 1, 2, 3):
            self.influence_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.influence_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch
        )
        influence_layout.addWidget(self.influence_table)
        explanation_row = QHBoxLayout()
        self.parameter_question = QLineEdit()
        self.parameter_question.setPlaceholderText(
            "Ask about the selected parameter evidence, for example: Which findings are only exploratory?"
        )
        self.parameter_question.setAccessibleName("Question about parameter evidence")
        self.explain_parameter_button = QPushButton("Explain parameter evidence")
        self.explain_parameter_button.setEnabled(False)
        self.explain_parameter_button.clicked.connect(self.explain_parameter_evidence)
        explanation_row.addWidget(self.parameter_question, 1)
        explanation_row.addWidget(self.explain_parameter_button)
        influence_layout.addLayout(explanation_row)
        self.parameter_explanation = QPlainTextEdit()
        self.parameter_explanation.setReadOnly(True)
        self.parameter_explanation.setPlaceholderText(
            "Optional local explanations appear here. They are interpretations, not scientific evidence."
        )
        self.parameter_explanation.setMaximumHeight(180)
        influence_layout.addWidget(self.parameter_explanation)
        layout.addWidget(influence)
        layout.addStretch(1)
        self._resize_evidence_table_to_entries(self.feasibility_table)
        self._resize_evidence_table_to_entries(self.influence_table)

        self.state.policy_state_changed.connect(lambda _status: self._update_policy_gate_state())
        if self.model_library is not None:
            self.model_library.changed.connect(self.refresh_policy_library)
        self.refresh_policy_library()

    def _request_policy_library_refresh(self) -> None:
        """Force a current disk/database read without changing policy lifecycle state."""

        if self.model_library is not None:
            self.model_library.refresh()
        else:
            self.refresh_policy_library()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if hasattr(self, "policy_table"):
            self._resize_policy_table_to_entries()

    def refresh_policy_library(self) -> None:
        selected_key = self._row_key(self._selected_row())
        governing = self.state.governing_policy_status()
        registered = [
            policy
            for policy in self.state.policy_registry.list(include_archived=False)
            if not policy.checkpoint_path.endswith(".resume.pt")
            and "_lineage" not in str(policy.checkpoint_path)
        ]
        registered_by_path = {
            str(Path(policy.checkpoint_path).expanduser().resolve()).casefold(): policy
            for policy in registered
        }
        assessment_by_policy = {
            str(item.get("policy_id", "")): item
            for item in self.state.policy_registry.feasibility_assessment_summaries()
            if not item.get("verification_error")
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
                        self._training_evaluation_text(policy.get("training_evaluations")),
                        "Not assessed",
                        (
                            "Training complete · import required"
                            if policy.get("policy_candidate")
                            else "Training complete · candidate unavailable"
                        ),
                        "Not admitted",
                        "Verified saved candidate" if not candidate_error else "Needs attention",
                    )
                else:
                    retained_assessment = assessment_by_policy.get(registered_policy.id, {})
                    retained_ratings = dict(
                        retained_assessment.get("feasibility_assessment", {}) or {}
                    )
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
                        (
                            f"{float(retained_ratings['overall_feasibility_score']):.1f}/100"
                            if "overall_feasibility_score" in retained_ratings
                            else "Legacy evidence"
                        ),
                        scientific_status,
                        (
                            "Feasibility assessment admitted"
                            if registered_policy.qualification_status
                            in {"assessed", "scientist_selected"}
                            else (
                                "Legacy qualification admitted"
                                if registered_policy.qualification_status == "qualified"
                                else "Feasibility assessment required"
                            )
                        ),
                        (
                            "Compatible"
                            if registered_policy.compatible_with(TSH_CALO_ALGORITHM_ID)
                            else "Not compatible"
                        ),
                    )
            else:
                retained_assessment = assessment_by_policy.get(policy.id, {})
                retained_ratings = dict(retained_assessment.get("feasibility_assessment", {}) or {})
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
                    (
                        f"{float(retained_ratings['overall_feasibility_score']):.1f}/100"
                        if "overall_feasibility_score" in retained_ratings
                        else "Legacy evidence"
                    ),
                    scientific_status,
                    (
                        "Feasibility assessment admitted"
                        if policy.qualification_status in {"assessed", "scientist_selected"}
                        else (
                            "Legacy qualification admitted"
                            if policy.qualification_status == "qualified"
                            else "Feasibility assessment required"
                        )
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
        # Rebuilding a QTableWidget can preserve the current row without emitting a
        # selection change. Evidence is read from disk/database, so a refresh must
        # recompute the detail blocks even when the same policy remains selected.
        self._policy_selection_changed()
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

    def _resize_evidence_table_to_entries(self, table: QTableWidget) -> None:
        """Expose every retained evidence row through the page's single scroll owner."""

        table.resizeRowsToContents()
        header = table.horizontalHeader()
        header_height = max(header.height(), header.sizeHint().height())
        body_height = sum(table.rowHeight(row) for row in range(table.rowCount()))
        frame_height = table.frameWidth() * 2
        table.setFixedHeight(header_height + body_height + frame_height)
        table.updateGeometry()
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
                # selectRow() is silent when the row is already current. Assessment
                # admission commonly updates that same row, so refresh its evidence
                # explicitly instead of waiting for a relaunch or a different click.
                self._policy_selection_changed()
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
                self.policy_activate_button.setVisible(False)
                self.policy_activate_button.setEnabled(False)
                self.policy_select_button.setText("Import before selection")
                self.policy_select_button.setEnabled(False)
                self.policy_delete_button.setToolTip(
                    "Permanently delete this unregistered completed campaign directory after "
                    "an exact-path confirmation."
                )
                self.policy_delete_button.setEnabled(bool(completed_training.get("directory", "")))
            else:
                self.policy_import_button.setText("Imported")
                self.policy_import_button.setEnabled(False)
                if policy.active:
                    self.policy_activate_button.setVisible(True)
                    self.policy_activate_button.setText("Active governing policy")
                    self.policy_activate_button.setEnabled(False)
                    self.policy_activate_button.setToolTip("This is the active governing policy.")
                else:
                    eligible = bool(
                        policy.usable
                        and policy.compatible_with(TSH_CALO_ALGORITHM_ID)
                        and policy.qualification_status in {"scientist_selected", "qualified"}
                    )
                    self.policy_activate_button.setVisible(eligible)
                    self.policy_activate_button.setText("Activate for experiments")
                    self.policy_activate_button.setEnabled(eligible)
                    self.policy_activate_button.setToolTip(
                        "" if eligible else "Select this assessed model for use before activation."
                    )
                selectable = policy.qualification_status == "assessed"
                self.policy_select_button.setText(
                    "Selected for use"
                    if policy.qualification_status in {"scientist_selected", "qualified"}
                    else "Select for use"
                )
                self.policy_select_button.setEnabled(selectable)
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
            self._refresh_feasibility_and_influence()
            return
        policy = self._selected_policy()
        self.policy_import_button.setText("Imported" if policy is not None else "Import policy")
        self.policy_import_button.setEnabled(policy is None)
        eligible = bool(
            policy
            and not policy.active
            and policy.usable
            and policy.compatible_with(TSH_CALO_ALGORITHM_ID)
            and policy.qualification_status in {"scientist_selected", "qualified"}
        )
        self.policy_activate_button.setText(
            "Active governing policy"
            if policy is not None and policy.active
            else "Activate for experiments"
        )
        self.policy_activate_button.setVisible(bool(policy and (policy.active or eligible)))
        self.policy_activate_button.setEnabled(eligible)
        self.policy_activate_button.setToolTip(
            (
                "This is the active governing policy."
                if policy is not None and policy.active
                else ("" if eligible else "Select an assessed model for use before activation.")
            )
        )
        selectable = bool(policy and policy.qualification_status == "assessed")
        self.policy_select_button.setText(
            "Selected for use"
            if policy is not None
            and policy.qualification_status in {"scientist_selected", "qualified"}
            else "Select for use"
        )
        self.policy_select_button.setEnabled(selectable)
        removal_blocker = "Select a model to delete."
        if policy is not None:
            try:
                removal_blocker = self.state.policy_registry.unqualified_candidate_removal_blocker(
                    policy.id
                )
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                removal_blocker = str(exc) or "The selected policy cannot be removed."
        self.policy_delete_button.setEnabled(policy is not None and not removal_blocker)
        self.policy_delete_button.setToolTip(
            removal_blocker
            or "Permanently remove this exact inactive, unqualified, unreferenced model file "
            "and its registry entry after confirmation."
        )
        if policy is not None:
            self.path.setText(policy.name)
        self._update_qualification_controls()
        self._refresh_feasibility_and_influence()

    def _training_campaign_for_policy(self, policy) -> dict | None:
        if policy is None or self.model_library is None:
            return None
        try:
            checkpoint = Path(policy.checkpoint_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        for campaign in self.model_library.completed_campaigns():
            try:
                candidate = Path(str(campaign.get("policy_candidate", ""))).expanduser().resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            if candidate == checkpoint:
                return campaign
        return None

    @staticmethod
    def _parsed_training_plan_result(
        campaign: dict | None,
    ) -> tuple[object | None, str]:
        if not campaign:
            return None, "no completed campaign matches the selected checkpoint"
        path = Path(str(campaign.get("plan", ""))).expanduser()
        if not str(campaign.get("plan", "")).strip():
            return None, "the completed campaign has no training-plan path"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None, "the authenticated training plan is not a JSON object"
            return parse_tsh_calo_extension_plan(payload), ""
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return None, str(exc) or type(exc).__name__

    @classmethod
    def _parsed_training_plan(cls, campaign: dict | None):
        return cls._parsed_training_plan_result(campaign)[0]

    def _refresh_feasibility_and_influence(self) -> None:
        self.feasibility_table.setRowCount(0)
        self.influence_table.setRowCount(0)
        self._current_influence_evidence = {}
        self.explain_parameter_button.setEnabled(False)
        self._resize_evidence_table_to_entries(self.feasibility_table)
        self._resize_evidence_table_to_entries(self.influence_table)
        policy = self._selected_policy()
        if policy is None:
            self.feasibility_status.setText(
                "Import and select a model to view feasibility measurements."
            )
            self.influence_status.setText(
                "Import and select a model to view immutable training values."
            )
            return
        summaries = self.state.policy_registry.feasibility_assessment_summaries(policy.id)
        assessment = next((item for item in summaries if not item.get("verification_error")), None)
        if assessment is None:
            error = next(
                (
                    str(item.get("verification_error", ""))
                    for item in summaries
                    if item.get("verification_error")
                ),
                "",
            )
            self.feasibility_status.setText(
                (
                    f"Feasibility ratings are unavailable because retained evidence failed verification: {error}"
                    if error
                    else "No completed feasibility assessment is admitted for this model."
                )
            )
        else:
            ratings = dict(assessment.get("feasibility_assessment", {}) or {})
            overall = dict(ratings.get("overall_ratings", {}) or {})
            rows = [
                (
                    "Overall full feasibility",
                    ratings.get("overall_feasibility_score"),
                    f"{ratings.get('candidate_cell_count', 0)} candidate cells",
                ),
                (
                    "First-feasible reach",
                    overall.get("first_feasible_reached"),
                    "Across candidate cells",
                ),
                (
                    "First-feasible efficiency",
                    overall.get("first_feasible_efficiency"),
                    "Budget-normalized; not reached scores zero",
                ),
                (
                    "Independent validation",
                    overall.get("independent_validation"),
                    "Retained candidate solutions",
                ),
                (
                    "Paired objective coverage",
                    overall.get("paired_feasible_objective_coverage"),
                    "Pairs where both arms were feasible",
                ),
            ]
            for case in list(ratings.get("case_ratings", [])):
                case_scores = dict(case.get("ratings", {}) or {})
                rows.append(
                    (
                        f"{case.get('case', 'Case')} full feasibility",
                        case_scores.get("full_feasibility"),
                        f"{case.get('n_candidate_cells', 0)} candidate cells; median first feasible FE "
                        f"{case.get('candidate_first_feasible_evaluation_median') or 'not reached'}",
                    )
                )
            self.feasibility_table.setRowCount(len(rows))
            tooltip = (
                f"Candidate SHA-256: {policy.sha256}\n"
                f"Assessment: {assessment.get('assessment_id', '')}\n"
                f"Rating schema: {ratings.get('schema_version', '')}\n\n"
                f"{ratings.get('score_definition', '')}\n\n"
                "The software makes no suitability recommendation. Scientist selection is separate."
            )
            for row, (label, score, observation) in enumerate(rows):
                try:
                    score_text = f"{float(score):.1f}/100"
                except (TypeError, ValueError):
                    score_text = "Not available"
                values = (label, score_text, observation, "Scientist decides")
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setToolTip(tooltip)
                    self.feasibility_table.setItem(row, column, item)
            self.feasibility_status.setText(
                f"Integrity-valid measurements for {policy.name}. Overall full feasibility "
                f"{float(ratings.get('overall_feasibility_score', 0.0)):.1f}/100 · "
                f"scientist decision: {'selected' if assessment.get('scientist_selected') else 'not decided'}."
            )
            self._resize_evidence_table_to_entries(self.feasibility_table)

        selected_campaign = self._training_campaign_for_policy(policy)
        if selected_campaign is None or assessment is None:
            self.influence_status.setText(
                "The selected model does not have both authenticated training and assessment "
                "evidence, so parameter associations are unavailable."
            )
            return
        try:
            selected_verified = verify_training_influence_campaign(
                selected_campaign,
                assessment,
                expected_candidate_sha256=policy.sha256,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.influence_status.setText(
                "The selected model's training values could not be bound to its assessment "
                f"evidence ({str(exc) or type(exc).__name__})."
            )
            return
        cohort: list[dict] = []
        if self.model_library is not None:
            policies_by_path = {
                str(Path(item.checkpoint_path).expanduser().resolve()).casefold(): item
                for item in self.state.policy_registry.list(include_archived=False)
            }
            for campaign in self.model_library.completed_campaigns():
                candidate_path = str(campaign.get("policy_candidate", ""))
                if not candidate_path:
                    continue
                try:
                    candidate_key = str(Path(candidate_path).expanduser().resolve()).casefold()
                except (OSError, RuntimeError, ValueError):
                    continue
                cohort_policy = policies_by_path.get(candidate_key)
                if cohort_policy is None:
                    continue
                retained = next(
                    (
                        item
                        for item in self.state.policy_registry.feasibility_assessment_summaries(
                            cohort_policy.id
                        )
                        if not item.get("verification_error")
                    ),
                    None,
                )
                if retained is None:
                    continue
                try:
                    verified = verify_training_influence_campaign(
                        campaign,
                        retained,
                        expected_candidate_sha256=cohort_policy.sha256,
                    )
                except (OSError, RuntimeError, TypeError, ValueError):
                    continue
                cohort.append(
                    {
                        "candidate_sha256": verified.candidate_sha256,
                        "plan": verified.plan,
                        "ratings": verified.ratings,
                        "assessment_comparison_protocol_sha256": (
                            verified.assessment_comparison_protocol_sha256
                        ),
                        "training_compatibility_sha256": verified.training_compatibility_sha256,
                    }
                )
        try:
            influence = build_training_parameter_influence(
                selected_candidate_sha256=policy.sha256,
                selected_plan=selected_verified.plan,
                cohort=cohort,
                selected_assessment_comparison_protocol_sha256=(
                    selected_verified.assessment_comparison_protocol_sha256
                ),
                selected_training_compatibility_sha256=(
                    selected_verified.training_compatibility_sha256
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.influence_status.setText(
                "Training values were found, but the read-only influence report could not be "
                f"constructed ({str(exc) or type(exc).__name__}). No parameter was changed."
            )
            return
        parameters = list(influence.get("parameters", []))
        self.influence_table.setRowCount(len(parameters))
        limitations = "\n".join(str(item) for item in influence.get("limitations", []))
        for row, item in enumerate(parameters):
            effect = item.get("association")
            values = (
                item.get("label") or item.get("parameter", ""),
                item.get("selected_value", ""),
                f"{float(effect):+.3f}" if effect is not None else "Not estimated",
                item.get("direction", "not_estimated"),
                (
                    f"{item.get('affected_rating', 'not_estimated')} · "
                    f"{item.get('evidence_classification', '')}"
                ),
            )
            rating_effects = "\n".join(
                f"{entry.get('rating', '')}: {float(entry.get('association', 0.0)):+.3f} "
                f"({entry.get('direction', '')})"
                for entry in item.get("rating_effects", [])
            )
            tooltip = (
                f"Observed campaigns: {item.get('observations', 0)}\n"
                f"Distinct values: {item.get('distinct_values', [])}\n"
                f"Association measure: {item.get('association_measure', 'not_estimated')}\n"
                f"Most associated rating: {item.get('affected_rating', '')}\n"
                f"All rating associations:\n{rating_effects or 'Not estimated'}\n"
                f"Evidence suitable for ranking: {bool(item.get('rankable', False))}\n\n{limitations}"
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setToolTip(tooltip)
                self.influence_table.setItem(row, column, cell)
        self._resize_evidence_table_to_entries(self.influence_table)
        self._current_influence_evidence = dict(influence)
        self.explain_parameter_button.setEnabled(True)
        self.influence_status.setText(
            f"Selected-model training values are shown. Comparative evidence: "
            f"{influence.get('evidence_classification', 'unavailable').replace('_', ' ')} · "
            f"{influence.get('compatible_campaign_count', 0)} compatible assessed campaigns. "
            "No parameter is changed automatically."
        )

    def _local_assistant_config(self) -> LocalAssistantConfig:
        if self.settings_manager is None:
            return LocalAssistantConfig(enabled=False)
        enabled = str(
            self.settings_manager.value("local_parameter_assistant_enabled", "false")
        ).lower() in {"1", "true", "yes"}
        return LocalAssistantConfig(
            enabled=enabled,
            endpoint=str(
                self.settings_manager.value(
                    "local_parameter_assistant_endpoint", "http://127.0.0.1:11434"
                )
            ),
            model=str(
                self.settings_manager.value("local_parameter_assistant_model", "qwen3.5:9b")
            ),
        )

    def explain_parameter_evidence(self) -> None:
        if not self._current_influence_evidence:
            self.parameter_explanation.setPlainText("No parameter evidence is selected.")
            return
        if self.state.task_status.busy:
            self.parameter_explanation.setPlainText(
                "Local explanations are unavailable while scientific work is active."
            )
            return
        try:
            config = self._local_assistant_config()
            config.validate()
            if not config.enabled:
                raise RuntimeError(
                    "Enable the local research assistant in Application Settings first."
                )
        except (RuntimeError, ValueError) as exc:
            self.parameter_explanation.setPlainText(str(exc))
            return
        if self._assistant_thread is not None and self._assistant_thread.isRunning():
            return
        question = self.parameter_question.text().strip() or (
            "Explain the strongest supported parameter findings, their limitations, and what "
            "controlled evidence should be collected next."
        )
        self.explain_parameter_button.setEnabled(False)
        self.parameter_explanation.setPlainText("Requesting a local explanation…")
        thread = QThread(self)
        worker = _ParameterExplanationWorker(
            config, dict(self._current_influence_evidence), question
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._parameter_explanation_completed)
        worker.failed.connect(self._parameter_explanation_failed)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._parameter_explanation_thread_finished)
        self._assistant_thread = thread
        self._assistant_worker = worker
        thread.start()

    def _parameter_explanation_completed(self, payload: dict) -> None:
        text = str(payload.get("text", "")).strip()
        model = str(payload.get("model", "local model")).strip()
        self.parameter_explanation.setPlainText(
            f"{text}\n\nLocal explanation: {model}. This text is not scientific evidence."
        )

    def _parameter_explanation_failed(self, message: str) -> None:
        self.parameter_explanation.setPlainText(
            f"Local explanation was unavailable: {message}. No CALO state was changed."
        )

    def _parameter_explanation_thread_finished(self) -> None:
        self._assistant_thread = None
        self._assistant_worker = None
        self.explain_parameter_button.setEnabled(bool(self._current_influence_evidence))

    def _reveal_influence_analysis(self) -> None:
        """Bring the completed assessment's decision-support block into the visible page."""

        ancestor = self.parentWidget()
        while ancestor is not None:
            if isinstance(ancestor, QScrollArea) and ancestor is not self:
                ancestor.ensureWidgetVisible(self.influence_group, 24, 24)
                return
            ancestor = ancestor.parentWidget()
        self.ensureWidgetVisible(self.influence_group, 24, 24)

    def _qualification_candidate_blocker(self, policy) -> str:
        if policy is None:
            return "Import and select a policy first."
        if policy.active:
            return "The active governing policy cannot enter a new qualification workflow."
        if policy.archived:
            return "Restore the archived policy first."
        if policy.qualification_status in {"assessed", "scientist_selected", "qualified"}:
            return "A complete assessment is already admitted for this policy."
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
        self._qualification_prior_incidents = []
        self._qualification_discardable_workspaces = []
        self.qualification_button.setEnabled(policy is not None and not process_running)
        retained = None
        resume_error = ""
        if policy is not None and not blocker and not process_running:
            try:
                candidate_artifact = self.state.policy_registry.inspect_qualification_candidate(
                    policy.id
                )
                retained = self._retained_qualification_resume(
                    self._automatic_qualification_base_directory(),
                    policy,
                    candidate_artifact,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                resume_error = str(exc)
        resume_available = retained is not None and not blocker and not process_running
        self.qualification_resume_button.setVisible(resume_available)
        self.qualification_resume_button.setEnabled(resume_available)
        self.qualification_resume_button.setToolTip(
            "Resume the uniquely verified paused assessment under its exact frozen plan."
            if resume_available
            else resume_error
        )
        self.qualification_button.setToolTip(
            (f"Feasibility assessment unavailable: {blocker}" if blocker else "")
            or (
                "Start a new frozen feasibility assessment. The confirmed fresh start permanently "
                "removes existing incomplete resumable assessment work for this exact candidate."
                if self._qualification_discardable_workspaces
                else "Start a new frozen feasibility assessment without an automated suitability "
                "decision. Scientist selection and activation remain explicit."
            )
        )
        if policy is not None and blocker and not process_running:
            self.qualification_workflow_status.setText(
                f"Feasibility workflow blocked for {policy.name}: {blocker}"
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

    @staticmethod
    def _next_fresh_assessment_workspace(
        qualification_base: Path, policy, source_commit: str
    ) -> tuple[int, AutomaticQualificationWorkspace]:
        """Allocate a new identity without overwriting any retained campaign directory."""

        restart_ordinal = 0
        while True:
            workspace = AutomaticQualificationWorkspace.create(
                qualification_base / "campaigns",
                candidate_sha256=policy.sha256,
                source_commit=source_commit,
                restart_ordinal=restart_ordinal,
            )
            if not workspace.root.exists():
                return restart_ordinal, workspace
            restart_ordinal += 1

    def _retained_qualification_resume(
        self,
        qualification_base: Path,
        policy,
        candidate_artifact,
        *,
        require_unique_paused: bool = True,
    ) -> (
        tuple[
            TSHCALOQualificationPlan,
            AutomaticQualificationWorkspace,
            AutomaticQualificationSourceSnapshot,
        ]
        | None
    ):
        """Find one safe paused state and inventory exact incomplete work eligible for discard."""

        self._qualification_prior_incidents = []
        self._qualification_discardable_workspaces = []
        campaigns = qualification_base / "campaigns"
        if not campaigns.is_dir():
            return None
        prefix = f"architecture-v2-{policy.sha256[:16].lower()}-"
        paused: list[
            tuple[
                TSHCALOQualificationPlan,
                AutomaticQualificationWorkspace,
                AutomaticQualificationSourceSnapshot,
            ]
        ] = []
        for root in campaigns.glob(f"{prefix}*"):
            plan_path = root / "formal_qualification_plan.json"
            output = root / "formal-qualification-evidence"
            if not root.is_dir() or not plan_path.is_file() or not output.is_dir():
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
            disposition = inspect_tsh_calo_qualification_resume_state(output)
            if disposition["fresh_run_required"]:
                self._qualification_prior_incidents.append(
                    {
                        "campaign_root": str(root.resolve()),
                        "qualification_output": str(output.resolve()),
                        "classification": disposition["classification"],
                        "reason": disposition["reason"],
                        "source_commit": plan.source_commit,
                        "qualification_plan_sha256": plan.execution_plan_sha256(),
                        "frozen_restart_design_sha256": (
                            frozen_qualification_restart_design_sha256(plan)
                        ),
                        "retained_read_only": True,
                    }
                )
                continue
            if not disposition.get("resumable"):
                # Completed evidence is never a fresh-start deletion target.
                continue
            workspace = AutomaticQualificationWorkspace(
                root=root.resolve(),
                workflow_plan=root.resolve() / "automatic_qualification_workflow.json",
                qualification_plan=root.resolve() / "formal_qualification_plan.json",
                qualification_output=root.resolve() / "formal-qualification-evidence",
            )
            verified_plan = inspect_incomplete_automatic_qualification_workspace(
                workspace,
                campaigns_directory=campaigns,
                candidate_path=policy.checkpoint_path,
                candidate_sha256=policy.sha256,
            )
            if verified_plan.execution_plan_sha256() != plan.execution_plan_sha256():
                raise ValueError("Incomplete assessment plan changed during discovery")
            plan = verified_plan
            self._qualification_discardable_workspaces.append((plan, workspace))
            if not require_unique_paused:
                continue
            if disposition.get("status_state") != "paused":
                continue
            if plan.candidate_contract != qualification_candidate_contract(candidate_artifact):
                raise ValueError(
                    "The paused assessment plan no longer matches the candidate architecture "
                    "and parameter contract"
                )
            inspect_verified_paused_automatic_qualification_workspace(
                workspace,
                campaigns_directory=campaigns,
                candidate_path=policy.checkpoint_path,
                candidate_sha256=policy.sha256,
            )
            snapshot_root = (qualification_base / "source-snapshots" / plan.source_commit).resolve()
            validate_repository_for_plan(plan, root=snapshot_root)
            manifest_path = snapshot_root / AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Paused assessment source manifest is unreadable") from exc
            if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
                raise ValueError("Paused assessment source manifest is incompatible")
            snapshot = AutomaticQualificationSourceSnapshot(
                root=snapshot_root,
                source_commit=plan.source_commit,
                worktree_sha256=str(manifest.get("worktree_sha256", "")),
                manifest_sha256=checkpoint_sha256(manifest_path),
                file_count=len(manifest["files"]),
            )
            paused.append((plan, workspace, snapshot))
        if len(paused) > 1:
            if require_unique_paused:
                raise ValueError(
                    "Multiple paused assessments exist for this candidate; exact resume is "
                    "ambiguous. Start a fresh assessment to discard the verified incomplete work."
                )
            return None
        return paused[0] if paused else None

    def resume_selected_assessment(self) -> None:
        """Resume only one uniquely verified safe pause under its retained frozen source."""

        policy = self._selected_policy()
        if policy is None or self._qualification_process is not None:
            return
        blocker = self._qualification_candidate_blocker(policy)
        if blocker:
            self._update_qualification_controls()
            return
        if self.state.task_status.busy:
            QMessageBox.information(
                self,
                "Another task is active",
                "Finish the active task before resuming feasibility assessment.",
            )
            return
        try:
            candidate_artifact = self.state.policy_registry.inspect_qualification_candidate(
                policy.id
            )
            retained = self._retained_qualification_resume(
                self._automatic_qualification_base_directory(),
                policy,
                candidate_artifact,
            )
            if retained is None:
                QMessageBox.information(
                    self,
                    "No paused assessment",
                    "No uniquely verified safely paused assessment exists for the selected model. "
                    "Start fresh assessment creates a new assessment instead.",
                )
                self._update_qualification_controls()
                return
            qualification_plan, workspace, source_snapshot = retained
        except (OSError, RuntimeError, ValueError) as exc:
            show_error(
                self,
                "Assessment cannot resume",
                "Exact resume verification rejected the retained assessment. No retained file was "
                "changed; start fresh only if you intend to discard incomplete assessment work.",
                exc,
                source="automatic feasibility resume verification",
            )
            self._update_qualification_controls()
            return
        disposition = inspect_tsh_calo_qualification_resume_state(workspace.qualification_output)
        workload = automatic_qualification_workload()
        durable_cells = int(disposition.get("success_artifact_count", 0)) + int(
            disposition.get("failure_artifact_count", 0)
        )
        answer = QMessageBox.warning(
            self,
            "Resume feasibility assessment",
            f"Resume the exact paused assessment for {policy.name!r}?\n\n"
            f"Durable progress: {durable_cells}/"
            f"{workload['qualification_cells']} cells. Source snapshot: "
            f"{source_snapshot.source_commit[:12]}. The candidate checksum, frozen cases, seeds, "
            "population, exact FE budgets, analysis, and protected-case closure remain unchanged.\n\n"
            "Resume continues the authenticated partial cell/checkpoint and does not create a new "
            "assessment identity.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._qualification_policy_id = policy.id
            self._qualification_workspace = workspace
            self._qualification_source_snapshot = source_snapshot
            self._continue_automatic_qualification(policy, qualification_plan, workspace)
        except AutomaticQualificationRejected as exc:
            self._record_qualification_rejection(policy.name, str(exc))
        except Exception as exc:
            show_error(
                self,
                "Assessment could not resume",
                "The retained paused assessment remains unchanged and no policy state changed.",
                exc,
                source="automatic feasibility resume",
            )
            self._update_qualification_controls()

    def qualify_selected_policy(self) -> None:
        """Start fresh feasibility; this action never activates or binds the policy."""

        policy = self._selected_policy()
        if policy is None:
            return
        blocker = self._qualification_candidate_blocker(policy)
        if blocker:
            message = (
                f"The selected policy cannot enter formal feasibility assessment:\n\n{blocker}\n\n"
                "No plan, assessment evidence, registry state, or model file was changed."
            )
            self.qualification_workflow_status.setText(
                f"Feasibility workflow blocked for {policy.name}: {blocker}"
            )
            self.activity_message.emit("WARNING", self.qualification_workflow_status.text())
            QMessageBox.information(self, "Feasibility assessment unavailable", message)
            return
        if self._qualification_process is not None:
            return
        if self.state.task_status.busy:
            QMessageBox.information(
                self,
                "Another task is active",
                "Finish or safely pause the active task before starting feasibility assessment.",
            )
            return
        try:
            candidate_artifact = self.state.policy_registry.inspect_qualification_candidate(
                policy.id
            )
            qualification_base = self._automatic_qualification_base_directory()
            self._retained_qualification_resume(
                qualification_base,
                policy,
                candidate_artifact,
                require_unique_paused=False,
            )
            source_snapshot = prepare_automatic_source_snapshot(
                QUALIFICATION_SOURCE_ROOT,
                qualification_base / "source-snapshots",
            )
            restart_ordinal, workspace = self._next_fresh_assessment_workspace(
                qualification_base,
                policy,
                source_snapshot.source_commit,
            )
            qualification_plan = build_automatic_formal_qualification_plan(
                candidate_path=policy.checkpoint_path,
                candidate_sha256=policy.sha256,
                source_commit=source_snapshot.source_commit,
                candidate_artifact=candidate_artifact,
                restart_ordinal=restart_ordinal,
            )
            fresh_design_sha256 = frozen_qualification_restart_design_sha256(qualification_plan)
            retained_designs = [
                item["frozen_restart_design_sha256"] for item in self._qualification_prior_incidents
            ] + [
                frozen_qualification_restart_design_sha256(plan)
                for plan, _workspace in self._qualification_discardable_workspaces
            ]
            if any(identity != fresh_design_sha256 for identity in retained_designs):
                raise ValueError(
                    "The fresh assessment would change the complete frozen scientific design"
                )
        except Exception as exc:
            show_error(
                self,
                "Assessment could not start",
                "Automatic preflight rejected the candidate before any measurement started. "
                "Correct the reported candidate, source, or architecture condition and try again.",
                exc,
                source="automatic feasibility preflight",
            )
            return

        workload = automatic_qualification_workload()
        if self._qualification_prior_incidents:
            action = "Start a fresh corrected-source feasibility assessment"
        else:
            action = "Start a new frozen architecture feasibility assessment"
        answer = QMessageBox.warning(
            self,
            "Assess policy feasibility",
            f"{action} for {policy.name!r}?\n\n"
            f"Frozen design: {workload['cases']} assessment cases (case30 and case57), "
            f"{workload['runs_per_case']} paired runs per case, "
            f"{workload['qualification_cells']} paired optimizer cells. Every optimizer cell has "
            f"exactly {workload['evaluations_per_cell']} evaluations.\n\n"
            "The frozen candidate contract verifies the policy architecture, state/action and "
            "training schemas, ensemble membership, feature contract, training-design identity, "
            "and exact model checksum. Product version labels and project lifecycle labels are "
            "not feasibility measurements.\n\n"
            f"Source snapshot: {source_snapshot.source_commit[:12]} from "
            f"{source_snapshot.file_count} non-ignored files. The working source tree is not "
            "modified.\n\n"
            + (
                f"Prior incident: {len(self._qualification_prior_incidents)} contradictory or "
                "infrastructure-aborted campaign(s) remain byte-for-byte retained and will not "
                "be resumed or admitted. Every operative scientific design field is unchanged; "
                "only the new run and corrected source provenance identities differ.\n\n"
                if self._qualification_prior_incidents
                else ""
            )
            + (
                f"Fresh-start removal: {len(self._qualification_discardable_workspaces)} exact "
                "incomplete assessment workspace(s), including any safely paused work, will be "
                "permanently removed after confirmation. Their durable cells and partial-cell "
                "checkpoints will not be merged into the new assessment. "
                "Completed assessments, source snapshots, and immutable infrastructure incidents "
                "will not be removed.\n\n"
                if self._qualification_discardable_workspaces
                else ""
            )
            + "Activity records a micro step every 500 evaluations. Pause safely commits the current "
            "optimizer state, so even a partial cell can continue later. Pause/resume has no count "
            "limit and never changes this finite budget. Completed integrity-valid measurements are "
            "admitted without a pass/fail suitability verdict. The scientist reviews the ratings and "
            "must select the model explicitly before the separate activation action is available.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            discarded_roots = [
                str(retained_workspace.root.resolve())
                for _retained_plan, retained_workspace in (
                    self._qualification_discardable_workspaces
                )
            ]
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
            workflow_payload["prior_infrastructure_incidents"] = list(
                self._qualification_prior_incidents
            )
            workflow_payload["discarded_incomplete_assessment_roots"] = discarded_roots
            workflow_payload["frozen_restart_design_sha256"] = (
                frozen_qualification_restart_design_sha256(qualification_plan)
            )
            freeze_plan(
                workspace.workflow_plan,
                workflow_payload,
            )
            freeze_plan(workspace.qualification_plan, qualification_plan.to_dict())
            for retained_plan, retained_workspace in list(
                self._qualification_discardable_workspaces
            ):
                discard_incomplete_automatic_qualification_workspace(
                    retained_workspace,
                    campaigns_directory=qualification_base / "campaigns",
                    candidate_path=policy.checkpoint_path,
                    candidate_sha256=policy.sha256,
                    expected_plan_sha256=retained_plan.execution_plan_sha256(),
                )
            if discarded_roots:
                self.activity_message.emit(
                    "INFO",
                    f"Fresh feasibility start removed {len(discarded_roots)} verified incomplete "
                    "assessment workspace(s); immutable incidents and completed evidence were "
                    "retained.",
                )
            self._qualification_policy_id = policy.id
            self._qualification_workspace = workspace
            self._qualification_source_snapshot = source_snapshot
            self._continue_automatic_qualification(policy, qualification_plan, workspace)
        except AutomaticQualificationRejected as exc:
            self._record_qualification_rejection(policy.name, str(exc))
        except Exception as exc:
            show_error(
                self,
                "Assessment was not admitted",
                "The automatic workflow stopped without changing policy activation or experiment "
                "settings. Any incomplete assessment work removed by the confirmed fresh-start "
                "choice is not recoverable.",
                exc,
                source="automatic feasibility assessment",
            )
            self._update_qualification_controls()

    def _continue_automatic_qualification(
        self, policy, qualification_plan, workspace: AutomaticQualificationWorkspace
    ) -> None:
        retained_state = (
            inspect_tsh_calo_qualification_resume_state(workspace.qualification_output)
            if workspace.qualification_output.is_dir()
            else {}
        )
        if retained_state.get("fresh_run_required"):
            raise AutomaticQualificationRejected(
                "Policy rejected: the retained formal campaign is an immutable infrastructure "
                "incident and cannot resume"
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
        success_names = (
            {path.name for path in (output / "records").glob("*.json")}
            if (output / "records").is_dir()
            else set()
        )
        failure_names = (
            {path.name for path in (output / "failures").glob("*.json")}
            if (output / "failures").is_dir()
            else set()
        )
        completed = len(success_names | failure_names)
        expected = max(1, int(self._qualification_expected_cells))
        completed = min(completed, expected)
        evaluations_per_cell = int(automatic_qualification_workload()["evaluations_per_cell"])
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
            request = request_tsh_calo_qualification_pause(workspace.qualification_output)
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
            self._consume_qualification_output_line(self._qualification_stdout_buffer.rstrip("\r"))
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
            self.activity_message.emit(
                "WARNING", "A qualification progress event was incompatible."
            )
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
                f"{first_feasible_text} | {rate_text} | cell ETA {eta_text} | "
                "live, not yet a committed cell",
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
            self.state.task_status.update(progress=100, detail="Feasibility evidence committed")
            self.activity_message.emit(
                "INFO", "Feasibility measurements committed; no suitability decision was made."
            )

    def _qualification_process_error(self, error) -> None:
        process = self._qualification_process
        if process is None or error != QProcess.ProcessError.FailedToStart:
            return
        message = process.errorString()
        self._qualification_progress_timer.stop()
        self._qualification_process = None
        self._qualification_pause_requested = False
        stage = self._qualification_process_stage or "automatic"
        self.state.task_status.fail(f"Feasibility {stage} process could not start")
        self.qualification_workflow_status.setText(
            f"Feasibility {stage} process could not start; no evidence was admitted or activated."
        )
        self.activity_message.emit("ERROR", message)
        self._update_qualification_controls()
        show_error(
            self,
            "Feasibility process could not start",
            "No evidence was admitted and no policy state changed.",
            message,
            source="formal feasibility process",
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
                f"Feasibility assessment paused safely | {completed}/{expected} cells durable | "
                "choose Resume assessment to continue the exact frozen plan, or Start fresh "
                "assessment to discard incomplete work"
            )
            self.state.task_status.paused(detail)
            self.qualification_workflow_status.setText(detail)
            self.activity_message.emit(
                "INFO",
                "Feasibility assessment paused at an authenticated boundary. No policy evidence was "
                "admitted or activated; the same finite plan remains resumable.",
            )
            self._update_qualification_controls()
            return
        if exit_code == 0:
            try:
                if not policy_id or workspace is None:
                    raise RuntimeError("Automatic feasibility process identity was lost")
                policy = self.state.policy_registry.get(policy_id)
                self._admit_automatic_qualification(policy, workspace)
                return
            except AutomaticQualificationRejected as exc:
                self._record_qualification_rejection(
                    getattr(locals().get("policy"), "name", policy_id), str(exc)
                )
            except Exception as exc:
                self.state.task_status.fail("Feasibility evidence verification failed")
                show_error(
                    self,
                    "Policy was not admitted",
                    "The completed process did not provide admissible evidence. The policy remains "
                    "inactive and experiment settings were not changed.",
                    exc,
                    source="automatic feasibility verification",
                )
        else:
            disposition = (
                inspect_tsh_calo_qualification_resume_state(workspace.qualification_output)
                if workspace is not None and workspace.qualification_output.is_dir()
                else {}
            )
            if disposition.get("fresh_run_required"):
                self.state.task_status.fail(
                    f"Feasibility {stage} retained as an infrastructure incident"
                )
                self.qualification_workflow_status.setText(
                    f"Feasibility {stage} stopped with code {exit_code}. The contradictory run "
                    "is retained read-only and cannot resume or admit evidence. Click Start fresh "
                    "assessment to prepare a new corrected-source run with the unchanged frozen design."
                )
            else:
                self.state.task_status.fail(
                    f"Feasibility {stage} stopped without a verified safe pause; fresh start required"
                )
                self.qualification_workflow_status.setText(
                    f"Feasibility {stage} stopped with code {exit_code}. No evidence was admitted "
                    "or activated. Resume is unavailable without an authenticated safe pause; "
                    "Start fresh assessment removes verified incomplete work after confirmation."
                )
            self.activity_message.emit("ERROR", self.qualification_workflow_status.text())
        self._qualification_pause_requested = False
        self._update_qualification_controls()

    def _confirmed_safe_qualification_pause(self) -> bool:
        workspace = self._qualification_workspace
        if workspace is None:
            return False
        try:
            plan = load_qualification_plan(workspace.qualification_plan)
            inspect_verified_paused_automatic_qualification_workspace(
                workspace,
                campaigns_directory=workspace.root.resolve().parent,
                candidate_path=plan.candidate_path,
                candidate_sha256=plan.candidate_sha256,
            )
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    def _admit_automatic_qualification(
        self, policy, workspace: AutomaticQualificationWorkspace
    ) -> None:
        try:
            verified = self.state.policy_registry.inspect_feasibility_assessment(
                policy.id, workspace.qualification_output
            )
        except Exception as exc:
            raise AutomaticQualificationRejected(
                f"Completed feasibility measurements failed integrity verification ({exc})"
            ) from exc
        try:
            admitted = self.state.policy_registry.admit_feasibility_assessment(
                policy.id, workspace.qualification_output
            )
        except Exception as exc:
            raise RuntimeError("Verified feasibility evidence could not be admitted") from exc
        self.state.task_status.finish("Feasibility assessment complete; scientist decision pending")
        qualified_status = (
            f"Assessment complete for {admitted.name}: overall full feasibility "
            f"{verified.score:.1f}/100. The software made no suitability decision. Review the "
            "Feasibility and Influence blocks, then use Select for use if you choose this model."
        )
        self.refresh_policy_library()
        self._select_policy_id(admitted.id)
        self.qualification_workflow_status.setText(qualified_status)
        self.activity_message.emit(
            "INFO",
            f"Integrity-valid feasibility measurements admitted for {admitted.name}; scientist "
            "selection and activation remain undone.",
        )
        QMessageBox.information(
            self,
            "Feasibility assessment complete",
            f"{admitted.name!r} has an integrity-valid feasibility score of {verified.score:.1f}/100. "
            "No pass/fail recommendation was made. Review the two evidence blocks and explicitly "
            "select the model if you decide it should become eligible for activation.",
        )
        QTimer.singleShot(0, self._reveal_influence_analysis)

    def _record_qualification_rejection(self, policy_name: str, reason: str) -> None:
        self.state.task_status.finish("Feasibility evidence was not admissible")
        self.qualification_workflow_status.setText(
            f"{reason}. No scientific score or scientist decision was recorded."
        )
        self.activity_message.emit("WARNING", self.qualification_workflow_status.text())
        self._update_qualification_controls()
        QMessageBox.information(
            self,
            "Feasibility evidence unavailable",
            f"{policy_name!r} has no admissible feasibility dossier.\n\n{reason}\n\nThe retained "
            "evidence remains immutable, and no model-quality judgment was inferred.",
        )

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
            f"Selected by scientist: {active.name} · integrity verified · Ready for experiments"
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

    def select_policy_for_use(self) -> None:
        """Record the scientist's evidence-bound choice without activating the model."""

        policy = self._selected_policy()
        if policy is None or policy.qualification_status != "assessed":
            return
        assessments = [
            item
            for item in self.state.policy_registry.feasibility_assessment_summaries(policy.id)
            if not item.get("verification_error")
        ]
        if not assessments:
            return
        assessment = dict(assessments[0].get("feasibility_assessment", {}) or {})
        score = float(assessment.get("overall_feasibility_score", 0.0))
        answer = QMessageBox.question(
            self,
            "Select model for use",
            f"Select {policy.name!r} for possible experiment use?\n\n"
            f"Overall full feasibility: {score:.1f}/100\n"
            f"Candidate SHA-256: {policy.sha256}\n\n"
            "The software makes no recommendation. This records your explicit scientist decision "
            "for this immutable candidate and assessment. It does not activate the model; activation "
            "remains a separate action.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            selected = self.state.policy_registry.select_assessed_policy(policy.id)
        except Exception as exc:
            show_error(
                self,
                "Model could not be selected",
                "No scientist decision or activation change was accepted.",
                exc,
                source="scientist policy selection",
            )
            return
        self.refresh_policy_library()
        self._select_policy_id(selected.id)
        self.qualification_workflow_status.setText(
            f"Scientist selected {selected.name}. The model remains inactive until Activate for "
            "experiments is used explicitly."
        )

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
            if self.state.policy_registry.inspect_checkpoint(checkpoint)["sha256"] != policy.sha256:
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
                if (checkpoint == directory or directory in checkpoint.parents) and (
                    registered_policy is None or policy.id != registered_policy.id
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
            f"and model file inside this exact directory?\n\n{directory}\n\n" + confirmation_scope,
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
