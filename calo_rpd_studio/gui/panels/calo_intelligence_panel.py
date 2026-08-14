"""Policy inventory and explicit TSH-CALO lifecycle controls."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.algorithms.calo.policy_readiness import policy_record_user_status
from calo_rpd_studio.algorithms.calo.policy_retirement import (
    PolicyRetirementManager,
    write_inventory,
    write_plan,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import TSH_CALO_ALGORITHM_ID
from calo_rpd_studio.gui.user_feedback import show_error
from calo_rpd_studio.gui.widgets.page_header import PageHeader
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.results.database import ResultDatabase


class CALOIntelligencePanel(ScrollablePage):
    """Manage policies without embedding or retaining a second training implementation."""

    stage_completed = pyqtSignal()
    independent_training_requested = pyqtSignal()

    def __init__(self, state, experiment_manager, model_library=None, parent=None) -> None:
        del experiment_manager
        content = QWidget()
        super().__init__(content, parent)
        self.state = state
        self.model_library = model_library
        self._policy_rows = []

        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 22)
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
        library_layout = QVBoxLayout(library)
        self.policy_table = QTableWidget(0, 5)
        self.policy_table.setHorizontalHeaderLabels(
            ("Active", "Policy", "Grade", "Scientific status", "Compatibility")
        )
        self.policy_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.policy_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.policy_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.policy_table.setAlternatingRowColors(True)
        self.policy_table.verticalHeader().setVisible(False)
        self.policy_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.policy_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (2, 3, 4):
            self.policy_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.policy_table.itemSelectionChanged.connect(self._policy_selection_changed)
        library_layout.addWidget(self.policy_table, 1)

        buttons = QHBoxLayout()
        self.policy_import_button = QPushButton("Import policy")
        self.policy_activate_button = QPushButton("Activate for experiments")
        self.policy_archive_button = QPushButton("Archive")
        self.policy_delete_button = QPushButton("Delete model files")
        self.policy_refresh_button = QPushButton("Refresh")
        self.show_archived_policies = QCheckBox("Show archived")
        self.policy_import_button.clicked.connect(self.import_policy)
        self.policy_activate_button.clicked.connect(self.activate_selected_policy)
        self.policy_archive_button.clicked.connect(self.archive_selected_policy)
        self.policy_delete_button.clicked.connect(self.prepare_policy_removal)
        self.policy_refresh_button.clicked.connect(self.refresh_policy_library)
        self.show_archived_policies.toggled.connect(lambda _checked: self.refresh_policy_library())
        for button in (
            self.policy_import_button,
            self.policy_activate_button,
            self.policy_archive_button,
            self.policy_delete_button,
            self.policy_refresh_button,
        ):
            buttons.addWidget(button)
        buttons.addWidget(self.show_archived_policies)
        buttons.addStretch(1)
        library_layout.addLayout(buttons)
        layout.addWidget(library, 1)

        controller = QGroupBox("Governing policy")
        self.policy_controller_group = controller
        form = QFormLayout(controller)
        self.path = QLineEdit()
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
                        "U",
                        (
                            "Training complete · import required"
                            if policy.get("policy_candidate")
                            else "Training complete · candidate unavailable"
                        ),
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
                        registered_policy.grade,
                        scientific_status,
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
                    policy.grade,
                    scientific_status,
                    compatibility,
                )
            for column, value in enumerate(values):
                self.policy_table.setItem(row, column, QTableWidgetItem(str(value)))
        if selected_key:
            self._select_row_key(selected_key)
        elif self._policy_rows:
            self.policy_table.selectRow(0)
        self._update_policy_gate_state()
        self.state.notify_policy_state_changed()

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
                self.policy_delete_button.setText("Delete model files")
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
                self.policy_delete_button.setText("Review removal")
                self.policy_delete_button.setToolTip(
                    "Registered policies use the evidence-backed retirement workflow; active "
                    "policies cannot be deleted."
                )
                self.policy_delete_button.setEnabled(True)
            self.path.setText(str(completed_training.get("campaign_id", "")))
            return
        policy = self._selected_policy()
        self.policy_import_button.setText("Import policy")
        self.policy_import_button.setEnabled(True)
        eligible = bool(policy and policy_record_user_status(policy) == "Eligible to select")
        self.policy_activate_button.setText(
            "Active governing policy"
            if policy is not None and policy.active
            else ("Activate for experiments" if eligible else "Qualification required")
        )
        self.policy_activate_button.setEnabled(eligible)
        self.policy_archive_button.setEnabled(policy is not None and not policy.active)
        self.policy_delete_button.setEnabled(policy is not None)
        self.policy_delete_button.setText("Review removal")
        self.policy_delete_button.setToolTip(
            "Registered policies use the evidence-backed retirement workflow; active policies "
            "cannot be deleted."
        )
        self.policy_archive_button.setText(
            "Restore archived" if policy is not None and policy.archived else "Archive"
        )
        if policy is not None:
            self.path.setText(policy.name)

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

    def prepare_policy_removal(self) -> None:
        completed_training = self._selected_completed_training()
        if completed_training is not None and self._selected_policy() is None:
            self._delete_completed_training(completed_training)
            return
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Save policy removal review",
            str(Path.cwd() / "policy-retirement.inventory.json"),
            "JSON evidence (*.json)",
        )
        if not selected:
            return
        try:
            store = Path(__file__).resolve().parents[2] / "data" / "trained_models"
            database = ResultDatabase(self.state.database.path, read_only=True)
            manager = PolicyRetirementManager(
                store, database, source_root=Path(__file__).resolve().parents[3]
            )
            inventory = manager.inventory()
            inventory_path = write_inventory(selected, inventory)
            plan_path = inventory_path.with_name(
                f"{inventory_path.stem.removesuffix('.inventory')}.plan.json"
            )
            write_plan(plan_path, manager.dry_run(inventory))
        except Exception as exc:
            show_error(
                self,
                "Removal plan could not be exported",
                "The read-only policy inventory could not be written.",
                exc,
                source="policy removal plan export",
            )
            return
        QMessageBox.information(
            self,
            "Removal review saved",
            "The policy removal review was saved. No policy was changed or deleted.",
        )

    def _delete_completed_training(self, completed_training: dict) -> None:
        if self.model_library is None:
            return
        directory_text = str(completed_training.get("directory", "")).strip()
        campaign_id = str(completed_training.get("campaign_id", "Completed model")).strip()
        if not directory_text:
            return
        try:
            directory = Path(directory_text).expanduser().resolve(strict=True)
            for policy in self.state.policy_registry.list(include_archived=True):
                checkpoint = Path(policy.checkpoint_path).expanduser().resolve()
                if checkpoint == directory or directory in checkpoint.parents:
                    raise ValueError(
                        "This completed campaign is registered in the policy library. Use the "
                        "reviewed policy-retirement workflow; model files were not deleted."
                    )
        except (OSError, RuntimeError, ValueError) as exc:
            show_error(
                self,
                "Completed model could not be deleted",
                "The exact completed-campaign target could not be verified.",
                exc,
                source="completed training deletion preflight",
            )
            return
        answer = QMessageBox.warning(
            self,
            "Permanently delete completed model files",
            f"Delete completed campaign {campaign_id!r} and every checkpoint, log, extension, "
            f"and model file inside this exact directory?\n\n{directory}\n\n"
            "This cannot be undone. No registered or active policy will be changed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            deleted = self.model_library.delete_completed_campaign(directory)
        except (OSError, RuntimeError, ValueError) as exc:
            show_error(
                self,
                "Completed model could not be deleted",
                "No verified completed-campaign deletion was accepted.",
                exc,
                source="completed training deletion",
            )
            return
        QMessageBox.information(
            self,
            "Completed model files deleted",
            f"The completed campaign directory was permanently deleted:\n{deleted}",
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
