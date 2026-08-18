"""Obsolete-model visibility and deletion controls for CALO Intelligence."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QCheckBox, QMessageBox, QTableWidgetItem

from calo_rpd_studio.ai.model_io import checkpoint_sha256
from calo_rpd_studio.algorithms.calo.policy_artifact_deletion import (
    record_permanent_artifact_deletion,
)
from calo_rpd_studio.gui.user_feedback import show_error

from .calo_intelligence_policy_controls import ScientistCALOIntelligencePanel


class ObsoleteAwareCALOIntelligencePanel(ScientistCALOIntelligencePanel):
    """Keep cleanup-only artifacts hidden until a scientist explicitly asks to see them."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.show_obsolete_models = QCheckBox("Show obsolete models")
        self.show_obsolete_models.setAccessibleName("Show obsolete policy models")
        self.show_obsolete_models.setToolTip(
            "Show interrupted, failed, incomplete or corrupt saved training and completed "
            "models whose saved artifact can no longer be verified."
        )
        layout = self.policy_center_group.layout()
        if layout is not None:
            layout.insertWidget(0, self.show_obsolete_models)
        self.show_obsolete_models.toggled.connect(self.refresh_policy_library)
        self.refresh_policy_library()

    def _obsolete_campaigns(self) -> tuple[dict, ...]:
        getter = getattr(self.model_library, "obsolete_campaigns", None)
        if not callable(getter):
            return ()
        try:
            return tuple(getter())
        except (OSError, RuntimeError, ValueError):
            return ()

    @staticmethod
    def _campaign_key(record: dict) -> str:
        try:
            return str(Path(str(record.get("directory", ""))).expanduser().resolve()).casefold()
        except (OSError, RuntimeError, ValueError):
            return str(record.get("directory", "")).casefold()

    def _registered_policies_inside(self, directory: Path) -> list:
        contained = []
        for policy in self.state.policy_registry.list(include_archived=True):
            try:
                checkpoint = Path(policy.checkpoint_path).expanduser().resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            if checkpoint == directory or directory in checkpoint.parents:
                contained.append(policy)
        return contained

    def refresh_policy_library(self) -> None:
        selected_key = self._row_key(self._selected_row())
        super().refresh_policy_library()
        obsolete = list(self._obsolete_campaigns())
        obsolete_by_key = {self._campaign_key(item): item for item in obsolete}

        focus = ""
        request = getattr(self.model_library, "policy_library_focus_request", None)
        if callable(request):
            focus = str(request() or "")
        try:
            focus_key = str(Path(focus).expanduser().resolve()).casefold() if focus else ""
        except (OSError, RuntimeError, ValueError):
            focus_key = ""

        show = bool(
            getattr(self, "show_obsolete_models", None)
            and self.show_obsolete_models.isChecked()
        )
        if focus_key and focus_key in obsolete_by_key and hasattr(self, "show_obsolete_models"):
            self.show_obsolete_models.blockSignals(True)
            self.show_obsolete_models.setChecked(True)
            self.show_obsolete_models.blockSignals(False)
            show = True

        snapshots = {}
        for row, entry in enumerate(self._policy_rows):
            snapshots[self._row_key(entry)] = tuple(
                (
                    self.policy_table.item(row, column).text(),
                    self.policy_table.item(row, column).toolTip(),
                )
                if self.policy_table.item(row, column) is not None
                else ("", "")
                for column in range(self.policy_table.columnCount())
            )

        registered = self.state.policy_registry.list(include_archived=False)
        obsolete_policy_ids: set[str] = set()
        obsolete_rows: list[dict] = []
        for record in obsolete:
            try:
                directory = Path(str(record.get("directory", ""))).expanduser().resolve()
            except (OSError, RuntimeError, ValueError):
                directory = Path(str(record.get("directory", "")))
            contained = []
            for policy in registered:
                try:
                    checkpoint = Path(policy.checkpoint_path).expanduser().resolve()
                except (OSError, RuntimeError, ValueError):
                    continue
                if checkpoint == directory or directory in checkpoint.parents:
                    contained.append(policy)
            policy = contained[0] if len(contained) == 1 else None
            if policy is not None:
                obsolete_policy_ids.add(policy.id)
            obsolete_rows.append(
                {
                    **record,
                    "row_kind": "obsolete_training",
                    "registered_policy": policy,
                    "contained_policy_count": len(contained),
                }
            )

        normal_rows = []
        for entry in self._policy_rows:
            if isinstance(entry, dict) and self._campaign_key(entry) in obsolete_by_key:
                continue
            if not isinstance(entry, dict) and getattr(entry, "id", "") in obsolete_policy_ids:
                continue
            normal_rows.append(entry)
        self._policy_rows = normal_rows + (obsolete_rows if show else [])
        self.policy_table.setRowCount(len(self._policy_rows))

        for row, entry in enumerate(self._policy_rows):
            if isinstance(entry, dict) and entry.get("row_kind") == "obsolete_training":
                policy = entry.get("registered_policy")
                retained = bool(
                    policy
                    and policy.qualification_status
                    in {"assessed", "scientist_selected", "qualified"}
                )
                values = (
                    "",
                    str(entry.get("campaign_id", getattr(policy, "name", "Obsolete model"))),
                    self._training_evaluation_text(entry.get("training_evaluations")),
                    "Retained history" if retained else "Not assessed",
                    str(entry.get("obsolete_status", "Obsolete model files")),
                    "Retained history" if retained else "Training files only",
                    "Not usable",
                )
                reason = str(entry.get("obsolete_reason", "")).strip()
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setToolTip(reason)
                    self.policy_table.setItem(row, column, item)
            else:
                snapshot = snapshots.get(self._row_key(entry), ())
                for column in range(self.policy_table.columnCount()):
                    text, tooltip = snapshot[column] if column < len(snapshot) else ("", "")
                    item = QTableWidgetItem(text)
                    item.setToolTip(tooltip)
                    self.policy_table.setItem(row, column, item)

        self.policy_table.horizontalHeaderItem(0).setText("Use status")
        self._resize_policy_table_to_entries()
        target_key = f"training:{focus}" if focus else selected_key
        if target_key:
            self._select_row_key(target_key)
        elif self._policy_rows:
            self.policy_table.selectRow(0)
        if focus:
            clear = getattr(self.model_library, "clear_policy_library_focus_request", None)
            if callable(clear):
                clear()
        self._policy_selection_changed()
        self._update_policy_gate_state()

    def _selected_obsolete_training(self) -> dict | None:
        selected = self._selected_row()
        return (
            selected
            if isinstance(selected, dict) and selected.get("row_kind") == "obsolete_training"
            else None
        )

    def _selected_completed_training(self) -> dict | None:
        selected = self._selected_row()
        return (
            selected
            if isinstance(selected, dict) and selected.get("row_kind") == "completed_training"
            else None
        )

    def _policy_selection_changed(self) -> None:
        obsolete = self._selected_obsolete_training()
        if obsolete is None:
            super()._policy_selection_changed()
            return
        reason = str(obsolete.get("obsolete_reason", "")).strip()
        status = str(obsolete.get("obsolete_status", "Obsolete model files")).strip()
        blocker = str(obsolete.get("deletion_blocker", "")).strip()
        if bool(getattr(self.state, "policy_training_active", False)):
            blocker = "Stop or safely pause policy training before deleting saved training files."
        if int(obsolete.get("contained_policy_count", 0) or 0) > 1:
            blocker = (
                "This training directory contains more than one registered policy. Delete those "
                "models individually before deleting the directory."
            )
        self.policy_import_button.setText("Obsolete files")
        self.policy_import_button.setEnabled(False)
        self.qualification_button.setText("Assessment unavailable")
        self.qualification_button.setEnabled(False)
        self.qualification_resume_button.setVisible(False)
        self.policy_select_button.setText("Not available for use")
        self.policy_select_button.setEnabled(False)
        self.policy_activate_button.setVisible(False)
        self.policy_activate_button.setEnabled(False)
        self.policy_delete_button.setEnabled(not bool(blocker))
        self.policy_delete_button.setToolTip(
            blocker
            or "Permanently delete this exact obsolete training directory after an irreversible warning."
        )
        self.qualification_workflow_status.setText(
            f"{status}: {reason} These files cannot govern experiments. "
            "Deletion is a scientist-controlled cleanup action."
        )
        self._refresh_feasibility_and_influence()
        self._update_policy_gate_state()

    def delete_selected_model_files(self) -> None:
        obsolete = self._selected_obsolete_training()
        if obsolete is not None:
            self._delete_obsolete_training(obsolete)
            return
        super().delete_selected_model_files()

    def _delete_standalone_policy_file(self, policy) -> None:
        """Allow exact deletion when a registered model file itself no longer verifies."""
        source = Path(policy.checkpoint_path).expanduser()
        try:
            if source.is_symlink():
                raise ValueError("Symbolic-link model targets cannot be permanently deleted.")
            checkpoint = source.resolve(strict=True)
            if not checkpoint.is_file():
                raise ValueError("The selected model target is not a regular file.")
            observed_sha256 = checkpoint_sha256(checkpoint)
            verification_error = ""
            try:
                inspected = self.state.policy_registry.inspect_checkpoint(checkpoint)
                if str(inspected.get("sha256", "")).lower() != policy.sha256.lower():
                    verification_error = "The model file no longer matches its registered SHA-256."
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                verification_error = str(exc) or type(exc).__name__
        except (OSError, RuntimeError, ValueError) as exc:
            show_error(
                self,
                "Model could not be deleted",
                "The exact standalone model target could not be verified as a regular file.",
                exc,
                source="standalone obsolete-model deletion preflight",
            )
            return
        if not verification_error:
            super()._delete_standalone_policy_file(policy)
            return
        answer = QMessageBox.warning(
            self,
            "Permanently delete unverified model file",
            f"Permanently delete {policy.name!r}?\n\n{checkpoint}\n\n"
            f"CALO cannot verify this file as the registered model: {verification_error}\n\n"
            "The exact file will be permanently removed. Its live policy will be deactivated "
            "and archived; retained assessment, lineage, and experiment records remain historical. "
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        artifact_deleted = False
        try:
            if checkpoint_sha256(checkpoint) != observed_sha256:
                raise RuntimeError("The unverified model file changed while confirmation was open")
            checkpoint.unlink()
            artifact_deleted = True
            record_permanent_artifact_deletion(
                self.state.policy_registry,
                policy.id,
                expected_sha256=policy.sha256,
                reason="scientist_permanent_unverified_model_deletion",
                deleted_scope="unverified_standalone_model_file",
            )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            self.state.notify_policy_state_changed()
            self.refresh_policy_library()
            show_error(
                self,
                "Model deletion needs attention",
                "The model file was deleted but retained policy history needs review."
                if artifact_deleted
                else "No model file was deleted.",
                exc,
                source="standalone unverified-model permanent deletion",
            )
            return
        self.state.notify_policy_state_changed()
        self.refresh_policy_library()
        QMessageBox.information(
            self,
            "Unverified model file permanently deleted",
            f"The exact model file was permanently deleted:\n{checkpoint}",
        )

    def _delete_obsolete_training(self, obsolete: dict) -> None:
        if self.model_library is None:
            return
        if bool(getattr(self.state, "policy_training_active", False)):
            return
        directory_text = str(obsolete.get("directory", "")).strip()
        if not directory_text:
            return
        try:
            validator = getattr(self.model_library, "validate_obsolete_campaign_deletion")
            directory = validator(directory_text)
            contained = self._registered_policies_inside(directory)
            if len(contained) > 1:
                raise ValueError(
                    "This obsolete training directory contains more than one registered policy."
                )
            registered_policy = contained[0] if contained else None
        except (AttributeError, OSError, RuntimeError, ValueError) as exc:
            show_error(
                self,
                "Obsolete files could not be deleted",
                "The exact obsolete training target could not be verified.",
                exc,
                source="obsolete training deletion preflight",
            )
            return
        campaign_id = str(obsolete.get("campaign_id", directory.name)).strip() or directory.name
        status = str(obsolete.get("obsolete_status", "Obsolete model files")).strip()
        reason = str(obsolete.get("obsolete_reason", "")).strip()
        resume_warning = (
            "This interrupted run has a verified recovery point. Deletion permanently removes "
            "that recovery point and the run can no longer be resumed.\n\n"
            if bool(obsolete.get("resumable", False))
            else ""
        )
        policy_warning = (
            "Its registered policy will be deactivated and archived; retained scientific history "
            "remains non-executable.\n\n"
            if registered_policy is not None
            else ""
        )
        answer = QMessageBox.warning(
            self,
            "Permanently delete obsolete model files",
            f"Permanently delete {campaign_id!r}?\n\n{directory}\n\n"
            f"Status: {status}\n{reason}\n\n"
            + resume_warning
            + policy_warning
            + "Every checkpoint, recovery file, partial model, log, extension, and other file in "
            "this exact training directory will be permanently removed. This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        artifact_deleted = False
        try:
            directory = validator(directory_text)
            contained_now = self._registered_policies_inside(directory)
            if [item.id for item in contained_now] != [item.id for item in contained]:
                raise RuntimeError("Registered policy references changed while confirmation was open")
            deleted = self.model_library.delete_obsolete_campaign(directory)
            artifact_deleted = True
            if registered_policy is not None and not registered_policy.archived:
                record_permanent_artifact_deletion(
                    self.state.policy_registry,
                    registered_policy.id,
                    expected_sha256=registered_policy.sha256,
                    reason="scientist_permanent_obsolete_training_deletion",
                    deleted_scope="obsolete_training_campaign",
                )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            self.state.notify_policy_state_changed()
            self.refresh_policy_library()
            show_error(
                self,
                "Obsolete model deletion needs attention",
                "The files were deleted but retained policy history needs review."
                if artifact_deleted and registered_policy is not None
                else "No obsolete training directory was deleted.",
                exc,
                source="obsolete training permanent deletion",
            )
            return
        self.state.notify_policy_state_changed()
        self.refresh_policy_library()
        QMessageBox.information(
            self,
            "Obsolete model files permanently deleted",
            f"The obsolete training directory was permanently deleted:\n{deleted}",
        )
