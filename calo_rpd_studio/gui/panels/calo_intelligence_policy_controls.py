"""Scientist-owned policy use-status and permanent model deletion controls."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QMessageBox, QTableWidgetItem

from calo_rpd_studio.algorithms.calo.policy_artifact_deletion import (
    permanent_artifact_deletion_blocker,
    record_permanent_artifact_deletion,
)
from calo_rpd_studio.algorithms.calo.policy_readiness import governing_policy_user_message
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import TSH_CALO_ALGORITHM_ID
from calo_rpd_studio.gui.user_feedback import show_error

from .calo_intelligence_panel import CALOIntelligencePanel as _BaseCALOIntelligencePanel


class ScientistCALOIntelligencePanel(_BaseCALOIntelligencePanel):
    """Present governing truthfully and let the scientist delete an exact model artifact."""

    @staticmethod
    def _policy_use_status(policy, governing) -> str:
        if policy is None:
            return ""
        if governing.ready and governing.policy_id == policy.id:
            return "Governing"
        if policy.active:
            return "Not governing"
        return ""

    def _configure_policy_activation_button(self, policy) -> None:
        governing = self.state.governing_policy_status()
        is_governing = bool(
            policy is not None and governing.ready and governing.policy_id == policy.id
        )
        if is_governing:
            self.policy_activate_button.setVisible(True)
            self.policy_activate_button.setText("Active governing policy")
            self.policy_activate_button.setEnabled(False)
            self.policy_activate_button.setToolTip(
                "This verified policy is currently governing new TSH-CALO experiments."
            )
            return
        if policy is not None and policy.active:
            self.policy_activate_button.setVisible(True)
            self.policy_activate_button.setText("Not currently governing")
            self.policy_activate_button.setEnabled(False)
            self.policy_activate_button.setToolTip(governing_policy_user_message(governing))
            return
        eligible = bool(
            policy
            and policy.usable
            and policy.compatible_with(TSH_CALO_ALGORITHM_ID)
            and policy.qualification_status in {"scientist_selected", "qualified"}
        )
        self.policy_activate_button.setVisible(eligible)
        self.policy_activate_button.setText("Activate for experiments")
        self.policy_activate_button.setEnabled(eligible)
        self.policy_activate_button.setToolTip(
            "" if eligible else "Select an assessed model for use before activation."
        )

    def refresh_policy_library(self) -> None:
        super().refresh_policy_library()
        self.policy_table.horizontalHeaderItem(0).setText("Use status")
        governing = self.state.governing_policy_status()
        for row, entry in enumerate(self._policy_rows):
            policy = entry.get("registered_policy") if isinstance(entry, dict) else entry
            if policy is None:
                continue
            self.policy_table.setItem(
                row, 0, QTableWidgetItem(self._policy_use_status(policy, governing))
            )
            if not policy.compatible_with(TSH_CALO_ALGORITHM_ID):
                self.policy_table.setItem(row, 6, QTableWidgetItem("Not compatible"))
        self._resize_policy_table_to_entries()

    def _policy_selection_changed(self) -> None:
        super()._policy_selection_changed()
        completed_training = self._selected_completed_training()
        policy = self._selected_policy()
        self._configure_policy_activation_button(policy)

        if completed_training is not None and policy is None:
            # The base panel already owns the enablement rules for an unregistered campaign.
            self._update_policy_gate_state()
            return

        deletion_blocker = "Select a model to delete."
        if policy is not None:
            try:
                deletion_blocker = permanent_artifact_deletion_blocker(policy)
                if completed_training is not None and not deletion_blocker:
                    deletion_blocker = self._registered_completed_deletion_blocker(
                        completed_training, policy
                    )
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                deletion_blocker = str(exc) or "The selected model cannot be deleted."
        self.policy_delete_button.setEnabled(policy is not None and not deletion_blocker)
        self.policy_delete_button.setToolTip(
            deletion_blocker
            or (
                "Permanently delete the exact model/training files after an irreversible "
                "confirmation. Historical assessment and experiment records remain retained."
                if completed_training is not None
                else "Permanently delete this exact model file after an irreversible confirmation. "
                "Historical assessment and experiment records remain retained."
            )
        )
        self._update_policy_gate_state()

    def _qualification_candidate_blocker(self, policy) -> str:
        if policy is None:
            return "Import and select a policy first."
        if policy.archived:
            return "Restore the archived policy first."
        if not policy.usable:
            return "The selected policy model file is unavailable."
        if not policy.compatible_with(TSH_CALO_ALGORITHM_ID):
            return "The selected policy is not compatible with the frozen TSH-CALO architecture."
        if policy.qualification_status in {"assessed", "scientist_selected", "qualified"}:
            return "A complete assessment is already admitted for this policy."
        if policy.active:
            return "The active policy cannot enter a new qualification workflow."
        if int(policy.metadata.get("ensemble_size", 1)) < 2:
            return "Formal qualification requires an epistemic ensemble policy."
        return ""

    def _registered_completed_deletion_blocker(self, completed_training: dict, policy) -> str:
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
            return permanent_artifact_deletion_blocker(policy)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            return str(exc) or "The selected model cannot be deleted."

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
            self.policy_gate_status.setText(governing_policy_user_message(status))
            self.path.clear()
            self.deterministic.setEnabled(False)
            self.apply_policy_button.setEnabled(False)
            return
        self.policy_gate_status.setText(
            f"Selected by scientist: {active.name} · integrity verified · Ready for experiments"
        )
        self.path.setText(active.name)
        self.deterministic.setEnabled(True)
        self.apply_policy_button.setEnabled(True)

    def _delete_standalone_policy_file(self, policy) -> None:
        source = Path(policy.checkpoint_path).expanduser()
        try:
            blocker = permanent_artifact_deletion_blocker(policy)
            if blocker:
                raise PermissionError(blocker)
            checkpoint = source.resolve(strict=True)
            inspected = self.state.policy_registry.inspect_checkpoint(checkpoint)
            if str(inspected.get("sha256", "")).lower() != policy.sha256.lower():
                raise RuntimeError("Model checksum changed after registration")
            direct_removal = not bool(
                self.state.policy_registry.unqualified_candidate_removal_blocker(policy.id)
            )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            show_error(
                self,
                "Model could not be deleted",
                "The exact standalone model target could not be verified.",
                exc,
                source="standalone policy deletion preflight",
            )
            return

        retained = []
        if policy.active:
            retained.append(
                "This model is recorded as active. Deletion immediately deactivates it; if it is "
                "currently governing, new experiments return to the policy-free safe path."
            )
        if policy.qualification_status in {"assessed", "scientist_selected", "qualified"} or (
            self.state.database.list_policy_qualifications(policy.id)
        ):
            retained.append(
                "Retained assessment/qualification records stay in history as non-executable "
                "scientific provenance."
            )
        reference_count = self.state.database.policy_reference_count(policy.id, policy.sha256)
        if reference_count:
            retained.append(
                f"{reference_count} retained experiment binding(s) keep the policy identity and "
                "checksum for historical provenance."
            )
        if self.state.database.get_policy_checkpoint_by_sha256(policy.sha256) is not None:
            retained.append(
                "A retained training-lineage checkpoint references this checksum. The lineage "
                "record remains historical but cannot load the deleted model artifact."
            )
        retained_text = "\n".join(retained)
        if retained_text:
            retained_text += "\n\n"
        answer = QMessageBox.warning(
            self,
            "Permanently delete model file",
            f"Permanently delete {policy.name!r}?\n\n{checkpoint}\n\n"
            + retained_text
            + "The model file itself will be permanently removed and cannot be restored by CALO. "
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        artifact_deleted = False
        registration_removed = False
        try:
            inspected = self.state.policy_registry.inspect_checkpoint(checkpoint)
            if str(inspected.get("sha256", "")).lower() != policy.sha256.lower():
                raise RuntimeError("Model checksum changed immediately before permanent deletion")
            if direct_removal:
                self.state.policy_registry.remove_unqualified_candidate(
                    policy.id, reason="scientist_permanent_standalone_model_deletion"
                )
                registration_removed = True
            checkpoint.unlink()
            artifact_deleted = True
            if not direct_removal:
                record_permanent_artifact_deletion(
                    self.state.policy_registry,
                    policy.id,
                    expected_sha256=policy.sha256,
                    reason="scientist_permanent_standalone_model_deletion",
                    deleted_scope="standalone_model_file",
                )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            self.state.notify_policy_state_changed()
            self.refresh_policy_library()
            show_error(
                self,
                "Model deletion needs attention",
                (
                    "The model file was permanently deleted, but its retained policy history could "
                    "not be archived automatically. The missing artifact cannot govern new "
                    "experiments; review the retained library record."
                    if artifact_deleted and not direct_removal
                    else (
                        "The unused policy registration was removed, but the model file remains at "
                        "the shown path and must be reviewed manually."
                        if registration_removed and not artifact_deleted
                        else "No model file was deleted."
                    )
                ),
                exc,
                source="standalone policy permanent deletion",
            )
            return
        self.state.notify_policy_state_changed()
        self.refresh_policy_library()
        QMessageBox.information(
            self,
            "Model permanently deleted",
            (
                f"The exact unused model file and its removable registration were permanently "
                f"deleted:\n{checkpoint}"
                if direct_removal
                else (
                    f"The exact model file was permanently deleted:\n{checkpoint}\n\n"
                    "Its live policy was deactivated and archived. Retained assessment and "
                    "experiment records remain historical and cannot execute the deleted model."
                )
            ),
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
        checkpoint = None
        direct_removal = False
        try:
            directory = self.model_library.validate_completed_campaign_deletion(directory_text)
            for other_policy in self.state.policy_registry.list(include_archived=True):
                other_checkpoint = Path(other_policy.checkpoint_path).expanduser().resolve()
                if (other_checkpoint == directory or directory in other_checkpoint.parents) and (
                    registered_policy is None or other_policy.id != registered_policy.id
                ):
                    raise ValueError(
                        "This completed campaign contains another registered policy. Select and "
                        "delete that policy separately before deleting the campaign."
                    )
            if registered_policy is not None:
                blocker = permanent_artifact_deletion_blocker(registered_policy)
                if blocker:
                    raise PermissionError(blocker)
                checkpoint = Path(registered_policy.checkpoint_path).expanduser().resolve(strict=True)
                if directory not in checkpoint.parents:
                    raise ValueError(
                        "The selected registered policy is not contained by this campaign."
                    )
                inspected = self.state.policy_registry.inspect_checkpoint(checkpoint)
                if str(inspected.get("sha256", "")).lower() != registered_policy.sha256.lower():
                    raise RuntimeError("Registered model checksum changed after selection")
                direct_removal = not bool(
                    self.state.policy_registry.unqualified_candidate_removal_blocker(
                        registered_policy.id
                    )
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

        if registered_policy is None:
            confirmation_scope = "No registered policy is attached to this campaign."
        elif direct_removal:
            confirmation_scope = (
                "Its unused policy registration will also be removed. No retained assessment or "
                "experiment record depends on that registration."
            )
        else:
            confirmation_scope = (
                "The selected policy will be deactivated and cannot govern new experiments. Its retained "
                "lineage, assessment/qualification, and prior experiment records remain as "
                "non-executable historical provenance."
            )
        answer = QMessageBox.warning(
            self,
            "Permanently delete completed model files",
            f"Permanently delete completed campaign {campaign_id!r}?\n\n{directory}\n\n"
            "Every checkpoint, training log, extension, and model file inside this exact campaign "
            "directory will be permanently removed.\n\n"
            + confirmation_scope
            + "\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        artifact_deleted = False
        registration_removed = False
        try:
            if registered_policy is not None:
                assert checkpoint is not None
                inspected = self.state.policy_registry.inspect_checkpoint(checkpoint)
                if str(inspected.get("sha256", "")).lower() != registered_policy.sha256.lower():
                    raise RuntimeError(
                        "Registered model checksum changed immediately before permanent deletion"
                    )
                if direct_removal:
                    self.state.policy_registry.remove_unqualified_candidate(
                        registered_policy.id,
                        reason="scientist_permanent_completed_campaign_deletion",
                    )
                    registration_removed = True
            deleted = self.model_library.delete_completed_campaign(directory)
            artifact_deleted = True
            if registered_policy is not None and not direct_removal:
                record_permanent_artifact_deletion(
                    self.state.policy_registry,
                    registered_policy.id,
                    expected_sha256=registered_policy.sha256,
                    reason="scientist_permanent_completed_campaign_deletion",
                    deleted_scope="completed_training_campaign",
                )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            self.state.notify_policy_state_changed()
            self.refresh_policy_library()
            show_error(
                self,
                "Completed model deletion needs attention",
                (
                    "The completed campaign files were permanently deleted, but the retained policy "
                    "history could not be archived automatically. The missing artifact cannot "
                    "govern new experiments; review the retained library record."
                    if artifact_deleted and registered_policy is not None and not direct_removal
                    else (
                        "The unused policy registration was removed, but the campaign files remain "
                        "and can be retried from the refreshed unregistered row."
                        if registration_removed and not artifact_deleted
                        else "No verified completed-campaign deletion was accepted."
                    )
                ),
                exc,
                source="completed training permanent deletion",
            )
            return
        self.state.notify_policy_state_changed()
        self.refresh_policy_library()
        QMessageBox.information(
            self,
            "Completed model files permanently deleted",
            (
                f"The completed campaign directory was permanently deleted:\n{deleted}\n\n"
                "The policy was deactivated and archived. Retained assessment and experiment "
                "records remain historical and cannot execute the deleted model."
                if registered_policy is not None and not direct_removal
                else f"The completed campaign directory was permanently deleted:\n{deleted}"
            ),
        )
