"""Scientist-facing discovery, navigation, and cleanup of obsolete policy-model files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QPushButton

from calo_rpd_studio.gui.widgets.context_pane import TrainingPathEditor as _BaseTrainingPathEditor

from .independent_training_panel import TrainingModelLibrary as _BaseTrainingModelLibrary


class ObsoleteAwareTrainingModelLibrary(_BaseTrainingModelLibrary):
    """Extend saved-training discovery without changing resume/extension semantics."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._policy_library_focus_directory = ""

    @staticmethod
    def _saved_json_object(path: Path) -> tuple[dict | None, str]:
        if not path.is_file():
            return None, "missing"
        if path.is_symlink():
            return None, "symbolic-link"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"{type(exc).__name__}: {exc}"
        if not isinstance(payload, dict):
            return None, "saved JSON is not an object"
        return payload, ""

    @staticmethod
    def _saved_modified_ns(paths: tuple[Path, ...]) -> int:
        modified = 0
        for path in paths:
            try:
                modified = max(modified, path.lstat().st_mtime_ns)
            except OSError:
                continue
        return modified

    @staticmethod
    def _completed_obsolete_status(candidate_error: str) -> str:
        text = str(candidate_error).casefold()
        if "integrity" in text:
            return "Model integrity failed"
        if "could not be read" in text:
            return "Saved model unreadable"
        if "invalid" in text:
            return "Saved model invalid"
        if "unavailable" in text:
            return "Saved model incomplete"
        return "Saved model unavailable"

    def obsolete_campaigns(self) -> tuple[dict, ...]:
        """Return interrupted or mechanically unusable managed training directories.

        Healthy running campaigns and completed campaigns with a verified saved candidate are not
        obsolete. Interrupted campaigns are included because the scientist may explicitly discard
        their retained recovery files instead of resuming them.
        """

        saved = {
            str(Path(item["directory"]).expanduser().resolve()).casefold(): item
            for item in super().saved_campaigns()
        }
        records: dict[str, dict] = {}
        terminal_states = {"failed", "cancelled", "canceled", "aborted", "error"}
        for root in self.scan_locations():
            try:
                resolved_root = root.resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            for candidate in self._candidate_directories(root):
                plan_path = candidate / self.PLAN_FILE
                status_path = candidate / self.STATUS_FILE
                manifest_path = candidate / self.MANIFEST_FILE
                control_path = candidate / self.CONTROL_FILE
                marker_paths = (plan_path, status_path, manifest_path, control_path)
                if not any(path.exists() or path.is_symlink() for path in marker_paths):
                    continue
                try:
                    resolved = candidate.resolve(strict=True)
                except (OSError, RuntimeError, ValueError):
                    continue
                key = str(resolved).casefold()
                saved_record = saved.get(key)
                record: dict | None = None

                if saved_record is not None:
                    state = str(saved_record.get("state", "")).strip().lower()
                    if state == "interrupted":
                        resumable = bool(saved_record.get("resumable", False))
                        record = {
                            **saved_record,
                            "obsolete": True,
                            "obsolete_kind": "interrupted",
                            "obsolete_status": "Interrupted training",
                            "obsolete_reason": (
                                "A verified recovery point is retained and can still be resumed; "
                                "permanent deletion removes that recovery path."
                                if resumable
                                else "Training stopped without a verified safe resume point."
                            ),
                        }
                    elif state == "completed" and str(
                        saved_record.get("candidate_error", "")
                    ).strip():
                        candidate_error = str(saved_record.get("candidate_error", "")).strip()
                        record = {
                            **saved_record,
                            "obsolete": True,
                            "obsolete_kind": "completed_unusable",
                            "obsolete_status": self._completed_obsolete_status(candidate_error),
                            "obsolete_reason": candidate_error,
                        }
                else:
                    plan, plan_error = self._saved_json_object(plan_path)
                    status, status_error = self._saved_json_object(status_path)
                    campaign_id = (
                        str(plan.get("campaign_id", "")).strip()
                        if isinstance(plan, dict)
                        else ""
                    ) or candidate.name
                    state = (
                        str(status.get("state", "")).strip().lower()
                        if isinstance(status, dict)
                        else ""
                    )
                    obsolete_status = ""
                    obsolete_reason = ""
                    obsolete_kind = ""
                    if plan_error:
                        obsolete_kind = "incomplete_or_corrupt"
                        obsolete_status = (
                            "Incomplete training records"
                            if plan_error == "missing"
                            else "Corrupted training plan"
                        )
                        obsolete_reason = (
                            "The saved training plan is missing."
                            if plan_error == "missing"
                            else f"The saved training plan could not be read ({plan_error})."
                        )
                    elif status_error:
                        obsolete_kind = "incomplete_or_corrupt"
                        obsolete_status = (
                            "Incomplete training records"
                            if status_error == "missing"
                            else "Corrupted training status"
                        )
                        obsolete_reason = (
                            "The saved training status is missing."
                            if status_error == "missing"
                            else f"The saved training status could not be read ({status_error})."
                        )
                    elif state in terminal_states:
                        obsolete_kind = "failed"
                        obsolete_status = (
                            "Failed training" if state in {"failed", "error"} else "Stopped training"
                        )
                        failure = status.get("failure") if isinstance(status, dict) else None
                        failure_message = ""
                        if isinstance(failure, dict):
                            failure_message = str(
                                failure.get("message", "") or failure.get("detail", "")
                            ).strip()
                        obsolete_reason = (
                            failure_message or f"The saved campaign ended with state {state!r}."
                        )
                    elif state and state not in self.DISCOVERABLE_STATES:
                        obsolete_kind = "unrecognized_state"
                        obsolete_status = "Unrecognized saved training"
                        obsolete_reason = (
                            f"The saved campaign state {state!r} is not a current resumable or "
                            "completed training state."
                        )

                    if obsolete_status:
                        record = {
                            "campaign_id": campaign_id,
                            "state": state or "unreadable",
                            "directory": str(resolved),
                            "plan": str(plan_path.resolve()) if plan_path.exists() else "",
                            "resumable": False,
                            "policy_candidate": "",
                            "candidate_error": obsolete_reason,
                            "training_evaluations": None,
                            "extendable": False,
                            "extension_error": "",
                            "extension_pending": False,
                            "progress": (
                                dict(status.get("progress", {}) or {})
                                if isinstance(status, dict)
                                else {}
                            ),
                            "pause": (
                                dict(status.get("pause", {}) or {})
                                if isinstance(status, dict)
                                else {}
                            ),
                            "obsolete": True,
                            "obsolete_kind": obsolete_kind,
                            "obsolete_status": obsolete_status,
                            "obsolete_reason": obsolete_reason,
                            "modified_ns": self._saved_modified_ns(marker_paths),
                        }

                if record is None:
                    continue
                blocker = ""
                if candidate.is_symlink():
                    blocker = "Symbolic-link campaign directories cannot be deleted."
                elif resolved == resolved_root:
                    blocker = "The configured model-library root itself cannot be deleted."
                elif resolved_root not in resolved.parents:
                    blocker = "The obsolete campaign is outside its configured model-library root."
                records[key] = {
                    **record,
                    "directory": str(resolved),
                    "deletion_blocker": blocker,
                    "deletable": not bool(blocker),
                    "modified_ns": int(
                        record.get("modified_ns", 0) or self._saved_modified_ns(marker_paths)
                    ),
                }

        return tuple(
            sorted(
                records.values(),
                key=lambda item: (
                    -int(item.get("modified_ns", 0)),
                    str(item.get("campaign_id", "")).casefold(),
                    str(item.get("directory", "")).casefold(),
                ),
            )
        )

    def validate_obsolete_campaign_deletion(self, directory: str | Path) -> Path:
        """Return one exact obsolete managed campaign target without changing it."""

        requested = Path(directory).expanduser()
        if requested.is_symlink():
            raise ValueError("A symbolic-link campaign directory cannot be deleted")
        target = requested.resolve(strict=True)
        if not target.is_dir():
            raise ValueError("The selected obsolete campaign directory is unavailable")
        roots = tuple(root.resolve() for root in self.scan_locations())
        if not any(root in target.parents for root in roots):
            raise ValueError(
                "The selected obsolete campaign must be a child of a configured model-library location"
            )
        known = {
            str(Path(item["directory"]).resolve()).casefold(): item
            for item in self.obsolete_campaigns()
        }
        record = known.get(str(target).casefold())
        if record is None:
            raise ValueError(
                "The selected directory is no longer classified as an obsolete saved campaign"
            )
        blocker = str(record.get("deletion_blocker", "")).strip()
        if blocker:
            raise ValueError(blocker)
        if str(record.get("state", "")).strip().lower() == "running":
            raise ValueError("A running training campaign cannot be deleted")
        if not any(
            (target / name).exists() or (target / name).is_symlink()
            for name in (self.PLAN_FILE, self.STATUS_FILE, self.MANIFEST_FILE, self.CONTROL_FILE)
        ):
            raise ValueError("The selected directory no longer contains CALO training files")
        return target

    def delete_obsolete_campaign(self, directory: str | Path) -> Path:
        """Permanently remove one exact, revalidated obsolete training directory."""

        target = self.validate_obsolete_campaign_deletion(directory)
        shutil.rmtree(target)
        self._candidate_integrity_cache.clear()
        self.changed.emit()
        return target

    def request_policy_library_focus(self, directory: str | Path) -> Path:
        """Retain one exact saved campaign for the next Policy-library navigation."""

        target = Path(directory).expanduser().resolve(strict=True)
        known = {
            str(Path(item["directory"]).expanduser().resolve()).casefold()
            for item in (*super().saved_campaigns(), *self.obsolete_campaigns())
        }
        if str(target).casefold() not in known:
            raise ValueError("The selected saved training directory is no longer in the model library")
        self._policy_library_focus_directory = str(target)
        return target

    def policy_library_focus_request(self) -> str:
        return str(self._policy_library_focus_directory)

    def clear_policy_library_focus_request(self) -> None:
        self._policy_library_focus_directory = ""


class SavedTrainingManagementEditor(_BaseTrainingPathEditor):
    """Route saved-training file management to the central Policy library."""

    open_requested = pyqtSignal(str)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.manage_saved_button = QPushButton("View / delete files")
        self.manage_saved_button.setAccessibleName("View or delete selected saved training files")
        self.manage_saved_button.setToolTip("Select a saved training run first.")
        self.manage_saved_button.clicked.connect(self._open_selected_saved_artifact)
        self.manage_saved_button.setEnabled(False)
        host = self.refresh_library_button.parentWidget()
        host_layout = host.layout() if host is not None else None
        button_row = host_layout.itemAt(1).layout() if host_layout is not None else None
        if button_row is not None:
            button_row.insertWidget(1, self.manage_saved_button)
        elif host_layout is not None:
            host_layout.addWidget(self.manage_saved_button)
        self._update_saved_management_button()

    def refresh_model_library(self, preferred_root=None) -> None:
        super().refresh_model_library(preferred_root)
        self._update_saved_management_button()

    def _library_selection_changed(self, index: int = -1) -> None:
        super()._library_selection_changed(index)
        self._update_saved_management_button()

    def _update_saved_management_button(self) -> None:
        if not hasattr(self, "manage_saved_button"):
            return
        record = self.library_picker.currentData()
        available = bool(isinstance(record, dict) and record.get("directory"))
        self.manage_saved_button.setEnabled(available)
        self.manage_saved_button.setToolTip(
            "Open the selected saved run in the Policy library, where its files can be inspected "
            "and permanently deleted after confirmation."
            if available
            else "Select a saved training run first."
        )

    def _open_selected_saved_artifact(self) -> None:
        record = self.library_picker.currentData()
        if not isinstance(record, dict) or not record.get("directory"):
            return
        request = getattr(self.model_library, "request_policy_library_focus", None)
        if not callable(request):
            return
        request(str(record["directory"]))
        self.open_requested.emit("calo_intelligence")
