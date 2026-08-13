"""Explicit GUI adapter for the independent new-policy plan/check/start command."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.user_feedback import show_error
from calo_rpd_studio.gui.widgets.page_header import PageHeader


class TrainingLaunchModel(QObject):
    changed = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.values = {
            "architecture": "tsh_calo",
            "plan": "",
            "output": "",
        }
        self.plan_payload: dict | None = None
        self.plan_error = ""

    @staticmethod
    def _current_source_commit() -> str:
        """Return the authenticated application source identity used by fresh candidates."""
        from calo_rpd_studio.compute.source_identity import resolve_source_identity

        source_identity = resolve_source_identity()
        source_commit = str(source_identity.source_commit).strip().lower()
        if len(source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in source_commit
        ):
            raise ValueError("Training requires an identifiable application source build")
        return source_commit

    def set_value(self, key: str, value: str) -> None:
        if key not in self.values:
            raise KeyError(f"Unknown scientist-facing training input: {key}")
        normalized = str(value).strip()
        if self.values.get(key) == normalized:
            return
        self.values[key] = normalized
        if key == "plan":
            self.plan_payload = None
            self.plan_error = ""
        self.changed.emit(dict(self.values))

    def load_plan(self) -> None:
        """Load and validate the selected plan without starting policy work."""
        source = Path(self.values.get("plan", "")).expanduser()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Training plan must be a JSON object")
            from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
                TSHCALOTrainingCampaignPlan,
            )

            plan = TSHCALOTrainingCampaignPlan.from_dict(payload)
            # Imported plans supply scientific settings only; they cannot confer
            # lifecycle or release authority through scientist-selected files.
            normalized = plan.to_dict()
            normalized["source_commit"] = self._current_source_commit()
            normalized["development_freeze_commit"] = ""
            normalized["development_freeze_sha256"] = ""
            normalized["phase4_acceptance_sha256"] = ""
            plan = TSHCALOTrainingCampaignPlan.from_dict(normalized)
        except (OSError, RuntimeError, json.JSONDecodeError, ValueError) as exc:
            self.plan_payload = None
            self.plan_error = str(exc)
        else:
            self.plan_payload = plan.to_dict()
            self.plan_error = ""
            self.values["architecture"] = "tsh_calo"
        self.changed.emit(dict(self.values))

    def create_plan(
        self,
        *,
        campaign_id: str,
        development_cases: list[str],
        member_count: int,
        master_seed: int,
        population_size: int,
        max_evaluations: int,
        requested_device: str,
        allow_cpu_fallback: bool,
        training: dict,
    ) -> None:
        """Build a new plan using the built-in TSH-CALO architecture and visible inputs."""
        try:
            source_commit = self._current_source_commit()
            from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
                TSHCALOTrainingCampaignPlan,
                TSHCALOTrainingEpisodePlan,
                TSHCALOTrainingHyperparameters,
                TSHCALOTrainingMemberPlan,
            )
            from calo_rpd_studio.algorithms.calo.tsh_calo_training_resources import (
                TSHCALOTrainingResourceEnvelope,
            )

            effective_campaign_id = campaign_id or f"tsh-calo-{uuid.uuid4().hex[:12]}"
            if not development_cases:
                raise ValueError("Select at least one non-protected training case")
            members = tuple(
                TSHCALOTrainingMemberPlan(
                    member_id=f"member-{index + 1:03d}",
                    training_seed=int(master_seed) + index,
                    episodes=tuple(
                        TSHCALOTrainingEpisodePlan(
                            session_id=f"{effective_campaign_id}-{index + 1:03d}-{case_index + 1:03d}",
                            case_identity=case_name,
                            seed=(
                                int(master_seed)
                                + 100_000
                                + index * len(development_cases)
                                + case_index
                            ),
                        )
                        for case_index, case_name in enumerate(development_cases)
                    ),
                )
                for index in range(int(member_count))
            )
            plan = TSHCALOTrainingCampaignPlan(
                campaign_id=effective_campaign_id,
                source_commit=source_commit,
                development_freeze_commit="",
                development_freeze_sha256="",
                phase4_acceptance_sha256="",
                development_cases=tuple(development_cases),
                members=members,
                resource_envelope=TSHCALOTrainingResourceEnvelope(
                    rollout_capacity=max(1, min(int(max_evaluations), 4096)),
                    maximum_population_size=int(population_size),
                    maximum_topology_nodes=300,
                    maximum_topology_edges=1_000,
                    maximum_topology_controls=256,
                    maximum_scenarios=64,
                ),
                population_size=int(population_size),
                max_evaluations=int(max_evaluations),
                training=TSHCALOTrainingHyperparameters(**training),
                requested_device=requested_device,
                allow_cpu_fallback=bool(allow_cpu_fallback),
            )
            plan.validate()
        except (
            OSError,
            RuntimeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            self.plan_payload = None
            self.plan_error = str(exc)
        else:
            self.plan_payload = plan.to_dict()
            self.plan_error = ""
        self.changed.emit(dict(self.values))

    def set_plan_value(self, *path: str, value) -> None:
        if self.plan_payload is None:
            return
        target = self.plan_payload
        for key in path[:-1]:
            target = target[key]
        if target.get(path[-1]) == value:
            return
        target[path[-1]] = value
        self.changed.emit(dict(self.values))

    def clear_loaded_plan(self) -> None:
        if self.plan_payload is None and not self.plan_error:
            return
        self.plan_payload = None
        self.plan_error = ""
        self.changed.emit(dict(self.values))

    def set_member_design(
        self, *, development_cases: list[str], member_count: int, master_seed: int
    ) -> None:
        if self.plan_payload is None:
            return
        campaign_id = str(self.plan_payload.get("campaign_id", "tsh-calo"))
        self.plan_payload["development_cases"] = list(development_cases)
        self.plan_payload["members"] = [
            {
                "member_id": f"member-{index + 1:03d}",
                "training_seed": int(master_seed) + index,
                "episodes": [
                    {
                        "session_id": f"{campaign_id}-{index + 1:03d}-{case_index + 1:03d}",
                        "case_identity": case_name,
                        "seed": int(master_seed)
                        + 100_000
                        + index * len(development_cases)
                        + case_index,
                    }
                    for case_index, case_name in enumerate(development_cases)
                ],
            }
            for index in range(int(member_count))
        ]
        self.changed.emit(dict(self.values))

    def prepared_plan_path(self) -> str:
        """Materialize edited inputs as a hash-addressed plan outside the source tree."""
        if self.plan_payload is None:
            return self.values["plan"]
        encoded = (json.dumps(self.plan_payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        directory = Path(tempfile.gettempdir()) / "calo-rpd-studio" / "training-plans"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{digest}.json"
        if not target.exists():
            temporary = target.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_bytes(encoded)
            os.replace(temporary, target)
        return str(target)

    def fingerprint(self) -> str:
        payload = json.dumps(
            {"paths": self.values, "plan": self.plan_payload},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def missing(self, *, include_output: bool) -> tuple[str, ...]:
        missing = []
        if include_output and not self.values.get("output"):
            missing.append("output")
        return tuple(missing)

    def arguments(self, *, check: bool, resume: bool = False) -> list[str]:
        if self.values.get("architecture") != "tsh_calo":
            raise ValueError("CALO is built in and does not require policy training")
        result = [
            "-m",
            "calo_rpd_studio.scripts.train_tsh_calo",
            self.prepared_plan_path(),
        ]
        if check:
            result.append("--check")
        else:
            result.extend(("--output", self.values["output"]))
            if resume:
                result.append("--resume")
        return result

    def load_resume_record(self, record: dict) -> None:
        """Prefill an independent campaign resume without starting or checking it."""

        state = dict(record.get("state", {}) or {})
        output = str(state.get("output_directory", "") or state.get("output_path", "")).strip()
        plan = str(state.get("plan_path", "") or state.get("training_plan_path", "")).strip()
        if not plan and output:
            output_path = Path(output).expanduser()
            if output_path.suffix.lower() not in {".pt", ".pth", ".json"}:
                plan = str(output_path / "training_plan.json")
        if not output or not plan:
            raise ValueError("Independent training resume record is missing its plan or output")
        self.values["architecture"] = "tsh_calo"
        self.values["plan"] = plan
        self.values["output"] = output
        self.plan_payload = None
        self.plan_error = ""
        self.changed.emit(dict(self.values))


class IndependentTrainingPanel(QWidget):
    """Never starts work on construction or navigation; all process actions are explicit."""

    activity_message = pyqtSignal(str, str)

    def __init__(self, state, model: TrainingLaunchModel, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.model = model
        self.process: QProcess | None = None
        self._operation = ""
        self._invocation_fingerprint = ""
        self._validated_fingerprint = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)
        layout.addWidget(
            PageHeader(
                "Policy training",
                "Train a new TSH-CALO policy after checking the selected inputs.",
            )
        )
        boundary = QLabel(
            "Opening this page does not start training. Check readiness reviews the selected inputs; "
            "Start training runs only after confirmation. A completed result is saved but is not "
            "selected for experiments automatically."
        )
        boundary.setObjectName("ResultBanner")
        boundary.setWordWrap(True)
        layout.addWidget(boundary)

        self.plan_summary = QLabel()
        self.plan_summary.setObjectName("SectionCard")
        self.plan_summary.setWordWrap(True)
        self.plan_summary.setMinimumHeight(70)
        layout.addWidget(self.plan_summary)

        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setAccessibleName("Training action summary")
        self.command_preview.setMaximumHeight(92)
        layout.addWidget(self.command_preview)

        self.resume = QCheckBox("Resume this compatible training run")
        self.resume.setToolTip(
            "Resume only when the saved run matches the selected inputs and output directory."
        )
        layout.addWidget(self.resume)

        buttons = QHBoxLayout()
        self.check_button = QPushButton("1. Check readiness")
        self.check_button.clicked.connect(self.check_readiness)
        self.start_button = QPushButton("2. Start new training")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_training)
        buttons.addWidget(self.check_button)
        buttons.addWidget(self.start_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.status = QLabel("Not checked · no work started")
        self.status.setObjectName("ContextValue")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.model.changed.connect(self._configuration_changed)
        self.resume.toggled.connect(lambda _checked: self._refresh_preview())
        self._configuration_changed(dict(self.model.values))

    def prepare_resume(self, record: dict) -> None:
        """Load an authenticated resume request; explicit readiness and start remain mandatory."""

        self.model.load_resume_record(record)
        self.resume.setChecked(True)
        self._validated_fingerprint = ""
        self.start_button.setEnabled(False)
        self.status.setText("Resume loaded · check readiness before starting")
        self.activity_message.emit(
            "INFO", "Independent training resume loaded; no process was started."
        )

    def _configuration_changed(self, values: dict) -> None:
        self._validated_fingerprint = ""
        self.start_button.setEnabled(False)
        trainable = values.get("architecture") == "tsh_calo"
        self.check_button.setEnabled(self.process is None and trainable)
        self.resume.setEnabled(trainable)
        missing = self.model.missing(include_output=False)
        output = values.get("output") or "not selected"
        architecture = "CALO" if values.get("architecture") == "calo" else "TSH-CALO"
        self.plan_summary.setText(
            f"Base architecture: {architecture}\n"
            f"Settings template: {values.get('plan') or 'not selected'}\n"
            f"Output: {output}"
        )
        if values.get("architecture") == "calo":
            self.resume.setChecked(False)
            self.status.setText("CALO is ready without policy training")
            self._refresh_preview()
            return
        self.status.setText(
            "Readiness invalidated by input change"
            if not missing
            else "Complete the required inputs"
        )
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if self.model.values.get("architecture") == "calo":
            self.command_preview.setPlainText(
                "CALO is ready to use and does not require policy training."
            )
            return
        self.command_preview.setPlainText(
            "Readiness reviews the current training inputs without starting training.\n"
            "Training starts only after readiness passes and you confirm."
        )

    def check_readiness(self) -> None:
        if self.process is not None:
            return
        if self.model.values.get("architecture") == "calo":
            QMessageBox.information(
                self,
                "No training required",
                "CALO is built in and can be selected directly for an experiment. "
                "Choose TSH-CALO to train a new policy.",
            )
            return
        missing = self.model.missing(include_output=False)
        if missing:
            QMessageBox.warning(self, "Readiness inputs required", f"Select: {', '.join(missing)}")
            return
        if self.model.plan_payload is None and self.model.values.get("plan"):
            self.model.load_plan()
        if self.model.plan_payload is None:
            QMessageBox.warning(
                self,
                "Training plan not ready",
                self.model.plan_error or "Review the scientific training inputs.",
            )
            return
        self._start_process("check", self.model.arguments(check=True))

    def start_training(self) -> None:
        if self.process is not None:
            return
        if self.model.values.get("architecture") == "calo":
            QMessageBox.information(
                self,
                "No training required",
                "CALO is built in and does not require or produce a trained policy.",
            )
            return
        if bool(getattr(self.state, "policy_training_active", False)):
            QMessageBox.warning(
                self,
                "Policy training already active",
                "Wait for the active training run to finish or stop safely before starting another.",
            )
            return
        if bool(getattr(self.state.task_status, "busy", False)):
            QMessageBox.warning(
                self,
                "Foreground task already active",
                "Finish or safely stop the current foreground task before starting independent training.",
            )
            return
        if self._validated_fingerprint != self.model.fingerprint():
            QMessageBox.warning(
                self,
                "Fresh readiness check required",
                "Inputs changed or have not passed the independent readiness check.",
            )
            self.start_button.setEnabled(False)
            return
        missing = self.model.missing(include_output=True)
        if missing:
            QMessageBox.warning(self, "Training output required", f"Select: {', '.join(missing)}")
            return
        output_path = Path(self.model.values["output"]).expanduser()
        if output_path.exists() and not self.resume.isChecked():
            QMessageBox.warning(
                self,
                "Output already exists",
                "Choose a new output directory or select the compatible resume option.",
            )
            return
        answer = QMessageBox.question(
            self,
            "Start independent policy training",
            "Start the checked training run now? The saved result will not be selected for "
            "experiments automatically.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._start_process(
            "training",
            self.model.arguments(check=False, resume=self.resume.isChecked()),
        )

    def _start_process(self, operation: str, arguments: list[str]) -> None:
        self._operation = operation
        self._invocation_fingerprint = self.model.fingerprint()
        self.check_button.setEnabled(False)
        self.start_button.setEnabled(False)
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(arguments)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.errorOccurred.connect(self._process_error)
        process.finished.connect(self._process_finished)
        self.process = process
        if operation == "training":
            self.status.setText("Training running · result is not selected for experiments")
            self.state.begin_policy_training("Independent TSH-CALO training process")
            self.state.task_status.begin(
                "Independent policy training",
                detail="The training result will not be selected automatically",
                progress=-1,
                cancellable=False,
            )
        else:
            self.status.setText("Checking readiness · no training started")
        self.activity_message.emit("INFO", self.status.text())
        process.start()

    def _read_output(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.activity_message.emit("INFO", text.rstrip())

    def _process_error(self, error) -> None:
        if self.process is None:
            return
        message = f"Process error: {self.process.errorString()} ({error})"
        self.activity_message.emit("ERROR", message)
        if error == QProcess.ProcessError.FailedToStart:
            operation = self._operation
            process = self.process
            self.process = None
            self._operation = ""
            self._invocation_fingerprint = ""
            self.check_button.setEnabled(True)
            self.start_button.setEnabled(False)
            self._validated_fingerprint = ""
            self.status.setText("Process could not start · no training result was accepted")
            if operation == "training":
                self.state.end_policy_training("Independent training process could not start")
                self.state.task_status.fail(self.status.text())
            show_error(
                self,
                "Independent process could not start",
                "The readiness or training process could not be launched.",
                message,
                source="independent policy process",
            )
            process.deleteLater()

    def _process_finished(self, exit_code: int, _exit_status) -> None:
        operation = self._operation
        invocation_fingerprint = self._invocation_fingerprint
        process = self.process
        if process is not None:
            self._read_output()
        self.process = None
        self._operation = ""
        self._invocation_fingerprint = ""
        self.check_button.setEnabled(True)
        passed = int(exit_code) == 0
        if operation == "check":
            current_fingerprint = self.model.fingerprint()
            if passed and invocation_fingerprint == current_fingerprint:
                self._validated_fingerprint = invocation_fingerprint
                self.status.setText("Readiness passed · training not started")
                self.start_button.setEnabled(not self.model.missing(include_output=True))
            elif passed:
                self._validated_fingerprint = ""
                self.start_button.setEnabled(False)
                self.status.setText(
                    "Readiness result rejected because inputs changed · run readiness again"
                )
            else:
                self._validated_fingerprint = ""
                self.status.setText(f"Readiness failed (exit {exit_code}) · training not started")
            severity = (
                "INFO"
                if passed and invocation_fingerprint == current_fingerprint
                else ("WARNING" if passed else "ERROR")
            )
            self.activity_message.emit(severity, self.status.text())
            if process is not None:
                process.deleteLater()
            return

        self.state.end_policy_training(
            "Independent training completed"
            if passed
            else "Independent training stopped with failure"
        )
        if passed:
            self.status.setText(
                "Training completed · result saved and not selected for experiments"
            )
            self.state.task_status.finish(self.status.text())
        else:
            self.status.setText(f"Training failed (exit {exit_code}) · no result was selected")
            self.state.task_status.fail(self.status.text())
        self.activity_message.emit("INFO" if passed else "ERROR", self.status.text())
        if process is not None:
            process.deleteLater()
