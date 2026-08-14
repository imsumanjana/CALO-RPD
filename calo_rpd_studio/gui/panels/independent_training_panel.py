"""Explicit GUI adapter for the independent new-policy plan/check/start command."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QStandardPaths, pyqtSignal
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


class TrainingModelLibrary(QObject):
    """Discover saved campaigns in per-user, explicitly registered locations."""

    changed = pyqtSignal()

    SETTINGS_KEY = "training/model_scan_locations"
    PLAN_FILE = "training_plan.json"
    STATUS_FILE = "training_status.json"
    MANIFEST_FILE = "training_manifest.json"
    RESUMABLE_STATES = frozenset({"running", "interrupted"})
    DISCOVERABLE_STATES = frozenset({*RESUMABLE_STATES, "completed"})
    MAX_SCAN_DEPTH = 2
    MAX_SCANNED_DIRECTORIES = 500

    def __init__(self, settings_manager, *, default_directory: str | Path | None = None) -> None:
        super().__init__()
        self.settings_manager = settings_manager
        if default_directory is None:
            local_data = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppLocalDataLocation
            )
            if not local_data:
                local_data = str(Path.home() / ".calo-rpd-studio")
            default_directory = Path(local_data).expanduser() / "training-models"
        self.default_directory = Path(default_directory).expanduser().resolve()
        self.default_directory_error = ""
        self._candidate_integrity_cache: dict[tuple, tuple[str, str]] = {}
        try:
            self.default_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            self.default_directory_error = str(exc)

    def scan_locations(self) -> tuple[Path, ...]:
        stored = self.settings_manager.value(self.SETTINGS_KEY, [])
        if isinstance(stored, str):
            stored = [stored] if stored.strip() else []
        locations = [self.default_directory]
        for item in stored or []:
            try:
                path = Path(str(item)).expanduser().resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            if path not in locations:
                locations.append(path)
        return tuple(locations)

    def add_scan_location(self, path: str | Path) -> Path:
        location = Path(path).expanduser().resolve(strict=True)
        if not location.is_dir():
            raise ValueError("Model-library location must be a directory")
        additional = [item for item in self.scan_locations() if item != self.default_directory]
        if location != self.default_directory and location not in additional:
            additional.append(location)
            self.settings_manager.set_value(self.SETTINGS_KEY, [str(item) for item in additional])
            sync = getattr(self.settings_manager, "sync", None)
            if callable(sync):
                sync()
        self.changed.emit()
        return location

    @classmethod
    def _candidate_directories(cls, root: Path) -> tuple[Path, ...]:
        if not root.is_dir():
            return (root,)
        candidates: list[Path] = []
        queue: list[tuple[Path, int]] = [(root, 0)]
        while queue and len(candidates) < cls.MAX_SCANNED_DIRECTORIES:
            candidate, depth = queue.pop(0)
            candidates.append(candidate)
            if depth >= cls.MAX_SCAN_DEPTH:
                continue
            try:
                children = sorted(
                    (item for item in candidate.iterdir() if item.is_dir()),
                    key=lambda item: item.name.casefold(),
                )
            except OSError:
                continue
            queue.extend((item, depth + 1) for item in children)
        return tuple(candidates)

    def _completed_candidate(self, candidate: Path) -> tuple[str, str]:
        manifest_path = candidate / TrainingModelLibrary.MANIFEST_FILE
        if not manifest_path.is_file():
            return "", "Training completed, but its saved-policy manifest is unavailable."
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "", "Training completed, but its saved-policy manifest could not be read."
        ensemble = dict(manifest.get("ensemble_candidate", {}) or {})
        candidate_name = str(ensemble.get("path", "")).strip()
        expected_sha256 = str(ensemble.get("sha256", "")).strip().lower()
        if not candidate_name or Path(candidate_name).name != candidate_name:
            return "", "Training completed, but its saved policy location is invalid."
        candidate_path = candidate / candidate_name
        if not candidate_path.is_file():
            return "", "Training completed, but its saved policy file is unavailable."
        try:
            stat = candidate_path.stat()
            cache_key = (
                str(candidate_path.resolve()).casefold(),
                stat.st_size,
                stat.st_mtime_ns,
                expected_sha256,
            )
            cached = self._candidate_integrity_cache.get(cache_key)
            if cached is not None:
                return cached
            digest = hashlib.sha256()
            with candidate_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            actual_sha256 = digest.hexdigest()
        except OSError:
            return "", "Training completed, but its saved policy file could not be read."
        if actual_sha256 != expected_sha256:
            result = (
                "",
                "Training completed, but its saved policy integrity could not be confirmed.",
            )
        else:
            result = (str(candidate_path.resolve()), "")
        self._candidate_integrity_cache[cache_key] = result
        return result

    def saved_campaigns(self) -> tuple[dict, ...]:
        campaigns: dict[str, dict] = {}
        for root in self.scan_locations():
            for candidate in self._candidate_directories(root):
                plan_path = candidate / self.PLAN_FILE
                status_path = candidate / self.STATUS_FILE
                if not plan_path.is_file() or not status_path.is_file():
                    continue
                try:
                    plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(plan, dict) or not isinstance(status, dict):
                    continue
                state = str(status.get("state", "")).strip().lower()
                if state not in self.DISCOVERABLE_STATES:
                    continue
                campaign_id = str(plan.get("campaign_id", "")).strip() or candidate.name
                resolved = str(candidate.resolve())
                policy_candidate = ""
                candidate_error = ""
                if state == "completed":
                    policy_candidate, candidate_error = self._completed_candidate(candidate)
                try:
                    modified_ns = max(plan_path.stat().st_mtime_ns, status_path.stat().st_mtime_ns)
                except OSError:
                    modified_ns = 0
                campaigns[resolved.casefold()] = {
                    "campaign_id": campaign_id,
                    "state": state,
                    "directory": resolved,
                    "plan": str(plan_path.resolve()),
                    "resumable": state in self.RESUMABLE_STATES,
                    "policy_candidate": policy_candidate,
                    "candidate_error": candidate_error,
                    "modified_ns": modified_ns,
                }
        return tuple(
            sorted(
                campaigns.values(),
                key=lambda item: (
                    -int(item["modified_ns"]),
                    item["campaign_id"].casefold(),
                    item["directory"].casefold(),
                ),
            )
        )

    def resumable_campaigns(self) -> tuple[dict, ...]:
        return tuple(
            {
                "campaign_id": item["campaign_id"],
                "state": item["state"],
                "directory": item["directory"],
                "plan": item["plan"],
            }
            for item in self.saved_campaigns()
            if item["resumable"]
        )

    def completed_policy_candidates(self) -> tuple[dict, ...]:
        return tuple(
            item
            for item in self.saved_campaigns()
            if item["state"] == "completed" and item["policy_candidate"]
        )

    def record_training_output(self, output_directory: str) -> None:
        output = Path(output_directory).expanduser().resolve()
        existing_roots = self.scan_locations()
        if not any(output == root or root in output.parents for root in existing_roots):
            self.add_scan_location(output.parent)
            return
        self.changed.emit()


class TrainingLaunchModel(QObject):
    changed = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.values = {
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

    def load_plan(self, *, preserve_identity: bool = False) -> None:
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
            if not preserve_identity:
                # Imported templates supply scientific settings only; they cannot confer
                # lifecycle or release authority through scientist-selected files. An exact
                # resume keeps its stored identity for the campaign runner to authenticate.
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
                    # Reset consumes the initial population; every retained policy transition
                    # consumes one subsequent population. Bound retained transitions, not raw
                    # candidate evaluations, so memory admission matches the actual PPO rollout.
                    rollout_capacity=self._rollout_capacity(population_size, max_evaluations),
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

    @staticmethod
    def _rollout_capacity(population_size: int, max_evaluations: int) -> int:
        population = max(1, int(population_size))
        return max(1, min(int(max_evaluations) // population - 1, 4096))

    def set_resource_design(self, *, population_size: int, max_evaluations: int) -> None:
        if self.plan_payload is None:
            return
        self.plan_payload["population_size"] = int(population_size)
        self.plan_payload["max_evaluations"] = int(max_evaluations)
        envelope = self.plan_payload["resource_envelope"]
        envelope["maximum_population_size"] = int(population_size)
        envelope["rollout_capacity"] = self._rollout_capacity(population_size, max_evaluations)
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
        self.values["plan"] = plan
        self.values["output"] = output
        self.plan_payload = None
        self.plan_error = ""
        self.changed.emit(dict(self.values))


class IndependentTrainingPanel(QWidget):
    """Never starts work on construction or navigation; all process actions are explicit."""

    activity_message = pyqtSignal(str, str)
    training_completed = pyqtSignal(str)

    def __init__(self, state, model: TrainingLaunchModel, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.model = model
        self.process: QProcess | None = None
        self._operation = ""
        self._invocation_fingerprint = ""
        self._validated_fingerprint = ""
        self._process_output: list[str] = []

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

        self.resume = QCheckBox("Resume selected interrupted training")
        self.resume.setToolTip(
            "Continue only the selected interrupted training directory after its saved settings, "
            "state, recovery point, and saved-file integrity pass compatibility checks."
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
        self.resume.toggled.connect(self._resume_intent_changed)
        self._configuration_changed(dict(self.model.values))

    def _resume_intent_changed(self, _checked: bool) -> None:
        self._validated_fingerprint = ""
        self.start_button.setEnabled(False)
        self.status.setText("Resume choice changed · check readiness again")
        self._refresh_preview()

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
        self.check_button.setEnabled(self.process is None)
        self.resume.setEnabled(self.process is None)
        missing = self.model.missing(include_output=False)
        output = values.get("output") or "not selected"
        self.plan_summary.setText(
            f"Settings template: {values.get('plan') or 'not selected'}\nOutput: {output}"
        )
        self.status.setText(
            "Readiness invalidated by input change"
            if not missing
            else "Complete the required inputs"
        )
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        self.command_preview.setPlainText(
            "Readiness reviews the current training inputs without starting training.\n"
            "New training saves recovery points automatically. Training starts only after "
            "readiness passes and you confirm."
        )

    def check_readiness(self) -> None:
        if self.process is not None:
            return
        missing = self.model.missing(include_output=False)
        if missing:
            QMessageBox.warning(self, "Readiness inputs required", f"Select: {', '.join(missing)}")
            return
        if self.model.plan_payload is None and self.model.values.get("plan"):
            self.model.load_plan(preserve_identity=self.resume.isChecked())
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
        if self.resume.isChecked() and not output_path.is_dir():
            QMessageBox.warning(
                self,
                "Resumable output required",
                "Select an existing interrupted training directory, or turn off resume and "
                "choose a new output directory.",
            )
            return
        if output_path.exists() and not self.resume.isChecked():
            QMessageBox.warning(
                self,
                "Output already exists",
                "Choose a new output directory, or select the interrupted run from Saved "
                "training before continuing it.",
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
        self._process_output = []
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
            cleaned = text.rstrip()
            self._process_output.append(cleaned)
            # Full command output belongs in Activity -> Logs. The status and warning feeds
            # receive the short user-facing result from _process_finished instead.
            self.activity_message.emit("DEBUG", cleaned)

    def _friendly_process_failure(self, operation: str, exit_code: int) -> str:
        technical_output = "\n".join(self._process_output).casefold()
        if "currently available cpu ram" in technical_output:
            return (
                "The selected training settings need more memory than is currently available. "
                "Free system memory or reduce evaluations per episode, hidden dimension, or graph "
                "steps. At a fixed evaluation budget, a scientifically appropriate larger "
                "population also retains fewer policy transitions. Check readiness again; "
                "training was not started."
            )
        if "currently free vram" in technical_output:
            return (
                "The selected training settings exceed the safe NVIDIA GPU memory limit. Free GPU "
                "memory, reduce the training size, or enable CPU fallback when sufficient system "
                "memory is available. Training was not started."
            )
        if "requires a clean non-ignored source tree" in technical_output:
            return (
                "Readiness stopped because this application source has uncommitted changes. "
                "Finish and commit the software changes, then check readiness again. "
                "Training was not started."
            )
        if "requires an inspectable git source tree" in technical_output:
            return (
                "This application version could not be verified for policy training. "
                "Use a complete installed or checked-out application version, then check "
                "readiness again. Training was not started."
            )
        if "source" in technical_output and (
            "mismatch" in technical_output
            or "does not match" in technical_output
            or "identity" in technical_output
        ):
            return (
                "The saved training source does not match this application version. "
                "Select a compatible saved training run or start a new one."
            )
        if operation == "check":
            return (
                f"Readiness could not be confirmed (code {exit_code}). Training was not started. "
                "See Activity -> Logs for technical details."
            )
        return (
            f"Training stopped before a usable result was produced (code {exit_code}). "
            "No policy was selected. See Activity -> Logs for technical details."
        )

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
                self.status.setText(self._friendly_process_failure(operation, int(exit_code)))
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
            self.training_completed.emit(str(self.model.values.get("output", "")))
        else:
            self.status.setText(self._friendly_process_failure(operation, int(exit_code)))
            self.state.task_status.fail(self.status.text())
        self.activity_message.emit("INFO" if passed else "ERROR", self.status.text())
        if process is not None:
            process.deleteLater()
