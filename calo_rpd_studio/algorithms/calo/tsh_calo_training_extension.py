"""Explicit finite extension segments for completed independent TSH-CALO campaigns.

An extension never mutates the completed parent campaign. It authenticates the parent's final
trainer checkpoints, continues every ensemble member under the same scientific and execution plan,
and writes a new immutable child segment. Any number of finite child segments may be requested
explicitly; no segment is started automatically and no candidate is qualified or activated here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from calo_rpd_studio.ai.model_io import checkpoint_sha256, load_trusted_resume

from .tsh_calo_policy_artifact import (
    TSHCALOCandidateArtifact,
    assemble_tsh_calo_ensemble_candidate,
    inspect_tsh_calo_candidate,
)
from .tsh_calo_schema import (
    TSH_CALO_ACTION_SCHEMA,
    TSH_CALO_ALGORITHM_ID,
    TSH_CALO_ALGORITHM_VERSION,
    TSH_CALO_STATE_SCHEMA,
    TSH_CALO_TRAINING_ENVIRONMENT,
)
from .tsh_calo_training import IndependentTSHCALOTrainer
from .tsh_calo_training_campaign import (
    IndependentTSHCALOTrainingCampaign,
    TSHCALOTrainingCampaignPlan,
    TSHCALOTrainingCampaignResult,
    TSHCALOTrainingEpisodePlan,
    TSHCALOTrainingPauseRequested,
    TSH_CALO_TRAINING_CAMPAIGN_STATUS_SCHEMA,
    _canonical_sha256,
    _read_json,
    _write_json,
    parse_tsh_calo_extension_plan,
    tsh_calo_model_state_schema_sha256,
    tsh_calo_training_compatibility_contract,
    validate_tsh_calo_training_compatibility_contract,
)
from .tsh_calo_training_session import (
    IndependentTSHCALOTrainingSession,
    TSHCALOTrainingSessionConfig,
)


TSH_CALO_TRAINING_EXTENSION_SCHEMA = "tsh-calo-training-extension-segment-v1"
TSH_CALO_TRAINING_EXTENSION_MANIFEST_SCHEMA = "tsh-calo-training-extension-manifest-v1"


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingExtensionParent:
    directory: Path
    manifest_path: Path
    manifest_sha256: str
    checkpoints: tuple[dict, ...]
    completed_extension_count: int
    cumulative_candidate_evaluations: int


def _finite_segment_evaluations(plan: TSHCALOTrainingCampaignPlan) -> int:
    return int(
        sum(len(member.episodes) for member in plan.members) * plan.max_evaluations
    )


def _validated_continuation_checkpoints(
    plan: TSHCALOTrainingCampaignPlan,
    directory: Path,
    manifest: dict,
) -> tuple[dict, ...]:
    records = manifest.get("continuation_checkpoints")
    if not isinstance(records, list) or len(records) != len(plan.members):
        raise ValueError("Completed training lacks authenticated continuation checkpoints")
    validated = []
    for member_index, member in enumerate(plan.members):
        record = records[member_index]
        if not isinstance(record, dict):
            raise ValueError("Continuation checkpoint record is invalid")
        name = str(record.get("path", ""))
        expected_sha256 = str(record.get("sha256", ""))
        if (
            int(record.get("member_index", -1)) != member_index
            or str(record.get("member_id", "")) != member.member_id
            or Path(name).name != name
            or not name
        ):
            raise ValueError("Continuation checkpoint does not match its ensemble member")
        path = directory / name
        if checkpoint_sha256(path) != expected_sha256:
            raise ValueError("Continuation checkpoint SHA-256 differs from its manifest")
        payload = load_trusted_resume(path, map_location="cpu")
        trainer = dict(payload.get("trainer", {}) or {})
        config = plan.training_config(member)
        environment_config = plan.environment_config(config, member.episodes[-1])
        for key, expected in (
            ("algorithm_id", TSH_CALO_ALGORITHM_ID),
            ("algorithm_version", TSH_CALO_ALGORITHM_VERSION),
            ("state_schema_version", TSH_CALO_STATE_SCHEMA),
            ("action_schema_version", TSH_CALO_ACTION_SCHEMA),
            ("training_environment_version", TSH_CALO_TRAINING_ENVIRONMENT),
        ):
            if str(trainer.get(key, "")) != expected:
                raise ValueError(
                    f"Continuation checkpoint training architecture {key} changed"
                )
        saved_training_config = trainer.get("training_config")
        expected_training_config = asdict(config)
        if not isinstance(saved_training_config, dict):
            raise ValueError("Continuation checkpoint training parameters are unavailable")
        if set(saved_training_config) != set(expected_training_config):
            raise ValueError(
                "Continuation checkpoint training parameter fields were added or removed"
            )
        if saved_training_config != expected_training_config:
            raise ValueError("Continuation checkpoint training parameter values changed")
        expected_compatibility = tsh_calo_training_compatibility_contract(plan)
        if tsh_calo_model_state_schema_sha256(trainer.get("model_state_dict")) != (
            expected_compatibility["policy_parameter_layout_sha256"]
        ):
            raise ValueError("Continuation checkpoint policy parameter layout changed")
        saved_session_config = payload.get("session_config")
        if not isinstance(saved_session_config, dict):
            raise ValueError("Continuation checkpoint session parameters are unavailable")
        expected_session_fields = set(asdict(TSHCALOTrainingSessionConfig("schema-check")))
        if set(saved_session_config) != expected_session_fields:
            raise ValueError(
                "Continuation checkpoint session parameter fields were added or removed"
            )
        if bool(saved_session_config.get("deterministic_policy")) != (
            plan.deterministic_policy
        ):
            raise ValueError("Continuation checkpoint session parameter values changed")
        environment = dict(payload.get("environment", {}) or {})
        if environment.get("environment_design_sha256") != (
            environment_config.scientific_design_hash(config)
        ):
            raise ValueError("Continuation checkpoint environment parameters changed")
        expected_session = TSHCALOTrainingSessionConfig(
            session_id=str(saved_session_config.get("session_id", "")),
            deterministic_policy=plan.deterministic_policy,
        )
        if payload.get("session_design_sha256") != expected_session.scientific_design_hash(
            config,
            environment_config,
        ):
            raise ValueError("Continuation checkpoint session design changed")
        if trainer.get("scientific_design_hash") != config.scientific_design_hash():
            raise ValueError("Continuation checkpoint scientific design changed")
        receipts = list(trainer.get("training_episode_receipts", ()))
        if len(receipts) != int(record.get("receipt_count", -1)):
            raise ValueError("Continuation checkpoint receipt count differs from its manifest")
        validated.append({**record, "absolute_path": str(path.resolve())})
    return tuple(validated)


def resolve_tsh_calo_training_extension_parent(
    plan: TSHCALOTrainingCampaignPlan,
    campaign_directory: str | Path,
) -> tuple[TSHCALOTrainingExtensionParent, Path | None]:
    """Validate the immutable completed chain and return its parent plus any pending segment."""

    root = Path(campaign_directory).expanduser().resolve(strict=True)
    stored_plan_payload = _read_json(root / IndependentTSHCALOTrainingCampaign.PLAN_FILE)
    stored_plan = parse_tsh_calo_extension_plan(stored_plan_payload)
    if stored_plan.execution_plan_sha256() != plan.execution_plan_sha256():
        raise ValueError("Completed training plan changed; exact extension is forbidden")
    root_status = _read_json(root / IndependentTSHCALOTrainingCampaign.STATUS_FILE)
    if root_status.get("state") != "completed":
        raise RuntimeError("Training must complete before it can be extended")
    manifest_path = root / IndependentTSHCALOTrainingCampaign.MANIFEST_FILE
    manifest = _read_json(manifest_path)
    if manifest.get("state") != "completed_unqualified":
        raise ValueError("Completed training manifest is incompatible with extension")
    if manifest.get("execution_plan_sha256") != plan.execution_plan_sha256():
        raise ValueError("Completed training manifest belongs to another execution plan")
    compatibility = manifest.get("training_compatibility_contract")
    if compatibility is not None:
        validate_tsh_calo_training_compatibility_contract(compatibility, plan)
    manifest_sha256 = checkpoint_sha256(manifest_path)
    if root_status.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Completed training status does not authenticate its manifest")
    checkpoints = _validated_continuation_checkpoints(plan, root, manifest)
    segment_evaluations = _finite_segment_evaluations(plan)
    parent = TSHCALOTrainingExtensionParent(
        root,
        manifest_path,
        manifest_sha256,
        checkpoints,
        0,
        segment_evaluations,
    )
    pending: Path | None = None
    extensions = root / "extensions"
    if not extensions.is_dir():
        return parent, None
    for segment in sorted(extensions.glob("segment-*")):
        if not segment.is_dir():
            continue
        extension_plan_path = segment / "extension_plan.json"
        extension_plan = _read_json(extension_plan_path)
        expected_number = parent.completed_extension_count + 1
        if (
            segment.name != f"segment-{expected_number:06d}"
            or extension_plan.get("schema_version") != TSH_CALO_TRAINING_EXTENSION_SCHEMA
            or int(extension_plan.get("segment_number", -1)) != expected_number
            or extension_plan.get("base_plan_sha256") != plan.execution_plan_sha256()
            or extension_plan.get("parent_manifest_sha256") != parent.manifest_sha256
        ):
            raise ValueError("Training extension chain is not contiguous and authenticated")
        status = _read_json(segment / IndependentTSHCALOTrainingCampaign.STATUS_FILE)
        status_extension = status.get("extension")
        if (
            not isinstance(status_extension, dict)
            or status_extension.get("extension_plan_sha256")
            != checkpoint_sha256(extension_plan_path)
            or status_extension.get("extension_id") != extension_plan.get("extension_id")
        ):
            raise ValueError("Training extension status does not authenticate its segment plan")
        state = str(status.get("state", ""))
        if state in {"running", "interrupted"}:
            if pending is not None:
                raise ValueError("Multiple pending training extension segments are forbidden")
            pending = segment
            continue
        if state == "failed":
            raise RuntimeError("A failed extension segment requires explicit review before retry")
        if state != "completed":
            raise ValueError("Training extension status is invalid")
        child_manifest_path = segment / IndependentTSHCALOTrainingCampaign.MANIFEST_FILE
        child_manifest = _read_json(child_manifest_path)
        if (
            child_manifest.get("schema_version")
            != TSH_CALO_TRAINING_EXTENSION_MANIFEST_SCHEMA
            or child_manifest.get("parent_manifest_sha256") != parent.manifest_sha256
            or int(child_manifest.get("segment_number", -1)) != expected_number
            or child_manifest.get("execution_plan_sha256") != plan.execution_plan_sha256()
            or child_manifest.get("extension_plan_sha256")
            != status_extension.get("extension_plan_sha256")
        ):
            raise ValueError("Training extension manifest breaks its authenticated lineage")
        child_compatibility = child_manifest.get("training_compatibility_contract")
        if child_compatibility is not None:
            validate_tsh_calo_training_compatibility_contract(child_compatibility, plan)
        child_sha256 = checkpoint_sha256(child_manifest_path)
        if status.get("manifest_sha256") != child_sha256:
            raise ValueError("Training extension status does not authenticate its manifest")
        checkpoints = _validated_continuation_checkpoints(plan, segment, child_manifest)
        cumulative = int(child_manifest.get("cumulative_candidate_evaluations", -1))
        if cumulative != parent.cumulative_candidate_evaluations + segment_evaluations:
            raise ValueError("Training extension cumulative evaluation accounting changed")
        parent = TSHCALOTrainingExtensionParent(
            segment,
            child_manifest_path,
            child_sha256,
            checkpoints,
            expected_number,
            cumulative,
        )
    return parent, pending


class IndependentTSHCALOTrainingExtension:
    """Continue a completed ensemble through one explicit, finite, authenticated segment."""

    def __init__(
        self,
        plan: TSHCALOTrainingCampaignPlan,
        campaign_directory: str | Path,
        *,
        problem_factory: Callable[[str], object] | None = None,
        event_callback: Callable[[dict], None] | None = None,
        transition_callback: Callable[[dict], None] | None = None,
        execution_source_commit: str | None = None,
    ) -> None:
        plan.validate()
        self.plan = plan
        self.campaign_directory = Path(campaign_directory).expanduser().resolve()
        self.problem_factory = problem_factory
        self.event_callback = event_callback
        self.transition_callback = transition_callback
        self.execution_source_commit = str(
            execution_source_commit or plan.source_commit
        ).lower()
        if len(self.execution_source_commit) != 40 or any(
            character not in "0123456789abcdef"
            for character in self.execution_source_commit
        ):
            raise ValueError("Training extension requires an exact execution source commit")
        self.parent: TSHCALOTrainingExtensionParent | None = None
        self.segment_directory: Path | None = None
        self.runner: IndependentTSHCALOTrainingCampaign | None = None

    def _extension_episode(
        self, episode: TSHCALOTrainingEpisodePlan, segment_number: int
    ) -> TSHCALOTrainingEpisodePlan:
        return replace(
            episode,
            session_id=f"{episode.session_id}:extension:{segment_number:06d}",
        )

    def _configure_runner(self, directory: Path) -> IndependentTSHCALOTrainingCampaign:
        runner = IndependentTSHCALOTrainingCampaign(
            self.plan,
            directory,
            problem_factory=self.problem_factory,
            event_callback=self.event_callback,
            transition_callback=self.transition_callback,
        )
        self.segment_directory = directory
        self.runner = runner
        return runner

    def start(self) -> TSHCALOTrainingCampaignResult:
        parent, pending = resolve_tsh_calo_training_extension_parent(
            self.plan, self.campaign_directory
        )
        if pending is not None:
            raise RuntimeError("A paused training extension must be resumed before another starts")
        segment_number = parent.completed_extension_count + 1
        segment = self.campaign_directory / "extensions" / f"segment-{segment_number:06d}"
        if segment.exists():
            raise FileExistsError("Training extension segment directory already exists")
        segment.mkdir(parents=True)
        runner = self._configure_runner(segment)
        self.parent = parent
        _write_json(runner._plan_path, self.plan.to_dict())
        extension_plan = {
            "schema_version": TSH_CALO_TRAINING_EXTENSION_SCHEMA,
            "segment_number": segment_number,
            "extension_id": f"{self.plan.campaign_id}:extension:{segment_number:06d}",
            "base_campaign_id": self.plan.campaign_id,
            "base_plan_sha256": self.plan.execution_plan_sha256(),
            "parent_manifest_path": str(parent.manifest_path),
            "parent_manifest_sha256": parent.manifest_sha256,
            "parent_completed_extension_count": parent.completed_extension_count,
            "prior_cumulative_candidate_evaluations": parent.cumulative_candidate_evaluations,
            "segment_candidate_evaluations": _finite_segment_evaluations(self.plan),
            "same_scientific_design_required": True,
            "same_execution_plan_required": True,
            "source_revision_is_compatibility_identity": False,
            "architecture_and_parameter_schema_required": True,
            "automatic_start": False,
            "training_compatibility_contract": tsh_calo_training_compatibility_contract(
                self.plan
            ),
            "origin_source_commit": self.plan.source_commit,
            "execution_source_commit": self.execution_source_commit,
        }
        extension_plan_sha256 = _write_json(segment / "extension_plan.json", extension_plan)
        status = {
            "state": "running",
            "extension": {
                **extension_plan,
                "extension_plan_sha256": extension_plan_sha256,
            },
            "current_member_index": 0,
            "current_episode_index": 0,
            "session_checkpoint": None,
            "uncommitted_cuda_window": None,
            "member_candidates": [],
            "continuation_checkpoints": [],
            "failure": None,
            "pause": None,
            "progress": None,
            "event_sequence": 0,
        }
        runner._record_event(
            status,
            "extension_started",
            details={
                "segment_number": segment_number,
                "control_directory": str(segment),
                "prior_cumulative_candidate_evaluations": parent.cumulative_candidate_evaluations,
                "total_candidate_evaluations": _finite_segment_evaluations(self.plan),
                "progress_percent": 0,
            },
        )
        return self._execute(status)

    def resume(self) -> TSHCALOTrainingCampaignResult:
        parent, pending = resolve_tsh_calo_training_extension_parent(
            self.plan, self.campaign_directory
        )
        if pending is None:
            raise RuntimeError("No paused training extension is available to resume")
        runner = self._configure_runner(pending)
        self.parent = parent
        status = _read_json(runner._status_path)
        if status.get("schema_version") != TSH_CALO_TRAINING_CAMPAIGN_STATUS_SCHEMA:
            raise ValueError("Training extension status schema is incompatible")
        if status.get("plan_sha256") != self.plan.execution_plan_sha256():
            raise ValueError("Training extension status belongs to another plan")
        if status.get("uncommitted_cuda_window") is not None:
            raise RuntimeError("Training extension stopped inside an uncommitted evaluation window")
        status["state"] = "running"
        status["pause"] = None
        status["failure"] = None
        runner._record_event(
            status,
            "extension_resumed",
            details={
                **dict(status.get("progress", {}) or {}),
                "segment_number": int(status["extension"]["segment_number"]),
                "control_directory": str(pending),
            },
        )
        return self._execute(status)

    def start_or_resume(self) -> TSHCALOTrainingCampaignResult:
        _parent, pending = resolve_tsh_calo_training_extension_parent(
            self.plan, self.campaign_directory
        )
        return self.resume() if pending is not None else self.start()

    def _parent_trainer(self, member_index: int) -> IndependentTSHCALOTrainer:
        assert self.parent is not None
        record = self.parent.checkpoints[member_index]
        payload = load_trusted_resume(record["absolute_path"], map_location="cpu")
        return IndependentTSHCALOTrainer.from_resume_state_dict(
            dict(payload.get("trainer", {})),
            expected_config=self.plan.training_config(self.plan.members[member_index]),
        )

    def _restore_segment_session(
        self, status: dict, member_index: int
    ) -> tuple[IndependentTSHCALOTrainingSession, int]:
        assert self.runner is not None
        checkpoint = dict(status["session_checkpoint"])
        episode_index = int(checkpoint.get("episode_index", -1))
        member = self.plan.members[member_index]
        if int(checkpoint.get("member_index", -1)) != member_index:
            raise ValueError("Extension checkpoint belongs to another member")
        if episode_index < 0 or episode_index >= len(member.episodes):
            raise ValueError("Extension checkpoint episode index is invalid")
        path = self.runner.output_directory / str(checkpoint.get("path", ""))
        if checkpoint_sha256(path) != str(checkpoint.get("sha256", "")):
            raise ValueError("Extension checkpoint SHA-256 differs from its status")
        segment_number = int(status["extension"]["segment_number"])
        episode = self._extension_episode(member.episodes[episode_index], segment_number)
        payload = load_trusted_resume(path, map_location="cpu")
        selected_device = str(
            dict(payload.get("trainer", {}) or {})
            .get("training_device_provenance", {})
            .get("memory_admission", {})
            .get("selected_device", self.plan.requested_device)
        )
        problem = self.runner._build_problem(episode.case_identity, device_hint=selected_device)
        training = self.plan.training_config(member)
        session = IndependentTSHCALOTrainingSession.load_resume(
            path,
            problem=problem,
            training_config=training,
            environment_config=self.plan.environment_config(training, episode),
            session_config=TSHCALOTrainingSessionConfig(
                session_id=episode.session_id,
                deterministic_policy=self.plan.deterministic_policy,
            ),
        )
        return session, episode_index

    def _run_member(self, status: dict, member_index: int) -> TSHCALOCandidateArtifact:
        assert self.runner is not None
        assert self.parent is not None
        member = self.plan.members[member_index]
        training = self.plan.training_config(member)
        episode_index = int(status.get("current_episode_index", 0))
        checkpoint = status.get("session_checkpoint")
        trainer: IndependentTSHCALOTrainer
        if isinstance(checkpoint, dict) and int(checkpoint.get("member_index", -1)) == member_index:
            session, restored_episode = self._restore_segment_session(status, member_index)
            trainer = session.trainer
            if restored_episode == episode_index:
                self.runner._active_session = session
                self.runner._advance_session(session, status, member_index, episode_index)
                episode_index += 1
                status["current_episode_index"] = episode_index
                self.runner._write_status(status)
            elif restored_episode != episode_index - 1 or not session.completed:
                raise ValueError("Extension checkpoint/status progression is inconsistent")
        else:
            trainer = self._parent_trainer(member_index)
        try:
            segment_number = int(status["extension"]["segment_number"])
            for episode_index in range(episode_index, len(member.episodes)):
                episode = self._extension_episode(member.episodes[episode_index], segment_number)
                session = self.runner._new_session(trainer, training, episode)
                self.runner._active_session = session
                self.runner._advance_session(session, status, member_index, episode_index)
                status["current_episode_index"] = episode_index + 1
                self.runner._write_status(status)
            candidate_path = self.runner._candidate_path(member_index)
            candidate = trainer.export_unqualified_candidate(
                candidate_path,
                source_commit=self.plan.source_commit,
                execution_source_commit=self.execution_source_commit,
            )
            expected_receipts = int(self.parent.checkpoints[member_index]["receipt_count"]) + len(
                member.episodes
            )
            if len(candidate.training_provenance["training_episode_receipts"]) != expected_receipts:
                raise ValueError("Extension candidate receipt accounting is incomplete")
            return candidate
        finally:
            trainer.close()
            self.runner._active_session = None

    def _execute(self, status: dict) -> TSHCALOTrainingCampaignResult:
        assert self.runner is not None
        assert self.parent is not None
        try:
            members: list[TSHCALOCandidateArtifact] = []
            for index, item in enumerate(status.get("member_candidates", ())):
                members.append(
                    inspect_tsh_calo_candidate(
                        self.runner._candidate_path(index),
                        expected_sha256=item["sha256"],
                    )
                )
            member_index = int(status.get("current_member_index", 0))
            if len(members) != member_index:
                raise ValueError("Extension candidate/status progression is inconsistent")
            while member_index < len(self.plan.members):
                candidate = self._run_member(status, member_index)
                members.append(candidate)
                status["member_candidates"] = [
                    {"path": Path(item.path).name, "sha256": item.sha256} for item in members
                ]
                checkpoint = dict(status.get("session_checkpoint", {}) or {})
                if int(checkpoint.get("member_index", -1)) != member_index:
                    raise ValueError("Extension member lacks a continuation checkpoint")
                status["continuation_checkpoints"].append(
                    {
                        "member_index": member_index,
                        "member_id": self.plan.members[member_index].member_id,
                        "path": str(checkpoint["path"]),
                        "sha256": str(checkpoint["sha256"]),
                        "receipt_count": len(
                            candidate.training_provenance["training_episode_receipts"]
                        ),
                    }
                )
                member_index += 1
                status["current_member_index"] = member_index
                status["current_episode_index"] = 0
                self.runner._record_event(
                    status,
                    "extension_member_completed",
                    details={
                        **dict(status.get("progress", {}) or {}),
                        "member_number": member_index,
                        "member_count": len(self.plan.members),
                    },
                )
            ensemble_path = self.runner.output_directory / "ensemble.candidate.pt"
            ensemble = assemble_tsh_calo_ensemble_candidate(
                ensemble_path,
                [(member.path, member.sha256) for member in members],
            )
            extension = dict(status["extension"])
            segment_evaluations = _finite_segment_evaluations(self.plan)
            cumulative = self.parent.cumulative_candidate_evaluations + segment_evaluations
            manifest = {
                "schema_version": TSH_CALO_TRAINING_EXTENSION_MANIFEST_SCHEMA,
                "state": "completed_unqualified_extension",
                "campaign_id": self.plan.campaign_id,
                "segment_number": int(extension["segment_number"]),
                "extension_id": extension["extension_id"],
                "extension_plan_sha256": extension["extension_plan_sha256"],
                "base_campaign_directory": str(self.campaign_directory),
                "source_commit": self.plan.source_commit,
                "execution_source_commit": self.execution_source_commit,
                "scientific_design_sha256": self.plan.scientific_design_hash(),
                "execution_plan_sha256": self.plan.execution_plan_sha256(),
                "seed_manifest_sha256": self.plan.seed_manifest_sha256(),
                "parent_manifest_path": extension["parent_manifest_path"],
                "parent_manifest_sha256": extension["parent_manifest_sha256"],
                "segment_candidate_evaluations": segment_evaluations,
                "cumulative_candidate_evaluations": cumulative,
                "completed_extension_count": self.parent.completed_extension_count + 1,
                "member_candidates": status["member_candidates"],
                "continuation_checkpoints": status["continuation_checkpoints"],
                "training_compatibility_contract": tsh_calo_training_compatibility_contract(
                    self.plan
                ),
                "ensemble_candidate": {
                    "path": Path(ensemble.path).name,
                    "sha256": ensemble.sha256,
                    "lifecycle_status": "candidate_unqualified",
                },
                "extension_contract": {
                    "repeatable_finite_segments": True,
                    "same_scientific_design": True,
                    "same_execution_plan": True,
                    "source_revision_is_compatibility_identity": False,
                    "architecture_and_parameter_schema_required": True,
                    "retained_state": [
                        "model_parameters",
                        "optimizer_state",
                        "numpy_generator_state",
                        "torch_generator_state",
                        "ppo_update_count",
                        "episode_receipts",
                        "device_and_memory_admission_provenance",
                        "session_environment_state",
                        "rollout_collector_state",
                        "exact_evaluation_accounting",
                    ],
                    "automatic_extension": False,
                },
                "authority_boundary": {
                    "training_only": True,
                    "registered": False,
                    "qualified": False,
                    "activated": False,
                    "experiment_bound": False,
                },
            }
            manifest_path = self.runner.output_directory / self.runner.MANIFEST_FILE
            manifest_sha256 = _write_json(manifest_path, manifest)
            status["state"] = "completed"
            status["manifest_sha256"] = manifest_sha256
            completed_progress = {
                **dict(status.get("progress", {}) or {}),
                "progress_percent": 100,
                "committed_candidate_evaluations": segment_evaluations,
                "total_candidate_evaluations": segment_evaluations,
                "cumulative_candidate_evaluations": cumulative,
            }
            status["progress"] = completed_progress
            self.runner._record_event(
                status,
                "extension_completed",
                details={**completed_progress, "manifest_sha256": manifest_sha256},
            )
            return TSHCALOTrainingCampaignResult(
                str(self.runner.output_directory),
                self.plan.execution_plan_sha256(),
                self.plan.seed_manifest_sha256(),
                tuple(members),
                ensemble,
                str(manifest_path),
                manifest_sha256,
            )
        except TSHCALOTrainingPauseRequested:
            raise
        except KeyboardInterrupt:
            status["state"] = "interrupted"
            self.runner._write_status(status)
            raise
        except Exception as exc:
            status["state"] = "failed"
            status["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "category": "non_resumable_training_or_integrity_failure",
                "resumable": False,
            }
            self.runner._write_status(status)
            raise


def extension_plan_summary(
    plan: TSHCALOTrainingCampaignPlan, campaign_directory: str | Path
) -> dict:
    """Return a read-only readiness summary without starting or extending training."""

    parent, pending = resolve_tsh_calo_training_extension_parent(plan, campaign_directory)
    segment_evaluations = _finite_segment_evaluations(plan)
    return {
        "schema_version": TSH_CALO_TRAINING_EXTENSION_SCHEMA,
        "campaign_id": plan.campaign_id,
        "base_plan_sha256": plan.execution_plan_sha256(),
        "parent_manifest_sha256": parent.manifest_sha256,
        "completed_extension_count": parent.completed_extension_count,
        "next_segment_number": parent.completed_extension_count + 1,
        "segment_candidate_evaluations": segment_evaluations,
        "prior_cumulative_candidate_evaluations": parent.cumulative_candidate_evaluations,
        "next_cumulative_candidate_evaluations": (
            parent.cumulative_candidate_evaluations + segment_evaluations
        ),
        "pending_segment": "" if pending is None else str(pending),
        "same_scientific_design_required": True,
        "same_execution_plan_required": True,
        "source_revision_is_compatibility_identity": False,
        "architecture_and_parameter_schema_required": True,
        "training_compatibility_contract": tsh_calo_training_compatibility_contract(plan),
        "automatic_start": False,
        "summary_sha256": _canonical_sha256(
            {
                "plan": plan.execution_plan_sha256(),
                "parent": parent.manifest_sha256,
                "segment": segment_evaluations,
            }
        ),
    }
