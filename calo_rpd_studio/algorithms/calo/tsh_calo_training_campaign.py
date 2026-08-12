"""Explicit, development-only campaign boundary for fresh TSH-CALO ensemble members.

Nothing in this module is called by a power-system experiment. Starting and resuming are explicit
caller actions; outputs are immutable unqualified candidates and an unqualified ensemble only.
Qualification, registration, activation, experiment binding, and inference remain separate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Callable

from calo_rpd_studio.accelerated.torch_orpd import AcceleratedORPDProblem
from calo_rpd_studio.ai.model_io import (
    checkpoint_sha256,
    durable_write_bytes,
    load_trusted_resume,
)
from calo_rpd_studio.orpd.problem import ORPDProblem
from calo_rpd_studio.power_system.case_identity import protected_holdout_matches
from calo_rpd_studio.power_system.case_loader import CaseLoader

from .tsh_calo_policy_artifact import (
    TSHCALOCandidateArtifact,
    assemble_tsh_calo_ensemble_candidate,
    inspect_tsh_calo_candidate,
)
from .tsh_calo_schema import TSHCALOFeatureFlags
from .tsh_calo_training import IndependentTSHCALOTrainer, TSHCALOTrainingConfig
from .tsh_calo_training_environment import (
    IndependentTSHCALOTrainingEnvironment,
    TSHCALOTrainingEnvironmentConfig,
)
from .tsh_calo_training_resources import TSHCALOTrainingResourceEnvelope
from .tsh_calo_training_session import (
    IndependentTSHCALOTrainingSession,
    TSHCALOTrainingSessionConfig,
    TSHCALOTrainingSessionResult,
)


CUDA_DURABLE_EVALUATION_WINDOW = 100


TSH_CALO_TRAINING_CAMPAIGN_SCHEMA = (
    "tsh-calo-independent-training-campaign-v4-phase4-acceptance-bound"
)
TSH_CALO_TRAINING_SEED_MANIFEST_SCHEMA = "tsh-calo-training-seed-manifest-v1"
TSH_CALO_TRAINING_CAMPAIGN_STATUS_SCHEMA = "tsh-calo-training-campaign-status-v2"


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _valid_commit(value: str) -> bool:
    text = str(value).strip().lower()
    return len(text) == 40 and all(character in "0123456789abcdef" for character in text)


def _write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    durable_write_bytes(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"TSH-CALO campaign record is unreadable: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"TSH-CALO campaign record must be an object: {path.name}")
    return payload


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingEpisodePlan:
    session_id: str
    case_identity: str
    seed: int

    def validate(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("TSH-CALO campaign episode requires a session ID")
        if not isinstance(self.case_identity, str) or not self.case_identity.strip():
            raise ValueError("TSH-CALO campaign episode requires a case identity")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("TSH-CALO campaign episode seed must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingMemberPlan:
    member_id: str
    training_seed: int
    episodes: tuple[TSHCALOTrainingEpisodePlan, ...]

    def validate(self) -> None:
        if not isinstance(self.member_id, str) or not self.member_id.strip():
            raise ValueError("TSH-CALO campaign member requires an ID")
        if (
            not isinstance(self.training_seed, int)
            or isinstance(self.training_seed, bool)
            or self.training_seed < 0
        ):
            raise ValueError("TSH-CALO campaign member seed must be a non-negative integer")
        if not self.episodes:
            raise ValueError("TSH-CALO campaign member requires at least one episode")
        for episode in self.episodes:
            episode.validate()
        session_ids = [episode.session_id for episode in self.episodes]
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("TSH-CALO campaign member session IDs must be unique")


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingHyperparameters:
    hidden_dim: int = 64
    graph_steps: int = 2
    learning_rate: float = 3e-4
    ppo_epochs: int = 4
    clip_ratio: float = 0.20
    value_weight: float = 0.50
    entropy_weight: float = 0.01
    gradient_norm: float = 0.50
    discount_factor: float = 0.99
    gae_lambda: float = 0.95

    def apply(self, values: dict) -> None:
        values.update(asdict(self))


@dataclass(frozen=True, slots=True)
class TSHCALOEnvironmentHyperparameters:
    feasible_archive_capacity: int = 32
    boundary_archive_capacity: int = 48
    memory_capacity: int = 256
    memory_decay: float = 0.97
    credit_decay: float = 0.90
    credit_floor: float = 0.02
    group_credit_decay: float = 0.90
    max_learning_lane_fraction: float = 0.92
    precision_start_radius: float = 0.04
    precision_min_radius: float = 5e-4
    precision_max_radius: float = 0.15
    epsilon_quantile: float = 0.75
    epsilon_control_fraction: float = 0.65
    epsilon_exponent: float = 2.0
    stagnation_window: int = 12
    memory_evidence_batches: int = 6
    recovery_diversity_threshold: float = 0.06
    recovery_fraction: float = 0.18

    def apply(self, values: dict) -> None:
        values.update(asdict(self))


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingCampaignPlan:
    campaign_id: str
    source_commit: str
    development_freeze_commit: str
    development_freeze_sha256: str
    phase4_acceptance_sha256: str
    development_cases: tuple[str, ...]
    members: tuple[TSHCALOTrainingMemberPlan, ...]
    resource_envelope: TSHCALOTrainingResourceEnvelope
    population_size: int
    max_evaluations: int
    training: TSHCALOTrainingHyperparameters = field(default_factory=TSHCALOTrainingHyperparameters)
    environment: TSHCALOEnvironmentHyperparameters = field(
        default_factory=TSHCALOEnvironmentHyperparameters
    )
    deterministic_policy: bool = False
    environment_deterministic: bool = False
    requested_device: str = "auto"
    allow_cpu_fallback: bool = True
    feature_flags: TSHCALOFeatureFlags = field(default_factory=TSHCALOFeatureFlags)
    schema_version: str = TSH_CALO_TRAINING_CAMPAIGN_SCHEMA

    def validate(self) -> None:
        if self.schema_version != TSH_CALO_TRAINING_CAMPAIGN_SCHEMA:
            raise ValueError("TSH-CALO campaign schema is incompatible")
        if not isinstance(self.campaign_id, str) or not self.campaign_id.strip():
            raise ValueError("TSH-CALO campaign requires an ID")
        if not _valid_commit(self.source_commit):
            raise ValueError("TSH-CALO campaign requires an exact 40-character source commit")
        if (
            not _valid_commit(self.development_freeze_commit)
            or self.development_freeze_commit.lower() != self.source_commit.lower()
        ):
            raise ValueError(
                "TSH-CALO campaign source must equal the accepted development-freeze commit"
            )
        if len(str(self.development_freeze_sha256)) != 64 or any(
            character not in "0123456789abcdef"
            for character in str(self.development_freeze_sha256).lower()
        ):
            raise ValueError("TSH-CALO campaign requires the development-freeze payload SHA-256")
        if len(str(self.phase4_acceptance_sha256)) != 64 or any(
            character not in "0123456789abcdef"
            for character in str(self.phase4_acceptance_sha256).lower()
        ):
            raise ValueError("TSH-CALO campaign requires the Phase 4 acceptance receipt SHA-256")
        if len(self.members) < 2:
            raise ValueError(
                "TSH-CALO epistemic training requires at least two independent members"
            )
        if not self.development_cases or len(set(self.development_cases)) != len(
            self.development_cases
        ):
            raise ValueError("TSH-CALO campaign development cases must be non-empty and unique")
        leaked = protected_holdout_matches(self.development_cases)
        if leaked:
            raise ValueError(
                "Protected holdout cases cannot enter TSH-CALO training: " + ", ".join(leaked)
            )
        for member in self.members:
            member.validate()
        member_ids = [member.member_id for member in self.members]
        member_seeds = [member.training_seed for member in self.members]
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("TSH-CALO campaign member IDs must be unique")
        if len(set(member_seeds)) != len(member_seeds):
            raise ValueError("TSH-CALO independent member training seeds must be unique")
        all_session_ids = [
            episode.session_id for member in self.members for episode in member.episodes
        ]
        if len(set(all_session_ids)) != len(all_session_ids):
            raise ValueError("TSH-CALO campaign session IDs must be globally unique")
        expected_cases = set(self.development_cases)
        reference_sequence = tuple(episode.case_identity for episode in self.members[0].episodes)
        if set(reference_sequence) != expected_cases:
            raise ValueError(
                "Every declared TSH-CALO development case must occur in the curriculum"
            )
        for member in self.members:
            sequence = tuple(episode.case_identity for episode in member.episodes)
            if sequence != reference_sequence:
                raise ValueError(
                    "TSH-CALO ensemble members require the same frozen case curriculum"
                )
            if any(case not in expected_cases for case in sequence):
                raise ValueError("TSH-CALO campaign episode case is undeclared")
        if not isinstance(self.deterministic_policy, bool) or not isinstance(
            self.environment_deterministic, bool
        ):
            raise ValueError("TSH-CALO campaign determinism controls must be Boolean")
        if not isinstance(self.allow_cpu_fallback, bool):
            raise ValueError("TSH-CALO campaign CPU-fallback control must be Boolean")
        self.resource_envelope.validate()
        self.feature_flags.validate()
        if (
            self.feature_flags.population_schedule
            or self.feature_flags.allow_experimental_components
        ):
            raise ValueError(
                "Experimental Change F is not admitted by production-candidate training"
            )
        for member in self.members:
            training = self.training_config(member)
            for episode in member.episodes:
                self.environment_config(training, episode).validate(training)

    def seed_manifest(self) -> dict:
        return {
            "schema_version": TSH_CALO_TRAINING_SEED_MANIFEST_SCHEMA,
            "campaign_id": self.campaign_id,
            "members": [
                {
                    "member_id": member.member_id,
                    "training_seed": member.training_seed,
                    "episodes": [asdict(episode) for episode in member.episodes],
                }
                for member in self.members
            ],
        }

    def seed_manifest_sha256(self) -> str:
        return _canonical_sha256(self.seed_manifest())

    def scientific_design_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("requested_device", None)
        payload.pop("allow_cpu_fallback", None)
        return _canonical_sha256(payload)

    def execution_plan_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["development_cases"] = list(self.development_cases)
        payload["members"] = [
            {
                "member_id": member.member_id,
                "training_seed": member.training_seed,
                "episodes": [asdict(episode) for episode in member.episodes],
            }
            for member in self.members
        ]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "TSHCALOTrainingCampaignPlan":
        values = dict(payload or {})
        try:
            values["development_cases"] = tuple(values.get("development_cases", ()))
            values["members"] = tuple(
                TSHCALOTrainingMemberPlan(
                    member_id=item["member_id"],
                    training_seed=item["training_seed"],
                    episodes=tuple(
                        TSHCALOTrainingEpisodePlan(**episode)
                        for episode in item.get("episodes", ())
                    ),
                )
                for item in values.get("members", ())
            )
            values["resource_envelope"] = TSHCALOTrainingResourceEnvelope(
                **dict(values.get("resource_envelope", {}))
            )
            values["training"] = TSHCALOTrainingHyperparameters(**dict(values.get("training", {})))
            values["environment"] = TSHCALOEnvironmentHyperparameters(
                **dict(values.get("environment", {}))
            )
            values["feature_flags"] = TSHCALOFeatureFlags(**dict(values.get("feature_flags", {})))
            plan = cls(**values)
        except (KeyError, TypeError) as exc:
            raise ValueError("TSH-CALO campaign plan fields are incomplete") from exc
        plan.validate()
        return plan

    def training_config(self, member: TSHCALOTrainingMemberPlan) -> TSHCALOTrainingConfig:
        values = {
            "training_run_id": f"{self.campaign_id}:{member.member_id}",
            "development_cases": self.development_cases,
            "seed_manifest_sha256": self.seed_manifest_sha256(),
            "resource_envelope": self.resource_envelope,
            "development_freeze_commit": self.development_freeze_commit,
            "development_freeze_sha256": self.development_freeze_sha256,
            "phase4_acceptance_sha256": self.phase4_acceptance_sha256,
            "seed": member.training_seed,
            "device": self.requested_device,
            "allow_cpu_fallback": self.allow_cpu_fallback,
            "feature_flags": self.feature_flags,
        }
        self.training.apply(values)
        config = TSHCALOTrainingConfig(**values)
        config.validate()
        return config

    def environment_config(
        self,
        training: TSHCALOTrainingConfig,
        episode: TSHCALOTrainingEpisodePlan,
    ) -> TSHCALOTrainingEnvironmentConfig:
        values = {
            "case_identity": episode.case_identity,
            "population_size": self.population_size,
            "max_evaluations": self.max_evaluations,
            "seed": episode.seed,
            "environment_deterministic": self.environment_deterministic,
        }
        self.environment.apply(values)
        config = TSHCALOTrainingEnvironmentConfig(**values)
        config.validate(training)
        return config


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingCampaignResult:
    output_directory: str
    plan_sha256: str
    seed_manifest_sha256: str
    member_candidates: tuple[TSHCALOCandidateArtifact, ...]
    ensemble_candidate: TSHCALOCandidateArtifact
    manifest_path: str
    manifest_sha256: str


class IndependentTSHCALOTrainingCampaign:
    """Explicit start/resume runner with no experiment or policy-lifecycle authority."""

    PLAN_FILE = "training_plan.json"
    STATUS_FILE = "training_status.json"
    MANIFEST_FILE = "training_manifest.json"

    def __init__(
        self,
        plan: TSHCALOTrainingCampaignPlan,
        output_directory: str | Path,
        *,
        problem_factory: Callable[[str], object] | None = None,
        transition_callback: Callable[[dict], None] | None = None,
    ) -> None:
        plan.validate()
        self.plan = plan
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.problem_factory = problem_factory
        self.transition_callback = transition_callback
        self._active_session: IndependentTSHCALOTrainingSession | None = None
        self._last_failure_provenance: dict | None = None
        self._last_failure_resumable = False

    def _build_problem(self, case_identity: str, *, device_hint: str | None = None):
        if self.problem_factory is not None:
            problem = self.problem_factory(case_identity)
        else:
            selected = str(device_hint or self.plan.requested_device).strip().lower()
            case = CaseLoader.load(case_identity)
            if selected == "cpu":
                problem = ORPDProblem(case)
            else:
                problem = AcceleratedORPDProblem(
                    case,
                    device=selected,
                    batch_size=max(CUDA_DURABLE_EVALUATION_WINDOW, self.plan.population_size),
                    device_resident=True,
                    cuda_resident_hot_loop=True,
                    cuda_cpu_fallback_enabled=self.plan.allow_cpu_fallback,
                )
        if not callable(getattr(problem, "evaluate_with_context", None)):
            raise TypeError("TSH-CALO campaign problem must provide counted evaluate_with_context")
        if int(getattr(problem, "dimension", 0)) < 1:
            raise TypeError("TSH-CALO campaign problem must expose a positive dimension")
        return problem

    @property
    def _plan_path(self) -> Path:
        return self.output_directory / self.PLAN_FILE

    @property
    def _status_path(self) -> Path:
        return self.output_directory / self.STATUS_FILE

    def _checkpoint_path(self, member_index: int) -> Path:
        return self.output_directory / f"member-{member_index + 1:03d}.resume"

    def _candidate_path(self, member_index: int) -> Path:
        return self.output_directory / f"member-{member_index + 1:03d}.candidate.pt"

    def _write_status(self, status: dict) -> None:
        status["schema_version"] = TSH_CALO_TRAINING_CAMPAIGN_STATUS_SCHEMA
        status["plan_sha256"] = self.plan.execution_plan_sha256()
        _write_json(self._status_path, status)

    def start(self) -> TSHCALOTrainingCampaignResult:
        if self.output_directory.exists():
            raise FileExistsError(
                "TSH-CALO campaign start requires a new output directory; use explicit resume"
            )
        self.output_directory.mkdir(parents=True)
        _write_json(self._plan_path, self.plan.to_dict())
        _write_json(
            self.output_directory / "seed_manifest.json",
            self.plan.seed_manifest(),
        )
        status = {
            "state": "running",
            "current_member_index": 0,
            "current_episode_index": 0,
            "session_checkpoint": None,
            "uncommitted_cuda_window": None,
            "member_candidates": [],
            "failure": None,
        }
        self._write_status(status)
        return self._execute(status)

    def resume(self) -> TSHCALOTrainingCampaignResult:
        if not self._plan_path.is_file() or not self._status_path.is_file():
            raise FileNotFoundError("TSH-CALO campaign has no resumable plan/status records")
        stored_plan = TSHCALOTrainingCampaignPlan.from_dict(_read_json(self._plan_path))
        if stored_plan.execution_plan_sha256() != self.plan.execution_plan_sha256():
            raise ValueError("TSH-CALO campaign plan changed; exact resume is forbidden")
        status = _read_json(self._status_path)
        if status.get("schema_version") != TSH_CALO_TRAINING_CAMPAIGN_STATUS_SCHEMA:
            raise ValueError("TSH-CALO campaign status schema is incompatible")
        if status.get("plan_sha256") != self.plan.execution_plan_sha256():
            raise ValueError("TSH-CALO campaign status belongs to another plan")
        if status.get("state") == "failed":
            raise RuntimeError("Failed TSH-CALO campaigns cannot retry under the same identity")
        if status.get("state") == "completed":
            raise RuntimeError("TSH-CALO campaign is already complete")
        if status.get("state") not in {"running", "interrupted"}:
            raise ValueError("TSH-CALO campaign status is not resumable")
        if status.get("uncommitted_cuda_window") is not None:
            status["state"] = "failed"
            status["failure"] = {
                "type": "UncommittedCudaWindow",
                "message": (
                    "The process stopped inside a grouped CUDA evaluation window; exact resume "
                    "under the same campaign identity is forbidden."
                ),
                "category": "non_resumable_training_or_integrity_failure",
                "resumable": False,
                "window": dict(status["uncommitted_cuda_window"]),
            }
            self._write_status(status)
            raise RuntimeError(
                "TSH-CALO campaign stopped inside an uncommitted CUDA evaluation window"
            )
        status["state"] = "running"
        self._write_status(status)
        return self._execute(status)

    def _new_session(
        self,
        trainer: IndependentTSHCALOTrainer,
        training: TSHCALOTrainingConfig,
        episode: TSHCALOTrainingEpisodePlan,
    ) -> IndependentTSHCALOTrainingSession:
        problem = self._build_problem(episode.case_identity, device_hint=str(trainer.device))
        environment_config = self.plan.environment_config(training, episode)
        environment = IndependentTSHCALOTrainingEnvironment(
            problem,
            training,
            environment_config,
        )
        return IndependentTSHCALOTrainingSession(
            trainer,
            environment,
            TSHCALOTrainingSessionConfig(
                session_id=episode.session_id,
                deterministic_policy=self.plan.deterministic_policy,
            ),
        )

    def _restore_checkpoint_session(
        self,
        checkpoint: dict,
        member_index: int,
        training: TSHCALOTrainingConfig,
    ) -> tuple[IndependentTSHCALOTrainingSession, int]:
        checkpoint_member = int(checkpoint.get("member_index", -1))
        checkpoint_episode = int(checkpoint.get("episode_index", -1))
        if checkpoint_member != member_index:
            raise ValueError("TSH-CALO campaign checkpoint belongs to another member")
        episodes = self.plan.members[member_index].episodes
        if checkpoint_episode < 0 or checkpoint_episode >= len(episodes):
            raise ValueError("TSH-CALO campaign checkpoint episode index is invalid")
        checkpoint_name = str(checkpoint.get("path", ""))
        if checkpoint_name != self._checkpoint_path(member_index).name:
            raise ValueError("TSH-CALO campaign checkpoint path is outside its frozen campaign")
        checkpoint_path = self.output_directory / checkpoint_name
        if checkpoint_sha256(checkpoint_path) != str(checkpoint.get("sha256", "")):
            raise ValueError("TSH-CALO campaign checkpoint SHA-256 differs from its status record")
        episode = episodes[checkpoint_episode]
        resume_payload = load_trusted_resume(checkpoint_path, map_location="cpu")
        selected_device = str(
            dict(resume_payload.get("trainer", {}) or {})
            .get("training_device_provenance", {})
            .get("memory_admission", {})
            .get("selected_device", self.plan.requested_device)
        )
        problem = self._build_problem(episode.case_identity, device_hint=selected_device)
        session = IndependentTSHCALOTrainingSession.load_resume(
            checkpoint_path,
            problem=problem,
            training_config=training,
            environment_config=self.plan.environment_config(training, episode),
            session_config=TSHCALOTrainingSessionConfig(
                session_id=episode.session_id,
                deterministic_policy=self.plan.deterministic_policy,
            ),
        )
        return session, checkpoint_episode

    def _advance_session(
        self,
        session: IndependentTSHCALOTrainingSession,
        status: dict,
        member_index: int,
        episode_index: int,
    ) -> TSHCALOTrainingSessionResult:
        if session.completed:
            if session.receipt is None:
                raise ValueError("Completed TSH-CALO campaign session lacks its receipt")
            return TSHCALOTrainingSessionResult(
                session.receipt,
                tuple(session.update_metrics),
                tuple(session.canonical_rewards),
            )
        # Persist only after a bounded group of counted transitions instead of serializing the
        # complete CUDA learner/environment state after every step. The window is derived from the
        # declared population so it targets about 100 candidate evaluations without changing any
        # transition, PPO, reward, or FE-accounting semantics. An exception inside an uncommitted
        # window fails the campaign identity; partial work is never silently resumed.
        evaluations_per_transition = max(1, int(session.environment.config.population_size))
        transitions_per_window = max(
            1, CUDA_DURABLE_EVALUATION_WINDOW // evaluations_per_transition
        )
        while True:
            status["uncommitted_cuda_window"] = {
                "member_index": int(member_index),
                "episode_index": int(episode_index),
                "starting_transition_count": int(session.transition_count),
                "maximum_transitions": int(transitions_per_window),
                "target_candidate_evaluations": int(CUDA_DURABLE_EVALUATION_WINDOW),
            }
            self._write_status(status)
            result = session.advance(max_transitions=transitions_per_window)
            checkpoint_path = self._checkpoint_path(member_index)
            checkpoint_sha256 = session.save_resume(checkpoint_path)
            status["session_checkpoint"] = {
                "path": checkpoint_path.name,
                "sha256": checkpoint_sha256,
                "member_index": member_index,
                "episode_index": episode_index,
                "transition_count": session.transition_count,
            }
            status["uncommitted_cuda_window"] = None
            self._write_status(status)
            if self.transition_callback is not None:
                self.transition_callback(dict(status))
            if result is not None:
                return result

    def _validate_member_candidate(
        self,
        artifact: TSHCALOCandidateArtifact,
        training: TSHCALOTrainingConfig,
        expected_receipts: int,
    ) -> None:
        provenance = artifact.training_provenance
        if (
            artifact.artifact_kind != "single_policy_member"
            or provenance.get("training_run_id") != training.training_run_id
            or provenance.get("training_design_sha256") != training.scientific_design_hash()
            or provenance.get("source_commit") != self.plan.source_commit
            or provenance.get("development_freeze_commit") != self.plan.development_freeze_commit
            or provenance.get("development_freeze_sha256") != self.plan.development_freeze_sha256
            or provenance.get("phase4_acceptance_sha256") != self.plan.phase4_acceptance_sha256
            or len(provenance.get("training_episode_receipts", ())) != expected_receipts
        ):
            raise ValueError(
                "Existing TSH-CALO member candidate does not match the frozen campaign"
            )

    def _existing_member_candidate(
        self,
        path: Path,
        training: TSHCALOTrainingConfig,
        expected_receipts: int,
    ) -> TSHCALOCandidateArtifact | None:
        if not path.is_file():
            return None
        artifact = inspect_tsh_calo_candidate(path)
        self._validate_member_candidate(artifact, training, expected_receipts)
        return artifact

    def _run_member(self, status: dict, member_index: int) -> TSHCALOCandidateArtifact:
        self._last_failure_provenance = None
        self._last_failure_resumable = False
        member = self.plan.members[member_index]
        training = self.plan.training_config(member)
        episode_index = int(status.get("current_episode_index", 0))
        checkpoint = status.get("session_checkpoint")
        session: IndependentTSHCALOTrainingSession | None = None
        trainer: IndependentTSHCALOTrainer | None = None
        if isinstance(checkpoint, dict) and int(checkpoint.get("member_index", -1)) == member_index:
            session, checkpoint_episode = self._restore_checkpoint_session(
                checkpoint, member_index, training
            )
            trainer = session.trainer
            if checkpoint_episode == episode_index:
                self._active_session = session
                self._advance_session(session, status, member_index, episode_index)
                episode_index += 1
                status["current_episode_index"] = episode_index
                self._write_status(status)
            elif checkpoint_episode != episode_index - 1 or not session.completed:
                raise ValueError("TSH-CALO campaign checkpoint/status progression is inconsistent")
        else:
            trainer = IndependentTSHCALOTrainer(training)
        assert trainer is not None
        try:
            for episode_index in range(episode_index, len(member.episodes)):
                session = self._new_session(trainer, training, member.episodes[episode_index])
                self._active_session = session
                self._advance_session(session, status, member_index, episode_index)
                status["current_episode_index"] = episode_index + 1
                self._write_status(status)
            candidate_path = self._candidate_path(member_index)
            candidate = self._existing_member_candidate(
                candidate_path, training, len(member.episodes)
            )
            if candidate is None:
                candidate = trainer.export_unqualified_candidate(
                    candidate_path,
                    source_commit=self.plan.source_commit,
                )
            return candidate
        except Exception as exc:
            if self._active_session is not None:
                self._last_failure_provenance = (
                    self._active_session.environment.scientific_provenance()
                )
                self._last_failure_resumable = bool(
                    isinstance(exc, OSError)
                    and not self._active_session.failed
                    and self._active_session.environment.accounting_complete
                )
            raise
        finally:
            trainer.close()
            self._active_session = None

    def _existing_ensemble(
        self, path: Path, members: list[TSHCALOCandidateArtifact]
    ) -> TSHCALOCandidateArtifact | None:
        if not path.is_file():
            return None
        artifact = inspect_tsh_calo_candidate(path)
        recorded = [
            item.get("source_candidate_sha256")
            for item in artifact.training_provenance.get("members", ())
        ]
        expected = [member.sha256 for member in members]
        if artifact.artifact_kind != "ensemble_policy" or recorded != expected:
            raise ValueError(
                "Existing TSH-CALO ensemble does not match the frozen campaign members"
            )
        return artifact

    def _execute(self, status: dict) -> TSHCALOTrainingCampaignResult:
        try:
            members = []
            for candidate_index, item in enumerate(status.get("member_candidates", ())):
                expected_path = self._candidate_path(candidate_index)
                if str(item.get("path", "")) != expected_path.name:
                    raise ValueError(
                        "TSH-CALO campaign candidate path is outside its frozen campaign"
                    )
                artifact = inspect_tsh_calo_candidate(
                    expected_path,
                    expected_sha256=item["sha256"],
                )
                member_plan = self.plan.members[candidate_index]
                self._validate_member_candidate(
                    artifact,
                    self.plan.training_config(member_plan),
                    len(member_plan.episodes),
                )
                members.append(artifact)
            member_index = int(status.get("current_member_index", 0))
            if len(members) != member_index:
                raise ValueError("TSH-CALO campaign candidate/status progression is inconsistent")
            while member_index < len(self.plan.members):
                candidate = self._run_member(status, member_index)
                members.append(candidate)
                status["member_candidates"] = [
                    {"path": Path(item.path).name, "sha256": item.sha256} for item in members
                ]
                member_index += 1
                status["current_member_index"] = member_index
                status["current_episode_index"] = 0
                self._write_status(status)
            ensemble_path = self.output_directory / "ensemble.candidate.pt"
            ensemble = self._existing_ensemble(ensemble_path, members)
            if ensemble is None:
                ensemble = assemble_tsh_calo_ensemble_candidate(
                    ensemble_path,
                    [(member.path, member.sha256) for member in members],
                )
            manifest = {
                "schema_version": TSH_CALO_TRAINING_CAMPAIGN_SCHEMA,
                "state": "completed_unqualified",
                "campaign_id": self.plan.campaign_id,
                "source_commit": self.plan.source_commit,
                "development_freeze_commit": self.plan.development_freeze_commit,
                "development_freeze_sha256": self.plan.development_freeze_sha256,
                "phase4_acceptance_sha256": self.plan.phase4_acceptance_sha256,
                "scientific_design_sha256": self.plan.scientific_design_hash(),
                "execution_plan_sha256": self.plan.execution_plan_sha256(),
                "seed_manifest_sha256": self.plan.seed_manifest_sha256(),
                "member_candidates": [
                    {"path": Path(item.path).name, "sha256": item.sha256} for item in members
                ],
                "ensemble_candidate": {
                    "path": Path(ensemble.path).name,
                    "sha256": ensemble.sha256,
                    "lifecycle_status": "candidate_unqualified",
                },
                "authority_boundary": {
                    "training_only": True,
                    "registered": False,
                    "qualified": False,
                    "activated": False,
                    "experiment_bound": False,
                },
            }
            manifest_path = self.output_directory / self.MANIFEST_FILE
            manifest_sha256 = _write_json(manifest_path, manifest)
            status["state"] = "completed"
            status["manifest_sha256"] = manifest_sha256
            self._write_status(status)
            return TSHCALOTrainingCampaignResult(
                str(self.output_directory),
                self.plan.execution_plan_sha256(),
                self.plan.seed_manifest_sha256(),
                tuple(members),
                ensemble,
                str(manifest_path),
                manifest_sha256,
            )
        except KeyboardInterrupt:
            status["state"] = "interrupted"
            self._write_status(status)
            raise
        except Exception as exc:
            status["state"] = "interrupted" if self._last_failure_resumable else "failed"
            status["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "category": (
                    "resumable_infrastructure_interruption"
                    if self._last_failure_resumable
                    else "non_resumable_training_or_integrity_failure"
                ),
                "resumable": self._last_failure_resumable,
                "environment_provenance": (
                    self._last_failure_provenance
                    if self._active_session is None
                    else self._active_session.environment.scientific_provenance()
                ),
            }
            self._write_status(status)
            raise
