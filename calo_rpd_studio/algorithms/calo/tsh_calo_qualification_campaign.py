"""Preregistered independent quality campaign for an unqualified TSH-CALO ensemble.

This module has no registry or activation authority.  It evaluates one immutable candidate under a
non-serializable qualification capability, compares it with frozen CALO under paired seeds and equal
FE, independently validates every retained solution, and emits a receipt only for a formal campaign
that satisfies every frozen gate.  Screening campaigns can never emit a qualification receipt.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
import uuid

import numpy as np

from calo_rpd_studio.ai.model_io import (
    checkpoint_sha256,
    durable_write_bytes,
    trusted_resume_sha_path,
)
from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.registry import SPECS, create_optimizer
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.experiment_runner import build_problem
from calo_rpd_studio.experiments.seed_manager import RunSeeds, SeedManager
from calo_rpd_studio.orpd.formulation_fingerprint import scientific_problem_fingerprint
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.power_system.case_identity import protected_holdout_matches
from calo_rpd_studio.statistics.paired import (
    DEFAULT_OBJECTIVE_SCALE_FLOOR,
    PAIRED_ANALYSIS_SCHEMA_VERSION,
    RELATIVE_IMPROVEMENT_VERSION,
    exact_keyed_pairs,
    matched_pairs_rank_biserial,
    pair_manifest,
    relative_objective_improvement,
    wilcoxon_signed_rank_evidence,
)
from calo_rpd_studio.statistics.posthoc import holm_correction

from .tsh_calo_inference import (
    QualificationCandidateAuthority,
    TSHCALOInferenceController,
)
from .tsh_calo_optimizer import TSHCALOOptimizer
from .tsh_calo_policy_artifact import (
    TSHCALOCandidateArtifact,
    inspect_tsh_calo_candidate,
)
from .tsh_calo_qualification import build_tsh_calo_qualification_receipt
from .tsh_calo_schema import TSH_CALO_ALGORITHM_ID, TSH_CALO_POLICY_ARCHITECTURE
from .tsh_calo_shield import OODCalibration, ood_calibration_sha256, topology_ood_signature
from .tsh_calo_training import TSHCALOTrainingConfig
from .tsh_calo_training_environment import (
    IndependentTSHCALOTrainingEnvironment,
    TSHCALOTrainingEnvironmentConfig,
)
from .tsh_calo_training_resources import TSHCALOTrainingResourceEnvelope


TSH_CALO_QUALIFICATION_PLAN_SCHEMA = "tsh-calo-qualification-plan-v2-exact-pairs"
TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA = "tsh-calo-qualification-evidence-v2-exact-pairs"
TSH_CALO_COMPONENT_EVIDENCE_SCHEMA = "tsh-calo-component-ablation-evidence-v2-exact-pairs"
_REQUIRED_COMPONENTS = ("A", "B", "C", "D", "E")
TSH_CALO_CANDIDATE_CONTRACT_SCHEMA = "tsh-calo-candidate-contract-v1"
TSH_CALO_QUALIFICATION_STATUS_SCHEMA = "tsh-calo-qualification-status-v1"
TSH_CALO_QUALIFICATION_CONTROL_SCHEMA = "tsh-calo-qualification-control-v1"
TSH_CALO_QUALIFICATION_EVENT_SCHEMA = "tsh-calo-qualification-progress-event-v1"
TSH_CALO_QUALIFICATION_PAUSE_EXIT_CODE = 75
QUALIFICATION_STATUS_FILE = "qualification_status.json"
QUALIFICATION_CONTROL_FILE = "qualification_control.json"
QUALIFICATION_EVENT_LOG_FILE = "qualification_events.jsonl"
QUALIFICATION_CHECKPOINT_INTERVAL = 500
_CANDIDATE_CONTRACT_FIELDS = {
    "schema_version",
    "candidate_sha256",
    "algorithm_id",
    "runtime_architecture_version",
    "policy_architecture_version",
    "state_schema_version",
    "action_schema_version",
    "training_environment_version",
    "artifact_kind",
    "ensemble_size",
    "feature_contract",
    "member_candidate_sha256",
    "member_training_design_sha256",
    "training_provenance_sha256",
}


class QualificationCampaignLeaseUnavailable(RuntimeError):
    """Raised when another process or thread owns the same evidence directory."""


class TSHCALOQualificationPauseRequested(RuntimeError):
    """The campaign acknowledged a pause at an authenticated durable boundary."""

    def __init__(self, event: dict) -> None:
        super().__init__("TSH-CALO qualification paused at a verified checkpoint")
        self.event = dict(event)


class _ExclusiveQualificationCampaignLease:
    """OS-released single-writer lease; a timeout cannot leave a stale ownership claim."""

    _guard = threading.RLock()
    _owned: set[str] = set()

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = (directory / "qualification_campaign.lock").resolve()
        self.key = str(self.path).lower()
        self.stream = None
        with self._guard:
            if self.key in self._owned:
                raise QualificationCampaignLeaseUnavailable(
                    "This process already owns the TSH-CALO qualification evidence directory"
                )
            stream = open(self.path, "a+b")  # noqa: SIM115 - held for campaign lifetime
            try:
                self._lock_stream(stream)
            except BaseException:
                stream.close()
                raise
            self.stream = stream
            self._owned.add(self.key)

    @staticmethod
    def _lock_stream(stream) -> None:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise QualificationCampaignLeaseUnavailable(
                "Another process owns the TSH-CALO qualification evidence directory"
            ) from exc

    def close(self) -> None:
        stream = self.stream
        if stream is None:
            return
        self.stream = None
        with self._guard:
            self._owned.discard(self.key)
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


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
        raise ValueError(f"TSH-CALO qualification record is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"TSH-CALO qualification record must be an object: {path}")
    return payload


def _append_json_line(path: Path, payload: dict) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def request_tsh_calo_qualification_pause(output_directory: str | Path) -> dict:
    """Record an idempotent safe-pause request for one running qualification plan."""

    output = Path(output_directory).expanduser().resolve(strict=True)
    plan = TSHCALOQualificationPlan.from_dict(_read_json(output / "qualification_plan.json"))
    status = _read_json(output / QUALIFICATION_STATUS_FILE)
    plan_sha256 = plan.execution_plan_sha256()
    if status.get("schema_version") != TSH_CALO_QUALIFICATION_STATUS_SCHEMA:
        raise ValueError("TSH-CALO qualification status schema is incompatible")
    if status.get("qualification_plan_sha256") != plan_sha256:
        raise ValueError("TSH-CALO qualification status belongs to another frozen plan")
    if status.get("state") != "running":
        raise RuntimeError("Only a running TSH-CALO qualification can accept a safe pause")
    control_path = output / QUALIFICATION_CONTROL_FILE
    if control_path.is_file():
        existing = _read_json(control_path)
        if (
            existing.get("schema_version") == TSH_CALO_QUALIFICATION_CONTROL_SCHEMA
            and existing.get("state") == "requested"
            and existing.get("action") == "pause"
            and existing.get("qualification_run_id") == plan.qualification_run_id
            and existing.get("qualification_plan_sha256") == plan_sha256
        ):
            return existing
    request = {
        "schema_version": TSH_CALO_QUALIFICATION_CONTROL_SCHEMA,
        "request_id": uuid.uuid4().hex,
        "action": "pause",
        "state": "requested",
        "qualification_run_id": plan.qualification_run_id,
        "qualification_plan_sha256": plan_sha256,
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(control_path, request)
    return request


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _mean_or_none(values) -> float | None:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)) if array.size else None


@dataclass(frozen=True, slots=True)
class TSHCALOQualificationPlan:
    qualification_run_id: str
    source_commit: str
    candidate_path: str
    candidate_sha256: str
    development_cases: tuple[str, ...]
    runs: int
    master_seed: int
    population_size: int
    max_evaluations: int
    source_tracked_clean: bool = False
    mode: str = "screening"  # screening | formal
    calibration_samples_per_case: int = 8
    calibration_population_size: int = 40
    calibration_quantile: float = 0.95
    minimum_neural_weight: float = 0.0
    inference_device: str = "auto"
    allow_cpu_fallback: bool = True
    statistical_alpha: float = 0.05
    minimum_feasible_probability: float = 0.95
    feasibility_noninferiority_margin: float = 0.05
    minimum_objective_pair_fraction: float = 0.80
    minimum_relative_improvement: float = 0.002
    minimum_win_rate: float = 0.60
    minimum_rank_biserial: float = 0.20
    anytime_regression_tolerance: float = 0.01
    anytime_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    bootstrap_resamples: int = 10_000
    candidate_contract: dict = field(default_factory=dict)
    component_evidence: dict[str, dict] = field(default_factory=dict)
    analysis_schema_version: str = PAIRED_ANALYSIS_SCHEMA_VERSION
    relative_improvement_version: str = RELATIVE_IMPROVEMENT_VERSION
    objective_scale_floor: float = DEFAULT_OBJECTIVE_SCALE_FLOOR
    schema_version: str = TSH_CALO_QUALIFICATION_PLAN_SCHEMA

    def validate(self) -> None:
        if self.schema_version != TSH_CALO_QUALIFICATION_PLAN_SCHEMA:
            raise ValueError("TSH-CALO qualification plan schema is incompatible")
        if self.analysis_schema_version != PAIRED_ANALYSIS_SCHEMA_VERSION:
            raise ValueError("TSH-CALO paired-analysis schema is incompatible")
        if self.relative_improvement_version != RELATIVE_IMPROVEMENT_VERSION:
            raise ValueError("TSH-CALO relative-improvement schema is incompatible")
        if (
            not math.isfinite(float(self.objective_scale_floor))
            or self.objective_scale_floor <= 0.0
        ):
            raise ValueError("TSH-CALO objective scale floor must be finite and positive")
        if not self.qualification_run_id.strip():
            raise ValueError("TSH-CALO qualification requires a run ID")
        commit = str(self.source_commit).strip().lower()
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError("TSH-CALO qualification requires an exact source commit")
        if not isinstance(self.source_tracked_clean, bool):
            raise ValueError("TSH-CALO qualification source clean state must be Boolean")
        if not _is_sha256(self.candidate_sha256):
            raise ValueError("TSH-CALO qualification candidate SHA-256 is invalid")
        if not self.development_cases or len(set(self.development_cases)) != len(
            self.development_cases
        ):
            raise ValueError(
                "TSH-CALO qualification development cases must be non-empty and unique"
            )
        leaked = protected_holdout_matches(self.development_cases)
        if leaked:
            raise ValueError(
                "Protected holdouts cannot enter TSH-CALO qualification: " + ", ".join(leaked)
            )
        if self.mode not in {"screening", "formal"}:
            raise ValueError("TSH-CALO qualification mode must be screening or formal")
        if self.runs < 2 or (self.mode == "formal" and self.runs < 30):
            raise ValueError(
                "Formal TSH-CALO qualification requires at least 30 paired runs per case"
            )
        if self.population_size < 2 or self.max_evaluations < 2 * self.population_size:
            raise ValueError("TSH-CALO qualification population/FE budget is too small")
        if self.max_evaluations % self.population_size:
            raise ValueError("TSH-CALO qualification FE budget must divide exactly by population")
        if self.calibration_samples_per_case < 4 or self.calibration_population_size < 2:
            raise ValueError("TSH-CALO OOD calibration requires at least four states per case")
        if not 0.50 <= self.calibration_quantile < 1.0:
            raise ValueError("TSH-CALO OOD calibration quantile must be within [0.5, 1)")
        if not 0.0 <= self.minimum_neural_weight <= 1.0:
            raise ValueError("TSH-CALO minimum neural weight must be within [0, 1]")
        if self.inference_device not in {"auto", "cpu", "cuda"} and not str(
            self.inference_device
        ).startswith("cuda:"):
            raise ValueError("TSH-CALO qualification inference device is invalid")
        probabilities = (
            self.statistical_alpha,
            self.minimum_feasible_probability,
            self.feasibility_noninferiority_margin,
            self.minimum_objective_pair_fraction,
            self.minimum_win_rate,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
            raise ValueError("TSH-CALO qualification probability controls must be within [0, 1]")
        if self.statistical_alpha <= 0.0 or self.minimum_rank_biserial < -1.0:
            raise ValueError("TSH-CALO qualification statistical controls are invalid")
        if self.minimum_relative_improvement < 0.0 or self.anytime_regression_tolerance < 0.0:
            raise ValueError("TSH-CALO qualification practical margins cannot be negative")
        if self.bootstrap_resamples < 1_000:
            raise ValueError("TSH-CALO qualification requires at least 1,000 bootstrap resamples")
        fractions = tuple(float(item) for item in self.anytime_fractions)
        if not fractions or tuple(sorted(set(fractions))) != fractions:
            raise ValueError("TSH-CALO anytime fractions must be unique and increasing")
        if fractions[-1] != 1.0 or any(not 0.0 < item <= 1.0 for item in fractions):
            raise ValueError("TSH-CALO anytime fractions must end at 1.0 and lie within (0, 1]")
        has_candidate_contract = bool(self.candidate_contract)
        has_legacy_component_evidence = bool(self.component_evidence)
        if self.mode == "formal" and not (
            has_candidate_contract or set(self.component_evidence) == set(_REQUIRED_COMPONENTS)
        ):
            raise ValueError(
                "Formal qualification requires a frozen candidate architecture contract"
            )
        if has_candidate_contract and has_legacy_component_evidence:
            raise ValueError(
                "A qualification plan cannot mix the current architecture contract with legacy "
                "component evidence"
            )
        if has_candidate_contract:
            if set(self.candidate_contract) != _CANDIDATE_CONTRACT_FIELDS:
                raise ValueError(
                    "Qualification candidate architecture contract fields are incomplete"
                )
            if self.candidate_contract.get("schema_version") != (
                TSH_CALO_CANDIDATE_CONTRACT_SCHEMA
            ):
                raise ValueError("Qualification candidate architecture contract is incompatible")
            if str(self.candidate_contract.get("candidate_sha256", "")).lower() != (
                self.candidate_sha256.lower()
            ):
                raise ValueError("Qualification candidate contract belongs to another checkpoint")
            if (
                self.candidate_contract.get("algorithm_id") != TSH_CALO_ALGORITHM_ID
                or self.candidate_contract.get("artifact_kind") != "ensemble_policy"
                or int(self.candidate_contract.get("ensemble_size", 0)) < 2
                or not isinstance(self.candidate_contract.get("feature_contract"), dict)
            ):
                raise ValueError("Qualification candidate architecture contract is invalid")
            member_sha = list(self.candidate_contract.get("member_candidate_sha256", []) or [])
            training_sha = list(
                self.candidate_contract.get("member_training_design_sha256", []) or []
            )
            if (
                len(member_sha) != int(self.candidate_contract["ensemble_size"])
                or len(training_sha) != int(self.candidate_contract["ensemble_size"])
                or any(not _is_sha256(item) for item in member_sha + training_sha)
                or not _is_sha256(self.candidate_contract.get("training_provenance_sha256", ""))
            ):
                raise ValueError("Qualification candidate member contract is invalid")

    def seed_manifest(self) -> dict:
        self.validate()
        runs = SeedManager(self.master_seed).generate(self.runs)
        calibration = SeedManager(self.master_seed + 1_000_003).generate(
            self.calibration_samples_per_case
        )
        return {
            "schema_version": "tsh-calo-qualification-seeds-v1",
            "qualification_run_id": self.qualification_run_id,
            "paired_runs": [asdict(item) for item in runs],
            "calibration_runs": [asdict(item) for item in calibration],
        }

    def seed_manifest_sha256(self) -> str:
        return _canonical_sha256(self.seed_manifest())

    def scientific_payload(self) -> dict:
        payload = self.to_dict()
        payload.pop("candidate_path", None)
        payload.pop("inference_device", None)
        payload.pop("allow_cpu_fallback", None)
        for value in payload.get("component_evidence", {}).values():
            value.pop("path", None)
        return payload

    def scientific_design_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.scientific_payload())

    def execution_plan_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["development_cases"] = list(self.development_cases)
        payload["anytime_fractions"] = list(self.anytime_fractions)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "TSHCALOQualificationPlan":
        values = dict(payload or {})
        values["development_cases"] = tuple(values.get("development_cases", ()))
        values["anytime_fractions"] = tuple(values.get("anytime_fractions", (0.25, 0.5, 0.75, 1.0)))
        values["component_evidence"] = {
            str(key): dict(value)
            for key, value in dict(values.get("component_evidence", {})).items()
        }
        values["candidate_contract"] = dict(values.get("candidate_contract", {}) or {})
        plan = cls(**values)
        plan.validate()
        return plan


class _QualificationCandidateOptimizer(TSHCALOOptimizer):
    """Exact TSH-CALO optimizer with a capability object that cannot arrive through JSON config."""

    def __init__(self, *args, qualification_authority: QualificationCandidateAuthority, **kwargs):
        self._qualification_authority = qualification_authority
        super().__init__(*args, **kwargs)

    def _build_inference_controller(
        self, parameters: dict, calibration: OODCalibration
    ) -> TSHCALOInferenceController:
        return TSHCALOInferenceController(
            parameters,
            ood_calibration=calibration,
            expected_ood_calibration_sha256=ood_calibration_sha256(calibration),
            deterministic=True,
            seed=int(parameters.get("ai_inference_seed", self.seed + 7919)),
            requested_device=str(parameters.get("inference_device", "auto")),
            allow_cpu_fallback=bool(parameters.get("allow_cpu_fallback", True)),
            baseline_fallback_permitted=False,
            _qualification_authority=self._qualification_authority,
        )


def _base_experiment_config(plan: TSHCALOQualificationPlan, case_name: str) -> ExperimentConfig:
    config = ExperimentConfig()
    config.name = f"{plan.qualification_run_id} development qualification"
    config.case_name = str(case_name)
    config.study_case_plan = [str(case_name)]
    config.algorithms = ["CALO", TSH_CALO_ALGORITHM_ID]
    config.runs = int(plan.runs)
    config.master_seed = int(plan.master_seed)
    config.population_size = int(plan.population_size)
    config.max_iterations = int(plan.max_evaluations)
    config.budget.max_evaluations = int(plan.max_evaluations)
    config.scientific_backend = "cpu_reference"
    config.runtime_compute_device = "cpu"
    config.execution_backend = "cpu_only"
    config.device_resident_execution = False
    return config


def _candidate_binding(artifact: TSHCALOCandidateArtifact, calibration: OODCalibration) -> dict:
    members = list(artifact.training_provenance.get("members", []) or [])
    return {
        "policy_algorithm_id": artifact.algorithm_id,
        "policy_id": "unregistered-qualification-candidate",
        "policy_checkpoint": artifact.path,
        "policy_sha256": artifact.sha256,
        "policy_architecture_version": artifact.algorithm_version,
        "policy_state_schema_version": artifact.state_schema_version,
        "policy_action_schema_version": artifact.action_schema_version,
        "policy_training_environment_version": artifact.training_environment_version,
        "policy_qualification_status": "candidate_unqualified",
        "policy_active_at_binding": False,
        "policy_artifact_kind": artifact.artifact_kind,
        "policy_ensemble_size": artifact.ensemble_size,
        "policy_ensemble_members": members,
        "policy_feature_flags": dict(artifact.feature_flags),
        "strict_policy_binding": True,
        "deterministic_policy": True,
        "policy_ood_calibration_sha256": ood_calibration_sha256(calibration),
        "ood_calibration": {
            "mean": np.asarray(calibration.mean, dtype=float).tolist(),
            "scale": np.asarray(calibration.scale, dtype=float).tolist(),
            "attenuation_start": float(calibration.attenuation_start),
            "minimum_neural_weight": float(calibration.minimum_neural_weight),
        },
    }


def _collect_calibration(
    plan: TSHCALOQualificationPlan,
    seeds: list[RunSeeds],
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[OODCalibration, dict]:
    signatures: list[np.ndarray] = []
    records: list[dict] = []
    envelope = TSHCALOTrainingResourceEnvelope(
        rollout_capacity=1,
        maximum_population_size=int(plan.calibration_population_size),
        maximum_topology_nodes=300,
        maximum_topology_edges=1_000,
        maximum_topology_controls=256,
        maximum_scenarios=64,
    )
    training = TSHCALOTrainingConfig(
        training_run_id=plan.qualification_run_id + "-calibration-state-collector",
        development_cases=tuple(plan.development_cases),
        seed_manifest_sha256=plan.seed_manifest_sha256(),
        resource_envelope=envelope,
        seed=int(plan.master_seed),
        device="cpu",
    )
    for case_name in plan.development_cases:
        config = _base_experiment_config(plan, case_name)
        for index, seed in enumerate(seeds):
            problem = build_problem(config, seed.scenario_seed)
            environment = IndependentTSHCALOTrainingEnvironment(
                problem,
                training,
                TSHCALOTrainingEnvironmentConfig(
                    case_identity=case_name,
                    population_size=int(plan.calibration_population_size),
                    max_evaluations=2 * int(plan.calibration_population_size),
                    seed=int(seed.algorithm_seed),
                    environment_deterministic=True,
                ),
            )
            observation = environment.reset()
            signature = topology_ood_signature(observation.policy_state)
            signatures.append(signature)
            records.append(
                {
                    "case": case_name,
                    "sample_index": index,
                    "algorithm_seed": int(seed.algorithm_seed),
                    "scenario_seed": int(seed.scenario_seed),
                    "candidate_evaluations": int(observation.candidate_evaluations),
                    "scenario_power_flow_calls": int(observation.scenario_power_flow_calls),
                    "problem_fingerprint": scientific_problem_fingerprint(problem),
                    "signature_sha256": hashlib.sha256(
                        np.ascontiguousarray(signature, dtype=np.float64).tobytes()
                    ).hexdigest(),
                }
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        **records[-1],
                        "completed_samples": len(records),
                        "total_samples": len(plan.development_cases) * len(seeds),
                    }
                )
    matrix = np.stack(signatures)
    mean = matrix.mean(axis=0)
    raw_scale = matrix.std(axis=0, ddof=1)
    scale = np.where(raw_scale > 1e-8, raw_scale, 1.0)
    scores = np.sqrt(np.mean(np.square((matrix - mean) / scale), axis=1))
    attenuation_start = max(1e-6, float(np.quantile(scores, plan.calibration_quantile)))
    calibration = OODCalibration(
        mean,
        scale,
        attenuation_start,
        float(plan.minimum_neural_weight),
    )
    calibration.validate()
    return calibration, {
        "schema_version": "tsh-calo-development-ood-calibration-evidence-v1",
        "fit_method": "coordinate_mean_and_sample_standard_deviation",
        "constant_coordinate_scale": 1.0,
        "attenuation_quantile": float(plan.calibration_quantile),
        "sample_count": int(len(matrix)),
        "signature_dimension": int(matrix.shape[1]),
        "within_development_score_quantiles": {
            "q50": float(np.quantile(scores, 0.50)),
            "q90": float(np.quantile(scores, 0.90)),
            "q95": float(np.quantile(scores, 0.95)),
            "maximum": float(np.max(scores)),
        },
        "candidate_evaluations": int(sum(item["candidate_evaluations"] for item in records)),
        "scenario_power_flow_calls": int(
            sum(item["scenario_power_flow_calls"] for item in records)
        ),
        "records": records,
        "calibration": {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "attenuation_start": attenuation_start,
            "minimum_neural_weight": float(plan.minimum_neural_weight),
        },
        "ood_calibration_sha256": ood_calibration_sha256(calibration),
    }


def _independent_validate(config: ExperimentConfig, seeds: RunSeeds, result) -> dict:
    problem = build_problem(config, seeds.scenario_seed)
    controlled, _ = problem.decoder.decode(np.asarray(result.best_vector, dtype=float))
    checks = []
    for scenario in problem.scenarios:
        formulation_case = scenario.apply(controlled)
        internal = run_ac_power_flow(formulation_case, config.power_flow)
        try:
            from calo_rpd_studio.power_system.independent_validator import validate_against_pypower
        except ModuleNotFoundError as exc:
            return {
                "available": False,
                "passed": False,
                "reason": f"independent_validator_unavailable:{exc}",
                "scenarios": [],
            }
        cross = validate_against_pypower(
            formulation_case, internal, power_flow_options=config.power_flow
        )
        checks.append(
            {
                "scenario": str(scenario.name),
                "available": bool(cross.available),
                "passed": bool(cross.passed),
                "message": str(cross.message),
                "max_vm_difference": _finite_or_none(cross.max_vm_difference),
                "max_va_difference_deg": _finite_or_none(cross.max_va_difference_deg),
                "loss_difference_mw": _finite_or_none(cross.loss_difference_mw),
            }
        )
    return {
        "available": bool(checks) and all(item["available"] for item in checks),
        "passed": bool(checks) and all(item["available"] and item["passed"] for item in checks),
        "scenarios": checks,
    }


def _history_at_fraction(result, fraction: float) -> dict:
    metadata = dict(result.metadata or {})
    evaluations = np.asarray(metadata.get("convergence_evaluations", []), dtype=int)
    objectives = np.asarray(metadata.get("best_feasible_objective_history", []), dtype=float)
    violations = np.asarray(metadata.get("best_constraint_violation_history", []), dtype=float)
    target = int(math.ceil(float(result.evaluations) * float(fraction)))
    indices = np.flatnonzero(evaluations <= target)
    index = int(indices[-1]) if len(indices) else -1
    if index < 0:
        return {"evaluations": target, "feasible": False, "objective": None, "violation": None}
    objective = float(objectives[index])
    violation = float(violations[index])
    return {
        "evaluations": int(evaluations[index]),
        "feasible": bool(math.isfinite(objective)),
        "objective": _finite_or_none(objective),
        "violation": _finite_or_none(violation),
    }


def _result_record(
    *,
    label: str,
    case_name: str,
    run_index: int,
    seeds: RunSeeds,
    result,
    config: ExperimentConfig,
    anytime_fractions: tuple[float, ...],
) -> dict:
    independent = _independent_validate(config, seeds, result)
    return {
        "label": label,
        "case": case_name,
        "run_index": int(run_index),
        "seeds": asdict(seeds),
        "feasible": bool(result.feasible),
        "objective": _finite_or_none(result.best_objective),
        "violation": _finite_or_none(result.total_constraint_violation),
        "evaluations": int(result.evaluations),
        "iterations": int(result.iterations),
        "first_feasible_evaluation": result.metadata.get("first_feasible_evaluation"),
        "runtime_seconds": float(result.runtime_seconds),
        "best_vector": np.asarray(result.best_vector, dtype=float).tolist(),
        "problem_fingerprint": scientific_problem_fingerprint(
            build_problem(config, seeds.scenario_seed)
        ),
        "scenario_manifest": dict(result.metadata.get("scenario_manifest", {}) or {}),
        "device_admission": dict(result.metadata.get("device_admission", {}) or {}),
        "candidate_evaluations": int(
            result.metadata.get("candidate_evaluations", result.evaluations)
        ),
        "scenario_power_flow_calls": int(
            result.metadata.get("scenario_power_flow_calls", result.evaluations)
        ),
        "anytime": {
            str(fraction): _history_at_fraction(result, fraction) for fraction in anytime_fractions
        },
        "independent_validation": independent,
    }


def _paired_bootstrap_interval(
    values: np.ndarray, *, seed: int, resamples: int, confidence: float = 0.95
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(resamples), len(values)))
    medians = np.median(values[indices], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(medians, alpha)), float(np.quantile(medians, 1.0 - alpha))


def _proportion_difference_interval(
    candidate: np.ndarray, baseline: np.ndarray, *, seed: int, resamples: int
) -> tuple[float, float]:
    differences = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    return _paired_bootstrap_interval(differences, seed=seed, resamples=resamples)


def _case_evidence(
    plan: TSHCALOQualificationPlan,
    case_name: str,
    records: list[dict],
    *,
    analysis_seed: int,
) -> dict:
    pairs = exact_keyed_pairs(
        [item for item in records if item["case"] == case_name and item["label"] == "candidate"],
        [item for item in records if item["case"] == case_name and item["label"] == "baseline"],
        key_fields=("case", "run_index"),
        expected_keys=((case_name, run_index) for run_index in range(plan.runs)),
    )
    candidate = [pair.candidate for pair in pairs]
    baseline = [pair.comparator for pair in pairs]
    candidate_feasible = np.asarray([item["feasible"] for item in candidate], dtype=float)
    baseline_feasible = np.asarray([item["feasible"] for item in baseline], dtype=float)
    feasible_interval = _proportion_difference_interval(
        candidate_feasible,
        baseline_feasible,
        seed=analysis_seed,
        resamples=plan.bootstrap_resamples,
    )
    paired_objectives: list[float] = []
    for cand, base in zip(candidate, baseline, strict=True):
        if cand["feasible"] and base["feasible"]:
            cand_value = float(cand["objective"])
            base_value = float(base["objective"])
            paired_objectives.append(
                relative_objective_improvement(
                    cand_value,
                    base_value,
                    scale_floor=float(plan.objective_scale_floor),
                )
            )
    improvements = np.asarray(paired_objectives, dtype=float)
    objective_ci = _paired_bootstrap_interval(
        improvements,
        seed=analysis_seed + 1,
        resamples=plan.bootstrap_resamples,
    )
    statistical_test = wilcoxon_signed_rank_evidence(improvements, alternative="greater")
    pvalue = statistical_test["p_value"]
    rank_biserial = matched_pairs_rank_biserial(improvements)
    anytime: dict[str, dict] = {}
    for fraction in plan.anytime_fractions:
        key = str(fraction)
        cand_rows = [item["anytime"][key] for item in candidate]
        base_rows = [item["anytime"][key] for item in baseline]
        cand_feasible = np.asarray([item["feasible"] for item in cand_rows], dtype=float)
        base_feasible = np.asarray([item["feasible"] for item in base_rows], dtype=float)
        objective_improvements = []
        for cand, base in zip(cand_rows, base_rows, strict=True):
            if cand["feasible"] and base["feasible"]:
                cvalue, bvalue = float(cand["objective"]), float(base["objective"])
                objective_improvements.append(
                    relative_objective_improvement(
                        cvalue,
                        bvalue,
                        scale_floor=float(plan.objective_scale_floor),
                    )
                )
        anytime[key] = {
            "candidate_feasible_probability": _mean_or_none(cand_feasible),
            "baseline_feasible_probability": _mean_or_none(base_feasible),
            "feasible_probability_difference": _mean_or_none(cand_feasible - base_feasible),
            "median_relative_objective_improvement": (
                float(np.median(objective_improvements)) if objective_improvements else None
            ),
            "paired_feasible_objective_count": len(objective_improvements),
        }
    return {
        "analysis_schema_version": PAIRED_ANALYSIS_SCHEMA_VERSION,
        "relative_improvement_version": RELATIVE_IMPROVEMENT_VERSION,
        "objective_scale_floor": float(plan.objective_scale_floor),
        "case": case_name,
        "n_pairs": len(candidate),
        "pair_manifest": pair_manifest(pairs, ("case", "run_index")),
        "candidate_feasible_probability": _mean_or_none(candidate_feasible),
        "baseline_feasible_probability": _mean_or_none(baseline_feasible),
        "feasible_probability_difference": _mean_or_none(candidate_feasible - baseline_feasible),
        "feasible_probability_difference_ci95": [
            _finite_or_none(value) for value in feasible_interval
        ],
        "paired_feasible_objective_count": int(len(improvements)),
        "paired_feasible_objective_fraction": float(len(improvements) / max(len(candidate), 1)),
        "median_relative_objective_improvement": (
            float(np.median(improvements)) if len(improvements) else None
        ),
        "relative_objective_improvement_ci95": [_finite_or_none(value) for value in objective_ci],
        "objective_win_rate": float(np.mean(improvements > 0.0)) if len(improvements) else 0.0,
        "paired_rank_biserial": rank_biserial,
        "wilcoxon_p_one_sided": _finite_or_none(pvalue) if pvalue is not None else None,
        "statistical_test": statistical_test,
        "holm_p": None,
        "all_candidate_independently_validated": bool(candidate)
        and all(item["independent_validation"]["passed"] for item in candidate),
        "all_baseline_independently_validated": bool(baseline)
        and all(item["independent_validation"]["passed"] for item in baseline),
        "equal_exact_fe": bool(candidate)
        and all(
            int(cand["evaluations"]) == int(base["evaluations"]) == plan.max_evaluations
            for cand, base in zip(candidate, baseline, strict=True)
        ),
        "anytime": anytime,
    }


def qualification_candidate_contract(artifact: TSHCALOCandidateArtifact) -> dict:
    """Freeze the ABI and training-design identity that makes a candidate comparable.

    Product/source revisions are deliberately absent. Compatibility changes only when the policy
    architecture, state/action/environment schemas, feature contract, ensemble structure, or
    authenticated training-design provenance changes.
    """

    provenance = dict(artifact.training_provenance or {})
    if provenance.get("source_kind") == "independent_policy_training_ensemble":
        members = list(provenance.get("members", []) or [])
        training_designs = sorted(
            str(
                dict(item.get("training_provenance", {}) or {}).get(
                    "training_design_sha256", ""
                )
            )
            for item in members
        )
        source_candidates = sorted(
            str(item.get("source_candidate_sha256", "")).lower() for item in members
        )
    else:
        training_designs = [str(provenance.get("training_design_sha256", ""))]
        source_candidates = [artifact.sha256]
    if any(not _is_sha256(item) for item in training_designs + source_candidates):
        raise ValueError("Candidate training or member identity is incomplete")
    contract = {
        "schema_version": TSH_CALO_CANDIDATE_CONTRACT_SCHEMA,
        "candidate_sha256": artifact.sha256.lower(),
        "algorithm_id": artifact.algorithm_id,
        "runtime_architecture_version": artifact.algorithm_version,
        "policy_architecture_version": TSH_CALO_POLICY_ARCHITECTURE,
        "state_schema_version": artifact.state_schema_version,
        "action_schema_version": artifact.action_schema_version,
        "training_environment_version": artifact.training_environment_version,
        "artifact_kind": artifact.artifact_kind,
        "ensemble_size": int(artifact.ensemble_size),
        "feature_contract": dict(artifact.feature_flags),
        "member_candidate_sha256": source_candidates,
        "member_training_design_sha256": training_designs,
        "training_provenance_sha256": _canonical_sha256(provenance),
    }
    return contract


def _verify_candidate_contract(
    plan: TSHCALOQualificationPlan, artifact: TSHCALOCandidateArtifact
) -> dict:
    actual = qualification_candidate_contract(artifact)
    if actual != dict(plan.candidate_contract or {}):
        raise ValueError(
            "Candidate architecture or training-parameter contract differs from the frozen plan"
        )
    return actual


def _verify_component_evidence(plan: TSHCALOQualificationPlan) -> dict[str, dict]:
    verified = {}
    for component, reference in sorted(plan.component_evidence.items()):
        path = Path(str(reference.get("path", ""))).expanduser().resolve()
        expected = str(reference.get("sha256", "")).lower()
        if not _is_sha256(expected) or not path.is_file():
            raise ValueError(f"TSH-CALO Change {component} evidence reference is unavailable")
        physical = checkpoint_sha256(path)
        if physical != expected:
            raise ValueError(f"TSH-CALO Change {component} evidence checksum mismatch")
        payload = _read_json(path)
        if payload.get("schema_version") != TSH_CALO_COMPONENT_EVIDENCE_SCHEMA:
            raise ValueError(f"TSH-CALO Change {component} evidence schema is incompatible")
        if str(payload.get("component", "")) != component or not bool(
            payload.get("accepted", False)
        ):
            raise ValueError(f"TSH-CALO Change {component} has not earned inclusion")
        if str(payload.get("source_policy_sha256", "")).lower() != plan.candidate_sha256:
            raise ValueError(f"TSH-CALO Change {component} evidence belongs to another policy")
        if str(payload.get("source_commit", "")).lower() != plan.source_commit.lower():
            raise ValueError(f"TSH-CALO Change {component} evidence belongs to another source")
        if payload.get("source_tracked_clean") is not True or not plan.source_tracked_clean:
            raise ValueError(f"TSH-CALO Change {component} evidence source is not clean")
        if payload.get("analysis_schema_version") != PAIRED_ANALYSIS_SCHEMA_VERSION:
            raise ValueError(f"TSH-CALO Change {component} paired analysis is incompatible")
        if payload.get("relative_improvement_version") != RELATIVE_IMPROVEMENT_VERSION:
            raise ValueError(f"TSH-CALO Change {component} improvement definition is incompatible")
        if float(payload.get("objective_scale_floor", float("nan"))) != float(
            plan.objective_scale_floor
        ):
            raise ValueError(f"TSH-CALO Change {component} objective scale is incompatible")
        if protected_holdout_matches(payload.get("development_cases", [])):
            raise ValueError(f"TSH-CALO Change {component} evidence leaked a protected holdout")
        if set(payload.get("development_cases", [])) != set(plan.development_cases):
            raise ValueError(f"TSH-CALO Change {component} evidence used another case design")
        if bool(payload.get("protected_cases_opened", True)):
            raise ValueError(f"TSH-CALO Change {component} evidence opened protected cases")
        for label in (
            "component_ablation_plan_sha256",
            "scientific_design_sha256",
            "seed_manifest_sha256",
        ):
            if not _is_sha256(str(payload.get(label, ""))):
                raise ValueError(f"TSH-CALO Change {component} evidence lacks a frozen {label}")
        if not list(payload.get("analysis", [])):
            raise ValueError(f"TSH-CALO Change {component} evidence lacks direct analysis")
        if (
            payload.get("authority_boundary")
            != "component_ablation_only_no_qualification_or_lifecycle"
        ):
            raise ValueError(f"TSH-CALO Change {component} evidence authority is incompatible")
        verified[component] = {"path": str(path), "sha256": physical, "accepted": True}
    return verified


def _grade(plan: TSHCALOQualificationPlan, cases: list[dict], failures: list[dict]) -> dict:
    reasons: list[str] = []
    if plan.mode != "formal":
        reasons.append("screening campaigns cannot qualify or emit a receipt")
    if plan.mode == "formal" and not plan.source_tracked_clean:
        reasons.append("formal qualification requires a clean tracked source identity")
    if failures:
        reasons.append("one or more initiated paired runs failed and were retained")
    for item in cases:
        label = str(item["case"])
        if not item["equal_exact_fe"]:
            reasons.append(f"{label}: exact equal-FE accounting failed")
        if not item["all_candidate_independently_validated"]:
            reasons.append(f"{label}: candidate independent validation is incomplete or failed")
        if not item["all_baseline_independently_validated"]:
            reasons.append(f"{label}: baseline independent validation is incomplete or failed")
        if (
            item["candidate_feasible_probability"] is None
            or item["candidate_feasible_probability"] < plan.minimum_feasible_probability
        ):
            reasons.append(f"{label}: candidate feasible probability is below the frozen minimum")
        lower_feasible = item["feasible_probability_difference_ci95"][0]
        if (
            lower_feasible is None
            or float(lower_feasible) < -plan.feasibility_noninferiority_margin
        ):
            reasons.append(f"{label}: feasibility non-inferiority confidence bound failed")
        if item["paired_feasible_objective_fraction"] < plan.minimum_objective_pair_fraction:
            reasons.append(f"{label}: too few paired feasible objectives for quality inference")
        improvement = item["median_relative_objective_improvement"]
        if improvement is None or float(improvement) < plan.minimum_relative_improvement:
            reasons.append(f"{label}: practical objective-improvement threshold failed")
        if item["objective_win_rate"] < plan.minimum_win_rate:
            reasons.append(f"{label}: paired objective win-rate threshold failed")
        if item["paired_rank_biserial"] < plan.minimum_rank_biserial:
            reasons.append(f"{label}: paired effect-size threshold failed")
        if item["holm_p"] is None or float(item["holm_p"]) > plan.statistical_alpha:
            reasons.append(f"{label}: Holm-controlled objective evidence failed")
        for fraction, anytime in item["anytime"].items():
            if anytime["feasible_probability_difference"] is None or (
                anytime["feasible_probability_difference"] < -plan.feasibility_noninferiority_margin
            ):
                reasons.append(f"{label}@{fraction}: anytime feasibility regressed")
            objective = anytime["median_relative_objective_improvement"]
            if objective is not None and float(objective) < -plan.anytime_regression_tolerance:
                reasons.append(f"{label}@{fraction}: anytime objective materially regressed")
    return {
        "passed": not reasons,
        "grade": "A" if not reasons else "U",
        "score": 100.0 if not reasons else 0.0,
        "reasons": reasons,
        "claim_scope": (
            "development-case qualification only; protected-test and superiority claims remain closed"
            if not reasons
            else "no qualification or policy-benefit claim"
        ),
    }


def grade_tsh_calo_qualification_evidence(
    plan: TSHCALOQualificationPlan,
    cases: list[dict],
    failures: list[dict],
) -> dict:
    """Reapply the canonical frozen qualification gates to retained evidence rows."""

    return _grade(plan, cases, failures)


class TSHCALOQualificationCampaign:
    """Execute or exactly resume one frozen qualification evidence directory."""

    def __init__(
        self,
        plan: TSHCALOQualificationPlan,
        output_directory: str | Path,
        *,
        event_callback: Callable[[dict], None] | None = None,
    ) -> None:
        plan.validate()
        self.plan = plan
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.event_callback = event_callback

    def _expected_cells(self) -> int:
        return len(self.plan.development_cases) * self.plan.runs * 2

    def _checkpoint_interval(self) -> int:
        bounded = min(QUALIFICATION_CHECKPOINT_INTERVAL, int(self.plan.max_evaluations))
        population = int(self.plan.population_size)
        return max(population, (bounded // population) * population)

    def _completed_cells(self) -> int:
        return sum(
            1
            for directory in (
                self.output_directory / "records",
                self.output_directory / "failures",
            )
            if directory.is_dir()
            for _path in directory.glob("*.json")
        )

    def _emit_event(self, event: str, **details) -> dict:
        payload = {
            "schema_version": TSH_CALO_QUALIFICATION_EVENT_SCHEMA,
            "event": str(event),
            "qualification_run_id": self.plan.qualification_run_id,
            "qualification_plan_sha256": self.plan.execution_plan_sha256(),
            "emitted_at": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        _append_json_line(self.output_directory / QUALIFICATION_EVENT_LOG_FILE, payload)
        if self.event_callback is not None:
            self.event_callback(dict(payload))
        return payload

    def _write_status(self, *, state: str, **details) -> dict:
        payload = {
            "schema_version": TSH_CALO_QUALIFICATION_STATUS_SCHEMA,
            "qualification_run_id": self.plan.qualification_run_id,
            "qualification_plan_sha256": self.plan.execution_plan_sha256(),
            "state": str(state),
            "completed_cells": self._completed_cells(),
            "total_cells": self._expected_cells(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        _write_json(self.output_directory / QUALIFICATION_STATUS_FILE, payload)
        return payload

    def _pending_pause_request(self) -> dict | None:
        path = self.output_directory / QUALIFICATION_CONTROL_FILE
        if not path.is_file():
            return None
        request = _read_json(path)
        if request.get("state") != "requested" or request.get("action") != "pause":
            return None
        if (
            request.get("schema_version") != TSH_CALO_QUALIFICATION_CONTROL_SCHEMA
            or request.get("qualification_run_id") != self.plan.qualification_run_id
            or request.get("qualification_plan_sha256")
            != self.plan.execution_plan_sha256()
        ):
            raise RuntimeError("Qualification pause request belongs to another frozen plan")
        return request

    def _acknowledge_pause(
        self,
        request: dict,
        *,
        boundary: str,
        durable_path: Path,
        cell: dict | None = None,
        evaluations: int = 0,
    ) -> None:
        if not durable_path.is_file():
            raise RuntimeError("Qualification pause boundary was not durably committed")
        digest = checkpoint_sha256(durable_path)
        acknowledged = {
            **request,
            "state": "acknowledged",
            "acknowledged_at": datetime.now(timezone.utc).isoformat(),
            "boundary": str(boundary),
            "durable_path": str(durable_path),
            "durable_sha256": digest,
            "evaluations": int(evaluations),
        }
        _write_json(self.output_directory / QUALIFICATION_CONTROL_FILE, acknowledged)
        event = self._emit_event(
            "campaign_paused",
            request_id=request["request_id"],
            boundary=str(boundary),
            durable_path=str(durable_path),
            durable_sha256=digest,
            evaluations=int(evaluations),
            completed_cells=self._completed_cells(),
            total_cells=self._expected_cells(),
            current_cell=dict(cell or {}),
            resumable=True,
        )
        self._write_status(
            state="paused",
            pause={
                "reason": "user_requested_safe_pause",
                "request_id": request["request_id"],
                "boundary": str(boundary),
                "durable_path": str(durable_path),
                "durable_sha256": digest,
                "evaluations": int(evaluations),
                "resumable": True,
            },
            current_cell=dict(cell or {}),
            last_event=event,
        )
        raise TSHCALOQualificationPauseRequested(event)

    @staticmethod
    def _remove_completed_cell_checkpoint(path: Path) -> None:
        for target in (path, trusted_resume_sha_path(path)):
            try:
                target.unlink(missing_ok=True)
            except OSError:
                # The committed JSON result is authoritative; stale recovery bytes are harmless
                # and are ignored whenever that result exists.
                pass

    def _preflight(self) -> tuple[TSHCALOCandidateArtifact, dict, dict[str, dict]]:
        artifact = inspect_tsh_calo_candidate(
            self.plan.candidate_path, expected_sha256=self.plan.candidate_sha256
        )
        if artifact.artifact_kind != "ensemble_policy" or artifact.ensemble_size < 2:
            raise ValueError("TSH-CALO qualification requires an immutable ensemble candidate")
        if artifact.algorithm_id != TSH_CALO_ALGORITHM_ID:
            raise ValueError("TSH-CALO qualification candidate algorithm is incompatible")
        if self.plan.candidate_contract:
            return artifact, _verify_candidate_contract(self.plan, artifact), {}
        # Backward compatibility for already-retained formal evidence. New plans use the
        # stage-neutral candidate architecture contract above.
        return artifact, {}, _verify_component_evidence(self.plan)

    def start(self) -> dict:
        if self.output_directory.exists():
            raise FileExistsError(
                "TSH-CALO qualification start requires a new output directory; use explicit resume"
            )
        self.output_directory.mkdir(parents=True)
        _write_json(self.output_directory / "qualification_plan.json", self.plan.to_dict())
        _write_json(self.output_directory / "seed_manifest.json", self.plan.seed_manifest())
        return self._run(resume=False)

    def resume(self) -> dict:
        if (self.output_directory / "campaign_integrity_failure.json").exists():
            raise RuntimeError("Failed-integrity TSH-CALO qualification campaigns cannot resume")
        stored = TSHCALOQualificationPlan.from_dict(
            _read_json(self.output_directory / "qualification_plan.json")
        )
        if stored.execution_plan_sha256() != self.plan.execution_plan_sha256():
            raise ValueError("TSH-CALO qualification plan changed; exact resume is forbidden")
        if (self.output_directory / "qualification_evidence.json").exists():
            raise RuntimeError("TSH-CALO qualification campaign is already complete")
        return self._run(resume=True)

    def _run(self, *, resume: bool) -> dict:
        with _ExclusiveQualificationCampaignLease(self.output_directory):
            return self._run_owned(resume=resume)

    def _run_owned(self, *, resume: bool) -> dict:
        artifact, candidate_contract, component_evidence = self._preflight()
        plan_hash = self.plan.execution_plan_sha256()
        seed_manifest = self.plan.seed_manifest()
        seed_hash = self.plan.seed_manifest_sha256()
        self._write_status(state="running", pause=None, current_cell=None)
        self._emit_event(
            "campaign_resumed" if resume else "campaign_started",
            completed_cells=self._completed_cells(),
            total_cells=self._expected_cells(),
            evaluations_per_cell=int(self.plan.max_evaluations),
            checkpoint_interval_evaluations=self._checkpoint_interval(),
        )
        calibration_path = self.output_directory / "ood_calibration_evidence.json"
        if resume and calibration_path.is_file():
            calibration_evidence = _read_json(calibration_path)
            payload = dict(calibration_evidence["calibration"])
            calibration = OODCalibration(
                np.asarray(payload["mean"], dtype=float),
                np.asarray(payload["scale"], dtype=float),
                float(payload["attenuation_start"]),
                float(payload["minimum_neural_weight"]),
            )
            if ood_calibration_sha256(calibration) != calibration_evidence.get(
                "ood_calibration_sha256"
            ):
                raise ValueError("TSH-CALO stored OOD calibration checksum mismatch")
        else:
            self._emit_event(
                "calibration_started",
                cases=len(self.plan.development_cases),
                samples_per_case=int(self.plan.calibration_samples_per_case),
            )
            calibration_seeds = [RunSeeds(**item) for item in seed_manifest["calibration_runs"]]
            calibration, calibration_evidence = _collect_calibration(
                self.plan,
                calibration_seeds,
                progress_callback=lambda progress: self._emit_event(
                    "calibration_progress", **progress
                ),
            )
            _write_json(calibration_path, calibration_evidence)
            self._emit_event(
                "calibration_completed",
                ood_calibration_sha256=ood_calibration_sha256(calibration),
            )
        authority = QualificationCandidateAuthority(
            self.plan.qualification_run_id,
            plan_hash,
            artifact.sha256,
            self.plan.source_commit,
            tuple(self.plan.development_cases),
            ood_calibration_sha256(calibration),
        )
        binding = _candidate_binding(artifact, calibration)
        binding.update(
            {
                "inference_device": self.plan.inference_device,
                "allow_cpu_fallback": self.plan.allow_cpu_fallback,
            }
        )
        records_directory = self.output_directory / "records"
        failures_directory = self.output_directory / "failures"
        records_directory.mkdir(exist_ok=True)
        failures_directory.mkdir(exist_ok=True)
        paired_seeds = [RunSeeds(**item) for item in seed_manifest["paired_runs"]]
        checkpoints_directory = self.output_directory / "checkpoints"
        checkpoints_directory.mkdir(exist_ok=True)
        for case_index, case_name in enumerate(self.plan.development_cases):
            config = _base_experiment_config(self.plan, case_name)
            for run_index, seeds in enumerate(paired_seeds):
                for label_index, label in enumerate(("baseline", "candidate")):
                    cell_index = (
                        case_index * self.plan.runs * 2 + run_index * 2 + label_index + 1
                    )
                    record_path = records_directory / f"{case_name}-{run_index:03d}-{label}.json"
                    failure_path = failures_directory / f"{case_name}-{run_index:03d}-{label}.json"
                    if resume and (record_path.is_file() or failure_path.is_file()):
                        continue
                    boundary_request = self._pending_pause_request()
                    if boundary_request is not None:
                        durable_records = sorted(
                            [
                                *records_directory.glob("*.json"),
                                *failures_directory.glob("*.json"),
                            ]
                        )
                        self._acknowledge_pause(
                            boundary_request,
                            boundary="cell_record" if durable_records else "calibration",
                            durable_path=(durable_records[-1] if durable_records else calibration_path),
                            cell={
                                "cell_index": cell_index,
                                "case": case_name,
                                "run_index": run_index,
                                "label": label,
                            },
                        )
                    checkpoint_path = (
                        checkpoints_directory
                        / f"{case_name}-{run_index:03d}-{label}.resume"
                    )
                    resumed_checkpoint = checkpoint_path.is_file()
                    checkpoint_interval = self._checkpoint_interval()
                    cell = {
                        "cell_index": cell_index,
                        "total_cells": self._expected_cells(),
                        "case": case_name,
                        "run_index": run_index,
                        "run_number": run_index + 1,
                        "runs_per_case": self.plan.runs,
                        "label": label,
                    }
                    self._write_status(
                        state="running",
                        pause=None,
                        current_cell={**cell, "resumed_checkpoint": resumed_checkpoint},
                    )
                    self._emit_event(
                        "cell_started",
                        **cell,
                        resumed_checkpoint=resumed_checkpoint,
                        committed_cells=self._completed_cells(),
                        max_evaluations=int(self.plan.max_evaluations),
                    )
                    pause_state: dict[str, dict | None] = {"request": None}
                    progress_state: dict[str, int | None] = {
                        "next_log_evaluation": checkpoint_interval,
                        "first_observed_evaluations": None,
                    }
                    cell_started = time.perf_counter()

                    def progress_callback(
                        progress: dict,
                        _cell=cell,
                        _resumed_checkpoint=resumed_checkpoint,
                        _checkpoint_interval=checkpoint_interval,
                        _cell_started=cell_started,
                        _pause_state=pause_state,
                        _progress_state=progress_state,
                    ) -> None:
                        evaluations = int(progress.get("evaluations", 0))
                        first_resumed_sample = False
                        if _progress_state["first_observed_evaluations"] is None:
                            _progress_state["first_observed_evaluations"] = evaluations
                            first_resumed_sample = _resumed_checkpoint
                            if _resumed_checkpoint:
                                _progress_state["next_log_evaluation"] = (
                                    (evaluations // _checkpoint_interval) + 1
                                ) * _checkpoint_interval
                        if (
                            first_resumed_sample
                            or evaluations
                            >= int(_progress_state["next_log_evaluation"] or 0)
                            or evaluations >= self.plan.max_evaluations
                        ):
                            elapsed = max(time.perf_counter() - _cell_started, 1e-9)
                            observed = (
                                0
                                if first_resumed_sample
                                else max(
                                    evaluations
                                    - (
                                        int(
                                            _progress_state[
                                                "first_observed_evaluations"
                                            ]
                                            or 0
                                        )
                                        if _resumed_checkpoint
                                        else 0
                                    ),
                                    self.plan.population_size,
                                )
                            )
                            evaluations_per_second = (
                                observed / elapsed if observed > 0 else None
                            )
                            remaining = max(0, self.plan.max_evaluations - evaluations)
                            self._emit_event(
                                "cell_progress",
                                **_cell,
                                live_evaluations=evaluations,
                                max_evaluations=int(self.plan.max_evaluations),
                                cell_percent=round(
                                    100.0 * evaluations / self.plan.max_evaluations, 1
                                ),
                                committed_cells=self._completed_cells(),
                                best_feasible_objective=_finite_or_none(
                                    progress.get("best_feasible_objective", float("nan"))
                                ),
                                best_constraint_violation=_finite_or_none(
                                    progress.get("best_constraint_violation", float("nan"))
                                ),
                                first_feasible_evaluation=progress.get(
                                    "first_feasible_evaluation"
                                ),
                                evaluations_per_second=(
                                    round(evaluations_per_second, 3)
                                    if evaluations_per_second is not None
                                    else None
                                ),
                                cell_eta_seconds=(
                                    round(remaining / evaluations_per_second, 1)
                                    if evaluations_per_second is not None
                                    and evaluations_per_second > 0.0
                                    else None
                                ),
                                durability="live_uncommitted_until_checkpoint_acknowledgement",
                            )
                            _progress_state["next_log_evaluation"] = (
                                (evaluations // _checkpoint_interval) + 1
                            ) * _checkpoint_interval
                        if _pause_state["request"] is None:
                            _pause_state["request"] = self._pending_pause_request()

                    parameters: dict
                    try:
                        problem = build_problem(config, seeds.scenario_seed)
                        if label == "baseline":
                            parameters = dict(SPECS["CALO"].default_parameters)
                            parameters.update(
                                {
                                    "use_ai": False,
                                    "strict_policy_binding": False,
                                    "strict_benchmark_mode": True,
                                    "use_historical_parameter_priors": False,
                                    "use_cross_algorithm_warm_start": False,
                                    "optimizer_backend": "legacy",
                                }
                            )
                            parameters.update(
                                {
                                    "run_checkpoint_path": str(checkpoint_path),
                                    "checkpoint_interval_evaluations": checkpoint_interval,
                                }
                            )
                            if resumed_checkpoint:
                                parameters["resume_run_checkpoint"] = str(checkpoint_path)
                            optimizer = create_optimizer(
                                "CALO",
                                problem,
                                OptimizerConfig(
                                    self.plan.population_size,
                                    self.plan.max_evaluations,
                                    self.plan.max_evaluations,
                                    parameters,
                                ),
                                seeds.algorithm_seed,
                                progress_callback=progress_callback,
                                cancel_callback=lambda _state=pause_state: (
                                    _state["request"] is not None
                                ),
                            )
                        else:
                            parameters = deepcopy(binding)
                            parameters["ai_inference_seed"] = int(seeds.ai_inference_seed)
                            parameters.update(
                                {
                                    "run_checkpoint_path": str(checkpoint_path),
                                    "checkpoint_interval_evaluations": checkpoint_interval,
                                }
                            )
                            if resumed_checkpoint:
                                parameters["resume_run_checkpoint"] = str(checkpoint_path)
                            optimizer = _QualificationCandidateOptimizer(
                                problem,
                                OptimizerConfig(
                                    self.plan.population_size,
                                    self.plan.max_evaluations,
                                    self.plan.max_evaluations,
                                    parameters,
                                ),
                                seeds.algorithm_seed,
                                qualification_authority=authority,
                                progress_callback=progress_callback,
                                cancel_callback=lambda _state=pause_state: (
                                    _state["request"] is not None
                                ),
                            )
                        result = optimizer.run()
                        if int(result.evaluations) < int(self.plan.max_evaluations):
                            request = pause_state["request"] or self._pending_pause_request()
                            if request is None:
                                raise RuntimeError(
                                    "Qualification cell stopped before its exact evaluation budget"
                                )
                            self._acknowledge_pause(
                                request,
                                boundary="optimizer_checkpoint",
                                durable_path=checkpoint_path,
                                cell=cell,
                                evaluations=int(result.evaluations),
                            )
                        record = _result_record(
                            label=label,
                            case_name=case_name,
                            run_index=run_index,
                            seeds=seeds,
                            result=result,
                            config=config,
                            anytime_fractions=self.plan.anytime_fractions,
                        )
                        record["qualification_plan_sha256"] = plan_hash
                        record["source_policy_sha256"] = artifact.sha256
                        _write_json(record_path, record)
                        self._remove_completed_cell_checkpoint(checkpoint_path)
                        self._emit_event(
                            "cell_completed",
                            **cell,
                            evaluations=int(result.evaluations),
                            committed_cells=self._completed_cells(),
                            total_cells=self._expected_cells(),
                            feasible=bool(result.feasible),
                            objective=_finite_or_none(result.best_objective),
                            violation=_finite_or_none(result.total_constraint_violation),
                        )
                    except TSHCALOQualificationPauseRequested:
                        raise
                    except Exception as exc:
                        failure = {
                            "schema_version": "tsh-calo-qualification-failure-v1",
                            "qualification_plan_sha256": plan_hash,
                            "source_policy_sha256": artifact.sha256,
                            "case": case_name,
                            "run_index": run_index,
                            "label": label,
                            "seeds": asdict(seeds),
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                            "retained": True,
                        }
                        _write_json(failure_path, failure)
                        self._emit_event(
                            "cell_failed",
                            **cell,
                            committed_cells=self._completed_cells(),
                            total_cells=self._expected_cells(),
                            exception_type=type(exc).__name__,
                            message=str(exc),
                        )
                    request_after_cell = (
                        pause_state["request"] or self._pending_pause_request()
                    )
                    if request_after_cell is not None:
                        durable_path = record_path if record_path.is_file() else failure_path
                        self._acknowledge_pause(
                            request_after_cell,
                            boundary="cell_record",
                            durable_path=durable_path,
                            cell=cell,
                            evaluations=int(self.plan.max_evaluations),
                        )
        records = [_read_json(path) for path in sorted(records_directory.glob("*.json"))]
        failures = [_read_json(path) for path in sorted(failures_directory.glob("*.json"))]
        expected = len(self.plan.development_cases) * self.plan.runs * 2
        if len(records) + len(failures) != expected:
            raise RuntimeError(
                "TSH-CALO qualification campaign did not retain every initiated cell"
            )
        cases = [
            _case_evidence(
                self.plan,
                case_name,
                records,
                analysis_seed=self.plan.master_seed + 2_000_003 + index,
            )
            for index, case_name in enumerate(self.plan.development_cases)
        ]
        finite_p = [
            float(item["wilcoxon_p_one_sided"])
            for item in cases
            if item["wilcoxon_p_one_sided"] is not None
        ]
        corrected = holm_correction(finite_p) if finite_p else []
        corrected_iter = iter(corrected)
        for item in cases:
            if item["wilcoxon_p_one_sided"] is not None:
                item["holm_p"] = float(next(corrected_iter))
        decision = grade_tsh_calo_qualification_evidence(self.plan, cases, failures)
        evidence = {
            "schema_version": TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA,
            "analysis_schema_version": PAIRED_ANALYSIS_SCHEMA_VERSION,
            "relative_improvement_version": RELATIVE_IMPROVEMENT_VERSION,
            "objective_scale_floor": float(self.plan.objective_scale_floor),
            "qualification_run_id": self.plan.qualification_run_id,
            "source_commit": self.plan.source_commit,
            "source_tracked_clean": self.plan.source_tracked_clean,
            "source_policy_sha256": artifact.sha256,
            "qualification_plan_sha256": plan_hash,
            "scientific_design_sha256": self.plan.scientific_design_sha256(),
            "seed_manifest_sha256": seed_hash,
            "ood_calibration_sha256": ood_calibration_sha256(calibration),
            "development_cases": list(self.plan.development_cases),
            "protected_cases_opened": False,
            "candidate_contract": candidate_contract,
            "component_evidence": component_evidence,
            "records": {
                "expected": expected,
                "completed": len(records),
                "failed": len(failures),
                "directory": str(records_directory),
                "failures_directory": str(failures_directory),
            },
            "case_evidence": cases,
            "decision": decision,
            "authority_boundary": "independent_qualification_only_no_registration_or_activation",
            "single_writer_semantics": "OS-released exclusive evidence-directory lease",
        }
        evidence_path = self.output_directory / "qualification_evidence.json"
        evidence_sha = _write_json(evidence_path, evidence)
        receipt = None
        if decision["passed"]:
            receipt = build_tsh_calo_qualification_receipt(
                qualification_run_id=self.plan.qualification_run_id,
                source_policy_sha256=artifact.sha256,
                source_commit=self.plan.source_commit,
                qualification_protocol_sha256=self.plan.scientific_design_sha256(),
                seed_manifest_sha256=seed_hash,
                evidence_artifact_sha256=evidence_sha,
                development_cases=self.plan.development_cases,
                ood_calibration=calibration,
            )
            _write_json(self.output_directory / "qualification_receipt.json", receipt)
        completed_event = self._emit_event(
            "campaign_completed",
            passed=bool(decision["passed"]),
            completed_cells=len(records) + len(failures),
            successful_cells=len(records),
            failed_cells=len(failures),
            total_cells=expected,
            evidence_path=str(evidence_path),
            evidence_sha256=evidence_sha,
        )
        self._write_status(
            state="completed_qualified" if decision["passed"] else "completed_not_qualified",
            pause=None,
            current_cell=None,
            last_event=completed_event,
            evidence_path=str(evidence_path),
            evidence_sha256=evidence_sha,
        )
        return {
            "qualification_run_id": self.plan.qualification_run_id,
            "passed": bool(decision["passed"]),
            "mode": self.plan.mode,
            "evidence_path": str(evidence_path),
            "evidence_sha256": evidence_sha,
            "receipt": receipt,
            "registration_performed": False,
            "activation_performed": False,
        }
