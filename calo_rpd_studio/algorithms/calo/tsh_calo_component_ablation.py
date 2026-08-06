"""Frozen, development-only paired evidence for approved TSH-CALO Changes A–E.

This module cannot train, qualify, register, activate, or deploy a policy.  It evaluates one
immutable unqualified ensemble through non-serializable removal profiles and emits checksum-bound
component evidence that the independent qualifier may later verify.  Experimental Change F is
deliberately absent from the production-candidate evidence set.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from calo_rpd_studio.ai.model_io import checkpoint_sha256, durable_write_bytes
from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.registry import SPECS, create_optimizer
from calo_rpd_studio.experiments.experiment_runner import build_problem
from calo_rpd_studio.experiments.seed_manager import RunSeeds, SeedManager
from calo_rpd_studio.power_system.case_identity import protected_holdout_matches
from calo_rpd_studio.statistics.paired import (
    DEFAULT_OBJECTIVE_SCALE_FLOOR,
    PAIRED_ANALYSIS_SCHEMA_VERSION,
    RELATIVE_IMPROVEMENT_VERSION,
)
from calo_rpd_studio.statistics.posthoc import holm_correction

from .tsh_calo_inference import (
    ComponentAblationAuthority,
    TSHCALOComponentAblationProfile,
    TSHCALOInferenceController,
)
from .tsh_calo_optimizer import TSHCALOOptimizer
from .tsh_calo_policy_artifact import TSHCALOCandidateArtifact, inspect_tsh_calo_candidate
from .tsh_calo_qualification_campaign import (
    TSH_CALO_COMPONENT_EVIDENCE_SCHEMA,
    _ExclusiveQualificationCampaignLease,
    _base_experiment_config,
    _candidate_binding,
    _case_evidence,
    _collect_calibration,
    _result_record,
)
from .tsh_calo_schema import TSH_CALO_ALGORITHM_ID, TSHCALOFeatureFlags
from .tsh_calo_shield import OODCalibration, ood_calibration_sha256


TSH_CALO_COMPONENT_ABLATION_PLAN_SCHEMA = "tsh-calo-component-ablation-plan-v2-exact-pairs"
TSH_CALO_COMPONENT_ABLATION_CAMPAIGN_SCHEMA = "tsh-calo-component-ablation-campaign-v2-exact-pairs"
_COMPONENTS = ("A", "B", "C", "D", "E")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    durable_write_bytes(path, encoded.encode("utf-8"))
    return str(checkpoint_sha256(path))


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"TSH-CALO ablation record is unreadable: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"TSH-CALO ablation record must be an object: {path.name}")
    return payload


@dataclass(frozen=True, slots=True)
class ComponentAcceptanceCriteria:
    statistical_alpha: float = 0.05
    feasibility_noninferiority_margin: float = 0.05
    minimum_objective_pair_fraction: float = 0.80
    minimum_relative_improvement: float = 0.002
    minimum_win_rate: float = 0.60
    minimum_rank_biserial: float = 0.20
    anytime_regression_tolerance: float = 0.01

    def validate(self) -> None:
        probabilities = (
            self.statistical_alpha,
            self.feasibility_noninferiority_margin,
            self.minimum_objective_pair_fraction,
            self.minimum_win_rate,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
            raise ValueError("TSH-CALO ablation probability controls must lie within [0, 1]")
        if self.statistical_alpha <= 0.0:
            raise ValueError("TSH-CALO ablation alpha must be positive")
        if not -1.0 <= self.minimum_rank_biserial <= 1.0:
            raise ValueError("TSH-CALO ablation rank-biserial threshold is invalid")
        if self.minimum_relative_improvement < 0.0 or self.anytime_regression_tolerance < 0.0:
            raise ValueError("TSH-CALO ablation practical margins cannot be negative")


@dataclass(frozen=True, slots=True)
class TSHCALOComponentAblationPlan:
    campaign_id: str
    source_commit: str
    candidate_path: str
    candidate_sha256: str
    development_cases: tuple[str, ...]
    runs: int
    master_seed: int
    population_size: int
    max_evaluations: int
    source_tracked_clean: bool = False
    calibration_samples_per_case: int = 8
    calibration_population_size: int = 40
    calibration_quantile: float = 0.95
    minimum_neural_weight: float = 0.0
    inference_device: str = "auto"
    allow_cpu_fallback: bool = True
    anytime_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    bootstrap_resamples: int = 10_000
    criteria: ComponentAcceptanceCriteria = field(default_factory=ComponentAcceptanceCriteria)
    analysis_schema_version: str = PAIRED_ANALYSIS_SCHEMA_VERSION
    relative_improvement_version: str = RELATIVE_IMPROVEMENT_VERSION
    objective_scale_floor: float = DEFAULT_OBJECTIVE_SCALE_FLOOR
    schema_version: str = TSH_CALO_COMPONENT_ABLATION_PLAN_SCHEMA

    @property
    def qualification_run_id(self) -> str:
        """Compatibility label for shared development-case config/calibration construction."""

        return self.campaign_id

    def validate(self) -> None:
        if self.schema_version != TSH_CALO_COMPONENT_ABLATION_PLAN_SCHEMA:
            raise ValueError("TSH-CALO component-ablation plan schema is incompatible")
        if self.analysis_schema_version != PAIRED_ANALYSIS_SCHEMA_VERSION:
            raise ValueError("TSH-CALO component-ablation analysis schema is incompatible")
        if self.relative_improvement_version != RELATIVE_IMPROVEMENT_VERSION:
            raise ValueError("TSH-CALO component-ablation improvement schema is incompatible")
        if (
            not math.isfinite(float(self.objective_scale_floor))
            or self.objective_scale_floor <= 0.0
        ):
            raise ValueError("TSH-CALO component-ablation objective scale floor is invalid")
        if not self.campaign_id.strip():
            raise ValueError("TSH-CALO component ablation requires a campaign ID")
        commit = str(self.source_commit).strip().lower()
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError("TSH-CALO component ablation requires an exact source commit")
        if not isinstance(self.source_tracked_clean, bool):
            raise ValueError("TSH-CALO component-ablation source clean state must be Boolean")
        if not _is_sha256(self.candidate_sha256):
            raise ValueError("TSH-CALO component ablation candidate SHA-256 is invalid")
        if not self.development_cases or len(set(self.development_cases)) != len(
            self.development_cases
        ):
            raise ValueError("TSH-CALO component ablation cases must be non-empty and unique")
        leaked = protected_holdout_matches(self.development_cases)
        if leaked:
            raise ValueError(
                "Protected holdouts cannot enter TSH-CALO component ablation: " + ", ".join(leaked)
            )
        if self.runs < 30:
            raise ValueError(
                "TSH-CALO component evidence requires at least 30 paired runs per case"
            )
        if self.population_size < 2 or self.max_evaluations < 2 * self.population_size:
            raise ValueError("TSH-CALO component-ablation population/FE budget is too small")
        if self.max_evaluations % self.population_size:
            raise ValueError("TSH-CALO component-ablation FE budget must divide by population")
        if self.calibration_samples_per_case < 4 or self.calibration_population_size < 2:
            raise ValueError(
                "TSH-CALO component ablation requires at least four calibration states"
            )
        if not 0.50 <= self.calibration_quantile < 1.0:
            raise ValueError("TSH-CALO component-ablation calibration quantile is invalid")
        if not 0.0 <= self.minimum_neural_weight <= 1.0:
            raise ValueError("TSH-CALO component-ablation neural floor is invalid")
        if self.inference_device not in {"auto", "cpu", "cuda"} and not str(
            self.inference_device
        ).startswith("cuda:"):
            raise ValueError("TSH-CALO component-ablation inference device is invalid")
        if not isinstance(self.allow_cpu_fallback, bool):
            raise ValueError("TSH-CALO component-ablation fallback control must be Boolean")
        fractions = tuple(float(value) for value in self.anytime_fractions)
        if not fractions or tuple(sorted(set(fractions))) != fractions or fractions[-1] != 1.0:
            raise ValueError("TSH-CALO ablation anytime fractions must be increasing and end at 1")
        if any(not 0.0 < value <= 1.0 for value in fractions):
            raise ValueError("TSH-CALO ablation anytime fractions must lie within (0, 1]")
        if self.bootstrap_resamples < 1_000:
            raise ValueError(
                "TSH-CALO component ablation requires at least 1,000 bootstrap samples"
            )
        self.criteria.validate()

    def seed_manifest(self) -> dict:
        self.validate()
        return {
            "schema_version": "tsh-calo-component-ablation-seeds-v1",
            "campaign_id": self.campaign_id,
            "paired_runs": [
                asdict(item) for item in SeedManager(self.master_seed).generate(self.runs)
            ],
            "calibration_runs": [
                asdict(item)
                for item in SeedManager(self.master_seed + 1_000_003).generate(
                    self.calibration_samples_per_case
                )
            ],
        }

    def seed_manifest_sha256(self) -> str:
        return _canonical_sha256(self.seed_manifest())

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["development_cases"] = list(self.development_cases)
        payload["anytime_fractions"] = list(self.anytime_fractions)
        return payload

    def scientific_design_sha256(self) -> str:
        payload = self.to_dict()
        payload.pop("candidate_path", None)
        payload.pop("inference_device", None)
        payload.pop("allow_cpu_fallback", None)
        return _canonical_sha256(payload)

    def execution_plan_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict) -> "TSHCALOComponentAblationPlan":
        values = dict(payload or {})
        values["development_cases"] = tuple(values.get("development_cases", ()))
        values["anytime_fractions"] = tuple(values.get("anytime_fractions", ()))
        values["criteria"] = ComponentAcceptanceCriteria(**dict(values.get("criteria", {})))
        try:
            plan = cls(**values)
        except TypeError as exc:
            raise ValueError("TSH-CALO component-ablation plan fields are incomplete") from exc
        plan.validate()
        return plan


_PROFILES = {
    "graph_context_only": TSHCALOComponentAblationProfile(
        "graph_context_only", True, False, False, False, False
    ),
    "hierarchical_actions_only": TSHCALOComponentAblationProfile(
        "hierarchical_actions_only", False, True, False, False, False
    ),
    "graph_plus_hierarchy": TSHCALOComponentAblationProfile(
        "graph_plus_hierarchy", True, True, False, False, False
    ),
    "uncertainty_shield": TSHCALOComponentAblationProfile(
        "uncertainty_shield", True, True, True, False, False
    ),
    "contextual_bandit_residual": TSHCALOComponentAblationProfile(
        "contextual_bandit_residual", True, True, True, True, False
    ),
    "full_approved_A_E": TSHCALOComponentAblationProfile(
        "full_approved_A_E", True, True, True, True, True
    ),
}
_EXECUTION_LABELS = (
    "frozen_calo",
    "canonical_refactor",
    *_PROFILES,
)


class _ComponentAblationOptimizer(TSHCALOOptimizer):
    def __init__(self, *args, ablation_authority: ComponentAblationAuthority, **kwargs):
        self._ablation_authority = ablation_authority
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
            _component_ablation_authority=self._ablation_authority,
        )

    def _physics_repair_enabled(self, features: TSHCALOFeatureFlags) -> bool:
        return bool(features.physics_repair and self._ablation_authority.profile.physics_repair)


def _calo_optimizer(plan, problem, algorithm_seed: int):
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
    return create_optimizer(
        "CALO",
        problem,
        OptimizerConfig(
            plan.population_size,
            plan.max_evaluations,
            plan.max_evaluations,
            parameters,
        ),
        algorithm_seed,
    )


def _record_identity(record: dict) -> dict:
    """Remove operational timing/admission values from Change-A exact parity."""

    keys = (
        "case",
        "run_index",
        "seeds",
        "feasible",
        "objective",
        "violation",
        "evaluations",
        "iterations",
        "first_feasible_evaluation",
        "best_vector",
        "problem_fingerprint",
        "scenario_manifest",
        "candidate_evaluations",
        "scenario_power_flow_calls",
        "anytime",
        "independent_validation",
    )
    return {key: record.get(key) for key in keys}


def _comparison_rows(records: list[dict], baseline: str, candidate: str) -> list[dict]:
    rows = []
    for item in records:
        if item["label"] not in {baseline, candidate}:
            continue
        row = deepcopy(item)
        row["label"] = "baseline" if item["label"] == baseline else "candidate"
        rows.append(row)
    return rows


def _comparison_analysis(
    plan: TSHCALOComponentAblationPlan,
    records: list[dict],
    *,
    comparison_id: str,
    baseline: str,
    candidate: str,
    seed_offset: int,
) -> dict:
    rows = _comparison_rows(records, baseline, candidate)
    cases = [
        _case_evidence(
            cast(Any, plan),
            case_name,
            rows,
            analysis_seed=plan.master_seed + seed_offset + index,
        )
        for index, case_name in enumerate(plan.development_cases)
    ]
    return {
        "comparison_id": comparison_id,
        "baseline": baseline,
        "candidate": candidate,
        "case_evidence": cases,
    }


def _case_reasons(plan: TSHCALOComponentAblationPlan, item: dict) -> list[str]:
    criteria = plan.criteria
    label = str(item["case"])
    reasons = []
    if not item["equal_exact_fe"]:
        reasons.append(f"{label}: exact equal-FE accounting failed")
    if not item["all_candidate_independently_validated"]:
        reasons.append(f"{label}: candidate independent validation failed")
    if not item["all_baseline_independently_validated"]:
        reasons.append(f"{label}: baseline independent validation failed")
    lower = item["feasible_probability_difference_ci95"][0]
    if lower is None or float(lower) < -criteria.feasibility_noninferiority_margin:
        reasons.append(f"{label}: feasibility non-inferiority failed")
    if item["paired_feasible_objective_fraction"] < criteria.minimum_objective_pair_fraction:
        reasons.append(f"{label}: insufficient paired feasible objectives")
    improvement = item["median_relative_objective_improvement"]
    if improvement is None or float(improvement) < criteria.minimum_relative_improvement:
        reasons.append(f"{label}: incremental practical improvement failed")
    if item["objective_win_rate"] < criteria.minimum_win_rate:
        reasons.append(f"{label}: incremental win-rate failed")
    if item["paired_rank_biserial"] < criteria.minimum_rank_biserial:
        reasons.append(f"{label}: incremental effect-size failed")
    if item["holm_p"] is None or float(item["holm_p"]) > criteria.statistical_alpha:
        reasons.append(f"{label}: Holm-controlled evidence failed")
    for fraction, anytime in item["anytime"].items():
        difference = anytime["feasible_probability_difference"]
        if difference is None or difference < -criteria.feasibility_noninferiority_margin:
            reasons.append(f"{label}@{fraction}: anytime feasibility regressed")
        objective = anytime["median_relative_objective_improvement"]
        if objective is not None and objective < -criteria.anytime_regression_tolerance:
            reasons.append(f"{label}@{fraction}: anytime objective regressed")
    return reasons


def _component_analyses(plan, records: list[dict]) -> dict[str, list[dict]]:
    return {
        "A": [],
        "B": [
            _comparison_analysis(
                plan,
                records,
                comparison_id="B_increment_given_hierarchy",
                baseline="hierarchical_actions_only",
                candidate="graph_plus_hierarchy",
                seed_offset=3_100_000,
            ),
            _comparison_analysis(
                plan,
                records,
                comparison_id="B_standalone_falsification",
                baseline="canonical_refactor",
                candidate="graph_context_only",
                seed_offset=3_200_000,
            ),
        ],
        "C": [
            _comparison_analysis(
                plan,
                records,
                comparison_id="C_increment_given_graph",
                baseline="graph_context_only",
                candidate="graph_plus_hierarchy",
                seed_offset=3_300_000,
            ),
            _comparison_analysis(
                plan,
                records,
                comparison_id="C_standalone_falsification",
                baseline="canonical_refactor",
                candidate="hierarchical_actions_only",
                seed_offset=3_400_000,
            ),
        ],
        "D": [
            _comparison_analysis(
                plan,
                records,
                comparison_id="D_uncertainty_increment",
                baseline="graph_plus_hierarchy",
                candidate="uncertainty_shield",
                seed_offset=3_500_000,
            ),
            _comparison_analysis(
                plan,
                records,
                comparison_id="D_bandit_increment",
                baseline="uncertainty_shield",
                candidate="contextual_bandit_residual",
                seed_offset=3_600_000,
            ),
        ],
        "E": [
            _comparison_analysis(
                plan,
                records,
                comparison_id="E_counted_physics_increment",
                baseline="contextual_bandit_residual",
                candidate="full_approved_A_E",
                seed_offset=3_700_000,
            )
        ],
    }


def _apply_holm(analyses: dict[str, list[dict]]) -> None:
    indexed = []
    pvalues = []
    for component in ("B", "C", "D", "E"):
        for comparison in analyses[component]:
            if comparison["comparison_id"].endswith("falsification"):
                continue
            for case in comparison["case_evidence"]:
                value = case["wilcoxon_p_one_sided"]
                if value is not None:
                    indexed.append(case)
                    pvalues.append(float(value))
    corrected = holm_correction(pvalues) if pvalues else []
    for case, value in zip(indexed, corrected, strict=True):
        case["holm_p"] = float(value)


def _change_a_evidence(plan, records: list[dict], failures: list[dict]) -> tuple[bool, dict]:
    mismatches = []
    by_key = {(item["case"], int(item["run_index"]), item["label"]): item for item in records}
    for case_name in plan.development_cases:
        for run_index in range(plan.runs):
            left = by_key.get((case_name, run_index, "frozen_calo"))
            right = by_key.get((case_name, run_index, "canonical_refactor"))
            if left is None or right is None or _record_identity(left) != _record_identity(right):
                mismatches.append({"case": case_name, "run_index": run_index})
    accepted = not failures and not mismatches
    return accepted, {
        "comparison_id": "A_canonical_refactor_exact_parity",
        "baseline": "frozen_calo",
        "candidate": "canonical_refactor",
        "expected_pairs": len(plan.development_cases) * plan.runs,
        "exact_pairs": len(plan.development_cases) * plan.runs - len(mismatches),
        "mismatches": mismatches,
    }


def _component_acceptance(plan, component: str, analyses: list[dict], failures: list[dict]):
    reasons = []
    if failures:
        reasons.append("one or more initiated matrix cells failed and were retained")
    primary = [item for item in analyses if not item["comparison_id"].endswith("falsification")]
    for comparison in primary:
        for case in comparison["case_evidence"]:
            reasons.extend(
                f"{comparison['comparison_id']}: {reason}" for reason in _case_reasons(plan, case)
            )
    if not primary:
        reasons.append(f"Change {component} has no incremental comparison")
    return not reasons, reasons


class TSHCALOComponentAblationCampaign:
    """Execute or resume one immutable development-only A–E evidence matrix."""

    PLAN_FILE = "component_ablation_plan.json"

    def __init__(self, plan: TSHCALOComponentAblationPlan, output_directory: str | Path) -> None:
        plan.validate()
        self.plan = plan
        self.output_directory = Path(output_directory).expanduser().resolve()

    def _preflight(self) -> TSHCALOCandidateArtifact:
        artifact = inspect_tsh_calo_candidate(
            self.plan.candidate_path, expected_sha256=self.plan.candidate_sha256
        )
        if artifact.artifact_kind != "ensemble_policy" or artifact.ensemble_size < 2:
            raise ValueError("TSH-CALO component ablation requires an immutable ensemble")
        if artifact.algorithm_id != TSH_CALO_ALGORITHM_ID:
            raise ValueError("TSH-CALO component-ablation candidate is incompatible")
        flags = TSHCALOFeatureFlags(**dict(artifact.feature_flags))
        flags.validate()
        if not flags.physics_repair:
            raise ValueError("A–E component ablation requires a candidate trained with Change E")
        if flags.population_schedule or flags.allow_experimental_components:
            raise ValueError("Experimental Change F cannot enter A–E component evidence")
        return artifact

    def start(self) -> dict:
        if self.output_directory.exists():
            raise FileExistsError("TSH-CALO component-ablation start requires a new directory")
        self.output_directory.mkdir(parents=True)
        _write_json(self.output_directory / self.PLAN_FILE, self.plan.to_dict())
        _write_json(self.output_directory / "seed_manifest.json", self.plan.seed_manifest())
        return self._run(resume=False)

    def resume(self) -> dict:
        stored = TSHCALOComponentAblationPlan.from_dict(
            _read_json(self.output_directory / self.PLAN_FILE)
        )
        if stored.execution_plan_sha256() != self.plan.execution_plan_sha256():
            raise ValueError("TSH-CALO component-ablation plan changed; resume is forbidden")
        if (self.output_directory / "campaign_evidence.json").exists():
            raise RuntimeError("TSH-CALO component-ablation campaign is already complete")
        return self._run(resume=True)

    def _run(self, *, resume: bool) -> dict:
        with _ExclusiveQualificationCampaignLease(self.output_directory):
            return self._run_owned(resume=resume)

    def _calibration(self, *, resume: bool) -> OODCalibration:
        path = self.output_directory / "ood_calibration_evidence.json"
        if resume and path.is_file():
            stored = _read_json(path)
            values = dict(stored["calibration"])
            calibration = OODCalibration(
                np.asarray(values["mean"], dtype=float),
                np.asarray(values["scale"], dtype=float),
                float(values["attenuation_start"]),
                float(values["minimum_neural_weight"]),
            )
            if ood_calibration_sha256(calibration) != stored.get("ood_calibration_sha256"):
                raise ValueError("Stored TSH-CALO ablation calibration checksum mismatch")
            return calibration
        seed_manifest = self.plan.seed_manifest()
        seeds = [RunSeeds(**item) for item in seed_manifest["calibration_runs"]]
        calibration, evidence = _collect_calibration(cast(Any, self.plan), seeds)
        _write_json(path, evidence)
        return calibration

    def _run_owned(self, *, resume: bool) -> dict:
        artifact = self._preflight()
        plan_hash = self.plan.execution_plan_sha256()
        calibration = self._calibration(resume=resume)
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
        seeds = [RunSeeds(**item) for item in self.plan.seed_manifest()["paired_runs"]]
        for case_name in self.plan.development_cases:
            config = _base_experiment_config(cast(Any, self.plan), case_name)
            for run_index, run_seeds in enumerate(seeds):
                for label in _EXECUTION_LABELS:
                    record_path = records_directory / f"{case_name}-{run_index:03d}-{label}.json"
                    failure_path = failures_directory / f"{case_name}-{run_index:03d}-{label}.json"
                    if resume and (record_path.is_file() or failure_path.is_file()):
                        continue
                    try:
                        problem = build_problem(config, run_seeds.scenario_seed)
                        if label in {"frozen_calo", "canonical_refactor"}:
                            optimizer = _calo_optimizer(
                                self.plan, problem, run_seeds.algorithm_seed
                            )
                        else:
                            profile = _PROFILES[label]
                            authority = ComponentAblationAuthority(
                                self.plan.campaign_id,
                                plan_hash,
                                artifact.sha256,
                                self.plan.source_commit,
                                tuple(self.plan.development_cases),
                                ood_calibration_sha256(calibration),
                                profile,
                            )
                            parameters = deepcopy(binding)
                            parameters["ai_inference_seed"] = int(run_seeds.ai_inference_seed)
                            optimizer = _ComponentAblationOptimizer(
                                problem,
                                OptimizerConfig(
                                    self.plan.population_size,
                                    self.plan.max_evaluations,
                                    self.plan.max_evaluations,
                                    parameters,
                                ),
                                run_seeds.algorithm_seed,
                                ablation_authority=authority,
                            )
                        result = optimizer.run()
                        record = _result_record(
                            label=label,
                            case_name=case_name,
                            run_index=run_index,
                            seeds=run_seeds,
                            result=result,
                            config=config,
                            anytime_fractions=self.plan.anytime_fractions,
                        )
                        record["component_ablation_plan_sha256"] = plan_hash
                        record["source_policy_sha256"] = artifact.sha256
                        _write_json(record_path, record)
                    except Exception as exc:
                        _write_json(
                            failure_path,
                            {
                                "schema_version": "tsh-calo-component-ablation-failure-v1",
                                "component_ablation_plan_sha256": plan_hash,
                                "source_policy_sha256": artifact.sha256,
                                "case": case_name,
                                "run_index": run_index,
                                "label": label,
                                "seeds": asdict(run_seeds),
                                "exception_type": type(exc).__name__,
                                "message": str(exc),
                                "retained": True,
                            },
                        )
        records = [_read_json(path) for path in sorted(records_directory.glob("*.json"))]
        failures = [_read_json(path) for path in sorted(failures_directory.glob("*.json"))]
        expected = len(self.plan.development_cases) * self.plan.runs * len(_EXECUTION_LABELS)
        if len(records) + len(failures) != expected:
            raise RuntimeError("TSH-CALO ablation did not retain every initiated matrix cell")
        analyses = _component_analyses(self.plan, records)
        _apply_holm(analyses)
        component_references = {}
        a_accepted, a_analysis = _change_a_evidence(self.plan, records, failures)
        for component in _COMPONENTS:
            if component == "A":
                accepted, reasons, component_analysis = (
                    a_accepted,
                    [] if a_accepted else ["canonical-refactor exact parity failed"],
                    [a_analysis],
                )
            else:
                accepted, reasons = _component_acceptance(
                    self.plan, component, analyses[component], failures
                )
                component_analysis = analyses[component]
            if not self.plan.source_tracked_clean:
                accepted = False
                reasons.append("component evidence requires a clean tracked source identity")
            evidence = {
                "schema_version": TSH_CALO_COMPONENT_EVIDENCE_SCHEMA,
                "analysis_schema_version": PAIRED_ANALYSIS_SCHEMA_VERSION,
                "relative_improvement_version": RELATIVE_IMPROVEMENT_VERSION,
                "objective_scale_floor": float(self.plan.objective_scale_floor),
                "component": component,
                "accepted": bool(accepted),
                "source_policy_sha256": artifact.sha256,
                "source_commit": self.plan.source_commit,
                "source_tracked_clean": self.plan.source_tracked_clean,
                "campaign_id": self.plan.campaign_id,
                "component_ablation_plan_sha256": plan_hash,
                "scientific_design_sha256": self.plan.scientific_design_sha256(),
                "seed_manifest_sha256": self.plan.seed_manifest_sha256(),
                "development_cases": list(self.plan.development_cases),
                "protected_cases_opened": False,
                "analysis": component_analysis,
                "reasons": reasons,
                "claim_scope": (
                    "development-only component inclusion evidence"
                    if accepted
                    else "no component-inclusion or policy-benefit claim"
                ),
                "authority_boundary": "component_ablation_only_no_qualification_or_lifecycle",
            }
            path = self.output_directory / f"component-{component}.evidence.json"
            sha256 = _write_json(path, evidence)
            component_references[component] = {
                "path": str(path),
                "sha256": sha256,
                "accepted": bool(accepted),
            }
        campaign = {
            "schema_version": TSH_CALO_COMPONENT_ABLATION_CAMPAIGN_SCHEMA,
            "analysis_schema_version": PAIRED_ANALYSIS_SCHEMA_VERSION,
            "relative_improvement_version": RELATIVE_IMPROVEMENT_VERSION,
            "objective_scale_floor": float(self.plan.objective_scale_floor),
            "campaign_id": self.plan.campaign_id,
            "source_commit": self.plan.source_commit,
            "source_tracked_clean": self.plan.source_tracked_clean,
            "source_policy_sha256": artifact.sha256,
            "execution_plan_sha256": plan_hash,
            "scientific_design_sha256": self.plan.scientific_design_sha256(),
            "seed_manifest_sha256": self.plan.seed_manifest_sha256(),
            "matrix_labels": list(_EXECUTION_LABELS),
            "experimental_F": "excluded_disabled_not_eligible_for_formal_A_E_evidence",
            "records": {"expected": expected, "completed": len(records), "failed": len(failures)},
            "component_evidence": component_references,
            "all_A_E_accepted": all(item["accepted"] for item in component_references.values()),
            "registration_performed": False,
            "activation_performed": False,
            "qualification_performed": False,
        }
        evidence_path = self.output_directory / "campaign_evidence.json"
        evidence_sha = _write_json(evidence_path, campaign)
        return {
            "campaign_id": self.plan.campaign_id,
            "all_A_E_accepted": campaign["all_A_E_accepted"],
            "evidence_path": str(evidence_path),
            "evidence_sha256": evidence_sha,
            "component_evidence": component_references,
            "registration_performed": False,
            "activation_performed": False,
            "qualification_performed": False,
        }
