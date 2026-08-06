"""Frozen full-comparison campaign planning."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from calo_rpd_studio.algorithms.registry import primary_algorithm_names
from calo_rpd_studio.experiments.evaluation_budget import BudgetPolicy
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.study_strength import (
    StudyStrength,
    recommend_paired_runs,
)
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableDecoder
from calo_rpd_studio.portfolio.models import EvidenceProfile, PortfolioKind
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.version import VERSION, FREEZE_MANIFEST
from .freeze import verify_freeze_manifest
from .suite import BenchmarkSuite, standard_benchmark_suite


@dataclass(slots=True)
class BenchmarkCampaignConfig:
    name: str = f"CALO-RPD v{VERSION} benchmark campaign"
    cases: tuple[str, ...] = ("case30", "case57", "case118", "case300")
    study_keys: tuple[str, ...] = (
        "deterministic",
        "mixed",
        "load_mean_risk",
        "renewable_cvar",
        "branch_worst_case",
    )
    runs: int = field(
        default_factory=lambda: (
            recommend_paired_runs(
                StudyStrength.STRONG,
                planned_comparisons=max(1, len(primary_algorithm_names()) - 1),
            ).runs
        )
    )
    standardized_effect: float = 0.50
    target_power: float = 0.95
    family_alpha: float = 0.05
    failure_allowance: float = 0.10
    run_planning_method: str = "normal_approximation_holm"
    power_evidence_sha256: str = ""
    require_protected_test: bool = True
    max_evaluations: int = 5000
    population_size: int = 50
    master_seed: int = 2026
    output_directory: str = "benchmark_v541"
    parallel_workers: int = 1
    execution_backend: str = "cuda_preferred"
    execution_purpose: str = "formal"
    requested_compute_device: str = "auto"
    freeze_manifest: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[1] / "data" / "frozen" / FREEZE_MANIFEST
        )
    )
    algorithms: tuple[str, ...] = field(default_factory=primary_algorithm_names)

    def validate(self, suite: BenchmarkSuite | None = None, *, verify_freeze: bool = True) -> None:
        suite = suite or standard_benchmark_suite()
        algorithms = tuple(str(name) for name in self.algorithms)
        if len(algorithms) < 2 or "CALO" not in algorithms:
            raise ValueError("A confirmatory campaign requires CALO and at least one comparator.")
        if len(set(algorithms)) != len(algorithms):
            raise ValueError("Benchmark algorithms must be unique.")
        registered = set(primary_algorithm_names())
        unknown_algorithms = set(algorithms) - registered
        if unknown_algorithms:
            raise ValueError(
                "Unregistered benchmark algorithms: " + ", ".join(sorted(unknown_algorithms))
            )
        planning_method = str(self.run_planning_method).strip().lower()
        if planning_method == "normal_approximation_holm":
            recommendation = recommend_paired_runs(
                StudyStrength.STRONG,
                standardized_effect=float(self.standardized_effect),
                target_power=float(self.target_power),
                family_alpha=float(self.family_alpha),
                failure_allowance=float(self.failure_allowance),
                planned_comparisons=len(algorithms) - 1,
            )
            if int(self.runs) < int(recommendation.runs):
                raise ValueError(
                    f"The preregistered power approximation requires at least "
                    f"{recommendation.runs} initiated paired runs for "
                    f"{len(algorithms) - 1} planned comparisons; requested {self.runs}."
                )
        elif planning_method == "pilot_simulation":
            digest = str(self.power_evidence_sha256).strip().lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "pilot_simulation run planning requires a 64-character power-evidence SHA-256."
                )
            if int(self.runs) < 2:
                raise ValueError(
                    "A paired confirmatory campaign requires at least two initiated runs."
                )
        else:
            raise ValueError(
                "run_planning_method must be normal_approximation_holm or pilot_simulation."
            )
        if self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be at least 1")
        if self.execution_backend != "cuda_preferred" or self.execution_purpose != "formal":
            raise ValueError(
                "A confirmatory benchmark campaign requires formal CUDA-preferred execution."
            )
        unknown_cases = set(self.cases) - set(suite.cases)
        if unknown_cases:
            raise ValueError(f"Unsupported benchmark cases: {sorted(unknown_cases)}")
        if self.require_protected_test and not any(
            suite.evidence_role(case_name) == "test" for case_name in self.cases
        ):
            raise ValueError(
                "A confirmatory campaign requires at least one protected test system. "
                "Use require_protected_test=False only for an explicitly labeled validation replay."
            )
        known_studies = {study.key for study in suite.studies}
        unknown_studies = set(self.study_keys) - known_studies
        if unknown_studies:
            raise ValueError(f"Unsupported benchmark studies: {sorted(unknown_studies)}")
        if verify_freeze:
            verification = verify_freeze_manifest(self.freeze_manifest)
            if not verification.passed:
                raise RuntimeError(verification.message)


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    task_index: int
    task_id: str
    case_name: str
    study_key: str
    study_label: str
    evidence_role: str
    config: ExperimentConfig

    @property
    def planned_jobs(self) -> int:
        return int(self.config.runs) * len(self.config.algorithms)


def _task_seed(master_seed: int, case_name: str, study_key: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{case_name}:{study_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def build_campaign(
    campaign: BenchmarkCampaignConfig,
    *,
    base_config: ExperimentConfig | None = None,
    suite: BenchmarkSuite | None = None,
    verify_freeze: bool = True,
) -> list[BenchmarkTask]:
    suite = suite or standard_benchmark_suite()
    campaign.validate(suite, verify_freeze=verify_freeze)
    base = deepcopy(base_config or ExperimentConfig())
    tasks: list[BenchmarkTask] = []
    index = 0
    for case_name in campaign.cases:
        for study_key in campaign.study_keys:
            study = suite.study(study_key)
            config = deepcopy(base)
            config.name = f"{campaign.name} · {case_name} · {study.label}"
            config.case_name = case_name
            config.algorithms = list(campaign.algorithms)
            config.runs = int(campaign.runs)
            # Final campaign repetitions are explicit evidence requirements. Synchronize the
            # embedded portfolio so validation cannot reduce or inflate the powered run plan.
            config.portfolio.kind = PortfolioKind.OVERALL_EXPERIMENT
            config.portfolio.evidence_profile = EvidenceProfile.CUSTOM
            config.portfolio.custom_runs = int(campaign.runs)
            config.master_seed = _task_seed(campaign.master_seed, case_name, study_key)
            config.population_size = int(campaign.population_size)
            config.budget.policy = BudgetPolicy.EQUAL_EVALUATIONS
            config.budget.max_evaluations = int(campaign.max_evaluations)
            config.max_iterations = max(int(config.max_iterations), int(campaign.max_evaluations))
            config.parallel_workers = int(campaign.parallel_workers)
            config.execution_backend = str(campaign.execution_backend)
            config.execution_purpose = str(campaign.execution_purpose)
            config.requested_compute_device = str(campaign.requested_compute_device)
            config.cuda_cpu_fallback_enabled = False
            config.output_directory = str(
                Path(campaign.output_directory) / "raw_arrays" / case_name / study_key
            )
            study.configure(config)
            config.validate()
            task_id = f"{case_name}__{study_key}"
            tasks.append(
                BenchmarkTask(
                    index,
                    task_id,
                    case_name,
                    study_key,
                    study.label,
                    suite.evidence_role(case_name),
                    config,
                )
            )
            index += 1
    return tasks


def write_campaign_plan(
    campaign: BenchmarkCampaignConfig, tasks: list[BenchmarkTask], destination: str | Path
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "campaign": {
            "name": campaign.name,
            "cases": list(campaign.cases),
            "study_keys": list(campaign.study_keys),
            "runs": campaign.runs,
            "run_design": {
                "method": campaign.run_planning_method,
                "standardized_effect": campaign.standardized_effect,
                "target_power": campaign.target_power,
                "family_alpha": campaign.family_alpha,
                "planned_comparisons": max(1, len(campaign.algorithms) - 1),
                "failure_allowance": campaign.failure_allowance,
                "power_evidence_sha256": campaign.power_evidence_sha256 or None,
                "initiated_runs_per_algorithm_task": campaign.runs,
            },
            "max_evaluations": campaign.max_evaluations,
            "population_size": campaign.population_size,
            "master_seed": campaign.master_seed,
            "algorithms": list(campaign.algorithms),
            "freeze_manifest": campaign.freeze_manifest,
            "require_protected_test": campaign.require_protected_test,
        },
        "tasks": [
            {
                "task_index": task.task_index,
                "task_id": task.task_id,
                "case_name": task.case_name,
                "study_key": task.study_key,
                "study_label": task.study_label,
                "evidence_role": task.evidence_role,
                "planned_jobs": task.planned_jobs,
                "formulation_manifest": ORPDVariableDecoder(
                    CaseLoader.load(task.case_name), task.config.variables
                ).formulation_manifest(),
                "scenario_configuration": task.config.to_dict()["scenarios"],
                "config": task.config.to_dict(),
                "experiment_id": None,
                "status": "planned",
            }
            for task in tasks
        ],
    }
    payload["design_sha256"] = campaign_plan_design_sha256(payload)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def campaign_plan_design_sha256(payload: dict) -> str:
    """Hash immutable campaign design while excluding run-time task state."""

    immutable_tasks = []
    for task in list(payload.get("tasks", []) or []):
        immutable_tasks.append(
            {
                key: value
                for key, value in dict(task).items()
                if key not in {"experiment_id", "status"}
            }
        )
    immutable = {
        "schema_version": int(payload.get("schema_version", 0) or 0),
        "campaign": dict(payload.get("campaign", {}) or {}),
        "tasks": immutable_tasks,
    }
    encoded = json.dumps(
        immutable,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_campaign_plan_design(source: str | Path) -> tuple[bool, str]:
    """Verify that immutable campaign design was not changed after planning."""

    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = str(payload.get("design_sha256", "") or "")
    actual = campaign_plan_design_sha256(payload)
    if not expected:
        return False, "Campaign plan has no immutable design SHA-256."
    if expected != actual:
        return False, "Campaign design changed after its plan was frozen."
    return True, "Campaign design SHA-256 verified."
