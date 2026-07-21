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
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableDecoder
from calo_rpd_studio.portfolio.models import EvidenceProfile, PortfolioKind
from calo_rpd_studio.power_system.case_loader import CaseLoader
from .freeze import verify_freeze_manifest
from .suite import BenchmarkSuite, standard_benchmark_suite


@dataclass(slots=True)
class BenchmarkCampaignConfig:
    name: str = "CALO-RPD v3.4 final benchmark"
    cases: tuple[str, ...] = ("case30", "case57", "case118", "case300")
    study_keys: tuple[str, ...] = (
        "deterministic",
        "mixed",
        "load_mean_risk",
        "renewable_cvar",
        "branch_worst_case",
    )
    runs: int = 30
    max_evaluations: int = 5000
    population_size: int = 50
    master_seed: int = 2026
    output_directory: str = "benchmark_v34"
    parallel_workers: int = 1
    execution_backend: str = "weighted_split"
    freeze_manifest: str = field(default_factory=lambda: str(Path(__file__).resolve().parents[1] / "data" / "frozen" / "calo_v410_freeze.json"))
    algorithms: tuple[str, ...] = field(default_factory=primary_algorithm_names)

    def validate(self, suite: BenchmarkSuite | None = None, *, verify_freeze: bool = True) -> None:
        suite = suite or standard_benchmark_suite()
        if not 30 <= int(self.runs) <= 50:
            raise ValueError("Final benchmark campaigns require 30–50 independent runs per task.")
        if self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")
        if tuple(self.algorithms) != tuple(primary_algorithm_names()):
            raise ValueError("The frozen v3.4 final benchmark must include exactly the 20 primary algorithms.")
        unknown_cases = set(self.cases) - set(suite.cases)
        if unknown_cases:
            raise ValueError(f"Unsupported benchmark cases: {sorted(unknown_cases)}")
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
            # Final campaign repetitions are explicit evidence requirements.  Synchronize the
            # embedded portfolio so validation can never reduce or inflate 30–50 requested runs.
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
            config.output_directory = str(Path(campaign.output_directory) / "raw_arrays" / case_name / study_key)
            study.configure(config)
            config.validate()
            task_id = f"{case_name}__{study_key}"
            tasks.append(BenchmarkTask(index, task_id, case_name, study_key, study.label, config))
            index += 1
    return tasks


def write_campaign_plan(campaign: BenchmarkCampaignConfig, tasks: list[BenchmarkTask], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "campaign": {
            "name": campaign.name,
            "cases": list(campaign.cases),
            "study_keys": list(campaign.study_keys),
            "runs": campaign.runs,
            "max_evaluations": campaign.max_evaluations,
            "population_size": campaign.population_size,
            "master_seed": campaign.master_seed,
            "algorithms": list(campaign.algorithms),
            "freeze_manifest": campaign.freeze_manifest,
        },
        "tasks": [
            {
                "task_index": task.task_index,
                "task_id": task.task_id,
                "case_name": task.case_name,
                "study_key": task.study_key,
                "study_label": task.study_label,
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
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination
