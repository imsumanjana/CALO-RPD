"""Serializable complete experiment configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path

import yaml

from calo_rpd_studio.orpd.objectives import ObjectiveConfig, ObjectiveKind
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig, ShuntControlDefinition
from calo_rpd_studio.robustness.robust_objectives import RobustAggregation, RobustObjectiveConfig
from .evaluation_budget import BudgetPolicy, EvaluationBudget


@dataclass(slots=True)
class RobustScenarioSettings:
    mode: str = "deterministic"
    count: int = 20
    active_load_std: float = 0.05
    reactive_load_std: float = 0.05
    branch_outages: list[int] = field(default_factory=list)
    generator_outages: list[int] = field(default_factory=list)
    renewable_bus: int = 0
    renewable_rated_mw: float = 0.0
    renewable_mean_capacity_factor: float = 0.5
    renewable_std_capacity_factor: float = 0.15


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "CALO-RPD comparative experiment"
    case_name: str = "case30"
    algorithms: list[str] = field(default_factory=lambda: ["CALO", "TLBO", "PSO"])
    runs: int = 5
    master_seed: int = 2026
    population_size: int = 50
    max_iterations: int = 1000
    budget: EvaluationBudget = field(default_factory=EvaluationBudget)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    variables: ORPDVariableConfig = field(default_factory=ORPDVariableConfig)
    robust_objective: RobustObjectiveConfig = field(default_factory=RobustObjectiveConfig)
    scenarios: RobustScenarioSettings = field(default_factory=RobustScenarioSettings)
    algorithm_parameters: dict[str, dict] = field(default_factory=dict)
    output_directory: str = "results_data"
    parallel_workers: int = 1
    execution_backend: str = "adaptive_hybrid"
    gpu_utilization_target: int = 70
    cpu_utilization_target: int = 50
    gpu_memory_limit: int = 85
    gpu_parallel_jobs: int = 4
    xpu_utilization_target: int = 70
    xpu_memory_limit: int = 85
    xpu_parallel_jobs: int = 2
    system_memory_limit: int = 85

    def validate(self) -> None:
        from calo_rpd_studio.algorithms.registry import SPECS

        if self.runs <= 0:
            raise ValueError("runs must be positive")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if not self.algorithms:
            raise ValueError("At least one algorithm must be selected")
        unknown = [name for name in self.algorithms if name not in SPECS]
        if unknown:
            raise ValueError(f"Unknown primary algorithms: {unknown}")
        if self.parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive")
        if self.execution_backend not in {"adaptive_hybrid", "cpu_only", "gpu_preferred"}:
            raise ValueError("Unsupported execution backend")
        if not 10 <= int(self.gpu_utilization_target) <= 100:
            raise ValueError("gpu_utilization_target must be between 10 and 100")
        if not 10 <= int(self.cpu_utilization_target) <= 100:
            raise ValueError("cpu_utilization_target must be between 10 and 100")
        if not 20 <= int(self.gpu_memory_limit) <= 100:
            raise ValueError("gpu_memory_limit must be between 20 and 100")
        if int(self.gpu_parallel_jobs) <= 0:
            raise ValueError("gpu_parallel_jobs must be positive")
        if not 10 <= int(self.xpu_utilization_target) <= 100:
            raise ValueError("xpu_utilization_target must be between 10 and 100")
        if not 20 <= int(self.xpu_memory_limit) <= 100:
            raise ValueError("xpu_memory_limit must be between 20 and 100")
        if int(self.xpu_parallel_jobs) <= 0:
            raise ValueError("xpu_parallel_jobs must be positive")
        if not 20 <= int(self.system_memory_limit) <= 100:
            raise ValueError("system_memory_limit must be between 20 and 100")
        self.budget.validate()

    def to_dict(self) -> dict:
        def convert(value):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))

    def save(self, path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if destination.suffix.lower() in {".yaml", ".yml"}:
            destination.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        else:
            destination.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        objective_data = data.get("objective", {})
        objective = ObjectiveConfig(
            ObjectiveKind(objective_data.get("kind", ObjectiveKind.ACTIVE_POWER_LOSS.value)),
            float(objective_data.get("weight_loss", 1)),
            float(objective_data.get("weight_voltage_deviation", 0)),
            float(objective_data.get("weight_l_index", 0)),
            float(objective_data.get("loss_scale", 1)),
            float(objective_data.get("voltage_deviation_scale", 1)),
            float(objective_data.get("l_index_scale", 1)),
        )
        variable_data = data.get("variables", {})
        shunts = tuple(
            ShuntControlDefinition(**item) for item in variable_data.get("shunt_controls", [])
        )
        variables = ORPDVariableConfig(
            bool(variable_data.get("generator_voltages", True)),
            bool(variable_data.get("transformer_taps", True)),
            bool(variable_data.get("shunt_compensation", True)),
            bool(variable_data.get("discrete_transformer_taps", True)),
            bool(variable_data.get("discrete_shunts", True)),
            float(variable_data.get("transformer_minimum", 0.9)),
            float(variable_data.get("transformer_maximum", 1.1)),
            float(variable_data.get("transformer_step", 0.0125)),
            shunts,
        )
        robust_data = data.get("robust_objective", {})
        robust = RobustObjectiveConfig(
            RobustAggregation(
                robust_data.get("aggregation", RobustAggregation.EXPECTED.value)
            ),
            float(robust_data.get("risk_lambda", 1)),
            float(robust_data.get("cvar_alpha", 0.95)),
        )
        budget_data = data.get("budget", {})
        budget = EvaluationBudget(
            BudgetPolicy(
                budget_data.get("policy", BudgetPolicy.EQUAL_EVALUATIONS.value)
            ),
            int(budget_data.get("max_evaluations", 5000)),
            budget_data.get("wall_clock_seconds"),
        )
        return cls(
            name=data.get("name", "CALO-RPD comparative experiment"),
            case_name=data.get("case_name", "case30"),
            algorithms=list(data.get("algorithms", ["CALO", "TLBO", "PSO"])),
            runs=int(data.get("runs", 5)),
            master_seed=int(data.get("master_seed", 2026)),
            population_size=int(data.get("population_size", 50)),
            max_iterations=int(data.get("max_iterations", 1000)),
            budget=budget,
            objective=objective,
            variables=variables,
            robust_objective=robust,
            scenarios=RobustScenarioSettings(**data.get("scenarios", {})),
            algorithm_parameters=dict(data.get("algorithm_parameters", {})),
            output_directory=data.get("output_directory", "results_data"),
            parallel_workers=int(data.get("parallel_workers", 1)),
            execution_backend=str(data.get("execution_backend", "adaptive_hybrid")),
            gpu_utilization_target=int(data.get("gpu_utilization_target", 70)),
            cpu_utilization_target=int(data.get("cpu_utilization_target", 50)),
            gpu_memory_limit=int(data.get("gpu_memory_limit", 85)),
            gpu_parallel_jobs=int(data.get("gpu_parallel_jobs", 4)),
            xpu_utilization_target=int(data.get("xpu_utilization_target", 70)),
            xpu_memory_limit=int(data.get("xpu_memory_limit", 85)),
            xpu_parallel_jobs=int(data.get("xpu_parallel_jobs", 2)),
            system_memory_limit=int(data.get("system_memory_limit", 85)),
        )

    @classmethod
    def load(cls, path) -> "ExperimentConfig":
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        data = (
            yaml.safe_load(text)
            if source.suffix.lower() in {".yaml", ".yml"}
            else json.loads(text)
        )
        return cls.from_dict(data)
