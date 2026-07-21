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
from calo_rpd_studio.portfolio.models import PortfolioConfig


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

    def validate(self) -> None:
        modes = {
            "deterministic",
            "load_uncertainty",
            "monte_carlo",
            "renewable_uncertainty",
            "branch_contingency",
            "generator_contingency",
        }
        if self.mode not in modes:
            raise ValueError(f"Unsupported scenario mode: {self.mode}")
        if self.mode != "deterministic" and int(self.count) <= 0:
            raise ValueError("Robust scenario count must be positive")
        if float(self.active_load_std) < 0 or float(self.reactive_load_std) < 0:
            raise ValueError("Load standard deviations must be non-negative")
        if self.mode == "renewable_uncertainty":
            if int(self.renewable_bus) <= 0 or float(self.renewable_rated_mw) <= 0:
                raise ValueError(
                    "Renewable uncertainty requires a positive bus number and rated MW"
                )
            if not 0.0 <= float(self.renewable_mean_capacity_factor) <= 1.0:
                raise ValueError("Renewable mean capacity factor must be between 0 and 1")
            if float(self.renewable_std_capacity_factor) < 0:
                raise ValueError(
                    "Renewable capacity-factor standard deviation must be non-negative"
                )
        if self.mode == "branch_contingency" and not self.branch_outages:
            raise ValueError("Branch contingency mode requires at least one branch outage index")
        if self.mode == "generator_contingency" and not self.generator_outages:
            raise ValueError(
                "Generator contingency mode requires at least one generator outage index"
            )
        if any(int(index) < 0 for index in (*self.branch_outages, *self.generator_outages)):
            raise ValueError("Contingency indices must be non-negative")


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
    execution_backend: str = "gpu_preferred"
    gpu_utilization_target: int = 70
    cpu_utilization_target: int = 50
    gpu_memory_limit: int = 85
    gpu_parallel_jobs: int = 4
    xpu_utilization_target: int = 70
    xpu_memory_limit: int = 85
    xpu_parallel_jobs: int = 2
    system_memory_limit: int = 85
    cuda_task_share: int = 100
    xpu_task_share: int = 0
    cpu_task_share: int = 0
    strict_device_shares: bool = True
    scientific_backend: str = "torch_fp64"
    device_resident_execution: bool = True
    cuda_priority_work_stealing: bool = True
    tensor_batch_size: int = 64
    require_backend_parity: bool = True
    parity_objective_tolerance: float = 1e-5
    parity_violation_tolerance: float = 1e-6
    parity_voltage_tolerance: float = 1e-5
    runtime_compute_device: str = "cpu"
    throughput_engine_enabled: bool = True
    persistent_accelerator_workers: bool = True
    cross_run_batching: bool = True
    cross_run_batch_window_ms: float = 4.0
    max_cross_run_batch: int = 4096
    automatic_batch_calibration: bool = True
    calibration_batch_sizes: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    calibration_repetitions: int = 1
    throughput_profile_path: str = "results_data/throughput_profile_v34.json"
    compile_stable_kernels: bool = False
    telemetry_iteration_interval: int = 10
    buffered_trace_writes: bool = True
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    portfolio_id: str = ""
    resume_enabled: bool = True
    resume_campaign_id: str = ""
    checkpoint_interval_evaluations: int = 500
    safe_pause: bool = True
    reuse_compatible_results: bool = True
    # v5 experiment-evolution metadata. These fields never alter the original experiment record;
    # they describe a new execution revision attached to the same scientific experiment identity.
    extension_experiment_id: str = ""
    experiment_revision_id: str = ""
    extension_mode: str = ""
    extension_publication_eligible: bool = True
    extension_run_indices: list[int] = field(default_factory=list)
    extension_algorithm_names: list[str] = field(default_factory=list)
    # exact_continue resumes an optimizer-state checkpoint; recompute_from_seed reruns the same
    # paired seed from FE=0 under the new horizon and stores it as a new evidence revision.
    extension_execution_strategy: str = "exact_continue"
    # Exact continuation may branch from any preserved horizon that has a complete optimizer
    # checkpoint. Recompute-from-seed always starts from FE=0 and ignores this field.
    extension_source_horizon: int = 0
    require_exact_run_checkpoint_for_horizon_extension: bool = True
    run_checkpoint_root: str = ""
    extension_checkpoint_paths: dict[str, str] = field(default_factory=dict)
    extension_existing_run_ids: dict[str, str] = field(default_factory=dict)

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
        if self.execution_backend not in {
            "cuda_priority",
            "cuda_only",
            "throughput_auto",
            "weighted_split",
            "adaptive_hybrid",
            "cpu_only",
            "gpu_preferred",
        }:
            raise ValueError("Unsupported execution backend")
        if self.scientific_backend not in {"torch_fp64", "cpu_reference"}:
            raise ValueError("scientific_backend must be torch_fp64 or cpu_reference")
        if self.scientific_backend == "cpu_reference" and self.execution_backend != "cpu_only":
            raise ValueError("The cpu_reference scientific backend requires CPU-only scheduling")
        if int(self.tensor_batch_size) <= 0:
            raise ValueError("tensor_batch_size must be positive")
        if float(self.cross_run_batch_window_ms) <= 0:
            raise ValueError("cross_run_batch_window_ms must be positive")
        if int(self.max_cross_run_batch) <= 0:
            raise ValueError("max_cross_run_batch must be positive")
        if int(self.calibration_repetitions) <= 0:
            raise ValueError("calibration_repetitions must be positive")
        if not self.calibration_batch_sizes or any(
            int(value) <= 0 for value in self.calibration_batch_sizes
        ):
            raise ValueError("calibration_batch_sizes must contain positive integers")
        if int(self.telemetry_iteration_interval) <= 0:
            raise ValueError("telemetry_iteration_interval must be positive")
        if int(self.checkpoint_interval_evaluations) <= 0:
            raise ValueError("checkpoint_interval_evaluations must be positive")
        self.scenarios.validate()
        if (
            self.robust_objective.aggregation is RobustAggregation.CVAR
            and not 0.0 < float(self.robust_objective.cvar_alpha) < 1.0
        ):
            raise ValueError("CVaR alpha must lie strictly between 0 and 1")
        if float(self.robust_objective.risk_lambda) < 0.0:
            raise ValueError("risk_lambda must be non-negative")
        for value, label in (
            (self.parity_objective_tolerance, "parity_objective_tolerance"),
            (self.parity_violation_tolerance, "parity_violation_tolerance"),
            (self.parity_voltage_tolerance, "parity_voltage_tolerance"),
        ):
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"{label} must be positive and below 1")
        self.portfolio.validate()
        # Portfolio requirements are a minimum, never a reason to silently reduce a user's
        # requested repetitions. A request for 31–50 runs must remain exactly 31–50.
        self.runs = max(int(self.runs), int(self.portfolio.required_runs()))
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
        shares = (int(self.cuda_task_share), int(self.xpu_task_share), int(self.cpu_task_share))
        if any(value < 0 or value > 100 for value in shares):
            raise ValueError("Device task shares must each be between 0 and 100")
        if sum(shares) != 100:
            raise ValueError("CUDA, XPU, and CPU task shares must sum to 100")
        if self.execution_backend == "cuda_priority" and shares != (80, 10, 10):
            raise ValueError("cuda_priority requires the fixed 80/10/10 CUDA/XPU/CPU share")
        if self.execution_backend in {"cuda_only", "gpu_preferred"} and shares != (100, 0, 0):
            raise ValueError(f"{self.execution_backend} requires the fixed 100/0/0 preferred share")
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
            str(variable_data.get("formulation_profile", "ieee-orpd-controls-v3.4.0")),
        )
        robust_data = data.get("robust_objective", {})
        robust = RobustObjectiveConfig(
            RobustAggregation(robust_data.get("aggregation", RobustAggregation.EXPECTED.value)),
            float(robust_data.get("risk_lambda", 1)),
            float(robust_data.get("cvar_alpha", 0.95)),
        )
        budget_data = data.get("budget", {})
        budget = EvaluationBudget(
            BudgetPolicy(budget_data.get("policy", BudgetPolicy.EQUAL_EVALUATIONS.value)),
            int(budget_data.get("max_evaluations", 5000)),
            float(budget_data["wall_clock_seconds"]) if "wall_clock_seconds" in budget_data and budget_data["wall_clock_seconds"] is not None else None,
        )
        execution_backend = str(data.get("execution_backend", "gpu_preferred"))
        preset_shares = (
            (100, 0, 0) if execution_backend in {"cuda_only", "gpu_preferred"} else (80, 10, 10)
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
            execution_backend=execution_backend,
            gpu_utilization_target=int(data.get("gpu_utilization_target", 70)),
            cpu_utilization_target=int(data.get("cpu_utilization_target", 50)),
            gpu_memory_limit=int(data.get("gpu_memory_limit", 85)),
            gpu_parallel_jobs=int(data.get("gpu_parallel_jobs", 4)),
            xpu_utilization_target=int(data.get("xpu_utilization_target", 70)),
            xpu_memory_limit=int(data.get("xpu_memory_limit", 85)),
            xpu_parallel_jobs=int(data.get("xpu_parallel_jobs", 2)),
            system_memory_limit=int(data.get("system_memory_limit", 85)),
            cuda_task_share=int(data.get("cuda_task_share", preset_shares[0])),
            xpu_task_share=int(data.get("xpu_task_share", preset_shares[1])),
            cpu_task_share=int(data.get("cpu_task_share", preset_shares[2])),
            strict_device_shares=bool(data.get("strict_device_shares", True)),
            scientific_backend=str(data.get("scientific_backend", "torch_fp64")),
            device_resident_execution=bool(data.get("device_resident_execution", True)),
            cuda_priority_work_stealing=bool(data.get("cuda_priority_work_stealing", True)),
            tensor_batch_size=int(data.get("tensor_batch_size", 64)),
            require_backend_parity=bool(data.get("require_backend_parity", True)),
            parity_objective_tolerance=float(data.get("parity_objective_tolerance", 1e-5)),
            parity_violation_tolerance=float(data.get("parity_violation_tolerance", 1e-6)),
            parity_voltage_tolerance=float(data.get("parity_voltage_tolerance", 1e-5)),
            runtime_compute_device=str(data.get("runtime_compute_device", "cpu")),
            throughput_engine_enabled=bool(data.get("throughput_engine_enabled", True)),
            persistent_accelerator_workers=bool(data.get("persistent_accelerator_workers", True)),
            cross_run_batching=bool(data.get("cross_run_batching", True)),
            cross_run_batch_window_ms=float(data.get("cross_run_batch_window_ms", 4.0)),
            max_cross_run_batch=int(data.get("max_cross_run_batch", 4096)),
            automatic_batch_calibration=bool(data.get("automatic_batch_calibration", True)),
            calibration_batch_sizes=[
                int(value) for value in data.get("calibration_batch_sizes", [16, 32, 64, 128, 256])
            ],
            calibration_repetitions=int(data.get("calibration_repetitions", 1)),
            throughput_profile_path=str(
                data.get("throughput_profile_path", "results_data/throughput_profile_v34.json")
            ),
            compile_stable_kernels=bool(data.get("compile_stable_kernels", False)),
            telemetry_iteration_interval=int(data.get("telemetry_iteration_interval", 10)),
            buffered_trace_writes=bool(data.get("buffered_trace_writes", True)),
            portfolio=PortfolioConfig.from_dict(data.get("portfolio", {})),
            portfolio_id=str(data.get("portfolio_id", "")),
            resume_enabled=bool(data.get("resume_enabled", True)),
            resume_campaign_id=str(data.get("resume_campaign_id", "")),
            checkpoint_interval_evaluations=int(data.get("checkpoint_interval_evaluations", 500)),
            safe_pause=bool(data.get("safe_pause", True)),
            reuse_compatible_results=bool(data.get("reuse_compatible_results", True)),
            extension_experiment_id=str(data.get("extension_experiment_id", "")),
            experiment_revision_id=str(data.get("experiment_revision_id", "")),
            extension_mode=str(data.get("extension_mode", "")),
            extension_publication_eligible=bool(data.get("extension_publication_eligible", True)),
            extension_run_indices=[int(v) for v in data.get("extension_run_indices", [])],
            extension_algorithm_names=[str(v) for v in data.get("extension_algorithm_names", [])],
            extension_execution_strategy=str(
                data.get("extension_execution_strategy", "exact_continue")
            ),
            extension_source_horizon=int(data.get("extension_source_horizon", 0) or 0),
            require_exact_run_checkpoint_for_horizon_extension=bool(
                data.get("require_exact_run_checkpoint_for_horizon_extension", True)
            ),
            run_checkpoint_root=str(data.get("run_checkpoint_root", "")),
            extension_checkpoint_paths={
                str(k): str(v) for k, v in dict(data.get("extension_checkpoint_paths", {})).items()
            },
            extension_existing_run_ids={
                str(k): str(v) for k, v in dict(data.get("extension_existing_run_ids", {})).items()
            },
        )

    @classmethod
    def load(cls, path) -> "ExperimentConfig":
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        data = (
            yaml.safe_load(text) if source.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
        )
        return cls.from_dict(data)
