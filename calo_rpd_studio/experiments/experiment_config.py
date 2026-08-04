"""Serializable complete experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
import json
from pathlib import Path
import math

import yaml

from calo_rpd_studio.orpd.objectives import ObjectiveConfig, ObjectiveKind
from calo_rpd_studio.orpd.constraints import ConstraintToleranceConfig
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig, ShuntControlDefinition
from calo_rpd_studio.robustness.robust_objectives import (
    RobustAggregation,
    RobustObjectiveConfig,
    ConstraintAggregation,
)
from calo_rpd_studio.power_system.ac_power_flow import PowerFlowOptions
from .evaluation_budget import BudgetPolicy, EvaluationBudget
from calo_rpd_studio.portfolio.models import PortfolioConfig


CURRENT_EXECUTION_BACKENDS = frozenset({"cuda_preferred", "cpu_only"})
LEGACY_CUDA_EXECUTION_BACKENDS = frozenset(
    {
        "cuda_priority",
        "cuda_only",
        "throughput_auto",
        "weighted_split",
        "adaptive_hybrid",
        "gpu_preferred",
    }
)
LEGACY_EXECUTION_TUNING_FIELDS = frozenset(
    {
        "gpu_utilization_target",
        "cpu_utilization_target",
        "gpu_memory_limit",
        "gpu_parallel_jobs",
        "system_memory_limit",
        "cuda_task_share",
        "cpu_task_share",
        "strict_device_shares",
        "cuda_priority_work_stealing",
    }
)


def migrate_execution_backend(value: object) -> str:
    """Return the current execution mode while keeping historical XPU plans view-only.

    Legacy CUDA and hybrid scheduling labels described implementation strategies rather than a
    scientific choice. They now migrate to one automatic CUDA-first mode. An XPU plan is not
    silently reinterpreted because that would falsify its historical execution intent; it remains
    readable and :meth:`ExperimentConfig.validate` rejects it until a current mode is selected.
    """

    backend = str(value or "cuda_preferred").strip().lower()
    if backend in LEGACY_CUDA_EXECUTION_BACKENDS:
        return "cuda_preferred"
    return backend


def _reject_unknown_keys(payload: dict, allowed: set[str], context: str) -> None:
    unknown = sorted(set(map(str, payload.keys())) - set(allowed))
    if unknown:
        raise ValueError(
            f"Unknown {context} configuration field(s): {', '.join(unknown)}. "
            "Configuration keys are validated strictly so spelling mistakes cannot silently fall back to defaults."
        )


def _field_names(cls) -> set[str]:
    return {item.name for item in fields(cls)}


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
        for value, label in (
            (self.active_load_std, "active_load_std"),
            (self.reactive_load_std, "reactive_load_std"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if self.mode == "renewable_uncertainty":
            if (
                int(self.renewable_bus) <= 0
                or not math.isfinite(float(self.renewable_rated_mw))
                or float(self.renewable_rated_mw) <= 0
            ):
                raise ValueError(
                    "Renewable uncertainty requires a positive bus number and rated MW"
                )
            if (
                not math.isfinite(float(self.renewable_mean_capacity_factor))
                or not 0.0 <= float(self.renewable_mean_capacity_factor) <= 1.0
            ):
                raise ValueError("Renewable mean capacity factor must be between 0 and 1")
            if (
                not math.isfinite(float(self.renewable_std_capacity_factor))
                or float(self.renewable_std_capacity_factor) < 0
            ):
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
    study_strength: str = "custom"
    study_case_plan: list[str] = field(default_factory=lambda: ["case30"])
    study_standardized_effect: float | None = None
    study_target_power: float | None = None
    study_family_alpha: float = 0.05
    study_failure_allowance: float = 0.10
    study_run_planning_method: str = "custom"
    algorithms: list[str] = field(default_factory=lambda: ["CALO", "TLBO", "PSO"])
    # The default portfolio is JOURNAL evidence, whose explicit minimum is 30 runs.
    # Keep the default intrinsically valid rather than relying on validate() to mutate it.
    runs: int = 30
    master_seed: int = 2026
    population_size: int = 50
    max_iterations: int = 1000
    budget: EvaluationBudget = field(default_factory=EvaluationBudget)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    variables: ORPDVariableConfig = field(default_factory=ORPDVariableConfig)
    robust_objective: RobustObjectiveConfig = field(default_factory=RobustObjectiveConfig)
    power_flow: PowerFlowOptions = field(default_factory=PowerFlowOptions)
    constraint_tolerances: ConstraintToleranceConfig = field(
        default_factory=ConstraintToleranceConfig
    )
    scenarios: RobustScenarioSettings = field(default_factory=RobustScenarioSettings)
    algorithm_parameters: dict[str, dict] = field(default_factory=dict)
    output_directory: str = "results_data"
    parallel_workers: int = 1
    execution_backend: str = "cuda_preferred"
    scientific_backend: str = "torch_fp64"
    device_resident_execution: bool = True
    # v6.9: cap the CUDA process at 80% of physical VRAM while keeping all active
    # CUDA-eligible numerical state resident inside that budget.
    cuda_vram_budget_fraction: float = 0.80
    cuda_oom_retry_count: int = 4
    cuda_minimum_microbatch: int = 1
    cuda_resident_hot_loop: bool = True
    cuda_cpu_fallback_enabled: bool = True
    # CUDA evaluation windows target at least 100 candidates before the one packed result transfer.
    # Scientific dependencies may still force a smaller final/iteration batch; OOM recovery only
    # reduces the active CUDA microbatch and records that exception explicitly.
    tensor_batch_size: int = 100
    require_backend_parity: bool = True
    parity_objective_tolerance: float = 1e-5
    parity_violation_tolerance: float = 1e-6
    parity_voltage_tolerance: float = 1e-5
    parity_angle_tolerance_deg: float = 1e-4
    runtime_compute_device: str = "cpu"
    throughput_engine_enabled: bool = True
    persistent_accelerator_workers: bool = True
    cross_run_batching: bool = True
    cross_run_batch_window_ms: float = 4.0
    max_cross_run_batch: int = 4096
    automatic_batch_calibration: bool = True
    calibration_batch_sizes: list[int] = field(default_factory=lambda: [100, 200, 400])
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

    def validate_policy_development(self) -> None:
        """Validate only the scientific formulation consumed by CALO Intelligence.

        Policy training is an independent workflow.  It may reuse an ExperimentConfig file as an
        immutable container for objective/controls/power-flow/scenario definitions, but it must not
        inherit Comparison Study portfolio repetition minima, benchmark run counts, execution-lane
        shares, campaign budgets, or other tab-specific execution constraints.
        """
        self.objective.validate()
        self.variables.validate()
        self.power_flow.validate()
        self.constraint_tolerances.validate()
        self.robust_objective.validate()
        self.scenarios.validate()
        if (
            self.robust_objective.aggregation is RobustAggregation.CVAR
            and not 0.0 < float(self.robust_objective.cvar_alpha) < 1.0
        ):
            raise ValueError("CVaR alpha must lie strictly between 0 and 1")
        if float(self.robust_objective.risk_lambda) < 0.0:
            raise ValueError("risk_lambda must be non-negative")

    def validate(self) -> None:
        from calo_rpd_studio.algorithms.registry import SPECS

        if self.runs <= 0:
            raise ValueError("runs must be positive")
        if self.study_strength not in {"custom", "low", "moderate", "good", "strong"}:
            raise ValueError("study_strength must be custom, low, moderate, good, or strong")
        if not self.study_case_plan or any(not str(name).strip() for name in self.study_case_plan):
            raise ValueError("study_case_plan must contain at least one non-empty case name")
        if self.study_standardized_effect is not None and not (
            math.isfinite(float(self.study_standardized_effect))
            and 0.0 < float(self.study_standardized_effect) <= 3.0
        ):
            raise ValueError("study_standardized_effect must lie in (0, 3] when provided")
        if self.study_target_power is not None and not (
            math.isfinite(float(self.study_target_power))
            and 0.50 <= float(self.study_target_power) < 1.0
        ):
            raise ValueError("study_target_power must lie in [0.50, 1) when provided")
        if not 0.0 < float(self.study_family_alpha) < 1.0:
            raise ValueError("study_family_alpha must lie in (0, 1)")
        if not 0.0 <= float(self.study_failure_allowance) < 0.50:
            raise ValueError("study_failure_allowance must lie in [0, 0.50)")
        if self.study_run_planning_method not in {
            "custom",
            "screening_floor",
            "paired_normal_holm_approximation",
        }:
            raise ValueError("Unsupported study_run_planning_method")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not self.algorithms:
            raise ValueError("At least one algorithm must be selected")
        unknown = [name for name in self.algorithms if name not in SPECS]
        if unknown:
            raise ValueError(f"Unknown primary algorithms: {unknown}")
        if self.parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive")
        if "xpu" in str(self.execution_backend).lower():
            raise ValueError(
                "This historical XPU execution plan is view-only. Select Accelerated when "
                "available or CPU only before starting a new experiment."
            )
        if self.execution_backend not in CURRENT_EXECUTION_BACKENDS:
            raise ValueError(
                "execution_backend must be cuda_preferred or cpu_only; legacy scheduler modes "
                "must be loaded through ExperimentConfig.from_dict for migration"
            )
        if self.scientific_backend not in {"torch_fp64", "cpu_reference"}:
            raise ValueError("scientific_backend must be torch_fp64 or cpu_reference")
        if self.scientific_backend == "cpu_reference" and self.execution_backend != "cpu_only":
            raise ValueError("The cpu_reference scientific backend requires CPU-only scheduling")
        if int(self.tensor_batch_size) <= 0:
            raise ValueError("tensor_batch_size must be positive")
        if not math.isclose(float(self.cuda_vram_budget_fraction), 0.80, abs_tol=1e-12):
            raise ValueError("cuda_vram_budget_fraction is fixed at 0.80 of currently free VRAM")
        if int(self.cuda_oom_retry_count) < 0:
            raise ValueError("cuda_oom_retry_count must be non-negative")
        if int(self.cuda_minimum_microbatch) <= 0:
            raise ValueError("cuda_minimum_microbatch must be positive")
        if (
            not math.isfinite(float(self.cross_run_batch_window_ms))
            or float(self.cross_run_batch_window_ms) <= 0
        ):
            raise ValueError("cross_run_batch_window_ms must be finite and positive")
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
        self.objective.validate()
        self.variables.validate()
        self.power_flow.validate()
        self.constraint_tolerances.validate()
        self.robust_objective.validate()
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
            (self.parity_angle_tolerance_deg, "parity_angle_tolerance_deg"),
        ):
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"{label} must be positive and below 1")
        self.portfolio.validate()
        # Validation is deliberately read-only. Portfolio repetition requirements must be
        # normalized explicitly by the caller or corrected by the user; validate() never mutates
        # the scientific configuration behind the GUI's back.
        required_runs = int(self.portfolio.required_runs())
        if int(self.runs) < required_runs:
            raise ValueError(
                f"runs={self.runs} is below the portfolio-required minimum of {required_runs}. "
                "Apply explicit portfolio normalization before execution."
            )
        self.budget.validate()
        if (
            self.budget.policy is BudgetPolicy.EQUAL_EVALUATIONS
            and int(self.budget.max_evaluations) % int(self.population_size) != 0
        ):
            raise ValueError(
                "Equal-evaluation publication fairness requires max_evaluations to be divisible "
                "by population_size so every optimizer, including CALO, consumes exactly the same FE budget."
            )

    def normalize_for_execution(self) -> "ExperimentConfig":
        """Apply explicit execution normalization and return ``self``.

        This method is intentionally separate from :meth:`validate` so validation remains
        read-only. At present only the portfolio minimum repetition count is normalized.
        """
        self.portfolio.normalize_for_execution()
        self.runs = max(int(self.runs), int(self.portfolio.required_runs()))
        return self

    def to_dict(self) -> dict:
        def convert(value):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        payload = convert(asdict(self))
        for field_name in LEGACY_EXECUTION_TUNING_FIELDS:
            payload.pop(field_name, None)
        return payload

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
    def from_dict(cls, data: dict, *, allow_unknown_fields: bool = False) -> "ExperimentConfig":
        if not isinstance(data, dict):
            raise TypeError("Experiment configuration must be a mapping/object")
        if not allow_unknown_fields:
            # Historical scheduler knobs remain accepted only at this migration boundary. They are
            # deliberately discarded below: current execution is automatic CUDA-first routing
            # with availability-based memory admission, never percentage/share based scheduling.
            _reject_unknown_keys(
                data,
                _field_names(cls) | set(LEGACY_EXECUTION_TUNING_FIELDS),
                "experiment",
            )
        objective_data = dict(data.get("objective", {}) or {})
        if not allow_unknown_fields:
            _reject_unknown_keys(objective_data, _field_names(ObjectiveConfig), "objective")
        objective = ObjectiveConfig(
            kind=ObjectiveKind(objective_data.get("kind", ObjectiveKind.ACTIVE_POWER_LOSS.value)),
            weight_loss=float(objective_data.get("weight_loss", 1)),
            weight_voltage_deviation=float(objective_data.get("weight_voltage_deviation", 0)),
            weight_l_index=float(objective_data.get("weight_l_index", 0)),
            loss_scale=float(objective_data.get("loss_scale", 1)),
            voltage_deviation_scale=float(objective_data.get("voltage_deviation_scale", 1)),
            l_index_scale=float(objective_data.get("l_index_scale", 1)),
        )
        variable_data = dict(data.get("variables", {}) or {})
        if not allow_unknown_fields:
            _reject_unknown_keys(variable_data, _field_names(ORPDVariableConfig), "variables")
        shunt_items = list(variable_data.get("shunt_controls", []) or [])
        if not allow_unknown_fields:
            for index, item in enumerate(shunt_items):
                if not isinstance(item, dict):
                    raise TypeError(f"variables.shunt_controls[{index}] must be an object")
                _reject_unknown_keys(
                    item,
                    _field_names(ShuntControlDefinition),
                    f"variables.shunt_controls[{index}]",
                )
        shunts = tuple(ShuntControlDefinition(**item) for item in shunt_items)
        variables = ORPDVariableConfig(
            generator_voltages=bool(variable_data.get("generator_voltages", True)),
            transformer_taps=bool(variable_data.get("transformer_taps", True)),
            shunt_compensation=bool(variable_data.get("shunt_compensation", True)),
            discrete_transformer_taps=bool(variable_data.get("discrete_transformer_taps", True)),
            discrete_shunts=bool(variable_data.get("discrete_shunts", True)),
            transformer_minimum=float(variable_data.get("transformer_minimum", 0.9)),
            transformer_maximum=float(variable_data.get("transformer_maximum", 1.1)),
            transformer_step=float(variable_data.get("transformer_step", 0.0125)),
            shunt_controls=shunts,
            formulation_profile=str(
                variable_data.get("formulation_profile", "ieee-orpd-controls-v3.4.0")
            ),
            generator_voltage_buses=(
                None
                if variable_data.get("generator_voltage_buses") is None
                else tuple(variable_data["generator_voltage_buses"])
            ),
            transformer_branch_indices=(
                None
                if variable_data.get("transformer_branch_indices") is None
                else tuple(variable_data["transformer_branch_indices"])
            ),
        )
        robust_data = dict(data.get("robust_objective", {}) or {})
        if not allow_unknown_fields:
            _reject_unknown_keys(
                robust_data, _field_names(RobustObjectiveConfig), "robust_objective"
            )
        robust = RobustObjectiveConfig(
            aggregation=RobustAggregation(
                robust_data.get("aggregation", RobustAggregation.EXPECTED.value)
            ),
            risk_lambda=float(robust_data.get("risk_lambda", 1)),
            cvar_alpha=float(robust_data.get("cvar_alpha", 0.95)),
            constraint_aggregation=ConstraintAggregation(
                robust_data.get(
                    "constraint_aggregation", ConstraintAggregation.ALL_SCENARIO_MAX.value
                )
            ),
        )
        pf_data = dict(data.get("power_flow", {}) or {})
        if not allow_unknown_fields:
            _reject_unknown_keys(pf_data, _field_names(PowerFlowOptions), "power_flow")
        power_flow = PowerFlowOptions(
            tolerance=float(pf_data.get("tolerance", 1e-8)),
            max_iterations=int(pf_data.get("max_iterations", 30)),
            enforce_q_limits=bool(pf_data.get("enforce_q_limits", True)),
            max_q_limit_rounds=int(pf_data.get("max_q_limit_rounds", 10)),
            q_limit_tolerance_mvar=float(pf_data.get("q_limit_tolerance_mvar", 1e-6)),
        )
        tolerance_data = dict(data.get("constraint_tolerances", {}) or {})
        if not allow_unknown_fields:
            _reject_unknown_keys(
                tolerance_data, _field_names(ConstraintToleranceConfig), "constraint_tolerances"
            )
        constraint_tolerances = ConstraintToleranceConfig(
            voltage_pu=float(tolerance_data.get("voltage_pu", 1e-7)),
            generator_p_mw=float(tolerance_data.get("generator_p_mw", 1e-6)),
            generator_q_mvar=float(tolerance_data.get("generator_q_mvar", 1e-6)),
            branch_loading_percent=float(tolerance_data.get("branch_loading_percent", 1e-6)),
            branch_angle_deg=float(tolerance_data.get("branch_angle_deg", 1e-6)),
            feasibility_total=float(tolerance_data.get("feasibility_total", 1e-12)),
            schema_version=str(
                tolerance_data.get("schema_version", "calo_rpd_constraint_tolerance_v5.9")
            ),
        )
        budget_data = dict(data.get("budget", {}) or {})
        if not allow_unknown_fields:
            _reject_unknown_keys(budget_data, _field_names(EvaluationBudget), "budget")
        budget = EvaluationBudget(
            BudgetPolicy(budget_data.get("policy", BudgetPolicy.EQUAL_EVALUATIONS.value)),
            int(budget_data.get("max_evaluations", 5000)),
            float(budget_data["wall_clock_seconds"])
            if "wall_clock_seconds" in budget_data and budget_data["wall_clock_seconds"] is not None
            else None,
        )
        scenario_data = dict(data.get("scenarios", {}) or {})
        portfolio_data = dict(data.get("portfolio", {}) or {})
        if not allow_unknown_fields:
            _reject_unknown_keys(scenario_data, _field_names(RobustScenarioSettings), "scenarios")
            _reject_unknown_keys(portfolio_data, _field_names(PortfolioConfig), "portfolio")
        execution_backend = migrate_execution_backend(
            data.get("execution_backend", "cuda_preferred")
        )
        return cls(
            name=data.get("name", "CALO-RPD comparative experiment"),
            case_name=data.get("case_name", "case30"),
            study_strength=str(data.get("study_strength", "custom")),
            study_case_plan=list(data.get("study_case_plan", [data.get("case_name", "case30")])),
            study_standardized_effect=(
                float(data["study_standardized_effect"])
                if data.get("study_standardized_effect") is not None
                else None
            ),
            study_target_power=(
                float(data["study_target_power"])
                if data.get("study_target_power") is not None
                else None
            ),
            study_family_alpha=float(data.get("study_family_alpha", 0.05)),
            study_failure_allowance=float(data.get("study_failure_allowance", 0.10)),
            study_run_planning_method=str(data.get("study_run_planning_method", "custom")),
            algorithms=list(data.get("algorithms", ["CALO", "TLBO", "PSO"])),
            runs=int(data.get("runs", 30)),
            master_seed=int(data.get("master_seed", 2026)),
            population_size=int(data.get("population_size", 50)),
            max_iterations=int(data.get("max_iterations", 1000)),
            budget=budget,
            objective=objective,
            variables=variables,
            robust_objective=robust,
            power_flow=power_flow,
            constraint_tolerances=constraint_tolerances,
            scenarios=RobustScenarioSettings(**scenario_data),
            algorithm_parameters=dict(data.get("algorithm_parameters", {})),
            output_directory=data.get("output_directory", "results_data"),
            parallel_workers=int(data.get("parallel_workers", 1)),
            execution_backend=execution_backend,
            scientific_backend=str(data.get("scientific_backend", "torch_fp64")),
            device_resident_execution=bool(data.get("device_resident_execution", True)),
            cuda_vram_budget_fraction=0.80,
            cuda_oom_retry_count=int(data.get("cuda_oom_retry_count", 4)),
            cuda_minimum_microbatch=int(data.get("cuda_minimum_microbatch", 1)),
            cuda_resident_hot_loop=bool(data.get("cuda_resident_hot_loop", True)),
            cuda_cpu_fallback_enabled=bool(data.get("cuda_cpu_fallback_enabled", True)),
            tensor_batch_size=int(data.get("tensor_batch_size", 100)),
            require_backend_parity=bool(data.get("require_backend_parity", True)),
            parity_objective_tolerance=float(data.get("parity_objective_tolerance", 1e-5)),
            parity_violation_tolerance=float(data.get("parity_violation_tolerance", 1e-6)),
            parity_voltage_tolerance=float(data.get("parity_voltage_tolerance", 1e-5)),
            parity_angle_tolerance_deg=float(data.get("parity_angle_tolerance_deg", 1e-4)),
            runtime_compute_device=str(data.get("runtime_compute_device", "cpu")),
            throughput_engine_enabled=bool(data.get("throughput_engine_enabled", True)),
            persistent_accelerator_workers=bool(data.get("persistent_accelerator_workers", True)),
            cross_run_batching=bool(data.get("cross_run_batching", True)),
            cross_run_batch_window_ms=float(data.get("cross_run_batch_window_ms", 4.0)),
            max_cross_run_batch=int(data.get("max_cross_run_batch", 4096)),
            automatic_batch_calibration=bool(data.get("automatic_batch_calibration", True)),
            calibration_batch_sizes=[
                int(value) for value in data.get("calibration_batch_sizes", [100, 200, 400])
            ],
            calibration_repetitions=int(data.get("calibration_repetitions", 1)),
            throughput_profile_path=str(
                data.get("throughput_profile_path", "results_data/throughput_profile_v34.json")
            ),
            compile_stable_kernels=bool(data.get("compile_stable_kernels", False)),
            telemetry_iteration_interval=int(data.get("telemetry_iteration_interval", 10)),
            buffered_trace_writes=bool(data.get("buffered_trace_writes", True)),
            portfolio=PortfolioConfig.from_dict(portfolio_data),
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
    def load(cls, path, *, allow_unknown_fields: bool = False) -> "ExperimentConfig":
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        data = (
            yaml.safe_load(text) if source.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
        )
        return cls.from_dict(data, allow_unknown_fields=allow_unknown_fields)
