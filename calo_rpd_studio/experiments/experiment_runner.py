"""Run construction, robust scenarios, budget enforcement, and failure isolation."""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json, time, traceback
from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.registry import SPECS, create_optimizer
from calo_rpd_studio.algorithms.result import OptimizerResult
from calo_rpd_studio.orpd.problem import ORPDProblem, ORPDProblemConfig
from calo_rpd_studio.accelerated.torch_orpd import AcceleratedORPDProblem
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.robustness.scenario import Scenario
from calo_rpd_studio.robustness.scenario_generator import (
    ScenarioGeneratorConfig,
    generate_load_scenarios,
)
from calo_rpd_studio.robustness.renewable_uncertainty import renewable_scenarios
from calo_rpd_studio.robustness.contingencies import (
    n_minus_one_branch_scenarios,
    n_minus_one_generator_scenarios,
)
from .evaluation_budget import BudgetPolicy
from .seed_manager import SeedManager, RunSeeds


@dataclass(slots=True)
class CompletedRun:
    algorithm: str
    run_index: int
    seeds: RunSeeds
    result: OptimizerResult


@dataclass(slots=True)
class FailedRun:
    algorithm: str
    run_index: int
    seeds: RunSeeds
    failure_type: str
    message: str
    traceback_text: str
    evaluation_count: int = 0
    numerical_state: dict = field(default_factory=dict)


def failed_run_from_exception(algorithm, run_index, seeds, exc):
    return FailedRun(
        algorithm,
        run_index,
        seeds,
        type(exc).__name__,
        str(exc),
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )


def build_scenarios(config, seed, case=None):
    """Build, validate, and normalize a non-empty robust scenario set."""
    import numpy as np

    settings = config.scenarios
    settings.validate()
    if settings.mode == "deterministic":
        scenarios = [Scenario("base", 1.0)]
    elif settings.mode in {"load_uncertainty", "monte_carlo"}:
        scenarios = generate_load_scenarios(
            ScenarioGeneratorConfig(
                settings.count, settings.active_load_std, settings.reactive_load_std
            ),
            seed,
        )
    elif settings.mode == "renewable_uncertainty":
        if settings.renewable_bus <= 0 or settings.renewable_rated_mw <= 0:
            raise ValueError(
                "Renewable uncertainty requires a valid bus number and positive rated power"
            )
        if case is not None and int(settings.renewable_bus) not in case.bus_index_map():
            raise ValueError(
                f"Renewable uncertainty bus {settings.renewable_bus} does not exist in {config.case_name}"
            )
        scenarios = renewable_scenarios(
            settings.count,
            settings.renewable_bus,
            settings.renewable_rated_mw,
            settings.renewable_mean_capacity_factor,
            settings.renewable_std_capacity_factor,
            np.random.default_rng(seed),
        )
    elif settings.mode == "branch_contingency":
        if case is not None and any(index >= case.n_branch for index in settings.branch_outages):
            raise ValueError("At least one branch-contingency index is outside the selected case")
        scenarios = n_minus_one_branch_scenarios(settings.branch_outages)
    elif settings.mode == "generator_contingency":
        if case is not None and any(index >= case.n_gen for index in settings.generator_outages):
            raise ValueError(
                "At least one generator-contingency index is outside the selected case"
            )
        scenarios = n_minus_one_generator_scenarios(settings.generator_outages)
    else:
        raise ValueError(f"Unsupported scenario mode: {settings.mode}")
    if not scenarios:
        raise ValueError("Scenario generation produced no scenarios")
    weights = np.asarray([float(item.weight) for item in scenarios], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("Scenario weights must be finite, non-negative, and have a positive sum")
    weights /= float(weights.sum())
    for item, weight in zip(scenarios, weights, strict=True):
        item.weight = float(weight)
        if case is not None:
            transformed = item.apply(case)
            if (
                transformed.n_bus != case.n_bus
                or transformed.n_branch != case.n_branch
                or transformed.n_gen != case.n_gen
            ):
                raise ValueError(f"Scenario {item.name!r} changed the case topology dimensions")
    return scenarios


def build_problem(config, scenario_seed):
    case = CaseLoader.load(config.case_name)
    problem_config = ORPDProblemConfig(config.objective, config.variables, config.robust_objective)
    scenarios = build_scenarios(config, scenario_seed, case)
    if str(getattr(config, "scientific_backend", "cpu_reference")) == "torch_fp64":
        return AcceleratedORPDProblem(
            case,
            problem_config,
            scenarios,
            device=str(getattr(config, "runtime_compute_device", "cpu")),
            dtype_name="float64",
            batch_size=int(getattr(config, "tensor_batch_size", 64)),
            device_resident=bool(getattr(config, "device_resident_execution", True)),
        )
    return ORPDProblem(case, problem_config, scenarios)


def run_single(config, algorithm, run_index, seeds, progress_callback=None, cancel_callback=None):
    problem = build_problem(config, seeds.scenario_seed)
    defaults = dict(SPECS[algorithm].default_parameters)
    defaults.update(config.algorithm_parameters.get(algorithm, {}))
    defaults.setdefault("execution_device", str(getattr(config, "runtime_compute_device", "cpu")))
    if str(getattr(config, "scientific_backend", "cpu_reference")) == "torch_fp64":
        defaults.setdefault("optimizer_backend", "torch")
    if algorithm == "CALO":
        defaults.setdefault("ai_inference_seed", seeds.ai_inference_seed)
        defaults.setdefault(
            "inference_device", str(getattr(config, "runtime_compute_device", "cpu"))
        )
    started = time.perf_counter()
    policy = config.budget.policy
    if policy is BudgetPolicy.EQUAL_WALL_CLOCK:
        max_evaluations = 2_000_000_000
        max_iterations = 2_000_000_000
    elif policy is BudgetPolicy.EQUAL_EVALUATIONS:
        max_evaluations = config.budget.max_evaluations
        max_iterations = max(config.max_iterations, config.budget.max_evaluations)
    else:
        max_evaluations = config.budget.max_evaluations
        max_iterations = config.max_iterations

    def cancel():
        if cancel_callback and cancel_callback():
            return True
        return bool(
            policy is BudgetPolicy.EQUAL_WALL_CLOCK
            and config.budget.wall_clock_seconds is not None
            and time.perf_counter() - started >= config.budget.wall_clock_seconds
        )

    opt = create_optimizer(
        algorithm,
        problem,
        OptimizerConfig(config.population_size, max_evaluations, max_iterations, defaults),
        seeds.algorithm_seed,
        progress_callback,
        cancel,
    )
    result = opt.run()
    decoder = getattr(problem, "decoder", None)
    if decoder is not None and hasattr(decoder, "formulation_manifest"):
        result.metadata["formulation_manifest"] = decoder.formulation_manifest()
    scenarios = list(getattr(problem, "scenarios", ()))
    scenario_payload = {
        "schema_version": 1,
        "mode": str(config.scenarios.mode),
        "count": len(scenarios),
        "names": [str(item.name) for item in scenarios],
        "weights": [float(item.weight) for item in scenarios],
        "seed": int(seeds.scenario_seed),
        "base_case_checksum": str(problem.case.checksum()),
        "transformed_case_checksums": [
            str(item.apply(problem.case).checksum()) for item in scenarios
        ],
    }
    encoded = json.dumps(
        scenario_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    scenario_payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    result.metadata["scenario_manifest"] = scenario_payload
    if (
        policy is BudgetPolicy.EQUAL_WALL_CLOCK
        and not (cancel_callback and cancel_callback())
        and config.budget.wall_clock_seconds
        and time.perf_counter() - started >= config.budget.wall_clock_seconds
    ):
        result.termination_reason = "wall_clock_budget"
    return CompletedRun(algorithm, run_index, seeds, result)


def run_sequential(config, progress_callback=None, cancel_callback=None):
    config.validate()
    seeds = SeedManager(config.master_seed).generate(config.runs)
    out = []
    for ri in range(config.runs):
        for algo in config.algorithms:
            if cancel_callback and cancel_callback():
                return out
            out.append(run_single(config, algo, ri, seeds[ri], progress_callback, cancel_callback))
    return out


def run_sequential_resilient(config, progress_callback=None, cancel_callback=None):
    config.validate()
    seeds = SeedManager(config.master_seed).generate(config.runs)
    done = []
    failed = []
    for ri in range(config.runs):
        for algo in config.algorithms:
            if cancel_callback and cancel_callback():
                return done, failed
            try:
                done.append(
                    run_single(config, algo, ri, seeds[ri], progress_callback, cancel_callback)
                )
            except Exception as exc:
                failed.append(failed_run_from_exception(algo, ri, seeds[ri], exc))
    return done, failed
