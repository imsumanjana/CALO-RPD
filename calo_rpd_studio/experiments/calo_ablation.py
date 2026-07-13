"""Predefined CALO Core v2 ablation suite."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.legacy_mtlbo import LegacyMTLBOOptimizer
from .experiment_runner import CompletedRun, build_problem, run_single


@dataclass(frozen=True, slots=True)
class AblationSpec:
    label: str
    algorithm: str = "CALO"
    parameters: dict | None = None


ABLATION_SPECS = (
    AblationSpec("Classical TLBO", "TLBO", {}),
    AblationSpec("Legacy Gaussian MTLBO", "LEGACY", {}),
    AblationSpec("CALO Core v2 without AI", "CALO", {"use_ai": False}),
    AblationSpec("CALO without epsilon-feasibility", "CALO", {"use_epsilon": False}),
    AblationSpec("CALO without dual archives", "CALO", {"use_dual_archives": False}),
    AblationSpec("CALO without mixed-variable learning", "CALO", {"use_mixed_variable": False}),
    AblationSpec("CALO without success memory", "CALO", {"use_memory": False}),
    AblationSpec("CALO without diversity recovery", "CALO", {"use_diversity_recovery": False}),
    AblationSpec("Complete CALO", "CALO", {}),
)


def run_ablation(config, spec, run_index, seeds, progress_callback=None, cancel_callback=None):
    if spec.algorithm != "LEGACY":
        local = deepcopy(config)
        local.algorithm_parameters = dict(local.algorithm_parameters)
        local.algorithm_parameters[spec.algorithm] = {
            **local.algorithm_parameters.get(spec.algorithm, {}),
            **(spec.parameters or {}),
        }
        completed = run_single(
            local,
            spec.algorithm,
            run_index,
            seeds,
            progress_callback,
            cancel_callback,
        )
        completed.result.algorithm = spec.label
        completed.algorithm = spec.label
        return completed
    problem = build_problem(config, seeds.scenario_seed)
    optimizer = LegacyMTLBOOptimizer(
        problem,
        OptimizerConfig(
            config.population_size,
            config.budget.max_evaluations,
            max(config.max_iterations, config.budget.max_evaluations),
            {},
        ),
        seeds.algorithm_seed,
        progress_callback,
        cancel_callback,
    )
    result = optimizer.run()
    result.algorithm = spec.label
    return CompletedRun(spec.label, run_index, seeds, result)
