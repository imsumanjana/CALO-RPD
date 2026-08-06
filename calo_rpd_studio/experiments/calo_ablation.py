"""Predefined CALO v4 scientific ablation suite."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass

from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.legacy_mtlbo import LegacyMTLBOOptimizer
from .evaluation_budget import BudgetPolicy
from .experiment_runner import (
    CompletedRun,
    RunExecutionFailure,
    _optimizer_failure_state,
    build_problem,
    run_single,
)
from calo_rpd_studio.compute.device_binding import (
    resolve_config_for_entrypoint,
    result_device_attestation,
)


@dataclass(frozen=True, slots=True)
class AblationSpec:
    label: str
    algorithm: str = "CALO"
    parameters: dict | None = None


ABLATION_SPECS = (
    AblationSpec("Classical TLBO", "TLBO", {}),
    AblationSpec("Legacy Gaussian MTLBO", "LEGACY", {}),
    AblationSpec("CALO v4 without AI policy", "CALO", {"use_ai": False}),
    AblationSpec("CALO v4 without adaptive epsilon", "CALO", {"use_epsilon": False}),
    AblationSpec("CALO v4 without dual archives", "CALO", {"use_dual_archives": False}),
    AblationSpec("CALO v4 without success memory", "CALO", {"use_memory": False}),
    AblationSpec("CALO v4 without HPEM", "CALO", {"use_hpem": False}),
    AblationSpec("CALO v4 without contextual credit", "CALO", {"use_contextual_credit": False}),
    AblationSpec(
        "CALO v4 without variable-group intelligence", "CALO", {"use_variable_intelligence": False}
    ),
    AblationSpec("CALO v4 without dual-lane learning", "CALO", {"use_dual_lane": False}),
    AblationSpec("CALO v4 without cognitive precision", "CALO", {"use_cognitive_precision": False}),
    AblationSpec("CALO v4 without diversity recovery", "CALO", {"use_diversity_recovery": False}),
    AblationSpec("Complete CALO v4", "CALO", {}),
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
    config = resolve_config_for_entrypoint(config)
    started = time.perf_counter()
    policy = config.budget.policy
    if policy is BudgetPolicy.EQUAL_WALL_CLOCK:
        max_eval = 2_000_000_000
        max_iter = 2_000_000_000
    else:
        max_eval = config.budget.max_evaluations
        max_iter = max(config.max_iterations, config.budget.max_evaluations)

    def _cancel():
        if cancel_callback and cancel_callback():
            return True
        return bool(
            policy is BudgetPolicy.EQUAL_WALL_CLOCK
            and config.budget.wall_clock_seconds is not None
            and time.perf_counter() - started >= config.budget.wall_clock_seconds
        )

    try:
        problem = build_problem(config, seeds.scenario_seed, _cancel)
    except Exception as exc:
        raise RunExecutionFailure(exc, _optimizer_failure_state(None, config, exc)) from exc

    try:
        optimizer = LegacyMTLBOOptimizer(
            problem,
            OptimizerConfig(config.population_size, max_eval, max_iter, {}),
            seeds.algorithm_seed,
            progress_callback,
            _cancel,
        )
    except Exception as exc:
        raise RunExecutionFailure(exc, _optimizer_failure_state(None, config, exc)) from exc
    try:
        result = optimizer.run()
        result.metadata["optimizer_device"] = "cpu_legacy_optimizer"
        result.metadata["device_attestation"] = result_device_attestation(config, problem, result)
        result.algorithm = spec.label
    except Exception as exc:
        raise RunExecutionFailure(exc, _optimizer_failure_state(optimizer, config, exc)) from exc
    return CompletedRun(spec.label, run_index, seeds, result)
