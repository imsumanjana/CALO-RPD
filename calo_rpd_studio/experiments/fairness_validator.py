"""Fairness audit for comparative experiments."""

from dataclasses import dataclass, field

from .evaluation_budget import BudgetPolicy


@dataclass(slots=True)
class FairnessReport:
    fair: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_fairness(config):
    errors = []
    warnings = []
    try:
        config.validate()
    except Exception as exc:
        errors.append(str(exc))
        return FairnessReport(False, errors, warnings)
    if (
        config.budget.policy is BudgetPolicy.EQUAL_EVALUATIONS
        and config.budget.max_evaluations < config.population_size
    ):
        errors.append("Equal-evaluation budget must be at least the population size.")
    if (
        config.budget.policy is BudgetPolicy.EQUAL_EVALUATIONS
        and int(config.budget.max_evaluations) % int(config.population_size) != 0
    ):
        errors.append(
            "Strict equal-evaluation fairness requires max_evaluations to be divisible by "
            "population_size so every algorithm is assigned the exact same requested FE budget."
        )
    if config.budget.policy is BudgetPolicy.ALGORITHM_NATIVE:
        warnings.append("Algorithm-native limits do not provide a universal equal-cost comparison.")
    if config.budget.policy is BudgetPolicy.EQUAL_WALL_CLOCK:
        warnings.append(
            "Wall-clock comparisons depend on hardware, operating-system load, and implementation details; retain full provenance."
        )
    if len(set(config.algorithms)) != len(config.algorithms):
        errors.append("Each primary algorithm may appear only once in one comparison protocol.")
    if config.parallel_workers > 1:
        warnings.append(
            "Parallel throughput mode runs independent optimizer jobs concurrently. Objective-quality comparisons remain valid under the common evaluation protocol, but per-run wall-clock times are affected by resource contention; use one worker for strict runtime ranking."
        )
    if config.execution_backend != "cpu_only":
        if str(getattr(config, "scientific_backend", "cpu_reference")) == "torch_fp64":
            warnings.append(
                "The v3 torch FP64 backend makes every primary optimizer job accelerator-compatible through a common batched AC power-flow/constraint evaluator and torch-native baseline kernels. CPU remains responsible for orchestration, persistence, and independent validation. Use one device and one worker for strict runtime ranking."
            )
            if bool(getattr(config, "require_backend_parity", True)):
                warnings.append(
                    "A CPU/accelerator numerical-parity audit is required before final benchmark execution."
                )
        else:
            warnings.append(
                "The legacy CPU-reference evaluator is selected; accelerator task shares cannot move physical evaluation to a GPU."
            )
    return FairnessReport(not errors, errors, warnings)
