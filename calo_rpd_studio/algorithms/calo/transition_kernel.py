"""Canonical behavior-preserving CALO v5.9 transition authority.

This module owns the stochastic candidate assignment/operator execution and the
post-evaluation learning, archive, selection, reward, and diagnostic update used by both deployed
CALO and the independent PPO rollout environment.  It intentionally implements the frozen v5.9
semantics; TSH-CALO extensions use a new versioned kernel rather than changing these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from calo_rpd_studio.orpd.feasibility_rules import better

from .cognitive_state import population_diversity
from .diagnostics import PopulationDiagnostics, population_diagnostics
from .environmental_selection import environmental_select, epsilon_better
from .learning_operators import (
    cognitive_teacher_learning,
    constraint_boundary_differential,
    diversity_recovery,
    feasible_elite_learning,
    mixed_variable_neighbourhood,
    success_distribution_memory,
)
from .operator_credit import blend_probabilities
from .reward import RewardComponents, calculate_reward


REGIME_OPERATOR_PRIORS = np.asarray(
    [
        [0.05, 0.33, 0.12, 0.08, 0.30, 0.12],
        [0.18, 0.24, 0.18, 0.14, 0.18, 0.08],
        [0.34, 0.08, 0.22, 0.20, 0.12, 0.04],
        [0.08, 0.15, 0.10, 0.10, 0.12, 0.45],
    ],
    dtype=float,
)
REGIME_MEMORY_PRIORS = np.asarray(
    [
        [0.05, 0.15, 0.30, 0.50],
        [0.10, 0.25, 0.40, 0.25],
        [0.40, 0.35, 0.20, 0.05],
        [0.05, 0.10, 0.20, 0.65],
    ],
    dtype=float,
)
DISCOVERY_OPERATOR_PRIOR = np.asarray([0.05, 0.28, 0.08, 0.05, 0.22, 0.32], dtype=float)
DISCOVERY_MEMORY_PRIOR = np.asarray([0.03, 0.07, 0.25, 0.65], dtype=float)


def normalise_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values) & (values >= 0.0), values, 0.0)
    total = float(values.sum())
    return values / total if total > 0.0 else np.full(values.shape, 1.0 / len(values))


def individual_regime(global_regime: int, context: int) -> int:
    if int(context) == 3:
        return 3
    if int(context) == 2 and int(global_regime) >= 2:
        return 1
    if int(context) <= 1 and int(global_regime) == 0:
        return 1
    return int(global_regime)


@dataclass(slots=True)
class CandidateBatch:
    offspring: np.ndarray
    assigned_operators: np.ndarray
    assigned_memory: np.ndarray
    assigned_groups: np.ndarray
    individual_regimes: np.ndarray
    precision_mask: np.ndarray
    learned_lanes: np.ndarray


@dataclass(slots=True)
class EvaluationBatch:
    evaluations: list[object]
    requested: int
    completed: int

    @property
    def complete(self) -> bool:
        return self.requested == self.completed


@dataclass(slots=True)
class TransitionResult:
    combined_population: np.ndarray
    combined_evaluations: list[object]
    selected_population: np.ndarray
    selected_evaluations: list[object]
    selected_indices: np.ndarray
    offspring_personal_best: np.ndarray
    offspring_personal_best_evaluations: list[object]
    successful: np.ndarray
    objective_gain: np.ndarray
    feasibility_gain: np.ndarray
    feasibility_transition: np.ndarray
    precision_attempts: int
    precision_successes: int
    new_diagnostics: PopulationDiagnostics
    new_diversity: float
    reward: RewardComponents


def evaluate_candidates(
    offspring: np.ndarray,
    evaluator: Callable[[np.ndarray], Sequence[object]],
) -> EvaluationBatch:
    """Evaluate one declared candidate batch without hiding partial completion."""

    requested = int(len(offspring))
    evaluations = list(evaluator(offspring))
    completed = int(len(evaluations))
    if completed > requested:
        raise RuntimeError(
            f"CALO evaluator returned {completed} results for {requested} requested candidates"
        )
    return EvaluationBatch(evaluations, requested, completed)


def _select_distinct(
    population: np.ndarray, index: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    candidates = [i for i in range(len(population)) if i != index]
    if len(candidates) < 2:
        return population[index].copy(), population[index].copy()
    chosen = rng.choice(candidates, size=2, replace=False)
    return population[int(chosen[0])].copy(), population[int(chosen[1])].copy()


def _focus_to_group(
    x: np.ndarray, candidate: np.ndarray, mask: np.ndarray, operator: int
) -> np.ndarray:
    if int(operator) == 5 or not np.any(mask):
        return np.clip(candidate, 0.0, 1.0)
    focused = np.asarray(x, float).copy()
    focused[mask] = np.asarray(candidate, float)[mask]
    return np.clip(focused, 0.0, 1.0)


def _propose_candidate(
    *,
    operator: int,
    index: int,
    population: np.ndarray,
    personal_best: np.ndarray,
    rng: np.random.Generator,
    dimension: int,
    variables,
    best: np.ndarray,
    mean: np.ndarray,
    memory,
    hpem,
    feasible_archive,
    boundary_archive,
    parameters: dict[str, float],
    regime: int,
    context: int,
    memory_level: int,
    group: int,
    group_intelligence,
    learned_lane: bool,
) -> np.ndarray:
    x = population[index]
    r1, r2 = _select_distinct(population, index, rng)
    feasible_teacher = feasible_archive.sample(rng, best)
    boundary_teacher = boundary_archive.sample(rng, best)
    memory_teacher = hpem.summary(memory_level, feasible_teacher) if len(hpem) else feasible_teacher
    group_mask = group_intelligence.mask(group, dimension)

    if operator == 0:
        teacher = (
            memory_teacher
            if learned_lane and len(hpem)
            else (feasible_teacher if len(feasible_archive) else boundary_teacher)
        )
        candidate = feasible_elite_learning(
            x,
            teacher,
            r1,
            r2,
            rng,
            parameters["attraction"],
            parameters["differential"],
        )
    elif operator == 1:
        candidate = constraint_boundary_differential(
            x,
            boundary_teacher,
            r1,
            r2,
            rng,
            parameters["attraction"],
            parameters["differential"],
        )
    elif operator == 2:
        if learned_lane and len(hpem):
            teacher = memory_teacher
        else:
            teacher = (
                feasible_teacher if regime >= 2 and len(feasible_archive) else boundary_teacher
            )
        candidate = cognitive_teacher_learning(
            x,
            teacher,
            mean,
            rng,
            parameters["attraction"],
            0.35 * parameters["exploration_sigma"],
        )
    elif operator == 3:
        direction = memory.sample_direction(
            dimension,
            rng,
            prefer_feasibility=regime <= 1,
            regime=regime,
            context=context,
            group=group,
        )
        candidate = success_distribution_memory(
            x,
            personal_best[index],
            direction,
            rng,
            0.55,
            parameters["memory_weight"],
        )
        if learned_lane and len(hpem):
            candidate = np.clip(
                candidate + 0.12 * parameters["attraction"] * (memory_teacher - candidate),
                0.0,
                1.0,
            )
    elif operator == 4:
        candidate = mixed_variable_neighbourhood(
            x,
            variables,
            rng,
            continuous_sigma=max(parameters["exploration_sigma"] * 0.35, 0.004),
            discrete_radius=2 if regime == 3 else 1,
        )
    else:
        reference = (
            boundary_teacher
            if regime <= 1
            else (hpem.summary(3, feasible_teacher) if len(hpem) else feasible_teacher)
        )
        candidate = diversity_recovery(
            reference,
            population,
            rng,
            sigma=max(parameters["exploration_sigma"], 0.05),
        )
    return _focus_to_group(x, candidate, group_mask, operator)


def generate_offspring(
    *,
    population: np.ndarray,
    evaluations: list[object],
    personal_best: np.ndarray,
    rng: np.random.Generator,
    dimension: int,
    variables,
    quality_order: Sequence[int],
    contexts: np.ndarray,
    learned_lanes: np.ndarray,
    global_regime: int,
    raw_operator: int,
    native_policy: bool,
    ai_operator_probabilities: np.ndarray,
    adaptive: dict[str, float],
    memory,
    hpem,
    feasible_archive,
    boundary_archive,
    credit,
    group_intelligence,
    precision,
    precision_active: bool,
    precision_fraction: float,
    forced_recovery: set[int],
    consensus: float,
    environment_deterministic: bool,
    use_mixed_variable: bool = True,
    use_diversity_recovery: bool = True,
    use_contextual_credit: bool = True,
    use_variable_intelligence: bool = True,
    ai_policy_weight: float = 0.35,
    ai_credit_blend: float = 0.65,
    out: np.ndarray | None = None,
) -> CandidateBatch:
    """Assign and execute one frozen CALO candidate batch."""

    del evaluations  # quality ordering is explicit so training/runtime cannot silently diverge.
    population = np.asarray(population, dtype=float)
    population_size = int(len(population))
    offspring = np.empty_like(population) if out is None else out
    assigned_operators = np.full(population_size, -1, dtype=np.int8)
    assigned_memory = np.zeros(population_size, dtype=np.int8)
    assigned_groups = np.zeros(population_size, dtype=np.int8)
    individual_regimes = np.zeros(population_size, dtype=np.int8)
    precision_mask = np.zeros(population_size, dtype=bool)
    lanes = np.asarray(learned_lanes, dtype=np.int8).copy()
    hierarchy = hpem.hierarchy() if len(hpem) else np.zeros((4, dimension))
    best = population[int(quality_order[0])]
    mean = population.mean(axis=0)

    for index in range(population_size):
        context = int(contexts[index])
        regime = individual_regime(global_regime, context)
        individual_regimes[index] = regime
        learned_lane = bool(lanes[index])

        memory_prior = REGIME_MEMORY_PRIORS[regime].copy()
        if not learned_lane:
            memory_prior = DISCOVERY_MEMORY_PRIOR.copy()
        memory_online = credit.memory_probabilities(regime, context)
        memory_probabilities = blend_probabilities(memory_prior, memory_online, alpha=0.65)
        memory_level = (
            int(np.argmax(memory_probabilities))
            if environment_deterministic
            else int(rng.choice(4, p=memory_probabilities))
        )
        assigned_memory[index] = memory_level

        group = (
            group_intelligence.choose(regime, rng, environment_deterministic)
            if use_variable_intelligence
            else -1
        )
        assigned_groups[index] = group

        should_precision = (
            precision_active
            and learned_lane
            and index not in forced_recovery
            and (
                environment_deterministic
                and index < int(round(population_size * precision_fraction))
                or (not environment_deterministic and rng.random() < precision_fraction)
            )
        )
        if should_precision and len(hpem):
            success_direction = memory.mean_direction(
                dimension,
                regime=regime,
                context=context,
                group=group,
            )
            group_mask = group_intelligence.mask(group, dimension)
            offspring[index] = precision.propose(
                hpem.best_vector,
                hierarchy,
                success_direction,
                variables,
                group_mask,
                rng,
                consensus,
            )
            precision_mask[index] = True
            continue

        if native_policy:
            # The raw neural operator is authoritative for ordinary learners. Contextual credit,
            # rule priors, and discovery priors remain diagnostics/learning memory.
            # they do not silently redefine the PPO action. Precision and forced recovery are
            # recorded interventions.
            selected_operator = int(raw_operator)
            if (selected_operator == 4 and not use_mixed_variable) or (
                selected_operator == 5 and not use_diversity_recovery
            ):
                allowed = np.ones(6, dtype=float)
                if not use_mixed_variable:
                    allowed[4] = 0.0
                if not use_diversity_recovery:
                    allowed[5] = 0.0
                selected_operator = int(
                    np.argmax(normalise_probabilities(ai_operator_probabilities * allowed))
                )
        else:
            base_prior = normalise_probabilities(REGIME_OPERATOR_PRIORS[regime])
            learned_policy = normalise_probabilities(
                ai_policy_weight * ai_operator_probabilities + (1.0 - ai_policy_weight) * base_prior
            )
            online = (
                credit.operator_probabilities(regime, context)
                if use_contextual_credit
                else np.full(6, 1.0 / 6.0)
            )
            operator_probabilities = blend_probabilities(
                learned_policy, online, alpha=ai_credit_blend
            )
            if not learned_lane:
                operator_probabilities = normalise_probabilities(
                    0.45 * operator_probabilities + 0.55 * DISCOVERY_OPERATOR_PRIOR
                )
            if not use_mixed_variable:
                operator_probabilities[4] = 0.0
            if not use_diversity_recovery:
                operator_probabilities[5] = 0.0
            operator_probabilities = normalise_probabilities(operator_probabilities)
            selected_operator = (
                int(np.argmax(operator_probabilities))
                if environment_deterministic
                else int(rng.choice(6, p=operator_probabilities))
            )

        if index in forced_recovery:
            executed_operator = 5
            lanes[index] = 0
            assigned_memory[index] = 3
        else:
            executed_operator = int(selected_operator)
        assigned_operators[index] = executed_operator
        offspring[index] = _propose_candidate(
            operator=executed_operator,
            index=index,
            population=population,
            personal_best=personal_best,
            rng=rng,
            dimension=dimension,
            variables=variables,
            best=best,
            mean=mean,
            memory=memory,
            hpem=hpem,
            feasible_archive=feasible_archive,
            boundary_archive=boundary_archive,
            parameters=adaptive,
            regime=regime,
            context=context,
            memory_level=int(assigned_memory[index]),
            group=group,
            group_intelligence=group_intelligence,
            learned_lane=bool(lanes[index]),
        )

    return CandidateBatch(
        offspring,
        assigned_operators,
        assigned_memory,
        assigned_groups,
        individual_regimes,
        precision_mask,
        lanes,
    )


def complete_transition(
    *,
    population: np.ndarray,
    evaluations: list[object],
    personal_best: np.ndarray,
    personal_best_evaluations: list[object],
    offspring: np.ndarray,
    offspring_evaluations: list[object],
    epsilon: float,
    assigned_operators: np.ndarray,
    assigned_memory: np.ndarray,
    assigned_groups: np.ndarray,
    individual_regimes: np.ndarray,
    contexts: np.ndarray,
    precision_mask: np.ndarray,
    memory,
    credit,
    group_intelligence,
    precision,
    feasible_archive,
    boundary_archive,
    hpem,
    old_diagnostics: PopulationDiagnostics,
    old_diversity: float,
    diversity_weight: float,
    population_size: int,
    use_memory: bool = True,
    use_contextual_credit: bool = True,
    use_variable_intelligence: bool = True,
    use_dual_archives: bool = True,
    use_hpem: bool = True,
) -> TransitionResult:
    """Apply the frozen learning, selection, archive, diagnostic, and reward transition."""

    successful = np.zeros(population_size, dtype=bool)
    objective_gain = np.zeros(population_size, dtype=float)
    feasibility_gain = np.zeros(population_size, dtype=float)
    feasibility_transition = np.zeros(population_size, dtype=float)
    step_norm = np.linalg.norm(offspring - population, axis=1)
    offspring_pb = personal_best.copy()
    offspring_pb_ev = list(personal_best_evaluations)
    precision_attempts = int(np.count_nonzero(precision_mask))
    precision_successes = 0

    for index, (child, child_ev) in enumerate(zip(offspring, offspring_evaluations)):
        parent_ev = evaluations[index]
        successful[index] = epsilon_better(child_ev, parent_ev, epsilon)
        if parent_ev.feasible and child_ev.feasible and np.isfinite(parent_ev.value):
            objective_gain[index] = max(
                (float(parent_ev.value) - float(child_ev.value))
                / max(abs(float(parent_ev.value)), 1.0),
                0.0,
            )
        parent_violation = float(parent_ev.violation)
        child_violation = float(child_ev.violation)
        if np.isposinf(parent_violation) and np.isfinite(child_violation):
            feasibility_gain[index] = np.inf
        elif np.isfinite(parent_violation) and np.isfinite(child_violation):
            feasibility_gain[index] = max(parent_violation - child_violation, 0.0)
        feasibility_transition[index] = float(not parent_ev.feasible and child_ev.feasible)

        if better(child_ev, offspring_pb_ev[index]):
            offspring_pb[index] = child.copy()
            offspring_pb_ev[index] = child_ev
        if successful[index] and use_memory:
            memory_operator = 6 if precision_mask[index] else int(assigned_operators[index])
            memory.add(
                child - population[index],
                memory_operator,
                objective_gain[index],
                feasibility_gain[index],
                regime=int(individual_regimes[index]),
                context=int(contexts[index]),
                group=int(assigned_groups[index]),
            )
        if precision_mask[index] and successful[index]:
            precision_successes += 1

    if use_contextual_credit:
        credit.batch_update(
            individual_regimes,
            contexts,
            assigned_operators,
            assigned_memory,
            successful,
            objective_gain,
            feasibility_gain,
            feasibility_transition,
        )
    if use_variable_intelligence:
        group_intelligence.batch_update(
            individual_regimes,
            assigned_groups,
            successful,
            objective_gain,
            feasibility_gain,
            step_norm,
        )
    precision.update(precision_attempts, precision_successes)

    combined_population = np.vstack([population, offspring])
    combined_evaluations = list(evaluations) + list(offspring_evaluations)
    selected_population, selected_evaluations, selected_indices = environmental_select(
        combined_population,
        combined_evaluations,
        population_size,
        epsilon,
        diversity_weight=float(diversity_weight),
        return_indices=True,
    )
    if use_dual_archives:
        feasible_archive.update(combined_population, combined_evaluations)
        boundary_archive.update(combined_population, combined_evaluations)
    else:
        feasible_archive.entries = []
        boundary_archive.entries = []
        feasible_archive.update(selected_population, selected_evaluations)
        boundary_archive.update(selected_population, selected_evaluations)
    if use_hpem:
        hpem.update(combined_population, combined_evaluations)

    new_diagnostics = population_diagnostics(selected_evaluations, epsilon)
    new_diversity = population_diversity(selected_population)
    reward = calculate_reward(
        old_diagnostics.best_feasible_objective,
        new_diagnostics.best_feasible_objective,
        old_diagnostics.best_violation,
        new_diagnostics.best_violation,
        old_diagnostics.feasible_ratio,
        new_diagnostics.feasible_ratio,
        old_diversity,
        new_diversity,
    )
    return TransitionResult(
        combined_population=np.asarray(combined_population),
        combined_evaluations=combined_evaluations,
        selected_population=np.asarray(selected_population),
        selected_evaluations=list(selected_evaluations),
        selected_indices=np.asarray(selected_indices, dtype=int),
        offspring_personal_best=offspring_pb,
        offspring_personal_best_evaluations=offspring_pb_ev,
        successful=successful,
        objective_gain=objective_gain,
        feasibility_gain=feasibility_gain,
        feasibility_transition=feasibility_transition,
        precision_attempts=precision_attempts,
        precision_successes=precision_successes,
        new_diagnostics=new_diagnostics,
        new_diversity=new_diversity,
        reward=reward,
    )
