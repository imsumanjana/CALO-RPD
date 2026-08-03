"""Versioned TSH-CALO candidate generation over the canonical transition authority."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ai_controller import PARAMETER_HIGH, PARAMETER_LOW, PARAMETER_NAMES
from .operator_credit import blend_probabilities
from .transition_kernel import (
    DISCOVERY_MEMORY_PRIOR,
    REGIME_MEMORY_PRIORS,
    CandidateBatch,
    _focus_to_group,
    _propose_candidate,
    complete_transition,
    individual_regime,
)
from .tsh_calo_physics_repair import (
    PhysicsRepairContext,
    PhysicsRepairOperator,
    PhysicsRepairProposal,
    PhysicsRepairStatus,
)
from .tsh_calo_schema import N_BOUNDED_PARAMETERS, N_CONTROL_GROUPS, N_OPERATORS


@dataclass(slots=True)
class TSHCandidateBatch:
    candidates: CandidateBatch
    group_parameter_values: np.ndarray
    physics_repair_proposals: tuple[PhysicsRepairProposal | None, ...]


def _group_parameters(raw_parameters) -> np.ndarray:
    raw = np.asarray(raw_parameters, dtype=float)
    if raw.shape != (N_CONTROL_GROUPS, N_BOUNDED_PARAMETERS):
        raise ValueError("TSH-CALO group parameter actions do not match the action ABI")
    if not np.all(np.isfinite(raw)) or np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError("TSH-CALO group parameter actions must be finite and within [0, 1]")
    return PARAMETER_LOW[None, :] + raw * (PARAMETER_HIGH - PARAMETER_LOW)[None, :]


def generate_tsh_offspring(
    *,
    population: np.ndarray,
    evaluations: list[object],
    personal_best: np.ndarray,
    rng: np.random.Generator,
    dimension: int,
    variables,
    quality_order,
    contexts: np.ndarray,
    learner_groups: np.ndarray,
    learned_lanes: np.ndarray,
    global_regime: int,
    learner_operators: np.ndarray,
    group_parameter_actions: np.ndarray,
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
    physics_repair_operator: PhysicsRepairOperator | None = None,
    physics_contexts: tuple[PhysicsRepairContext | None, ...] | None = None,
    out: np.ndarray | None = None,
) -> TSHCandidateBatch:
    """Execute independently shielded learner actions without redefining frozen CALO."""

    del evaluations
    population = np.asarray(population, dtype=float)
    population_size = len(population)
    contexts = np.asarray(contexts, dtype=int)
    groups = np.asarray(learner_groups, dtype=int)
    operators = np.asarray(learner_operators, dtype=int)
    lanes = np.asarray(learned_lanes, dtype=np.int8).copy()
    if any(array.shape != (population_size,) for array in (contexts, groups, operators, lanes)):
        raise ValueError("TSH-CALO learner action vectors must align with the population")
    if np.any((groups < 0) | (groups >= N_CONTROL_GROUPS)):
        raise ValueError("TSH-CALO learner group is outside the action ABI")
    if np.any((operators < 0) | (operators >= N_OPERATORS)):
        raise ValueError("TSH-CALO learner operator is outside the action ABI")
    parameter_values = _group_parameters(group_parameter_actions)
    offspring = np.empty_like(population) if out is None else out
    if offspring.shape != population.shape:
        raise ValueError("TSH-CALO offspring buffer shape is incompatible")
    assigned_operators = np.full(population_size, -1, dtype=np.int8)
    assigned_memory = np.zeros(population_size, dtype=np.int8)
    assigned_groups = groups.astype(np.int8, copy=True)
    individual_regimes = np.zeros(population_size, dtype=np.int8)
    precision_mask = np.zeros(population_size, dtype=bool)
    repair_traces: list[PhysicsRepairProposal | None] = [None] * population_size
    hierarchy = hpem.hierarchy() if len(hpem) else np.zeros((4, dimension))
    best = population[int(quality_order[0])]
    mean = population.mean(axis=0)
    physics_contexts = physics_contexts or tuple(None for _ in range(population_size))
    if len(physics_contexts) != population_size:
        raise ValueError("TSH-CALO physics contexts must align with the population")

    for index in range(population_size):
        context = int(contexts[index])
        regime = individual_regime(global_regime, context)
        individual_regimes[index] = regime
        memory_prior = (
            REGIME_MEMORY_PRIORS[regime].copy()
            if bool(lanes[index])
            else DISCOVERY_MEMORY_PRIOR.copy()
        )
        memory_online = credit.memory_probabilities(regime, context)
        memory_probabilities = blend_probabilities(memory_prior, memory_online, alpha=0.65)
        memory_level = (
            int(np.argmax(memory_probabilities))
            if environment_deterministic
            else int(rng.choice(4, p=memory_probabilities))
        )
        assigned_memory[index] = memory_level
        group = int(groups[index])
        should_precision = (
            precision_active
            and bool(lanes[index])
            and index not in forced_recovery
            and (
                environment_deterministic
                and index < int(round(population_size * precision_fraction))
                or (not environment_deterministic and rng.random() < precision_fraction)
            )
        )
        if should_precision and len(hpem):
            success_direction = memory.mean_direction(
                dimension, regime=regime, context=context, group=group
            )
            offspring[index] = precision.propose(
                hpem.best_vector,
                hierarchy,
                success_direction,
                variables,
                group_intelligence.mask(group, dimension),
                rng,
                consensus,
            )
            precision_mask[index] = True
            continue

        selected_operator = 5 if index in forced_recovery else int(operators[index])
        if index in forced_recovery:
            lanes[index] = 0
            assigned_memory[index] = 3
        assigned_operators[index] = selected_operator
        if selected_operator == 6:
            if physics_repair_operator is None:
                raise RuntimeError("Shield selected physics repair without an enabled operator")
            proposal = physics_repair_operator.propose(
                population[index], physics_contexts[index], variables
            )
            repair_traces[index] = proposal
            if proposal.status is not PhysicsRepairStatus.PROPOSED or proposal.candidate is None:
                raise RuntimeError(
                    "Shield-selected physics repair became unavailable: " + proposal.reason
                )
            offspring[index] = _focus_to_group(
                population[index], proposal.candidate, group_intelligence.mask(group, dimension), 6
            )
            continue
        values = parameter_values[group]
        adaptive = {name: float(value) for name, value in zip(PARAMETER_NAMES, values)}
        offspring[index] = _propose_candidate(
            operator=selected_operator,
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

    batch = CandidateBatch(
        offspring,
        assigned_operators,
        assigned_memory,
        assigned_groups,
        individual_regimes,
        precision_mask,
        lanes,
    )
    return TSHCandidateBatch(batch, parameter_values, tuple(repair_traces))


def complete_tsh_transition(**kwargs):
    """Use channel 7 for precision because TSH operator 6 is physics repair."""

    return complete_transition(**kwargs, precision_memory_operator=7)
