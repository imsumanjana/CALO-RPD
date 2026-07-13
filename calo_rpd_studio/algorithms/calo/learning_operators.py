"""CALO Core v2 learning operators shared by training and runtime."""
from __future__ import annotations

import numpy as np

OPERATOR_NAMES = (
    "feasible_elite_learning",
    "constraint_boundary_differential",
    "cognitive_teacher_learning",
    "success_distribution_memory",
    "mixed_variable_neighbourhood",
    "diversity_recovery",
)


def feasible_elite_learning(x, pbest, r1, r2, rng, attraction: float, differential: float):
    return np.clip(
        x + attraction * rng.random(x.shape) * (pbest - x) + differential * (r1 - r2),
        0,
        1,
    )


def constraint_boundary_differential(x, boundary, r1, r2, rng, attraction: float,
                                     differential: float):
    return np.clip(
        x + attraction * rng.random(x.shape) * (boundary - x) + differential * (r1 - r2),
        0,
        1,
    )


def cognitive_teacher_learning(x, teacher, mean, rng, alpha: float, beta: float):
    z1 = np.abs(rng.normal(size=x.shape))
    z2 = rng.normal(size=x.shape)
    return np.clip(x + alpha * z1 * (teacher - x) + beta * z2 * (teacher - mean), 0, 1)


def success_distribution_memory(x, personal, sampled_direction, rng, personal_weight: float,
                                memory_weight: float):
    return np.clip(
        x
        + personal_weight * rng.random(x.shape) * (personal - x)
        + memory_weight * sampled_direction,
        0,
        1,
    )


def mixed_variable_neighbourhood(x, variables, rng, continuous_sigma: float = 0.03,
                                 discrete_radius: int = 1):
    candidate = np.asarray(x, float).copy()
    if not variables:
        return np.clip(candidate + continuous_sigma * rng.normal(size=candidate.shape), 0, 1)
    chosen = int(rng.integers(len(candidate)))
    variable = variables[chosen]
    kind = str(getattr(variable.kind, "value", variable.kind))
    values = tuple(getattr(variable, "values", ()) or ())
    if kind == "discrete" and len(values) > 1:
        current_index = int(np.rint(candidate[chosen] * (len(values) - 1)))
        step = int(rng.choice([-1, 1])) * int(rng.integers(1, max(2, discrete_radius + 1)))
        new_index = int(np.clip(current_index + step, 0, len(values) - 1))
        candidate[chosen] = new_index / (len(values) - 1)
    else:
        candidate[chosen] = np.clip(
            candidate[chosen] + continuous_sigma * rng.normal(), 0.0, 1.0
        )
    # Occasionally perturb a second variable to escape one-coordinate plateaus.
    if len(candidate) > 1 and rng.random() < 0.25:
        second = int(rng.integers(len(candidate)))
        if second != chosen:
            candidate[second] = np.clip(
                candidate[second] + 0.5 * continuous_sigma * rng.normal(), 0.0, 1.0
            )
    return candidate


def diversity_recovery(reference, population, rng, sigma: float = 0.12):
    population = np.asarray(population, float)
    if len(population) == 0:
        return np.clip(reference + sigma * rng.normal(size=reference.shape), 0, 1)
    # Opposition-guided seed plus controlled perturbation. This explores an underused region
    # instead of repeatedly perturbing the same elite point.
    centroid = population.mean(axis=0)
    opposition = 1.0 - centroid
    anchor = opposition if rng.random() < 0.55 else rng.random(reference.shape)
    return np.clip(0.55 * anchor + 0.45 * reference + sigma * rng.normal(size=reference.shape), 0, 1)


# Compatibility aliases retained for historical ablation imports.
def teacher_guided(x, best, mean, rng, alpha, beta):
    return cognitive_teacher_learning(x, best, mean, rng, alpha, beta)


def contrastive_peer(x, better_peer, diverse_peer, rng, gamma, delta):
    return feasible_elite_learning(x, better_peer, x, diverse_peer, rng, gamma, delta)


def self_reflective_memory(x, personal, memory_direction, rng, eta, mu):
    return success_distribution_memory(x, personal, memory_direction, rng, eta, mu)


def adaptive_exploration(reference, rng, sigma):
    return np.clip(reference + sigma * rng.normal(size=reference.shape), 0, 1)


def feasibility_recovery(x, feasible_elite, low_violation, rng, intensity):
    return np.clip(
        x
        + intensity * rng.random(x.shape) * (feasible_elite - x)
        + 0.35 * intensity * rng.normal(size=x.shape) * (low_violation - x),
        0,
        1,
    )


def stagnation_escape(elite, rng, sigma):
    return np.clip(elite + max(sigma, 0.02) * rng.normal(size=elite.shape), 0, 1)
