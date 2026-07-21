"""Adaptive epsilon-feasible environmental selection for CALO Core v2."""

from __future__ import annotations

import numpy as np


def epsilon_sort_key(evaluation, epsilon: float):
    eps_feasible = np.isfinite(evaluation.violation) and evaluation.violation <= epsilon + 1e-12
    if eps_feasible:
        return (0, float(evaluation.value), float(evaluation.violation))
    return (1, float(evaluation.violation), float(evaluation.value))


def epsilon_better(a, b, epsilon: float, tol: float = 1e-12) -> bool:
    if b is None:
        return True
    a_eps = np.isfinite(a.violation) and a.violation <= epsilon + tol
    b_eps = np.isfinite(b.violation) and b.violation <= epsilon + tol
    if a_eps and not b_eps:
        return True
    if b_eps and not a_eps:
        return False
    if a_eps and b_eps:
        if a.value < b.value - tol:
            return True
        if abs(a.value - b.value) <= tol:
            return a.violation < b.violation - tol
        return False
    if a.violation < b.violation - tol:
        return True
    if abs(a.violation - b.violation) <= tol:
        return a.value < b.value - tol
    return False


def environmental_select(
    vectors,
    evaluations,
    population_size: int,
    epsilon: float,
    diversity_weight: float = 0.18,
    return_indices: bool = False,
):
    vectors = np.asarray(vectors, dtype=float)
    order = sorted(range(len(evaluations)), key=lambda i: epsilon_sort_key(evaluations[i], epsilon))
    population_size = min(int(population_size), len(order))
    if population_size <= 0:
        empty = np.empty((0, vectors.shape[1]))
        return (empty, [], np.empty(0, dtype=int)) if return_indices else (empty, [])

    quality_count = max(1, population_size // 2)
    selected = order[:quality_count]
    remaining = order[quality_count:]
    rank_position = {index: rank for rank, index in enumerate(order)}

    while remaining and len(selected) < population_size:
        best_candidate = remaining[0]
        best_score = float("inf")
        for index in remaining:
            rank_score = rank_position[index] / max(len(order) - 1, 1)
            min_distance = min(np.linalg.norm(vectors[index] - vectors[j]) for j in selected)
            # Low is better. Diversity reduces the score while quality rank still dominates.
            score = rank_score - float(diversity_weight) * min_distance
            if score < best_score:
                best_score = score
                best_candidate = index
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    selected = selected[:population_size]
    result = (vectors[selected].copy(), [evaluations[i] for i in selected])
    if return_indices:
        return result[0], result[1], np.asarray(selected, dtype=int)
    return result
