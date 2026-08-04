"""L-SHADE 1.0.1 comparator with the common constrained-ORPD wrapper.

The search equations follow Tanabe and Fukunaga's corrected reference implementation: historical
memories for ``F`` and ``CR``, current-to-pbest/1/bin, an external archive, parent-midpoint bound
repair, and linear population-size reduction to four individuals. Selection uses the repository's
common Deb feasibility-first ordering. Successful-parameter weights therefore use feasible
objective improvement or infeasible violation improvement, an explicit constrained-task adapter.
"""

from __future__ import annotations

import time

import numpy as np

from calo_rpd_studio.orpd.feasibility_rules import better, sort_key
from calo_rpd_studio.orpd.problem import Evaluation

from .base_optimizer import BaseOptimizer


def constrained_improvement(parent: Evaluation, child: Evaluation) -> float:
    """Positive improvement magnitude consistent with feasibility-first selection."""

    if not better(child, parent):
        return 0.0
    parent_feasible = sort_key(parent)[0] == 0
    child_feasible = sort_key(child)[0] == 0
    if parent_feasible and child_feasible:
        magnitude = float(parent.value) - float(child.value)
    elif not parent_feasible and not child_feasible:
        magnitude = float(parent.violation) - float(child.violation)
        if magnitude <= 0.0:
            magnitude = float(parent.value) - float(child.value)
    else:
        magnitude = max(float(parent.violation), 1.0)
    epsilon: float = np.finfo(np.float64).eps.item()
    return max(float(magnitude), epsilon)


def parent_midpoint_repair(candidate, parent):
    """Apply the JADE/L-SHADE parent-midpoint repair on the normalized unit box."""

    child = np.asarray(candidate, dtype=float).copy()
    parent = np.asarray(parent, dtype=float)
    below = child < 0.0
    above = child > 1.0
    child[below] = parent[below] / 2.0
    child[above] = (1.0 + parent[above]) / 2.0
    return child


def positive_round(value: float) -> int:
    """Match C++ ``round`` for the non-negative sizes used by L-SHADE."""

    return int(np.floor(float(value) + 0.5))


class LSHADEOptimizer(BaseOptimizer):
    """Corrected L-SHADE mechanics under the shared optimizer result contract."""

    name = "L-SHADE"

    def _sample_scaling_factor(self, mean: float) -> float:
        value = -1.0
        while value <= 0.0:
            value = float(mean) + 0.1 * float(self.rng.standard_cauchy())
        return min(value, 1.0)

    def _archive_insert(self, archive: list[np.ndarray], parent, capacity: int) -> None:
        if capacity <= 1:
            return
        value = np.asarray(parent, dtype=float).copy()
        if len(archive) < capacity:
            archive.append(value)
        else:
            archive[int(self.rng.integers(capacity))] = value

    def _trial_population(
        self,
        population: np.ndarray,
        evaluations: list,
        archive: list[np.ndarray],
        memory_f: np.ndarray,
        memory_cr: np.ndarray,
        p_best_rate: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        size, dimension = population.shape
        order = self.order(evaluations)
        p_count = min(size, max(2, positive_round(size * p_best_rate)))
        trials = np.empty_like(population)
        sampled_f = np.empty(size, dtype=float)
        sampled_cr = np.empty(size, dtype=float)
        union = population if not archive else np.vstack((population, np.asarray(archive)))

        for target in range(size):
            memory_index = int(self.rng.integers(len(memory_f)))
            mean_cr = float(memory_cr[memory_index])
            cr = 0.0 if mean_cr < 0.0 else float(np.clip(self.rng.normal(mean_cr, 0.1), 0.0, 1.0))
            scale = self._sample_scaling_factor(float(memory_f[memory_index]))
            pbest = int(order[int(self.rng.integers(p_count))])

            r1_candidates = [index for index in range(size) if index != target]
            r1 = int(self.rng.choice(r1_candidates))
            while True:
                r2 = int(self.rng.integers(len(union)))
                if r2 != target and r2 != r1:
                    break
            mutant = (
                population[target]
                + scale * (population[pbest] - population[target])
                + scale * (population[r1] - union[r2])
            )
            mask = self.rng.random(dimension) < cr
            mask[int(self.rng.integers(dimension))] = True
            raw_trial = np.where(mask, mutant, population[target])
            outside = (raw_trial < 0.0) | (raw_trial > 1.0)
            self.source_repair_candidate_count += int(bool(np.any(outside)))
            self.source_repair_coordinate_count += int(np.count_nonzero(outside))
            self.source_repair_total_coordinates += int(dimension)
            trials[target] = parent_midpoint_repair(raw_trial, population[target])
            sampled_f[target] = scale
            sampled_cr[target] = cr
        return trials, sampled_f, sampled_cr

    @staticmethod
    def _update_memory(
        memory_f: np.ndarray,
        memory_cr: np.ndarray,
        memory_position: int,
        successful_f: list[float],
        successful_cr: list[float],
        improvements: list[float],
    ) -> int:
        if not successful_f:
            return memory_position
        weights = np.asarray(improvements, dtype=float)
        weights /= float(weights.sum())
        values_f = np.asarray(successful_f, dtype=float)
        values_cr = np.asarray(successful_cr, dtype=float)
        denominator_f = float(np.sum(weights * values_f))
        memory_f[memory_position] = float(np.sum(weights * values_f**2) / denominator_f)
        denominator_cr = float(np.sum(weights * values_cr))
        memory_cr[memory_position] = (
            -1.0
            if denominator_cr <= 0.0
            else float(np.sum(weights * values_cr**2) / denominator_cr)
        )
        return (memory_position + 1) % len(memory_f)

    def run(self):
        started = time.perf_counter()
        requested_size = max(4, int(self.config.population_size))
        population = self.random_population(requested_size)
        evaluations = self.evaluate_population(population)
        population = population[: len(evaluations)]
        if not evaluations:
            raise RuntimeError("L-SHADE could not evaluate its initial population")

        initial_size = len(population)
        minimum_size = 4
        self.source_repair_candidate_count = 0
        self.source_repair_coordinate_count = 0
        self.source_repair_total_coordinates = 0
        memory_size = max(1, int(self.config.parameters.get("memory_size", 5)))
        p_best_rate = float(self.config.parameters.get("p_best_rate", 0.11))
        archive_rate = max(0.0, float(self.config.parameters.get("archive_rate", 1.4)))
        if not 0.0 < p_best_rate <= 1.0:
            raise ValueError("L-SHADE p_best_rate must lie in (0, 1]")
        memory_f: np.ndarray = np.full(memory_size, 0.5, dtype=float)
        memory_cr: np.ndarray = np.full(memory_size, 0.5, dtype=float)
        memory_position = 0
        archive: list[np.ndarray] = []
        population_history = [initial_size]

        while (
            self.iteration < self.config.max_iterations
            and self.can_evaluate()
            and len(population) >= 4
        ):
            self.iteration += 1
            trials, sampled_f, sampled_cr = self._trial_population(
                population,
                evaluations,
                archive,
                memory_f,
                memory_cr,
                p_best_rate,
            )
            trial_evaluations = self.evaluate_population(trials)
            successful_f: list[float] = []
            successful_cr: list[float] = []
            improvements: list[float] = []
            archive_capacity = positive_round(len(population) * archive_rate)

            for index, child_evaluation in enumerate(trial_evaluations):
                parent_evaluation = evaluations[index]
                improvement = constrained_improvement(parent_evaluation, child_evaluation)
                if improvement > 0.0:
                    self._archive_insert(archive, population[index], archive_capacity)
                    successful_f.append(float(sampled_f[index]))
                    successful_cr.append(float(sampled_cr[index]))
                    improvements.append(improvement)
                if improvement > 0.0 or sort_key(child_evaluation) == sort_key(parent_evaluation):
                    population[index] = trials[index]
                    evaluations[index] = child_evaluation

            memory_position = self._update_memory(
                memory_f,
                memory_cr,
                memory_position,
                successful_f,
                successful_cr,
                improvements,
            )
            planned_size = positive_round(
                (minimum_size - initial_size)
                * self.evaluations
                / max(int(self.config.max_evaluations), 1)
                + initial_size
            )
            planned_size = max(minimum_size, planned_size)
            if len(population) > planned_size:
                retained = self.order(evaluations)[:planned_size]
                population = population[retained]
                evaluations = [evaluations[index] for index in retained]
            next_archive_capacity = positive_round(len(population) * archive_rate)
            while len(archive) > next_archive_capacity:
                archive.pop(int(self.rng.integers(len(archive))))
            population_history.append(len(population))
            self.record(
                {
                    "population_size": len(population),
                    "successful_parameters": len(successful_f),
                }
            )

        metadata = {
            "source_algorithm": "L-SHADE 1.0.1 corrected reference",
            "source_doi": "10.1109/CEC.2014.6900380",
            "mutation": "current-to-pbest/1/bin",
            "boundary_strategy": "parent_midpoint_unit_box",
            "boundary_repair_policy": "parent_midpoint_to_unit_box_relative_to_target",
            "source_boundary_repair_candidate_count": self.source_repair_candidate_count,
            "source_boundary_repair_coordinate_count": self.source_repair_coordinate_count,
            "source_boundary_repair_coordinate_rate": float(self.source_repair_coordinate_count)
            / max(self.source_repair_total_coordinates, 1),
            "constraint_adapter": "Deb feasibility-first selection; feasible objective or infeasible violation improvement weights",
            "memory_size": memory_size,
            "p_best_rate": p_best_rate,
            "archive_rate": archive_rate,
            "minimum_population_size": minimum_size,
            "population_size_history": population_history,
            "memory_f_final": memory_f.tolist(),
            "memory_cr_final": memory_cr.tolist(),
        }
        return self.finalize(population, metadata=metadata, started=started)
