"""Canonical low-dimensional runtime state for CALO v4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CALOTensorState:
    population: np.ndarray
    evaluations: list[object]
    personal_best: np.ndarray
    personal_best_evaluations: list[object]
    lineage_ids: np.ndarray
    age: np.ndarray
    stagnation: np.ndarray

    @classmethod
    def initialize(cls, population: np.ndarray, evaluations: list[object]) -> "CALOTensorState":
        population = np.asarray(population, dtype=np.float64)
        n = len(population)
        return cls(
            population.copy(),
            list(evaluations),
            population.copy(),
            list(evaluations),
            np.arange(n, dtype=np.int64),
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.int32),
        )

    def select_from_combined(
        self,
        combined_population: np.ndarray,
        combined_evaluations: list[object],
        selected_indices: np.ndarray,
        offspring_personal_best: np.ndarray,
        offspring_personal_best_evaluations: list[object],
    ) -> None:
        n_parent = len(self.population)
        parent_pb = self.personal_best.copy()
        parent_pb_ev = list(self.personal_best_evaluations)
        parent_lineage = self.lineage_ids.copy()
        parent_age = self.age.copy()
        parent_stagnation = self.stagnation.copy()

        combined_pb = np.vstack([parent_pb, offspring_personal_best])
        combined_pb_ev = parent_pb_ev + list(offspring_personal_best_evaluations)
        combined_lineage = np.concatenate([parent_lineage, parent_lineage])
        combined_age = np.concatenate([parent_age + 1, parent_age + 1])
        combined_stagnation = np.concatenate([parent_stagnation + 1, parent_stagnation + 1])

        selected = np.asarray(selected_indices, dtype=int)
        self.population = np.asarray(combined_population, dtype=float)[selected].copy()
        self.evaluations = [combined_evaluations[i] for i in selected]
        self.personal_best = combined_pb[selected].copy()
        self.personal_best_evaluations = [combined_pb_ev[i] for i in selected]
        self.lineage_ids = combined_lineage[selected].copy()
        self.age = combined_age[selected].copy()
        self.stagnation = combined_stagnation[selected].copy()

        # A child inherits the parent's lineage when it replaces that parent. If both parent
        # and child survive global environmental selection, the child becomes a branch lineage
        # so every active learner retains an unambiguous identity while preserving inherited pbest.
        selected_parent_sources = {int(i) for i in selected if int(i) < n_parent}
        next_lineage = int(parent_lineage.max(initial=-1)) + 1
        for pos, source_index in enumerate(selected):
            source_index = int(source_index)
            if source_index >= n_parent:
                parent_index = source_index - n_parent
                if parent_index in selected_parent_sources:
                    self.lineage_ids[pos] = next_lineage
                    next_lineage += 1
                if (
                    offspring_personal_best_evaluations[parent_index]
                    is not parent_pb_ev[parent_index]
                ):
                    self.stagnation[pos] = 0

    def restart_indices(
        self, indices: np.ndarray, new_vectors: np.ndarray, evaluations: list[object]
    ) -> None:
        indices = np.asarray(indices, dtype=int)
        new_vectors = np.asarray(new_vectors, dtype=float)
        if len(indices) != len(new_vectors) or len(indices) != len(evaluations):
            raise ValueError("Restart state sizes must match")
        next_lineage = int(self.lineage_ids.max(initial=-1)) + 1
        for offset, index in enumerate(indices):
            self.population[index] = new_vectors[offset]
            self.evaluations[index] = evaluations[offset]
            self.personal_best[index] = new_vectors[offset]
            self.personal_best_evaluations[index] = evaluations[offset]
            self.lineage_ids[index] = next_lineage + offset
            self.age[index] = 0
            self.stagnation[index] = 0
