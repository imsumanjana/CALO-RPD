"""Hierarchical Prefix Elite Memory (HPEM) for CALO v4.

A single canonical seven-row feasible-elite store is interpreted at four prefix
resolutions (Best-1/3/5/7).  The implementation deliberately stores each elite
vector once and derives all hierarchy summaries by prefix-weighted reductions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

PREFIX_SIZES = (1, 3, 5, 7)


@dataclass(slots=True)
class HPEMSnapshot:
    vectors: np.ndarray
    objectives: np.ndarray
    hierarchy: np.ndarray
    occupancy: int


def _normalised_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    return float(np.linalg.norm(a - b) / np.sqrt(a.size))


class HierarchicalPrefixEliteMemory:
    """Quality-diversity bounded feasible-elite memory.

    Slot semantics are intentionally asymmetric:
    * slot 0: absolute best feasible solution;
    * slots 1-2: additional strongest objective elites;
    * slots 3-4: quality-diversity structural elites;
    * slots 5-6: strong diverse alternatives.

    The store only admits exact-feasible finite-objective candidates. Constraint-
    boundary knowledge remains the responsibility of ``ConstraintBoundaryArchive``.
    """

    def __init__(
        self,
        dimension: int,
        capacity: int = 7,
        duplicate_tol: float = 1e-9,
        variables=None,
    ) -> None:
        if int(capacity) != 7:
            raise ValueError("CALO v4 HPEM canonical capacity is fixed at seven elites")
        self.dimension = int(dimension)
        self.capacity = 7
        self.duplicate_tol = float(max(duplicate_tol, 0.0))
        self.variables = list(variables or [])
        self.vectors = np.empty((0, self.dimension), dtype=np.float64)
        self.evaluations: list[object] = []
        self._hierarchy = np.zeros((4, self.dimension), dtype=np.float64)

    def _mixed_representation(self, vector: np.ndarray) -> np.ndarray:
        """Map normalized controls to a distance representation matching decoder semantics.

        Continuous coordinates remain normalized. Discrete coordinates are mapped to their
        decoded lattice index, normalized to [0,1]. Thus two normalized values decoding to
        the same tap/shunt state are identical for duplicate/diversity decisions.
        """
        x = np.asarray(vector, dtype=np.float64).copy()
        if len(self.variables) != self.dimension:
            return x
        for i, variable in enumerate(self.variables):
            values = tuple(getattr(variable, "values", ()) or ())
            kind = str(getattr(getattr(variable, "kind", None), "value", getattr(variable, "kind", "")))
            if kind == "discrete" and values:
                n = len(values)
                index = min(int(np.floor(np.clip(x[i], 0.0, 1.0) * n)), n - 1)
                x[i] = index / max(n - 1, 1)
        return x

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return _normalised_distance(self._mixed_representation(a), self._mixed_representation(b))

    def __len__(self) -> int:
        return len(self.evaluations)

    @property
    def best_vector(self) -> np.ndarray | None:
        return None if not len(self) else self.vectors[0].copy()

    @property
    def best_evaluation(self):
        return None if not len(self) else self.evaluations[0]

    @property
    def occupancy(self) -> float:
        return len(self) / self.capacity

    def _is_duplicate(self, vector: np.ndarray, kept: list[int], pool_vectors: np.ndarray) -> bool:
        return any(self._distance(vector, pool_vectors[index]) <= self.duplicate_tol for index in kept)

    @staticmethod
    def _objective(evaluation) -> float:
        value = float(getattr(evaluation, "value", np.inf))
        return value if np.isfinite(value) else np.inf

    def update(self, vectors: Iterable[np.ndarray], evaluations: Iterable[object]) -> None:
        candidates: list[tuple[np.ndarray, object]] = [
            (self.vectors[i].copy(), self.evaluations[i]) for i in range(len(self))
        ]
        for vector, evaluation in zip(vectors, evaluations):
            if bool(getattr(evaluation, "feasible", False)) and np.isfinite(
                float(getattr(evaluation, "value", np.inf))
            ):
                x = np.asarray(vector, dtype=np.float64).reshape(-1)
                if x.shape == (self.dimension,) and np.all(np.isfinite(x)):
                    candidates.append((np.clip(x, 0.0, 1.0).copy(), evaluation))

        if not candidates:
            self.vectors = np.empty((0, self.dimension), dtype=np.float64)
            self.evaluations = []
            self._hierarchy.fill(0.0)
            return

        # Stable objective ordering creates deterministic paired-seed replay.
        candidates.sort(key=lambda item: self._objective(item[1]))
        pool_vectors = np.asarray([item[0] for item in candidates], dtype=np.float64)
        pool_evaluations = [item[1] for item in candidates]

        unique_order: list[int] = []
        for index, vector in enumerate(pool_vectors):
            if self._is_duplicate(vector, unique_order, pool_vectors):
                continue
            unique_order.append(index)
        if not unique_order:
            return

        selected: list[int] = [unique_order[0]]  # absolute Best-1 is always protected
        remaining = unique_order[1:]

        # Slots 1-2: strongest distinct objective elites.
        while remaining and len(selected) < 3:
            selected.append(remaining.pop(0))

        def choose_quality_diverse(quality_weight: float) -> int:
            best_pos = 0
            best_score = -np.inf
            denom = max(len(unique_order) - 1, 1)
            rank = {index: position for position, index in enumerate(unique_order)}
            for pos, index in enumerate(remaining):
                quality = 1.0 - rank[index] / denom
                diversity = min(
                    self._distance(pool_vectors[index], pool_vectors[j]) for j in selected
                )
                score = quality_weight * quality + (1.0 - quality_weight) * diversity
                if score > best_score + 1e-15:
                    best_score = score
                    best_pos = pos
            return best_pos

        # Slots 3-4: retain objective quality while adding structural spread.
        while remaining and len(selected) < 5:
            selected.append(remaining.pop(choose_quality_diverse(0.70)))
        # Slots 5-6: stronger diversity pressure, but quality still matters.
        while remaining and len(selected) < 7:
            selected.append(remaining.pop(choose_quality_diverse(0.45)))

        self.vectors = pool_vectors[selected].copy()
        self.evaluations = [pool_evaluations[index] for index in selected]
        self._refresh_hierarchy()

    def _refresh_hierarchy(self) -> None:
        self._hierarchy.fill(0.0)
        count = len(self)
        if count == 0:
            return
        # Rank weights are scale-free and robust when objective magnitudes differ across cases.
        rank_weights = 1.0 / np.arange(1, count + 1, dtype=float)
        for level, requested in enumerate(PREFIX_SIZES):
            n = min(requested, count)
            weights = rank_weights[:n]
            weights = weights / weights.sum()
            self._hierarchy[level] = np.sum(weights[:, None] * self.vectors[:n], axis=0)

    def hierarchy(self) -> np.ndarray:
        return self._hierarchy.copy()

    def summary(self, level: int, fallback: np.ndarray | None = None) -> np.ndarray:
        level = int(np.clip(level, 0, 3))
        if not len(self):
            if fallback is None:
                return np.zeros(self.dimension, dtype=np.float64)
            return np.asarray(fallback, dtype=np.float64).copy()
        return self._hierarchy[level].copy()

    def consensus(self, reference: np.ndarray) -> float:
        """Return hierarchical directional agreement in [0, 1].

        Consensus is damped by occupancy so a single early feasible point cannot look
        artificially like four independent agreeing memories.
        """
        if len(self) < 2:
            return 0.0
        ref = np.asarray(reference, dtype=float).reshape(-1)
        if ref.shape != (self.dimension,):
            return 0.0
        directions = self._hierarchy - ref[None, :]
        norms = np.linalg.norm(directions, axis=1)
        valid = norms > 1e-12
        if np.count_nonzero(valid) < 2:
            return float(self.occupancy)
        unit = directions[valid] / norms[valid, None]
        similarities = unit @ unit.T
        upper = similarities[np.triu_indices_from(similarities, k=1)]
        if upper.size == 0:
            return 0.0
        # Negative alignment is disagreement; positive alignment maps linearly to confidence.
        agreement = float(np.clip((np.mean(upper) + 1.0) * 0.5, 0.0, 1.0))
        return float(np.clip(agreement * self.occupancy, 0.0, 1.0))

    def dispersion(self, level: int) -> float:
        if not len(self):
            return 0.0
        n = min(PREFIX_SIZES[int(np.clip(level, 0, 3))], len(self))
        if n <= 1:
            return 0.0
        center = self._hierarchy[int(np.clip(level, 0, 3))]
        distances = np.linalg.norm(self.vectors[:n] - center[None, :], axis=1) / np.sqrt(
            self.dimension
        )
        return float(np.mean(distances))

    def snapshot(self) -> HPEMSnapshot:
        return HPEMSnapshot(
            self.vectors.copy(),
            np.asarray([self._objective(ev) for ev in self.evaluations], dtype=float),
            self._hierarchy.copy(),
            len(self),
        )
