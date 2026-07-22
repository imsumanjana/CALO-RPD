"""Dual archives for feasible elites and diverse constraint-boundary candidates."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .diagnostics import components_of


@dataclass(slots=True)
class ArchiveEntry:
    vector: np.ndarray
    evaluation: object


class FeasibleEliteArchive:
    def __init__(self, capacity: int = 32) -> None:
        self.capacity = max(2, int(capacity))
        self.entries: list[ArchiveEntry] = []

    def update(self, vectors, evaluations) -> None:
        pool = list(self.entries)
        for vector, evaluation in zip(vectors, evaluations):
            if evaluation.feasible and np.isfinite(evaluation.value):
                pool.append(ArchiveEntry(np.asarray(vector, float).copy(), evaluation))
        pool.sort(key=lambda entry: float(entry.evaluation.value))
        kept: list[ArchiveEntry] = []
        for entry in pool:
            if any(np.linalg.norm(entry.vector - other.vector) < 1e-9 for other in kept):
                continue
            kept.append(entry)
            if len(kept) >= self.capacity:
                break
        self.entries = kept

    def sample(self, rng: np.random.Generator, fallback: np.ndarray) -> np.ndarray:
        if not self.entries:
            return np.asarray(fallback, float).copy()
        # Rank-biased sampling retains exploitation without always selecting the same point.
        ranks = np.arange(1, len(self.entries) + 1, dtype=float)
        weights = 1.0 / ranks
        weights /= weights.sum()
        return self.entries[int(rng.choice(len(self.entries), p=weights))].vector.copy()

    def sample_many(self, rng: np.random.Generator, fallback: np.ndarray, count: int) -> np.ndarray:
        """Vectorized rank-biased teacher sampling for a learner batch."""
        count = max(0, int(count))
        fallback = np.asarray(fallback, float)
        if count == 0:
            return np.empty((0, fallback.size), dtype=float)
        if not self.entries:
            return np.repeat(fallback[None, :], count, axis=0)
        ranks = np.arange(1, len(self.entries) + 1, dtype=float)
        weights = (1.0 / ranks); weights /= weights.sum()
        indices = rng.choice(len(self.entries), size=count, p=weights)
        bank = np.asarray([entry.vector for entry in self.entries], dtype=float)
        return bank[np.asarray(indices, dtype=int)].copy()

    @property
    def best(self) -> ArchiveEntry | None:
        return self.entries[0] if self.entries else None

    def __len__(self) -> int:
        return len(self.entries)


class ConstraintBoundaryArchive:
    """Stores diverse low-violation routes toward the feasible boundary.

    Candidates are retained using a quality/diversity score in decision space and their
    constraint-component profiles, so the archive does not collapse to a single lowest-CV point.
    """

    def __init__(self, capacity: int = 48) -> None:
        self.capacity = max(4, int(capacity))
        self.entries: list[ArchiveEntry] = []

    @staticmethod
    def _profile(entry: ArchiveEntry) -> np.ndarray:
        components = components_of(entry.evaluation)
        values = np.asarray(list(components.values()), dtype=float)
        norm = np.linalg.norm(values)
        return values / norm if norm > 1e-12 else values

    def update(self, vectors, evaluations) -> None:
        pool = list(self.entries)
        for vector, evaluation in zip(vectors, evaluations):
            if np.isfinite(evaluation.violation) and not evaluation.feasible:
                pool.append(ArchiveEntry(np.asarray(vector, float).copy(), evaluation))
        if not pool:
            return
        pool.sort(key=lambda entry: float(entry.evaluation.violation))
        seed_count = min(max(2, self.capacity // 4), len(pool))
        kept = pool[:seed_count]
        remaining = pool[seed_count:]
        while remaining and len(kept) < self.capacity:
            best_index = 0
            best_score = -float("inf")
            best_violation = max(float(pool[0].evaluation.violation), 1e-12)
            for index, entry in enumerate(remaining):
                decision_distance = min(
                    np.linalg.norm(entry.vector - other.vector) for other in kept
                )
                profile = self._profile(entry)
                profile_distance = min(
                    np.linalg.norm(profile - self._profile(other)) for other in kept
                )
                quality = 1.0 / (1.0 + float(entry.evaluation.violation) / best_violation)
                score = 0.55 * quality + 0.30 * decision_distance + 0.15 * profile_distance
                if score > best_score:
                    best_score = score
                    best_index = index
            kept.append(remaining.pop(best_index))
        self.entries = kept[: self.capacity]

    def sample(self, rng: np.random.Generator, fallback: np.ndarray) -> np.ndarray:
        if not self.entries:
            return np.asarray(fallback, float).copy()
        index = int(rng.integers(len(self.entries)))
        return self.entries[index].vector.copy()

    def sample_many(self, rng: np.random.Generator, fallback: np.ndarray, count: int) -> np.ndarray:
        """Vectorized uniform boundary-teacher sampling for a learner batch."""
        count = max(0, int(count))
        fallback = np.asarray(fallback, float)
        if count == 0:
            return np.empty((0, fallback.size), dtype=float)
        if not self.entries:
            return np.repeat(fallback[None, :], count, axis=0)
        indices = rng.integers(0, len(self.entries), size=count)
        bank = np.asarray([entry.vector for entry in self.entries], dtype=float)
        return bank[np.asarray(indices, dtype=int)].copy()

    @property
    def best(self) -> ArchiveEntry | None:
        if not self.entries:
            return None
        return min(self.entries, key=lambda entry: float(entry.evaluation.violation))

    def __len__(self) -> int:
        return len(self.entries)
