"""Bounded success-distribution memory for CALO Core v2."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SuccessRecord:
    direction: np.ndarray
    operator: int
    step_norm: float
    objective_gain: float
    feasibility_gain: float
    recency_index: int


class SuccessMemory:
    """Stores successful directions without averaging opposing moves into cancellation."""

    def __init__(self, capacity: int = 256, decay: float = 0.97) -> None:
        self.records = deque(maxlen=int(capacity))
        self.decay = float(decay)
        self._counter = 0

    def add(self, direction, operator, objective_gain=0.0, feasibility_gain=0.0) -> None:
        direction = np.asarray(direction, float).copy()
        self._counter += 1
        self.records.append(
            SuccessRecord(
                direction,
                int(operator),
                float(np.linalg.norm(direction)),
                float(objective_gain),
                float(feasibility_gain),
                self._counter,
            )
        )

    def sample_direction(self, dimension: int, rng: np.random.Generator,
                         prefer_feasibility: bool = False) -> np.ndarray:
        if not self.records:
            return np.zeros(dimension, dtype=float)
        records = list(self.records)
        age = np.asarray([self._counter - record.recency_index for record in records], dtype=float)
        recency = np.power(self.decay, age)
        gains = np.asarray([
            (record.feasibility_gain if prefer_feasibility else record.objective_gain)
            + 0.35 * (record.objective_gain if prefer_feasibility else record.feasibility_gain)
            + 1e-9
            for record in records
        ])
        weights = recency * np.maximum(gains, 1e-9)
        weights /= weights.sum()
        record = records[int(rng.choice(len(records), p=weights))]
        return record.direction.copy()

    def direction(self, dimension: int) -> np.ndarray:
        """Compatibility accessor returning the most recent successful direction."""
        if not self.records:
            return np.zeros(dimension, dtype=float)
        return self.records[-1].direction.copy()

    def success_rates(self, n_operators: int = 6) -> np.ndarray:
        rates = np.zeros(n_operators)
        counts = np.zeros(n_operators)
        for record in self.records:
            counts[record.operator] += 1
            rates[record.operator] += (
                max(record.objective_gain, 0.0) + max(record.feasibility_gain, 0.0) + 1e-6
            )
        mask = counts > 0
        rates[mask] /= counts[mask]
        if rates.max() > 0:
            rates /= rates.max()
        return rates

    def __len__(self) -> int:
        return len(self.records)
