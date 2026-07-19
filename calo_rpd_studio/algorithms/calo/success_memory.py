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


def _nonnegative_gain(value: float) -> float:
    """Return a non-negative gain while preserving +Inf as a recovery sentinel.

    A transition from a failed/non-converged parent (infinite violation) to a
    finite child can legitimately produce ``+Inf`` feasibility gain. That is a
    *strong success*, not missing data. NaN, -Inf, and non-positive values carry
    no usable reward and are mapped to zero. Sampling handles +Inf explicitly
    without ever performing Inf/Inf normalization.
    """
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if np.isnan(number) or number <= 0.0 or np.isneginf(number):
        return 0.0
    return number


class SuccessMemory:
    """Stores successful directions without averaging opposing moves into cancellation."""

    def __init__(self, capacity: int = 256, decay: float = 0.97) -> None:
        self.records = deque(maxlen=int(capacity))
        self.decay = float(decay)
        self._counter = 0

    def add(self, direction, operator, objective_gain=0.0, feasibility_gain=0.0) -> None:
        direction = np.asarray(direction, dtype=float).reshape(-1).copy()
        # A non-finite displacement is not a scientifically usable search direction.
        # Ignore it rather than contaminating every later memory sample.
        if direction.size == 0 or not np.all(np.isfinite(direction)):
            return
        step_norm = float(np.linalg.norm(direction))
        if not np.isfinite(step_norm):
            return

        self._counter += 1
        self.records.append(
            SuccessRecord(
                direction,
                int(operator),
                step_norm,
                _nonnegative_gain(objective_gain),
                _nonnegative_gain(feasibility_gain),
                self._counter,
            )
        )

    def sample_direction(
        self,
        dimension: int,
        rng: np.random.Generator,
        prefer_feasibility: bool = False,
    ) -> np.ndarray:
        dimension = int(dimension)
        if not self.records:
            return np.zeros(dimension, dtype=float)

        # A reused optimizer/checkpoint can encounter a changed decision-vector width.
        # Never return an incompatible direction.
        records = [record for record in self.records if record.direction.size == dimension]
        if not records:
            return np.zeros(dimension, dtype=float)

        age = np.asarray(
            [max(self._counter - record.recency_index, 0) for record in records],
            dtype=float,
        )
        decay = self.decay if np.isfinite(self.decay) and self.decay > 0.0 else 0.97
        decay = min(decay, 1.0)
        recency = np.power(decay, age)
        recency = np.where(np.isfinite(recency) & (recency > 0.0), recency, 0.0)

        objective = np.asarray(
            [_nonnegative_gain(record.objective_gain) for record in records], dtype=float
        )
        feasibility = np.asarray(
            [_nonnegative_gain(record.feasibility_gain) for record in records], dtype=float
        )

        # +Inf is a legitimate sentinel for recovery from an infinite-violation parent.
        # Give such records maximum reward mass explicitly instead of allowing Inf/Inf.
        infinite_gain = np.isposinf(objective) | np.isposinf(feasibility)
        if np.any(infinite_gain):
            gains = np.where(infinite_gain, 1.0, 1e-12)
        else:
            # Normalize finite reward components before combining them. This is
            # probability-equivalent up to the tiny exploration floor and avoids
            # overflow for extremely large but finite penalties.
            reward_scale = max(
                float(np.max(objective, initial=0.0)),
                float(np.max(feasibility, initial=0.0)),
                1.0,
            )
            objective /= reward_scale
            feasibility /= reward_scale
            if prefer_feasibility:
                gains = feasibility + 0.35 * objective
            else:
                gains = objective + 0.35 * feasibility
            gains += 1e-12

        weights = recency * gains
        valid = np.isfinite(weights) & (weights >= 0.0)
        weights = np.where(valid, weights, 0.0)

        # Scale before summation so neither overflow nor an all-zero denominator can
        # produce NaNs. Fallbacks preserve recency preference, then uniform exploration.
        peak = float(np.max(weights, initial=0.0))
        if peak > 0.0 and np.isfinite(peak):
            weights /= peak
            total = float(np.sum(weights))
        else:
            total = 0.0

        if not np.isfinite(total) or total <= 0.0:
            weights = recency.copy()
            peak = float(np.max(weights, initial=0.0))
            if peak > 0.0 and np.isfinite(peak):
                weights /= peak
                total = float(np.sum(weights))
            else:
                total = 0.0

        if not np.isfinite(total) or total <= 0.0:
            weights = np.full(len(records), 1.0 / len(records), dtype=float)
        else:
            weights /= total

        record = records[int(rng.choice(len(records), p=weights))]
        return record.direction.copy()

    def direction(self, dimension: int) -> np.ndarray:
        """Compatibility accessor returning the most recent compatible successful direction."""
        dimension = int(dimension)
        for record in reversed(self.records):
            if record.direction.size == dimension and np.all(np.isfinite(record.direction)):
                return record.direction.copy()
        return np.zeros(dimension, dtype=float)

    def success_rates(self, n_operators: int = 6) -> np.ndarray:
        rates = np.zeros(n_operators, dtype=float)
        counts = np.zeros(n_operators, dtype=float)
        infinite_success = np.zeros(n_operators, dtype=bool)
        for record in self.records:
            if not 0 <= record.operator < n_operators:
                continue
            counts[record.operator] += 1.0
            infinite_success[record.operator] |= bool(
                np.isposinf(record.objective_gain) or np.isposinf(record.feasibility_gain)
            )
            rates[record.operator] += (
                (0.0 if np.isposinf(record.objective_gain) else _nonnegative_gain(record.objective_gain))
                + (0.0 if np.isposinf(record.feasibility_gain) else _nonnegative_gain(record.feasibility_gain))
                + 1e-6
            )
        mask = counts > 0.0
        rates[mask] /= counts[mask]
        maximum = float(np.max(rates, initial=0.0))
        if np.isfinite(maximum) and maximum > 0.0:
            rates /= maximum
        else:
            rates.fill(0.0)
        # A recovery from infinite violation is the strongest observed success.
        rates[infinite_success] = 1.0
        return rates

    def __len__(self) -> int:
        return len(self.records)
