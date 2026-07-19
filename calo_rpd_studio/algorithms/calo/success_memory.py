"""Finite, bounded 3D success-direction memory for CALO v4.

Canonical persistent layout is ``[operator, history_slot, decision_variable]``.
Only successful compressed directions are retained; full trajectories are never
historized.  Probability construction is explicitly NaN/Inf safe.
"""
from __future__ import annotations

import numpy as np


def _nonnegative_gain(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if np.isnan(number) or number <= 0.0 or np.isneginf(number):
        return 0.0
    return number


class SuccessMemory:
    def __init__(self, capacity: int = 256, decay: float = 0.97, n_operators: int = 6) -> None:
        self.capacity = max(1, int(capacity))
        self.decay = float(decay)
        self.n_operators = max(1, int(n_operators))
        self.slots = int(np.ceil(self.capacity / self.n_operators))
        self.dimension: int | None = None
        self.directions: np.ndarray | None = None
        shape = (self.n_operators, self.slots)
        self.valid = np.zeros(shape, dtype=bool)
        self.step_norm = np.zeros(shape, dtype=np.float64)
        self.objective_gain = np.zeros(shape, dtype=np.float64)
        self.feasibility_gain = np.zeros(shape, dtype=np.float64)
        self.recency = np.zeros(shape, dtype=np.int64)
        self.regime = np.full(shape, -1, dtype=np.int8)
        self.context = np.full(shape, -1, dtype=np.int8)
        self.group = np.full(shape, -1, dtype=np.int8)
        self._counter = 0
        self._size = 0

    def _ensure_dimension(self, dimension: int) -> bool:
        if self.dimension is None:
            self.dimension = int(dimension)
            self.directions = np.zeros(
                (self.n_operators, self.slots, self.dimension), dtype=np.float64
            )
            return True
        return self.dimension == int(dimension)

    def _evict_global_oldest(self) -> None:
        if not np.any(self.valid):
            return
        ages = np.where(self.valid, self.recency, np.iinfo(np.int64).max)
        flat = int(np.argmin(ages))
        index = np.unravel_index(flat, ages.shape)
        self.valid[index] = False
        self._size -= 1

    def add(
        self,
        direction,
        operator,
        objective_gain=0.0,
        feasibility_gain=0.0,
        *,
        regime: int = -1,
        context: int = -1,
        group: int = -1,
    ) -> None:
        vector = np.asarray(direction, dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            return
        if not self._ensure_dimension(vector.size):
            return
        operator = int(operator)
        if not 0 <= operator < self.n_operators:
            return
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm):
            return
        if self._size >= self.capacity:
            self._evict_global_oldest()

        free = np.where(~self.valid[operator])[0]
        if free.size:
            slot = int(free[0])
        else:
            slot = int(np.argmin(self.recency[operator]))
            if self.valid[operator, slot]:
                self._size -= 1

        self._counter += 1
        assert self.directions is not None
        self.directions[operator, slot] = vector
        self.step_norm[operator, slot] = norm
        self.objective_gain[operator, slot] = _nonnegative_gain(objective_gain)
        self.feasibility_gain[operator, slot] = _nonnegative_gain(feasibility_gain)
        self.recency[operator, slot] = self._counter
        self.regime[operator, slot] = int(np.clip(regime, -1, 127))
        self.context[operator, slot] = int(np.clip(context, -1, 127))
        self.group[operator, slot] = int(np.clip(group, -1, 127))
        self.valid[operator, slot] = True
        self._size += 1

    def _candidate_indices(
        self,
        dimension: int,
        operator: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.dimension != int(dimension) or self.directions is None or not np.any(self.valid):
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        op_idx, slot_idx = np.where(self.valid)
        if operator is not None:
            mask = op_idx == int(operator)
            op_idx, slot_idx = op_idx[mask], slot_idx[mask]
        return op_idx, slot_idx

    def sample_direction(
        self,
        dimension: int,
        rng: np.random.Generator,
        prefer_feasibility: bool = False,
        *,
        operator: int | None = None,
        regime: int | None = None,
        context: int | None = None,
        group: int | None = None,
    ) -> np.ndarray:
        op_idx, slot_idx = self._candidate_indices(dimension, operator)
        if op_idx.size == 0:
            return np.zeros(int(dimension), dtype=np.float64)

        age = np.maximum(self._counter - self.recency[op_idx, slot_idx], 0).astype(float)
        decay = self.decay if np.isfinite(self.decay) and self.decay > 0.0 else 0.97
        decay = min(decay, 1.0)
        recency = np.power(decay, age)

        objective = self.objective_gain[op_idx, slot_idx].copy()
        feasibility = self.feasibility_gain[op_idx, slot_idx].copy()
        inf_mask = np.isposinf(objective) | np.isposinf(feasibility)
        if np.any(inf_mask):
            gains = np.where(inf_mask, 1.0, 1e-12)
        else:
            scale = max(
                float(np.max(objective, initial=0.0)),
                float(np.max(feasibility, initial=0.0)),
                1.0,
            )
            objective /= scale
            feasibility /= scale
            gains = (
                feasibility + 0.35 * objective
                if prefer_feasibility
                else objective + 0.35 * feasibility
            )
            gains += 1e-12

        context_boost = np.ones_like(gains)
        if regime is not None:
            context_boost *= np.where(self.regime[op_idx, slot_idx] == int(regime), 1.35, 1.0)
        if context is not None:
            context_boost *= np.where(self.context[op_idx, slot_idx] == int(context), 1.35, 1.0)
        if group is not None:
            context_boost *= np.where(self.group[op_idx, slot_idx] == int(group), 1.20, 1.0)

        weights = recency * gains * context_boost
        weights = np.where(np.isfinite(weights) & (weights >= 0.0), weights, 0.0)
        peak = float(np.max(weights, initial=0.0))
        if peak > 0.0 and np.isfinite(peak):
            weights /= peak
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= total
        chosen = int(rng.choice(len(weights), p=weights))
        assert self.directions is not None
        return self.directions[op_idx[chosen], slot_idx[chosen]].copy()

    def direction(self, dimension: int) -> np.ndarray:
        op_idx, slot_idx = self._candidate_indices(dimension, None)
        if op_idx.size == 0:
            return np.zeros(int(dimension), dtype=np.float64)
        latest = int(np.argmax(self.recency[op_idx, slot_idx]))
        assert self.directions is not None
        return self.directions[op_idx[latest], slot_idx[latest]].copy()

    def mean_direction(
        self,
        dimension: int,
        *,
        operator: int | None = None,
        regime: int | None = None,
        context: int | None = None,
        group: int | None = None,
    ) -> np.ndarray:
        op_idx, slot_idx = self._candidate_indices(dimension, operator)
        if op_idx.size == 0:
            return np.zeros(int(dimension), dtype=np.float64)
        mask = np.ones(len(op_idx), dtype=bool)
        if regime is not None:
            match = self.regime[op_idx, slot_idx] == int(regime)
            if np.any(match):
                mask &= match
        if context is not None:
            match = self.context[op_idx, slot_idx] == int(context)
            if np.any(match):
                mask &= match
        if group is not None:
            match = self.group[op_idx, slot_idx] == int(group)
            if np.any(match):
                mask &= match
        assert self.directions is not None
        vectors = self.directions[op_idx[mask], slot_idx[mask]]
        if len(vectors) == 0:
            return np.zeros(int(dimension), dtype=np.float64)
        weights = self.recency[op_idx[mask], slot_idx[mask]].astype(float)
        weights -= weights.min(initial=0.0)
        weights += 1.0
        mean = np.average(vectors, axis=0, weights=weights)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 1e-12 and np.isfinite(norm) else np.zeros(int(dimension))

    def success_rates(self, n_operators: int = 6) -> np.ndarray:
        n_operators = int(n_operators)
        rates = np.zeros(n_operators, dtype=float)
        for operator in range(min(n_operators, self.n_operators)):
            slots = np.where(self.valid[operator])[0]
            if not len(slots):
                continue
            objective = self.objective_gain[operator, slots]
            feasibility = self.feasibility_gain[operator, slots]
            if np.any(np.isposinf(objective) | np.isposinf(feasibility)):
                rates[operator] = 1.0
            else:
                # Scale components before addition so extreme finite gains cannot overflow.
                scale = max(
                    float(np.max(objective, initial=0.0)),
                    float(np.max(feasibility, initial=0.0)),
                    1.0,
                )
                finite = np.where(np.isfinite(objective), objective / scale, 0.0)
                finite += np.where(np.isfinite(feasibility), feasibility / scale, 0.0)
                rates[operator] = float(np.mean(finite + 1e-6))
        peak = float(np.max(rates, initial=0.0))
        if peak > 0.0 and np.isfinite(peak):
            rates /= peak
        return rates

    @property
    def density(self) -> float:
        return float(self._size / self.capacity)

    def __len__(self) -> int:
        return int(self._size)
