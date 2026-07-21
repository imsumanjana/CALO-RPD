"""Batch-updated contextual operator and memory-depth credit for CALO v4."""

from __future__ import annotations

import numpy as np

CONTEXT_NAMES = (
    "feasible_high_diversity",
    "feasible_low_diversity",
    "infeasible_improving",
    "infeasible_stagnated",
)


def _normalise(values: np.ndarray, floor: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values) & (values >= 0.0), values, 0.0)
    values = np.maximum(values, float(max(floor, 0.0)))
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(values.shape, 1.0 / max(values.size, 1), dtype=float)
    return values / total


def classify_contexts(population, evaluations, constraint_improving: bool) -> np.ndarray:
    """Classify each learner into one of four dense, reusable contexts."""
    population = np.asarray(population, dtype=float)
    n = len(evaluations)
    if n == 0:
        return np.empty(0, dtype=np.int8)
    centroid = population.mean(axis=0)
    distance = np.linalg.norm(population - centroid[None, :], axis=1)
    diversity_cut = float(np.median(distance))
    violations = np.asarray([float(getattr(ev, "violation", np.inf)) for ev in evaluations])
    finite = violations[np.isfinite(violations)]
    violation_cut = float(np.median(finite)) if finite.size else np.inf
    contexts = np.empty(n, dtype=np.int8)
    for index, ev in enumerate(evaluations):
        if bool(getattr(ev, "feasible", False)):
            contexts[index] = 0 if diversity_cut > 1e-12 and distance[index] >= diversity_cut else 1
        else:
            improving = constraint_improving or (
                np.isfinite(violations[index]) and violations[index] <= violation_cut
            )
            contexts[index] = 2 if improving else 3
    return contexts


class ContextualCredit:
    """Compact 3D credit tensors updated once per evaluated batch."""

    def __init__(
        self,
        n_regimes: int = 4,
        n_operators: int = 6,
        n_contexts: int = 4,
        n_memory_levels: int = 4,
        decay: float = 0.90,
        floor: float = 0.02,
    ) -> None:
        self.n_regimes = int(n_regimes)
        self.n_operators = int(n_operators)
        self.n_contexts = int(n_contexts)
        self.n_memory_levels = int(n_memory_levels)
        self.decay = float(np.clip(decay, 0.0, 1.0)) if np.isfinite(decay) else 0.90
        self.floor = float(max(floor, 1e-8))
        self.operator_credit = np.ones(
            (self.n_regimes, self.n_operators, self.n_contexts), dtype=np.float64
        )
        self.memory_credit = np.ones(
            (self.n_regimes, self.n_memory_levels, self.n_contexts), dtype=np.float64
        )
        self.attempts = np.zeros(self.n_operators, dtype=np.int64)
        self.successes = np.zeros(self.n_operators, dtype=np.int64)

    def operator_probabilities(self, regime: int, context: int) -> np.ndarray:
        return _normalise(self.operator_credit[int(regime), :, int(context)], self.floor)

    def memory_probabilities(self, regime: int, context: int) -> np.ndarray:
        return _normalise(self.memory_credit[int(regime), :, int(context)], self.floor)

    @staticmethod
    def _scaled(gains: np.ndarray) -> np.ndarray:
        gains = np.asarray(gains, dtype=float)
        positive_inf = np.isposinf(gains)
        finite = np.where(np.isfinite(gains) & (gains > 0.0), gains, 0.0)
        peak = float(np.max(finite, initial=0.0))
        scaled = finite / peak if peak > 0.0 and np.isfinite(peak) else np.zeros_like(finite)
        # Recovery from an infinite-violation state is maximal finite evidence, never NaN.
        scaled[positive_inf] = 1.0
        return scaled

    def batch_update(
        self,
        regime,
        contexts,
        operators,
        memory_levels,
        successful,
        objective_gain,
        feasibility_gain,
        feasibility_transition,
    ) -> None:
        """Aggregate the entire batch before applying one EMA update per occupied cell."""
        regimes = np.asarray(regime, dtype=np.int64)
        contexts = np.asarray(contexts, dtype=np.int64)
        operators = np.asarray(operators, dtype=np.int64)
        memory_levels = np.asarray(memory_levels, dtype=np.int64)
        successful = np.asarray(successful, dtype=bool)
        objective = self._scaled(np.asarray(objective_gain, dtype=float))
        feasibility = self._scaled(np.asarray(feasibility_gain, dtype=float))
        transitions = np.asarray(feasibility_transition, dtype=float)
        transitions = np.where(np.isfinite(transitions) & (transitions > 0.0), 1.0, 0.0)
        if regimes.ndim == 0:
            regimes = np.full(len(contexts), int(regimes), dtype=np.int64)
        n = min(
            len(regimes),
            len(contexts),
            len(operators),
            len(memory_levels),
            len(successful),
            len(objective),
            len(feasibility),
            len(transitions),
        )
        if n == 0:
            return
        regimes = np.clip(regimes[:n], 0, self.n_regimes - 1)
        contexts = contexts[:n]
        operators = operators[:n]
        memory_levels = memory_levels[:n]
        successful = successful[:n]
        objective = objective[:n]
        feasibility = feasibility[:n]
        transitions = transitions[:n]

        for operator in range(self.n_operators):
            mask_op = operators == operator
            self.attempts[operator] += int(np.count_nonzero(mask_op))
            self.successes[operator] += int(np.count_nonzero(mask_op & successful))

        for regime in range(self.n_regimes):
            mask_regime = regimes == regime
            if not np.any(mask_regime):
                continue
            for context in range(self.n_contexts):
                mask_context = mask_regime & (contexts == context)
                if not np.any(mask_context):
                    continue
                # In feasible contexts objective quality dominates; in infeasible contexts
                # feasibility progress dominates. Success remains a common reliability signal.
                if context <= 1:
                    weights = (0.55, 0.15, 0.20, 0.10)
                else:
                    weights = (0.15, 0.45, 0.20, 0.20)

                for operator in range(self.n_operators):
                    mask = mask_context & (operators == operator)
                    if not np.any(mask):
                        continue
                    reward = (
                        weights[0] * float(np.mean(objective[mask]))
                        + weights[1] * float(np.mean(feasibility[mask]))
                        + weights[2] * float(np.mean(successful[mask]))
                        + weights[3] * float(np.mean(transitions[mask]))
                    )
                    old = float(self.operator_credit[regime, operator, context])
                    self.operator_credit[regime, operator, context] = self.decay * old + (
                        1.0 - self.decay
                    ) * max(reward, 0.0)

                for level in range(self.n_memory_levels):
                    mask = mask_context & (memory_levels == level)
                    if not np.any(mask):
                        continue
                    reward = (
                        weights[0] * float(np.mean(objective[mask]))
                        + weights[1] * float(np.mean(feasibility[mask]))
                        + weights[2] * float(np.mean(successful[mask]))
                        + weights[3] * float(np.mean(transitions[mask]))
                    )
                    old = float(self.memory_credit[regime, level, context])
                    self.memory_credit[regime, level, context] = self.decay * old + (
                        1.0 - self.decay
                    ) * max(reward, 0.0)

        self.operator_credit = np.where(
            np.isfinite(self.operator_credit),
            np.maximum(self.operator_credit, self.floor),
            self.floor,
        )
        self.memory_credit = np.where(
            np.isfinite(self.memory_credit), np.maximum(self.memory_credit, self.floor), self.floor
        )

    def success_rates(self) -> np.ndarray:
        rates = np.zeros(self.n_operators, dtype=float)
        mask = self.attempts > 0
        rates[mask] = self.successes[mask] / self.attempts[mask]
        return rates

    def global_operator_probabilities(self) -> np.ndarray:
        values = np.mean(self.operator_credit, axis=(0, 2))
        return _normalise(values, self.floor)
