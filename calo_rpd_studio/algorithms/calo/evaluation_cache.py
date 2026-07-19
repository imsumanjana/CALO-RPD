"""Exact decoded-control deduplication/cache for CALO v4.

The cache can reduce physical solver calls but never changes requested function-
evaluation accounting. Keys use exact IEEE-754 hexadecimal representations of
all decoded physical controls; there is no approximate nearest-neighbour reuse.
"""
from __future__ import annotations

from collections import OrderedDict
import copy

import numpy as np


class ExactEvaluationCache:
    def __init__(self, problem, capacity: int = 4096) -> None:
        self.problem = problem
        self.capacity = max(0, int(capacity))
        self._cache: OrderedDict[tuple, object] = OrderedDict()
        self.physical_solver_calls = 0
        self.cache_hits = 0

    def key(self, vector: np.ndarray) -> tuple:
        z = np.clip(np.asarray(vector, dtype=np.float64), 0.0, 1.0)
        decoder = getattr(self.problem, "decoder", None)
        if decoder is None or not callable(getattr(decoder, "decode", None)):
            return tuple(float(value).hex() for value in z)
        _, physical = decoder.decode(z)
        return tuple((str(name), float(value).hex()) for name, value in sorted(physical.items()))

    def evaluate_requests(self, optimizer, population: np.ndarray) -> list[object]:
        population = np.asarray(population, dtype=float)
        if population.ndim == 1:
            population = population[None, :]
        remaining = max(0, int(optimizer.config.max_evaluations) - int(optimizer.evaluations))
        population = np.clip(population[:remaining], 0.0, 1.0)
        if len(population) == 0 or optimizer.cancelled():
            return []

        keys = [self.key(row) for row in population]
        local_results: dict[tuple, object] = {}
        missing_order: list[tuple] = []
        representative: dict[tuple, np.ndarray] = {}
        for key, row in zip(keys, population):
            cached = self._cache.get(key)
            if cached is not None:
                local_results[key] = cached
                self._cache.move_to_end(key)
                continue
            if key not in representative:
                representative[key] = row
                missing_order.append(key)

        if missing_order:
            unique = np.asarray([representative[key] for key in missing_order], dtype=float)
            evaluator = getattr(self.problem, "evaluate_population", None)
            solved = list(evaluator(unique)) if callable(evaluator) else [self.problem.evaluate(row) for row in unique]
            if len(solved) != len(unique):
                raise RuntimeError("Common evaluator returned an incomplete exact-deduplication batch")
            self.physical_solver_calls += len(unique)
            for key, evaluation in zip(missing_order, solved):
                local_results[key] = evaluation
                if self.capacity > 0:
                    self._cache[key] = copy.deepcopy(evaluation)
                    self._cache.move_to_end(key)
                    while len(self._cache) > self.capacity:
                        self._cache.popitem(last=False)

        seen_in_request: set[tuple] = set()
        out: list[object] = []
        for row, key in zip(population, keys):
            if key in seen_in_request or key not in missing_order:
                self.cache_hits += 1
            seen_in_request.add(key)
            evaluation = copy.deepcopy(local_results[key])
            out.append(optimizer._register_evaluation(row, evaluation))
        return out
