from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from calo_rpd_studio.algorithms.calo.evaluation_cache import ExactEvaluationCache
from calo_rpd_studio.orpd.problem import Evaluation


class _Problem:
    decoder = None
    def __init__(self):
        self.calls = 0
    def evaluate_population(self, population):
        self.calls += len(population)
        return [Evaluation(float(np.sum(row)), True, 0.0, {}, {}) for row in population]


class _Optimizer:
    def __init__(self, max_evaluations=1000):
        self.config = SimpleNamespace(max_evaluations=max_evaluations)
        self.evaluations = 0
    def cancelled(self):
        return False
    def _register_evaluation(self, _row, evaluation):
        self.evaluations += 1
        return evaluation


def test_exact_cache_deduplicates_physical_work_without_reducing_fe_accounting():
    problem = _Problem()
    optimizer = _Optimizer(max_evaluations=10)
    cache = ExactEvaluationCache(problem, capacity=32, adaptive=False)
    population = np.asarray([[0.1, 0.2], [0.1, 0.2], [0.3, 0.4], [0.1, 0.2]])
    results = cache.evaluate_requests(optimizer, population)
    assert len(results) == 4
    assert optimizer.evaluations == 4  # every requested FE remains counted
    assert cache.physical_solver_calls == 2  # only exact unique controls are solved
    assert problem.calls == 2


def test_low_persistent_hit_rate_disables_cross_batch_cache_but_keeps_exact_dedup():
    problem = _Problem()
    optimizer = _Optimizer(max_evaluations=100)
    cache = ExactEvaluationCache(
        problem,
        capacity=32,
        adaptive=True,
        minimum_requests_before_adaptation=4,
        minimum_persistent_hit_rate=0.50,
    )
    cache.evaluate_requests(optimizer, np.asarray([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08]]))
    assert cache.persistent_enabled is False
    before = cache.physical_solver_calls
    cache.evaluate_requests(optimizer, np.asarray([[0.2, 0.3], [0.2, 0.3]]))
    assert cache.physical_solver_calls == before + 1
    assert optimizer.evaluations == 6
