from __future__ import annotations

import numpy as np

from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.cma_es import feasibility_first_ranks
from calo_rpd_studio.algorithms.registry import create_optimizer
from calo_rpd_studio.orpd.problem import Evaluation


class SphereProblem:
    dimension = 5

    def evaluate(self, x):
        vector = np.asarray(x, dtype=float)
        value = float(np.sum((vector - 0.25) ** 2))
        return Evaluation(value, True, 0.0, {"sphere": value}, {})

    def evaluate_population(self, population):
        return [self.evaluate(candidate) for candidate in population]

    def solution_state(self, x):
        return {"normalized_decision_vector": np.asarray(x).tolist(), "scenarios": []}


def test_feasibility_first_ranks_are_dense_and_preserve_exact_ties():
    evaluations = [
        Evaluation(1.0, False, 0.4),
        Evaluation(100.0, True, 0.0),
        Evaluation(1.0, False, 0.2),
        Evaluation(100.0, True, 0.0),
    ]
    assert feasibility_first_ranks(evaluations) == [2.0, 0.0, 1.0, 0.0]


def test_cma_es_is_seed_reproducible_and_uses_official_reference_engine():
    config = OptimizerConfig(
        population_size=8,
        max_evaluations=40,
        max_iterations=40,
        parameters={"sigma": 0.3, "active_covariance": True},
    )
    first = create_optimizer("CMA-ES", SphereProblem(), config, seed=91).run()
    second = create_optimizer("CMA-ES", SphereProblem(), config, seed=91).run()
    np.testing.assert_allclose(first.best_vector, second.best_vector, rtol=0, atol=0)
    assert first.best_objective == second.best_objective
    assert first.evaluations == second.evaluations == 40
    assert first.metadata["source_package"] == "cma"
    assert first.metadata["source_package_version"] == "4.4.4"
    assert first.metadata["constraint_adapter"].startswith("Deb feasibility-first")


def test_cma_es_partial_budget_is_evaluated_but_not_used_for_covariance_update():
    result = create_optimizer(
        "CMA-ES",
        SphereProblem(),
        OptimizerConfig(8, 3, 20, {"sigma": 0.3}),
        seed=7,
    ).run()
    assert result.evaluations == 3
    assert result.iterations == 0
    assert result.metadata["partial_final_generation"] is True
