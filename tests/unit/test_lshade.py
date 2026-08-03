from __future__ import annotations

import numpy as np
import pytest

from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.lshade import (
    LSHADEOptimizer,
    constrained_improvement,
    parent_midpoint_repair,
    positive_round,
)
from calo_rpd_studio.algorithms.registry import create_optimizer
from calo_rpd_studio.orpd.problem import Evaluation


class SphereProblem:
    dimension = 4

    def evaluate(self, x):
        vector = np.asarray(x, dtype=float)
        value = float(np.sum((vector - 0.25) ** 2))
        return Evaluation(value, True, 0.0, {"sphere": value}, {})

    def evaluate_population(self, population):
        return [self.evaluate(candidate) for candidate in population]

    def solution_state(self, x):
        return {"normalized_decision_vector": np.asarray(x).tolist(), "scenarios": []}


def test_parent_midpoint_repair_matches_reference_equation():
    candidate = np.array([-0.2, 0.4, 1.3, 1.0])
    parent = np.array([0.8, 0.7, 0.2, 0.6])
    repaired = parent_midpoint_repair(candidate, parent)
    np.testing.assert_allclose(repaired, [0.4, 0.4, 0.6, 1.0], rtol=0, atol=0)


def test_positive_round_matches_reference_cpp_size_rounding():
    assert positive_round(16.5) == 17
    assert positive_round(16.49) == 16


def test_success_history_update_uses_weighted_lehmer_means():
    memory_f = np.full(3, 0.5)
    memory_cr = np.full(3, 0.5)
    next_position = LSHADEOptimizer._update_memory(
        memory_f,
        memory_cr,
        1,
        successful_f=[0.2, 0.8],
        successful_cr=[0.4, 0.6],
        improvements=[1.0, 3.0],
    )
    weights = np.array([0.25, 0.75])
    expected_f = np.sum(weights * np.array([0.2, 0.8]) ** 2) / np.sum(
        weights * np.array([0.2, 0.8])
    )
    expected_cr = np.sum(weights * np.array([0.4, 0.6]) ** 2) / np.sum(
        weights * np.array([0.4, 0.6])
    )
    assert memory_f[1] == pytest.approx(expected_f)
    assert memory_cr[1] == pytest.approx(expected_cr)
    assert next_position == 2


def test_constrained_improvement_respects_feasibility_first_adapter():
    infeasible = Evaluation(1.0, False, 0.5)
    less_infeasible = Evaluation(4.0, False, 0.2)
    feasible = Evaluation(8.0, True, 0.0)
    improved_feasible = Evaluation(7.0, True, 0.0)
    assert constrained_improvement(infeasible, less_infeasible) == pytest.approx(0.3)
    assert constrained_improvement(less_infeasible, feasible) > 0.0
    assert constrained_improvement(feasible, improved_feasible) == pytest.approx(1.0)
    assert constrained_improvement(improved_feasible, feasible) == 0.0


@pytest.mark.parametrize("optimizer_backend", ["legacy", "torch"])
def test_lshade_accounts_exact_budget_and_reduces_population(optimizer_backend):
    parameters = {
        "memory_size": 5,
        "p_best_rate": 0.11,
        "archive_rate": 1.4,
        "optimizer_backend": optimizer_backend,
        "execution_device": "cpu",
    }
    result = create_optimizer(
        "L-SHADE",
        SphereProblem(),
        OptimizerConfig(12, 120, 120, parameters),
        seed=2026,
    ).run()
    history = result.metadata["population_size_history"]
    assert result.evaluations == 120
    assert history[0] == 12
    assert history[-1] == 4
    assert all(next_size <= size for size, next_size in zip(history, history[1:]))
    assert result.metadata["boundary_repair_policy"].startswith("parent_midpoint")
    assert result.metadata["source_doi"] == "10.1109/CEC.2014.6900380"


def test_lshade_tiny_budget_finishes_without_invalid_mutation():
    result = create_optimizer(
        "L-SHADE",
        SphereProblem(),
        OptimizerConfig(8, 3, 10, {}),
        seed=5,
    ).run()
    assert result.evaluations == 3
    assert result.iterations == 0
