from __future__ import annotations

import numpy as np
import pytest

from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.calo.contextual_credit import ContextualCredit
from calo_rpd_studio.algorithms.calo.hierarchical_memory import HierarchicalPrefixEliteMemory
from calo_rpd_studio.algorithms.calo.optimizer import CALOOptimizer
from calo_rpd_studio.algorithms.calo.tensor_state import CALOTensorState
from calo_rpd_studio.algorithms.registry import SPECS, create_optimizer
from calo_rpd_studio.orpd.problem import Evaluation


def _ev(value: float, violation: float = 0.0) -> Evaluation:
    return Evaluation(float(value), violation <= 1e-12, float(violation), {}, {})


class CountingSphere:
    dimension = 4

    def __init__(self):
        self.physical_calls = 0

    def evaluate(self, x):
        self.physical_calls += 1
        vector = np.asarray(x, dtype=float)
        value = float(np.sum((vector - 0.25) ** 2))
        return _ev(value)

    def evaluate_population(self, population):
        return [self.evaluate(row) for row in population]

    def solution_state(self, x):
        return {"normalized_decision_vector": np.asarray(x).tolist(), "scenarios": []}


def test_hpem_uses_one_canonical_best7_store_and_four_prefix_summaries():
    rng = np.random.default_rng(4)
    vectors = rng.random((20, 6))
    evaluations = [_ev(float(i + 1)) for i in range(20)]
    hpem = HierarchicalPrefixEliteMemory(6)
    hpem.update(vectors, evaluations)
    assert hpem.vectors.shape == (7, 6)
    assert hpem.hierarchy().shape == (4, 6)
    np.testing.assert_allclose(hpem.hierarchy()[0], hpem.vectors[0])
    assert hpem.best_evaluation.value == 1.0
    # Canonical storage is seven rows, not duplicated 1+3+5+7 = 16 rows.
    assert hpem.vectors.size == 7 * 6


def test_hpem_rejects_infeasible_and_near_duplicate_elites():
    vectors = np.asarray(
        [
            [0.1, 0.2],
            [0.1, 0.2],
            [0.2, 0.3],
            [0.9, 0.8],
        ]
    )
    evaluations = [_ev(1.0), _ev(0.5), _ev(2.0, 0.1), _ev(1.5)]
    hpem = HierarchicalPrefixEliteMemory(2)
    hpem.update(vectors, evaluations)
    assert len(hpem) == 2
    assert hpem.best_evaluation.value == pytest.approx(0.5)
    assert all(ev.feasible for ev in hpem.evaluations)


def test_persistent_personal_best_is_not_reset_by_environmental_selection():
    population = np.asarray([[0.1, 0.1], [0.8, 0.8]])
    evaluations = [_ev(1.0), _ev(2.0)]
    state = CALOTensorState.initialize(population, evaluations)
    # Child 0 is worse than its historical pbest; child 1 improves.
    offspring = np.asarray([[0.6, 0.6], [0.2, 0.2]])
    child_ev = [_ev(3.0), _ev(0.5)]
    offspring_pb = state.personal_best.copy()
    offspring_pb_ev = list(state.personal_best_evaluations)
    offspring_pb[1] = offspring[1]
    offspring_pb_ev[1] = child_ev[1]
    combined = np.vstack([population, offspring])
    combined_ev = evaluations + child_ev
    # Select child 0 and child 1 deliberately to exercise lineage inheritance.
    state.select_from_combined(combined, combined_ev, np.asarray([2, 3]), offspring_pb, offspring_pb_ev)
    np.testing.assert_allclose(state.personal_best[0], population[0])
    np.testing.assert_allclose(state.personal_best[1], offspring[1])


def test_contextual_credit_batch_update_is_order_invariant():
    inputs = dict(
        regime=2,
        contexts=np.asarray([0, 0, 1, 1]),
        operators=np.asarray([0, 1, 0, 1]),
        memory_levels=np.asarray([0, 1, 2, 3]),
        successful=np.asarray([True, False, True, True]),
        objective_gain=np.asarray([0.2, 0.0, 0.1, 0.3]),
        feasibility_gain=np.zeros(4),
        feasibility_transition=np.zeros(4),
    )
    a = ContextualCredit()
    a.batch_update(**inputs)
    order = np.asarray([3, 1, 0, 2])
    b = ContextualCredit()
    b.batch_update(
        regime=2,
        contexts=inputs["contexts"][order],
        operators=inputs["operators"][order],
        memory_levels=inputs["memory_levels"][order],
        successful=inputs["successful"][order],
        objective_gain=inputs["objective_gain"][order],
        feasibility_gain=inputs["feasibility_gain"][order],
        feasibility_transition=inputs["feasibility_transition"][order],
    )
    np.testing.assert_allclose(a.operator_credit, b.operator_credit)
    np.testing.assert_allclose(a.memory_credit, b.memory_credit)


def test_v4_single_run_never_exceeds_common_requested_fe_budget_and_starts_fresh():
    problem = CountingSphere()
    params = dict(SPECS["CALO"].default_parameters)
    params.update({"use_ai": False, "use_exact_evaluation_cache": True})
    config = OptimizerConfig(population_size=8, max_evaluations=40, max_iterations=40, parameters=params)
    result1 = create_optimizer("CALO", problem, config, seed=7).run()
    result2 = create_optimizer("CALO", CountingSphere(), config, seed=7).run()
    assert result1.evaluations <= 40
    assert result2.evaluations == result1.evaluations
    np.testing.assert_allclose(result1.best_vector, result2.best_vector)
    assert result1.best_objective == pytest.approx(result2.best_objective, abs=1e-15)
    assert result1.metadata["calo_version"] == "v4.1"
    assert result1.metadata["hpem"]["hierarchy_shape"][0] == 4


def test_exact_cache_counts_duplicate_requests_but_reuses_physical_solver_work():
    class DuplicateCALO(CALOOptimizer):
        def random_population(self, n=None):
            return np.full((n or self.config.population_size, self.problem.dimension), 0.5)

    problem = CountingSphere()
    config = OptimizerConfig(
        population_size=4,
        max_evaluations=4,
        max_iterations=1,
        parameters={"use_ai": False, "use_exact_evaluation_cache": True},
    )
    result = DuplicateCALO(problem, config, seed=1).run()
    assert result.evaluations == 4
    assert problem.physical_calls == 1
    assert result.metadata["physical_solver_calls"] == 1


def test_strict_benchmark_mode_blocks_historical_runtime_warm_start():
    problem = CountingSphere()
    config = OptimizerConfig(
        population_size=4,
        max_evaluations=8,
        max_iterations=1,
        parameters={
            "use_ai": False,
            "strict_benchmark_mode": True,
            "historical_repository": "dummy.json",
            "use_historical_parameter_priors": True,
        },
    )
    with pytest.raises(ValueError, match="Strict benchmark mode"):
        CALOOptimizer(problem, config, seed=1).run()


def test_hpem_duplicate_detection_follows_discrete_decoder_lattice():
    from calo_rpd_studio.orpd.decision_variables import DecisionVariable, VariableKind

    variables = [
        DecisionVariable("Tap", 0.9, 1.1, VariableKind.DISCRETE, (0.9, 1.0, 1.1)),
        DecisionVariable("Vg", 0.9, 1.1),
    ]
    hpem = HierarchicalPrefixEliteMemory(2, variables=variables)
    # First coordinate differs numerically but both values decode to discrete lattice index 0.
    hpem.update(
        np.asarray([[0.01, 0.5], [0.20, 0.5], [0.80, 0.5]]),
        [_ev(1.0), _ev(0.9), _ev(1.1)],
    )
    assert len(hpem) == 2
    assert hpem.best_evaluation.value == pytest.approx(0.9)


def test_contextual_credit_attributes_batch_outcomes_to_individual_regimes():
    credit = ContextualCredit(decay=0.0)
    credit.batch_update(
        regime=np.asarray([0, 2]),
        contexts=np.asarray([2, 0]),
        operators=np.asarray([1, 3]),
        memory_levels=np.asarray([3, 0]),
        successful=np.asarray([True, True]),
        objective_gain=np.asarray([0.0, 1.0]),
        feasibility_gain=np.asarray([1.0, 0.0]),
        feasibility_transition=np.asarray([1.0, 0.0]),
    )
    # Only the occupied regime/context cells should receive their batch evidence.
    assert credit.operator_credit[0, 1, 2] != pytest.approx(1.0)
    assert credit.operator_credit[2, 3, 0] != pytest.approx(1.0)
    assert credit.operator_credit[1, 1, 2] == pytest.approx(1.0)
