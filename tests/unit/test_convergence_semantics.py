import math

from calo_rpd_studio.algorithms.base_optimizer import BaseOptimizer, OptimizerConfig


class Evaluation:
    def __init__(self, value, violation, feasible):
        self.value = value
        self.violation = violation
        self.feasible = feasible
        self.physical_controls = {}
        self.components = {}


class Problem:
    dimension = 1

    def __init__(self):
        self.items = iter(
            [
                Evaluation(1.0, 0.5, False),
                Evaluation(4.0, 0.2, False),
                Evaluation(3.0, 0.0, True),
                Evaluation(2.5, 0.0, True),
            ]
        )

    def evaluate(self, x):
        return next(self.items)

    def solution_state(self, x):
        return {}


class Dummy(BaseOptimizer):
    name = "DUMMY"

    def run(self):
        raise NotImplementedError


def test_feasible_objective_and_violation_histories_are_separate_and_monotonic():
    opt = Dummy(Problem(), OptimizerConfig(population_size=1, max_evaluations=10), seed=1)
    for _ in range(4):
        opt.evaluate([0.5])
        opt.record()
    assert math.isnan(opt.best_feasible_objective_history[0])
    assert math.isnan(opt.best_feasible_objective_history[1])
    assert opt.best_feasible_objective_history[2:] == [3.0, 2.5]
    assert opt.best_constraint_violation_history == [0.5, 0.2, 0.0, 0.0]
    # The feasibility-first incumbent objective can rise while violation improves; this is why it
    # must not be presented as ordinary monotonic best-objective convergence.
    assert opt.history[:2] == [1.0, 4.0]
