import numpy as np
from calo_rpd_studio.algorithms.registry import create_optimizer, SPECS
from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.orpd.problem import Evaluation
from calo_rpd_studio.experiments.seed_manager import SeedManager


class P:
    dimension = 4

    def evaluate(self, x):
        return Evaluation(float(np.sum((np.asarray(x) - 0.3) ** 2)), True, 0)

    def solution_state(self, x):
        return {"x": np.asarray(x).tolist()}


def test_seed_tuple_and_optimizer_reproducibility():
    s1 = SeedManager(2026).generate(3)
    s2 = SeedManager(2026).generate(3)
    assert s1 == s2
    cfg = OptimizerConfig(8, 40, 40, dict(SPECS["PSO"].default_parameters))
    a = create_optimizer("PSO", P(), cfg, seed=123).run()
    b = create_optimizer("PSO", P(), cfg, seed=123).run()
    assert np.allclose(a.best_vector, b.best_vector)
    assert a.best_objective == b.best_objective
