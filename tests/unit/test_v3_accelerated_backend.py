from __future__ import annotations

import numpy as np

from calo_rpd_studio.accelerated.torch_orpd import AcceleratedORPDProblem, parity_check
from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.registry import SPECS, create_optimizer
from calo_rpd_studio.orpd.problem import Evaluation, ORPDProblem


def test_torch_fp64_orpd_backend_matches_cpu_reference(toy_case):
    reference = ORPDProblem(toy_case)
    accelerated = AcceleratedORPDProblem(toy_case, device="cpu", batch_size=8)
    candidates = np.random.default_rng(2026).random((6, reference.dimension))
    report = parity_check(reference, accelerated, candidates)
    assert report.passed
    assert report.feasibility_mismatches == 0
    assert report.max_voltage_error < 1e-8
    assert report.max_objective_error < 1e-7


def test_accelerated_population_evaluation_reports_backend(toy_case):
    problem = AcceleratedORPDProblem(toy_case, device="cpu", batch_size=4)
    candidates = np.random.default_rng(7).random((5, problem.dimension))
    results = problem.evaluate_population(candidates)
    assert len(results) == 5
    assert all(
        result.metadata["scientific_backend"] == "torch_batched_dense_newton_raphson"
        for result in results
    )
    assert all(result.metadata["dtype"] == "float64" for result in results)


class SphereProblem:
    dimension = 4

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        value = float(np.sum((x - 0.25) ** 2))
        return Evaluation(value, True, 0.0, {"sphere": value}, {})

    def evaluate_population(self, population):
        return [self.evaluate(x) for x in population]

    def solution_state(self, x):
        return {
            "normalized_decision_vector": np.asarray(x).tolist(),
            "scenarios": [{"converged": True}],
        }


def test_all_nineteen_baselines_have_torch_canonical_kernels():
    for name in SPECS:
        if name == "CALO":
            continue
        parameters = dict(SPECS[name].default_parameters)
        parameters.update({"execution_device": "cpu", "optimizer_backend": "torch"})
        optimizer = create_optimizer(
            name,
            SphereProblem(),
            OptimizerConfig(8, 40, 40, parameters),
            seed=123,
        )
        result = optimizer.run()
        assert result.evaluations <= 40
        assert np.isfinite(result.best_objective)
        assert result.metadata["optimizer_kernel"] == "torch_canonical"
        assert result.metadata["optimizer_dtype"] == "float64"
