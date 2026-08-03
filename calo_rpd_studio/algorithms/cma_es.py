"""Official pycma ask-tell comparator under the common constrained-ORPD contract."""

from __future__ import annotations

from importlib.metadata import version
import time

import numpy as np

from calo_rpd_studio.orpd.feasibility_rules import sort_key

from .base_optimizer import BaseOptimizer


def feasibility_first_ranks(evaluations) -> list[float]:
    """Return dense scalar ranks without weakening the shared lexicographic ordering.

    CMA-ES is comparison based, so telling it feasibility-first ranks retains Deb ordering without
    inventing a scale-dependent objective/violation penalty. Exact ordering ties receive equal rank.
    """

    keys = [sort_key(evaluation) for evaluation in evaluations]
    rank_by_key = {key: rank for rank, key in enumerate(sorted(set(keys)))}
    return [float(rank_by_key[key]) for key in keys]


class CMAESOptimizer(BaseOptimizer):
    """Active CMA-ES from the authors' maintained pycma implementation."""

    name = "CMA-ES"

    def run(self):
        import cma

        started = time.perf_counter()
        population_size = max(4, int(self.config.population_size))
        sigma = float(self.config.parameters.get("sigma", 0.30))
        if not 0.0 < sigma <= 1.0:
            raise ValueError("CMA-ES sigma must lie in (0, 1]")
        active = bool(self.config.parameters.get("active_covariance", True))
        options = {
            "bounds": [0.0, 1.0],
            "BoundaryHandler": "BoundTransform",
            "CMA_active": active,
            "popsize": population_size,
            "randn": lambda *shape: self.rng.standard_normal(shape),
            # A custom generator owns stochasticity; pycma must not reseed NumPy's global RNG.
            "seed": np.nan,
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
        }
        strategy = cma.CMAEvolutionStrategy(
            np.full(self.problem.dimension, 0.5, dtype=float), sigma, options
        )
        last_population = np.empty((0, self.problem.dimension), dtype=float)
        native_stop: dict[str, object] = {}
        incomplete_generation = False

        while self.iteration < self.config.max_iterations and self.can_evaluate():
            remaining = int(self.config.max_evaluations) - int(self.evaluations)
            requested = min(population_size, remaining)
            candidates = np.asarray(strategy.ask(number=requested), dtype=float)
            evaluations = self.evaluate_population(candidates)
            candidates = candidates[: len(evaluations)]
            last_population = candidates
            if not evaluations:
                break
            if len(evaluations) < population_size:
                # The partial final generation still supplies legitimate budgeted incumbents, but
                # pycma covariance adaptation requires a complete declared population.
                incomplete_generation = True
                self.record({"population_size": len(evaluations), "covariance_updated": False})
                break
            strategy.tell(candidates.tolist(), feasibility_first_ranks(evaluations))
            self.iteration += 1
            self.record({"population_size": len(evaluations), "covariance_updated": True})
            native_stop = {str(key): value for key, value in strategy.stop().items()}
            if native_stop:
                break

        if last_population.size == 0:
            raise RuntimeError("CMA-ES could not evaluate an initial population")
        metadata = {
            "source_algorithm": "CMA-ES/pycma",
            "source_package": "cma",
            "source_package_version": version("cma"),
            "source_doi": "10.5281/zenodo.2559635",
            "optimizer_kernel": "pycma_reference_cpu_control",
            "optimizer_control_residency": "cpu_ram",
            "scientific_formulation": "active CMA-ES + feasibility-first dense-rank constrained wrapper",
            "constraint_adapter": "Deb feasibility-first dense ranks; no scalar penalty coefficient",
            "latent_mixed_variable_semantics": "continuous normalized search decoded by the common mixed-variable decoder",
            "boundary_strategy": "pycma_BoundTransform_[0,1]",
            "boundary_repair_policy": "pycma_smooth_bound_transform_to_unit_box",
            "population_size": population_size,
            "initial_sigma": sigma,
            "active_covariance": active,
            "native_stop": native_stop,
            "partial_final_generation": incomplete_generation,
        }
        return self.finalize(last_population, metadata=metadata, started=started)
