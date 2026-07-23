"""Common optimizer interface, budget accounting, and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time

import numpy as np

from calo_rpd_studio.orpd.feasibility_rules import better, sort_key
from .result import OptimizerResult


@dataclass(slots=True)
class OptimizerConfig:
    population_size: int = 50
    max_evaluations: int = 5000
    max_iterations: int = 1000
    parameters: dict[str, Any] = field(default_factory=dict)


class BaseOptimizer:
    """Shared optimizer mechanics and scientifically explicit convergence telemetry.

    The optimizer's final incumbent is selected with Deb-style feasibility-first rules.
    Consequently, the *objective value of that incumbent* is not guaranteed to be monotonic
    while the search remains infeasible: a lower-violation candidate may legitimately have a
    larger raw objective. For this reason, three histories are tracked separately:

    * ``incumbent_objective_history``: objective of the feasibility-first incumbent;
    * ``best_feasible_objective_history``: monotonic best feasible objective (NaN until feasible);
    * ``best_constraint_violation_history``: monotonic minimum normalized violation.

    Comparative convergence plots should use the latter two histories against objective-function
    evaluations, not raw iteration count.
    """

    name = "BASE"

    def __init__(self, problem, config=None, seed=0, progress_callback=None, cancel_callback=None):
        self.problem = problem
        self.config = config or OptimizerConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.progress_callback = progress_callback
        self.cancel_callback = cancel_callback
        self.evaluations = 0
        self.iteration = 0
        self.best_evaluation = None
        self.best_vector = None

        # Legacy feasibility-first incumbent objective history retained for compatibility.
        self.history: list[float] = []
        # Scientifically explicit histories used by live/statistical convergence displays.
        self.evaluation_history: list[int] = []
        self.best_feasible_objective_history: list[float] = []
        self.best_constraint_violation_history: list[float] = []
        self._best_feasible_objective = float("inf")
        self._best_constraint_violation = float("inf")
        self._best_constraint_evaluation = None
        self.first_feasible_evaluation: int | None = None
        self.constraint_component_histories: dict[str, list[float]] = {}
        self.repair_candidate_count = 0
        self.repair_coordinate_count = 0
        self.repair_total_coordinates = 0

    def cancelled(self):
        return bool(self.cancel_callback and self.cancel_callback())

    def can_evaluate(self, n=1):
        return not self.cancelled() and self.evaluations + n <= self.config.max_evaluations

    def _repair_to_bounds(self, x):
        raw = np.asarray(x, dtype=float)
        clipped = np.clip(raw, 0.0, 1.0)
        changed = np.not_equal(raw, clipped) | ~np.isfinite(raw)
        self.repair_total_coordinates += int(raw.size)
        if np.any(changed):
            self.repair_candidate_count += 1
            self.repair_coordinate_count += int(np.count_nonzero(changed))
        return clipped

    def _register_evaluation(self, clipped, ev):
        """Account for one already-computed candidate evaluation.

        Accelerator backends evaluate populations in batches.  Centralizing incumbent and
        convergence bookkeeping here keeps the evaluation budget and feasibility-first semantics
        identical between scalar CPU and batched CUDA/XPU execution.
        """
        clipped = np.clip(np.asarray(clipped, float), 0, 1)
        self.evaluations += 1
        if better(ev, self.best_evaluation):
            self.best_evaluation = ev
            self.best_vector = clipped.copy()
        if ev.feasible and np.isfinite(ev.value):
            self._best_feasible_objective = min(self._best_feasible_objective, float(ev.value))
            if self.first_feasible_evaluation is None:
                self.first_feasible_evaluation = int(self.evaluations)
        if np.isfinite(ev.violation) and ev.violation < self._best_constraint_violation:
            self._best_constraint_violation = float(ev.violation)
            self._best_constraint_evaluation = ev
        return ev

    def evaluate(self, x):
        if not self.can_evaluate():
            return None
        clipped = self._repair_to_bounds(x)
        ev = self.problem.evaluate(clipped)
        return self._register_evaluation(clipped, ev)

    def evaluate_population(self, pop):
        population = np.asarray(pop, dtype=float)
        if population.ndim == 1:
            population = population[None, :]
        remaining = max(0, int(self.config.max_evaluations) - int(self.evaluations))
        if remaining <= 0 or self.cancelled():
            return []
        raw_population = population[:remaining]
        population = np.asarray([self._repair_to_bounds(x) for x in raw_population], dtype=float)
        batch_evaluator = getattr(self.problem, "evaluate_population", None)
        if callable(batch_evaluator):
            evaluations = list(batch_evaluator(population))
            return [self._register_evaluation(x, ev) for x, ev in zip(population, evaluations)]
        out = []
        # ``population`` has already passed through the single common repair authority above.
        # Do not call ``evaluate()`` here because that would repair/count the same coordinates twice.
        for x in population:
            if not self.can_evaluate():
                break
            ev = self.problem.evaluate(x)
            out.append(self._register_evaluation(x, ev))
        return out

    def random_population(self, n=None):
        return self.rng.random((n or self.config.population_size, self.problem.dimension))

    def record(self, extra=None):
        incumbent = (
            float("inf") if self.best_evaluation is None else float(self.best_evaluation.value)
        )
        feasible_best = (
            float("nan")
            if not np.isfinite(self._best_feasible_objective)
            else float(self._best_feasible_objective)
        )
        best_violation = (
            float("inf")
            if not np.isfinite(self._best_constraint_violation)
            else float(self._best_constraint_violation)
        )

        self.history.append(incumbent)
        self.evaluation_history.append(int(self.evaluations))
        self.best_feasible_objective_history.append(feasible_best)
        self.best_constraint_violation_history.append(best_violation)
        components = {}
        if self._best_constraint_evaluation is not None:
            components = dict(
                (getattr(self._best_constraint_evaluation, "metadata", {}) or {}).get(
                    "constraint_components", {}
                )
            )
        known = set(self.constraint_component_histories) | set(components)
        for key in sorted(known):
            value = float(components.get(key, 0.0))
            self.constraint_component_histories.setdefault(key, []).append(value)

        if self.progress_callback:
            self.progress_callback(
                {
                    "algorithm": self.name,
                    "iteration": self.iteration,
                    "evaluations": self.evaluations,
                    # Backward-compatible field: feasibility-first incumbent objective.
                    "best_objective": incumbent,
                    "best_feasible_objective": feasible_best,
                    "best_constraint_violation": best_violation,
                    "constraint_components": components,
                    "first_feasible_evaluation": self.first_feasible_evaluation,
                    "feasible": False
                    if self.best_evaluation is None
                    else self.best_evaluation.feasible,
                    **(extra or {}),
                }
            )

    def order(self, evaluations):
        return sorted(range(len(evaluations)), key=lambda i: sort_key(evaluations[i]))

    def run(self):
        raise NotImplementedError

    def finalize(
        self, population=None, reason="budget_or_iteration_limit", metadata=None, started=None
    ):
        if self.best_evaluation is None or self.best_vector is None:
            raise RuntimeError(f"{self.name} completed without an evaluated candidate")
        runtime = 0.0 if started is None else time.perf_counter() - started
        md = dict(metadata or {})
        # Post-run state reconstruction is not part of the optimizer evaluation budget.
        md["solution_state"] = self.problem.solution_state(self.best_vector)
        md["convergence_evaluations"] = list(self.evaluation_history)
        md["best_feasible_objective_history"] = list(self.best_feasible_objective_history)
        md["best_constraint_violation_history"] = list(self.best_constraint_violation_history)
        md["incumbent_objective_history"] = list(self.history)
        md["constraint_component_histories"] = {
            key: list(values) for key, values in self.constraint_component_histories.items()
        }
        md["first_feasible_evaluation"] = self.first_feasible_evaluation
        md["boundary_repair_policy"] = "componentwise_clip_to_[0,1]"
        md["boundary_repair_candidate_count"] = int(self.repair_candidate_count)
        md["boundary_repair_coordinate_count"] = int(self.repair_coordinate_count)
        md["boundary_repair_coordinate_rate"] = (
            float(self.repair_coordinate_count) / max(int(self.repair_total_coordinates), 1)
        )
        md["convergence_definition"] = (
            "best feasible objective and minimum normalized constraint violation versus objective-function evaluations"
        )
        md["telemetry_durability"] = {
            "scientific_authority": "committed_optimizer_history_checkpoints_and_stored_results",
            "live_ui_transient_points": "ephemeral_non_publication_evidence",
            "hard_crash_semantics": "UI-only transient points after the last committed scientific state may be lost without changing authoritative stored evidence",
        }
        ev = self.best_evaluation
        return OptimizerResult(
            self.name,
            self.seed,
            dict(self.config.parameters),
            self.best_vector.copy(),
            dict(ev.physical_controls),
            float(ev.value),
            dict(ev.components),
            float(ev.violation),
            bool(ev.feasible),
            self.evaluations,
            self.iteration,
            list(self.history),
            runtime,
            None if population is None else np.asarray(population).copy(),
            reason,
            md,
        )
