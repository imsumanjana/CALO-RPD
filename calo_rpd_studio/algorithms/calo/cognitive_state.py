"""Constraint-aware CALO Core v2 cognitive state."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .diagnostics import CONSTRAINT_COMPONENTS, population_diagnostics, transformed_violation

STATE_DIM = 24
REGIME_NAMES = ("feasibility", "transition", "objective_refinement", "recovery")


@dataclass(slots=True)
class CognitiveState:
    diversity: float
    elite_spread: float
    feasible_ratio: float
    epsilon_feasible_ratio: float
    mean_violation: float
    best_violation: float
    component_violations: np.ndarray
    constraint_improvement: float
    objective_improvement: float
    constraint_stagnation: float
    objective_stagnation: float
    remaining_budget: float
    feasible_archive_fill: float
    boundary_archive_fill: float
    operator_credit: np.ndarray

    def vector(self) -> np.ndarray:
        vector = np.r_[
            self.diversity,
            self.elite_spread,
            self.feasible_ratio,
            self.epsilon_feasible_ratio,
            self.mean_violation,
            self.best_violation,
            self.component_violations,
            self.constraint_improvement,
            self.objective_improvement,
            self.constraint_stagnation,
            self.objective_stagnation,
            self.remaining_budget,
            self.feasible_archive_fill,
            self.boundary_archive_fill,
            np.asarray(self.operator_credit, float),
        ]
        if vector.shape != (STATE_DIM,):
            raise ValueError(f"CALO cognitive state must have {STATE_DIM} values, got {vector.shape}")
        return np.clip(np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)


def population_diversity(population) -> float:
    values = np.asarray(population, float)
    if values.ndim != 2 or not len(values):
        return 0.0
    return float(
        np.mean(np.linalg.norm(values - values.mean(axis=0), axis=1))
        / max(np.sqrt(values.shape[1]), 1e-12)
    )


def elite_diversity(population, evaluations, fraction: float = 0.2) -> float:
    if not len(population):
        return 0.0
    order = sorted(
        range(len(evaluations)),
        key=lambda i: (0 if evaluations[i].feasible else 1,
                       evaluations[i].value if evaluations[i].feasible else evaluations[i].violation),
    )
    count = max(2, int(np.ceil(len(order) * fraction)))
    elite = np.asarray(population)[order[:count]]
    return population_diversity(elite)


def _relative_improvement(previous: float, current: float) -> float:
    if not np.isfinite(previous) or not np.isfinite(current):
        return 0.0
    scale = max(abs(previous), abs(current), 1e-12)
    return float(np.clip((previous - current) / scale, -1.0, 1.0))


def build_cognitive_state(
    population,
    evaluations,
    *,
    epsilon: float,
    previous_best_violation: float,
    previous_best_objective: float,
    constraint_stagnation: float,
    objective_stagnation: float,
    remaining_budget: float,
    operator_credit,
    feasible_archive_size: int = 0,
    feasible_archive_capacity: int = 1,
    boundary_archive_size: int = 0,
    boundary_archive_capacity: int = 1,
) -> CognitiveState:
    diagnostics = population_diagnostics(evaluations, epsilon)
    components = np.asarray(
        [transformed_violation(diagnostics.component_best.get(name, 0.0)) for name in CONSTRAINT_COMPONENTS],
        dtype=float,
    )
    return CognitiveState(
        diversity=population_diversity(population),
        elite_spread=elite_diversity(population, evaluations),
        feasible_ratio=diagnostics.feasible_ratio,
        epsilon_feasible_ratio=diagnostics.epsilon_feasible_ratio,
        mean_violation=transformed_violation(diagnostics.mean_violation),
        best_violation=transformed_violation(diagnostics.best_violation),
        component_violations=components,
        constraint_improvement=_relative_improvement(previous_best_violation, diagnostics.best_violation),
        objective_improvement=_relative_improvement(
            previous_best_objective, diagnostics.best_feasible_objective
        ),
        constraint_stagnation=float(np.clip(constraint_stagnation, 0.0, 1.0)),
        objective_stagnation=float(np.clip(objective_stagnation, 0.0, 1.0)),
        remaining_budget=float(np.clip(remaining_budget, 0.0, 1.0)),
        feasible_archive_fill=float(
            np.clip(feasible_archive_size / max(feasible_archive_capacity, 1), 0.0, 1.0)
        ),
        boundary_archive_fill=float(
            np.clip(boundary_archive_size / max(boundary_archive_capacity, 1), 0.0, 1.0)
        ),
        operator_credit=np.asarray(operator_credit, float),
    )


def rule_based_regime_prior(state: CognitiveState) -> np.ndarray:
    """Transparent prior blended with the learned regime policy.

    The prior never hard-disables the AI controller. It only supplies scientifically interpretable
    guidance when exact feasibility is absent or a stagnation state is detected.
    """
    if max(state.constraint_stagnation, state.objective_stagnation) >= 0.95:
        prior = np.asarray([0.10, 0.15, 0.15, 0.60])
    elif state.feasible_ratio <= 0.0:
        prior = np.asarray([0.68, 0.24, 0.03, 0.05])
    elif state.feasible_ratio < 0.35:
        prior = np.asarray([0.30, 0.50, 0.15, 0.05])
    else:
        prior = np.asarray([0.08, 0.17, 0.68, 0.07])
    return prior / prior.sum()
