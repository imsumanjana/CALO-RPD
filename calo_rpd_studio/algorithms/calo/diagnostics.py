"""Constraint-aware diagnostics for CALO Core v2.

The diagnostic layer deliberately keeps objective progress and constraint progress separate.
This prevents a feasibility transition from being misinterpreted as objective deterioration and
provides the controller with the physical source of infeasibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

CONSTRAINT_COMPONENTS = (
    "bus_voltage",
    "generator_q",
    "generator_p",
    "branch_thermal",
    "power_flow",
)


def components_of(evaluation) -> dict[str, float]:
    metadata = getattr(evaluation, "metadata", {}) or {}
    raw = metadata.get("constraint_components", {}) or {}
    return {name: float(raw.get(name, 0.0)) for name in CONSTRAINT_COMPONENTS}


def transformed_violation(value: float) -> float:
    if not np.isfinite(value):
        return 1.0
    return float(np.tanh(max(float(value), 0.0)))


@dataclass(slots=True)
class PopulationDiagnostics:
    feasible_ratio: float
    epsilon_feasible_ratio: float
    best_violation: float
    mean_violation: float
    best_feasible_objective: float
    component_best: dict[str, float]
    component_mean: dict[str, float]


def population_diagnostics(evaluations: Iterable, epsilon: float = 0.0) -> PopulationDiagnostics:
    items = list(evaluations)
    if not items:
        return PopulationDiagnostics(0.0, 0.0, float("inf"), float("inf"), float("inf"),
                                     {k: float("inf") for k in CONSTRAINT_COMPONENTS},
                                     {k: float("inf") for k in CONSTRAINT_COMPONENTS})
    violations = np.asarray([float(e.violation) for e in items], dtype=float)
    finite_violations = np.where(np.isfinite(violations), violations, 1e12)
    best_index = int(np.argmin(finite_violations))
    feasible_values = [float(e.value) for e in items if e.feasible and np.isfinite(e.value)]
    component_rows = [components_of(e) for e in items]
    return PopulationDiagnostics(
        feasible_ratio=float(np.mean([bool(e.feasible) for e in items])),
        epsilon_feasible_ratio=float(np.mean(finite_violations <= max(float(epsilon), 0.0) + 1e-12)),
        best_violation=float(finite_violations[best_index]),
        mean_violation=float(np.mean(finite_violations)),
        best_feasible_objective=min(feasible_values) if feasible_values else float("inf"),
        component_best=dict(component_rows[best_index]),
        component_mean={
            key: float(np.mean([row.get(key, 0.0) for row in component_rows]))
            for key in CONSTRAINT_COMPONENTS
        },
    )


def diagnostic_history_template() -> dict[str, list[float]]:
    return {
        "best_total_violation": [],
        "mean_total_violation": [],
        "feasible_ratio": [],
        "epsilon_feasible_ratio": [],
        "population_diversity": [],
        "elite_diversity": [],
        "epsilon": [],
        **{f"best_{key}": [] for key in CONSTRAINT_COMPONENTS},
        **{f"mean_{key}": [] for key in CONSTRAINT_COMPONENTS},
    }
