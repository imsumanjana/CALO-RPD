"""Scenario aggregation for robust ORPD with explicit constraint semantics."""

from __future__ import annotations
from dataclasses import dataclass
import math
from enum import Enum
import numpy as np
from .cvar import weighted_cvar


class RobustAggregation(str, Enum):
    EXPECTED = "expected"
    MEAN_RISK = "mean_risk"
    WORST_CASE = "worst_case"
    CVAR = "cvar"


class ConstraintAggregation(str, Enum):
    """How scenario constraint violations define robust feasibility.

    ``ALL_SCENARIO_MAX`` is the publication-safe default: every scenario must satisfy the
    constraint tolerance. ``EXPECTED_WEIGHTED`` is available only for explicitly declared
    expected/risk formulations and must never be described as all-scenario robust feasibility.
    """

    ALL_SCENARIO_MAX = "all_scenario_max"
    EXPECTED_WEIGHTED = "expected_weighted"


@dataclass(slots=True)
class RobustObjectiveConfig:
    aggregation: RobustAggregation = RobustAggregation.EXPECTED
    risk_lambda: float = 1.0
    cvar_alpha: float = 0.95
    constraint_aggregation: ConstraintAggregation = ConstraintAggregation.ALL_SCENARIO_MAX

    def __post_init__(self) -> None:
        self.aggregation = RobustAggregation(self.aggregation)
        self.constraint_aggregation = ConstraintAggregation(self.constraint_aggregation)
        self.validate()

    def validate(self) -> None:
        if not math.isfinite(float(self.risk_lambda)) or float(self.risk_lambda) < 0.0:
            raise ValueError("risk_lambda must be finite and non-negative")
        if not math.isfinite(float(self.cvar_alpha)) or not 0.0 < float(self.cvar_alpha) < 1.0:
            raise ValueError("cvar_alpha must be finite and lie strictly between 0 and 1")


def normalize_scenario_weights(weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("Scenario weights must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(w)):
        raise ValueError("Scenario weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("Scenario weights must be non-negative")
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Scenario weights must have a positive finite sum")
    return w / total


def aggregate_robust(values, weights, config):
    config.validate()
    v = np.asarray(values, float)
    w = normalize_scenario_weights(weights)
    if v.ndim != 1 or v.size != w.size:
        raise ValueError("Scenario objective values and weights must have the same length")
    if not np.all(np.isfinite(v)):
        return float("inf")
    mean = float(np.sum(v * w))
    if config.aggregation is RobustAggregation.EXPECTED:
        return mean
    if config.aggregation is RobustAggregation.MEAN_RISK:
        return mean + config.risk_lambda * float(np.sqrt(np.sum(w * (v - mean) ** 2)))
    if config.aggregation is RobustAggregation.WORST_CASE:
        return float(np.max(v))
    return weighted_cvar(v, w, config.cvar_alpha)


def aggregate_constraint_violation(violations, weights, config: RobustObjectiveConfig) -> float:
    """Aggregate scenario violations without silently diluting a violated scenario."""
    config.validate()
    v = np.asarray(violations, dtype=float)
    w = normalize_scenario_weights(weights)
    if v.ndim != 1 or v.size != w.size:
        raise ValueError("Scenario constraint violations and weights must have the same length")
    if not np.all(np.isfinite(v)):
        return float("inf")
    mode = ConstraintAggregation(config.constraint_aggregation)
    if mode is ConstraintAggregation.ALL_SCENARIO_MAX:
        return float(np.max(v))
    return float(np.sum(w * v))
