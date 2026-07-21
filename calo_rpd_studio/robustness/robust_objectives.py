"""Scenario aggregation for robust ORPD."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .cvar import weighted_cvar


class RobustAggregation(str, Enum):
    EXPECTED = "expected"
    MEAN_RISK = "mean_risk"
    WORST_CASE = "worst_case"
    CVAR = "cvar"


@dataclass(slots=True)
class RobustObjectiveConfig:
    aggregation: RobustAggregation = RobustAggregation.EXPECTED
    risk_lambda: float = 1.0
    cvar_alpha: float = 0.95


def aggregate_robust(values, weights, config):
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    w = w / w.sum()
    mean = float(np.sum(v * w))
    if config.aggregation is RobustAggregation.EXPECTED:
        return mean
    if config.aggregation is RobustAggregation.MEAN_RISK:
        return mean + config.risk_lambda * float(np.sqrt(np.sum(w * (v - mean) ** 2)))
    if config.aggregation is RobustAggregation.WORST_CASE:
        return float(np.max(v))
    return weighted_cvar(v, w, config.cvar_alpha)
