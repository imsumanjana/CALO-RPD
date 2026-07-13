"""Separated objective/constraint reward used by CALO Core v2 and PPO training."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RewardComponents:
    objective_improvement: float
    constraint_improvement: float
    feasible_ratio_improvement: float
    diversity_recovery: float
    overhead_penalty: float

    @property
    def total(self) -> float:
        return (
            0.85 * self.objective_improvement
            + 1.20 * self.constraint_improvement
            + 0.75 * self.feasible_ratio_improvement
            + 0.15 * self.diversity_recovery
            - 0.04 * self.overhead_penalty
        )


def _gain(old: float, new: float) -> float:
    if not np.isfinite(old) or not np.isfinite(new):
        return 0.0
    scale = max(abs(old), abs(new), 1e-12)
    return float(np.clip((old - new) / scale, -1.0, 1.0))


def calculate_reward(
    old_objective: float,
    new_objective: float,
    old_violation: float,
    new_violation: float,
    old_feasible_ratio: float,
    new_feasible_ratio: float,
    old_diversity: float,
    new_diversity: float,
    *,
    overhead: float = 0.0,
) -> RewardComponents:
    return RewardComponents(
        objective_improvement=_gain(old_objective, new_objective),
        constraint_improvement=_gain(old_violation, new_violation),
        feasible_ratio_improvement=float(np.clip(new_feasible_ratio - old_feasible_ratio, -1.0, 1.0)),
        diversity_recovery=float(np.clip(new_diversity - old_diversity, -0.5, 0.5)),
        overhead_penalty=max(float(overhead), 0.0),
    )
