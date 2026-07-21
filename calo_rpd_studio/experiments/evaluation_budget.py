"""Evaluation-budget policies."""

from dataclasses import dataclass
from enum import Enum


class BudgetPolicy(str, Enum):
    EQUAL_EVALUATIONS = "equal_evaluations"
    EQUAL_WALL_CLOCK = "equal_wall_clock"
    ALGORITHM_NATIVE = "algorithm_native"


@dataclass(slots=True)
class EvaluationBudget:
    policy: BudgetPolicy = BudgetPolicy.EQUAL_EVALUATIONS
    max_evaluations: int = 5000
    wall_clock_seconds: float | None = None

    def validate(self):
        if self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")
        if self.policy is BudgetPolicy.EQUAL_WALL_CLOCK and (
            self.wall_clock_seconds is None or self.wall_clock_seconds <= 0
        ):
            raise ValueError("A positive wall-clock budget is required")
