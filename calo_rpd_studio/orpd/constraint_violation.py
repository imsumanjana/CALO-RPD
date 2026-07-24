"""Constraint violation aggregation."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class ConstraintViolation:
    total: float
    components: dict[str, float]
    feasibility_tolerance: float = 1e-12

    @property
    def feasible(self) -> bool:
        """Use the same persisted feasibility tolerance as the problem evaluator."""
        return bool(self.total <= float(self.feasibility_tolerance))
