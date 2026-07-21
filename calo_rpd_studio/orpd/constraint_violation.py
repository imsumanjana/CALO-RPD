"""Constraint violation aggregation."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class ConstraintViolation:
    total: float
    components: dict[str, float]

    @property
    def feasible(self):
        return self.total <= 1e-12
