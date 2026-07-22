"""Immutable, validated scenario transformations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    weight: float = 1.0
    transform: Callable | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("Scenario name must be non-empty")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("Scenario weight must be finite and non-negative")
        object.__setattr__(self, "weight", weight)
        if self.transform is not None and not callable(self.transform):
            raise TypeError("Scenario transform must be callable or None")

    def apply(self, case):
        transformed = case.clone() if self.transform is None else self.transform(case.clone())
        if transformed is None:
            raise ValueError(f"Scenario {self.name!r} transform returned None")
        return transformed
