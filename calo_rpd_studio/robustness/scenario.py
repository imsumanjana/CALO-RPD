"""Immutable scenario transformations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class Scenario:
    name: str
    weight: float = 1.0
    transform: Callable | None = None

    def apply(self, case):
        return case.clone() if self.transform is None else self.transform(case.clone())
