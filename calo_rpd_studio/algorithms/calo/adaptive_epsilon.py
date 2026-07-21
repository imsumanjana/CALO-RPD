"""Behavior-driven epsilon-feasibility controller for CALO v4."""

from __future__ import annotations

import numpy as np


class AdaptiveEpsilonController:
    def __init__(
        self, initial: float, control_fraction: float = 0.65, exponent: float = 2.0
    ) -> None:
        self.initial = float(max(initial, 0.0)) if np.isfinite(initial) else 0.0
        self.control_fraction = float(np.clip(control_fraction, 0.10, 1.0))
        self.exponent = float(max(exponent, 0.25)) if np.isfinite(exponent) else 2.0
        self.current = self.initial

    def value(
        self,
        evaluations: int,
        max_evaluations: int,
        feasible_ratio: float,
        violation_improving: bool,
        constraint_stagnation: float,
    ) -> float:
        progress = float(np.clip(evaluations / max(max_evaluations, 1), 0.0, 1.0))
        control_end = self.control_fraction
        if progress >= control_end or self.initial <= 0.0:
            self.current = 0.0
            return 0.0
        ratio = max(0.0, 1.0 - progress / max(control_end, 1e-12))
        scheduled = self.initial * ratio**self.exponent
        feasible_ratio = float(np.clip(feasible_ratio, 0.0, 1.0))
        stagnation = float(np.clip(constraint_stagnation, 0.0, 1.0))
        if feasible_ratio >= 0.65:
            factor = 0.55
        elif feasible_ratio >= 0.25:
            factor = 0.80
        elif violation_improving:
            factor = 1.05
        elif stagnation >= 0.75:
            factor = 1.15
        else:
            factor = 1.0
        target = min(self.initial, scheduled * factor)
        # Avoid abrupt oscillation while still allowing faster tightening when feasibility is stable.
        if target < self.current:
            self.current = max(target, 0.70 * self.current)
        else:
            self.current = min(target, 1.05 * self.current if self.current > 0 else target)
        if progress >= 0.90 * control_end and feasible_ratio >= 0.50:
            self.current = min(self.current, scheduled * 0.35)
        return float(max(self.current, 0.0))
