"""CALO-native, mixed-variable-aware cognitive precision refinement."""
from __future__ import annotations

import numpy as np


class CognitivePrecisionEngine:
    def __init__(self, initial_radius: float = 0.04, min_radius: float = 5e-4, max_radius: float = 0.15) -> None:
        self.radius = float(np.clip(initial_radius, min_radius, max_radius))
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.attempts = 0
        self.successes = 0

    def active(self, feasible_ratio: float, objective_stagnation: float, progress: float, hpem_size: int) -> bool:
        return bool(
            hpem_size >= 3
            and feasible_ratio >= 0.50
            and progress >= 0.55
            and (objective_stagnation >= 0.25 or progress >= 0.78)
        )

    @staticmethod
    def _legal_discrete_neighbor(value: float, variable, rng: np.random.Generator) -> float:
        values = tuple(getattr(variable, "values", ()) or ())
        if len(values) <= 1:
            return float(np.clip(value, 0.0, 1.0))
        index = int(np.rint(np.clip(value, 0.0, 1.0) * (len(values) - 1)))
        candidates = [j for j in (index - 1, index + 1) if 0 <= j < len(values)]
        if not candidates:
            return float(index / (len(values) - 1))
        return float(int(rng.choice(candidates)) / (len(values) - 1))

    def propose(
        self,
        anchor: np.ndarray,
        hierarchy: np.ndarray,
        success_direction: np.ndarray,
        variables,
        group_mask: np.ndarray,
        rng: np.random.Generator,
        consensus: float,
    ) -> np.ndarray:
        anchor = np.asarray(anchor, dtype=float)
        h = np.asarray(hierarchy, dtype=float)
        success = np.asarray(success_direction, dtype=float)
        candidate = anchor.copy()
        # Best-3 local geometry and Best-5 structural direction complement the exact Best-1 anchor.
        d3 = h[1] - h[0]
        d5 = h[2] - h[1]
        direction = 0.55 * d3 + 0.30 * d5 + 0.15 * success
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12 and np.isfinite(norm):
            direction = direction / norm
        else:
            direction = np.zeros_like(anchor)
        confidence = float(np.clip(consensus, 0.0, 1.0))
        noise_scale = self.radius * (0.35 + 0.65 * (1.0 - confidence))
        variables = variables or []
        for index in np.where(group_mask)[0]:
            variable = variables[index] if index < len(variables) else None
            kind = str(getattr(getattr(variable, "kind", "continuous"), "value", getattr(variable, "kind", "continuous")))
            if kind == "discrete" and tuple(getattr(variable, "values", ()) or ()):
                if rng.random() < 0.65:
                    candidate[index] = self._legal_discrete_neighbor(anchor[index], variable, rng)
            else:
                candidate[index] = anchor[index] + self.radius * direction[index] + noise_scale * rng.normal()
        return np.clip(candidate, 0.0, 1.0)

    def update(self, attempted: int, successful: int) -> None:
        attempted = int(max(attempted, 0))
        successful = int(np.clip(successful, 0, attempted))
        if attempted == 0:
            return
        self.attempts += attempted
        self.successes += successful
        rate = successful / attempted
        if rate >= 0.25:
            self.radius = min(self.max_radius, self.radius * 1.08)
        elif rate <= 0.05:
            self.radius = max(self.min_radius, self.radius * 0.72)
        else:
            self.radius = max(self.min_radius, self.radius * 0.95)
