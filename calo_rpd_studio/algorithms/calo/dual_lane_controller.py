"""Single-budget blind-discovery / learned-search lane controller for CALO v4."""

from __future__ import annotations

import numpy as np

# Memory-readiness weights: feasibility evidence, HPEM coverage, success-rate density,
# evidence maturity (fraction of budget used), and population-wide consensus.
_READINESS_WEIGHTS = (0.28, 0.24, 0.18, 0.18, 0.12)
# Base learning-fraction blend: a floor component plus a progress-driven ramp.
_PROGRESS_BLEND = (0.35, 0.65)
# Diversity threshold below which stagnation triggers discovery-reopening.
_STAGNATION_DIVERSITY_THRESHOLD = 0.05
# Fraction by which the learning lane is cut when stagnated in a low-diversity basin.
_STAGNATION_DISCOVERY_PENALTY = 0.70


class DualLaneController:
    def __init__(self, min_learning: float = 0.0, max_learning: float = 0.92) -> None:
        self.min_learning = float(np.clip(min_learning, 0.0, 1.0))
        self.max_learning = float(np.clip(max_learning, self.min_learning, 1.0))

    @staticmethod
    def memory_readiness(
        feasible_ratio: float,
        hpem_occupancy: float,
        success_density: float,
        evidence_fraction: float,
        consensus: float,
    ) -> float:
        w = _READINESS_WEIGHTS
        score = (
            w[0] * np.clip(feasible_ratio, 0.0, 1.0)
            + w[1] * np.clip(hpem_occupancy, 0.0, 1.0)
            + w[2] * np.clip(success_density, 0.0, 1.0)
            + w[3] * np.clip(evidence_fraction, 0.0, 1.0)
            + w[4] * np.clip(consensus, 0.0, 1.0)
        )
        return float(np.clip(score, 0.0, 1.0))

    def learning_fraction(
        self,
        readiness: float,
        progress: float,
        diversity: float,
        stagnated: bool,
    ) -> float:
        progress = float(np.clip(progress, 0.0, 1.0))
        readiness = float(np.clip(readiness, 0.0, 1.0))
        # The learning lane cannot dominate before the run has produced meaningful evidence.
        base = readiness * (_PROGRESS_BLEND[0] + _PROGRESS_BLEND[1] * progress)
        if stagnated and diversity < _STAGNATION_DIVERSITY_THRESHOLD:
            base *= _STAGNATION_DISCOVERY_PENALTY
        return float(np.clip(base, self.min_learning, self.max_learning))

    def assign(
        self,
        population_size: int,
        learning_fraction: float,
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> np.ndarray:
        if deterministic:
            n_learning = int(round(population_size * learning_fraction))
            lane = np.zeros(population_size, dtype=np.int8)
            lane[:n_learning] = 1
            return lane
        return (rng.random(population_size) < float(learning_fraction)).astype(np.int8)
