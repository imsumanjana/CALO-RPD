"""Seeded load-uncertainty scenario generation."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .scenario import Scenario


@dataclass(slots=True)
class ScenarioGeneratorConfig:
    count: int = 20
    active_load_std: float = 0.05
    reactive_load_std: float = 0.05
    clip_low: float = 0.5
    clip_high: float = 1.5


def generate_load_scenarios(config, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(config.count):
        p = float(np.clip(rng.normal(1, config.active_load_std), config.clip_low, config.clip_high))
        q = float(
            np.clip(rng.normal(1, config.reactive_load_std), config.clip_low, config.clip_high)
        )

        def transform(case, p=p, q=q):
            case.bus[:, 2] *= p
            case.bus[:, 3] *= q
            return case

        out.append(Scenario(f"load_{i + 1:03d}", 1 / config.count, transform))
    return out
