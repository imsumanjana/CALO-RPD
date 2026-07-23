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


def generate_stratified_load_scenarios(config):
    """Deterministic stratified aggregate-load sensitivity scenarios.

    Unlike Monte Carlo sampling, this mode is seed-independent and symmetrically spans the declared
    P/Q uncertainty bands. It represents system-wide correlated load-scaling sensitivity, not a
    nodal stochastic covariance model.
    """
    count = max(1, int(config.count))
    if count == 1:
        z_values = np.asarray([0.0])
    else:
        z_values = np.linspace(-2.0, 2.0, count)
    out = []
    for i, z in enumerate(z_values):
        # Pair Q in reverse order so the set covers both same-direction and opposing P/Q stress.
        zq = z_values[-(i + 1)] if count > 1 else 0.0
        p = float(np.clip(1.0 + float(config.active_load_std) * z, config.clip_low, config.clip_high))
        q = float(np.clip(1.0 + float(config.reactive_load_std) * zq, config.clip_low, config.clip_high))

        def transform(case, p=p, q=q):
            case.bus[:, 2] *= p
            case.bus[:, 3] *= q
            return case

        out.append(Scenario(f"load_sensitivity_{i + 1:03d}", 1.0 / count, transform))
    return out
