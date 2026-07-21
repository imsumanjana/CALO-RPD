"""Whale Optimization Algorithm."""

from __future__ import annotations
import time, numpy as np
from .base_optimizer import BaseOptimizer


class WhaleOptimizer(BaseOptimizer):
    name = "WOA"

    def run(self):
        started = time.perf_counter()
        p = self.random_population()
        ev = self.evaluate_population(p)
        b = float(self.config.parameters.get("spiral_b", 1.0))
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            best = p[self.order(ev)[0]].copy()
            a = 2 * (1 - min(self.iteration / max(self.config.max_iterations, 1), 1))
            new = []
            for x in p:
                A = 2 * a * self.rng.random() - a
                C = 2 * self.rng.random()
                if self.rng.random() < 0.5:
                    ref = best if abs(A) < 1 else p[int(self.rng.integers(len(p)))]
                    cand = ref - A * np.abs(C * ref - x)
                else:
                    dist = np.abs(best - x)
                    spiral_parameter = self.rng.uniform(-1, 1)
                    cand = (
                        dist * np.exp(b * spiral_parameter) * np.cos(2 * np.pi * spiral_parameter)
                        + best
                    )
                new.append(np.clip(cand, 0, 1))
            p = np.asarray(new)
            ev = self.evaluate_population(p)
            self.record()
        return self.finalize(p, started=started)
