"""Continuous Dragonfly Algorithm."""

from __future__ import annotations
import time, numpy as np
from .base_optimizer import BaseOptimizer


class DragonflyOptimizer(BaseOptimizer):
    name = "DA"

    def run(self):
        started = time.perf_counter()
        p = self.random_population()
        step = np.zeros_like(p)
        ev = self.evaluate_population(p)
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            order = self.order(ev)
            food = p[order[0]]
            enemy = p[order[-1]]
            mean = p.mean(0)
            mean_step = step.mean(0)
            progress = min(self.iteration / max(self.config.max_iterations, 1), 1)
            w = 0.9 - 0.5 * progress
            s = 0.1 * (mean - p)
            a = 0.1 * mean_step
            c = 0.1 * (mean - p)
            f = 0.5 * (food - p)
            e = 0.1 * (p - enemy)
            step = np.clip(w * step + s + a + c + f + e, -0.2, 0.2)
            p = np.clip(p + step, 0, 1)
            ev = self.evaluate_population(p)
            self.record()
        return self.finalize(p, started=started)
