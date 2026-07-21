"""Firefly Algorithm."""

from __future__ import annotations
import time, numpy as np
from .base_optimizer import BaseOptimizer
from calo_rpd_studio.orpd.feasibility_rules import better


class FireflyOptimizer(BaseOptimizer):
    name = "FA"

    def run(self):
        started = time.perf_counter()
        p = self.random_population()
        ev = self.evaluate_population(p)
        alpha = float(self.config.parameters.get("alpha", 0.2))
        beta0 = float(self.config.parameters.get("beta0", 1.0))
        gamma = float(self.config.parameters.get("gamma", 1.0))
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            for i in range(len(p)):
                for j in range(len(p)):
                    if not self.can_evaluate():
                        break
                    if better(ev[j], ev[i]):
                        r2 = float(np.sum((p[i] - p[j]) ** 2))
                        beta = beta0 * np.exp(-gamma * r2)
                        x = np.clip(
                            p[i]
                            + beta * (p[j] - p[i])
                            + alpha * (self.rng.random(self.problem.dimension) - 0.5),
                            0,
                            1,
                        )
                        e = self.evaluate(x)
                        if e is not None and better(e, ev[i]):
                            p[i] = x
                            ev[i] = e
                if not self.can_evaluate():
                    break
            alpha *= 0.98
            self.record()
        return self.finalize(p, started=started)
