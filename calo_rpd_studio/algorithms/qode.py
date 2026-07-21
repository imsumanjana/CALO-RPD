"""Quasi-oppositional differential evolution."""

from __future__ import annotations
import time, numpy as np
from .base_optimizer import BaseOptimizer
from calo_rpd_studio.orpd.feasibility_rules import better


class QODEOptimizer(BaseOptimizer):
    name = "QODE"

    def run(self):
        started = time.perf_counter()
        n = self.config.population_size
        p = self.random_population()
        opp = 1 - p
        mid = 0.5
        lo = np.minimum(opp, mid)
        hi = np.maximum(opp, mid)
        quasi = self.rng.uniform(lo, hi)
        pool = np.vstack([p, quasi])
        ev = self.evaluate_population(pool)
        order = self.order(ev)[:n]
        p = pool[order]
        ev = [ev[i] for i in order]
        f = float(self.config.parameters.get("f", 0.5))
        cr = float(self.config.parameters.get("cr", 0.9))
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            for i in range(n):
                if not self.can_evaluate():
                    break
                ids = [j for j in range(n) if j != i]
                a, b, c = self.rng.choice(ids, 3, False)
                m = np.clip(p[a] + f * (p[b] - p[c]), 0, 1)
                mask = self.rng.random(self.problem.dimension) < cr
                mask[self.rng.integers(self.problem.dimension)] = True
                x = np.where(mask, m, p[i])
                e = self.evaluate(x)
                if e is not None and better(e, ev[i]):
                    p[i] = x
                    ev[i] = e
            self.record()
        return self.finalize(p, started=started)
