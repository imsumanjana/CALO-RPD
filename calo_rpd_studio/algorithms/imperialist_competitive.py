"""Imperialist Competitive Algorithm for bounded continuous search."""

from __future__ import annotations
import time, numpy as np
from .base_optimizer import BaseOptimizer
from calo_rpd_studio.orpd.feasibility_rules import better


class ImperialistCompetitiveOptimizer(BaseOptimizer):
    name = "ICA"

    def run(self):
        started = time.perf_counter()
        p = self.random_population()
        ev = self.evaluate_population(p)
        n_imp = max(
            1, min(int(self.config.parameters.get("imperialists", max(1, len(p) // 5))), len(p) - 1)
        )
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            order = self.order(ev)
            imperials = order[:n_imp]
            colonies = order[n_imp:]
            for ci in colonies:
                if not self.can_evaluate():
                    break
                imp = int(self.rng.choice(imperials))
                x = np.clip(
                    p[ci]
                    + self.rng.uniform(0, 2) * (p[imp] - p[ci])
                    + self.rng.normal(0, 0.02, self.problem.dimension),
                    0,
                    1,
                )
                e = self.evaluate(x)
                if e is not None and better(e, ev[ci]):
                    p[ci] = x
                    ev[ci] = e
            self.record()
        return self.finalize(p, started=started)
