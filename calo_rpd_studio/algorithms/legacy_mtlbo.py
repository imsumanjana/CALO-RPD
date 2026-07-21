"""Legacy Gaussian MTLBO retained only for CALO ablation analysis."""

from __future__ import annotations
import time, numpy as np
from .base_optimizer import BaseOptimizer
from calo_rpd_studio.orpd.feasibility_rules import better


class LegacyMTLBOOptimizer(BaseOptimizer):
    name = "Legacy Gaussian MTLBO"

    def run(self):
        started = time.perf_counter()
        p = self.random_population()
        ev = self.evaluate_population(p)
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            best = p[self.order(ev)[0]]
            mean = p.mean(0)
            cand = np.clip(p + self.rng.normal(0, 1, p.shape) * (best - mean), 0, 1)
            ce = self.evaluate_population(cand)
            for i, e in enumerate(ce):
                if better(e, ev[i]):
                    p[i] = cand[i]
                    ev[i] = e
            self.record()
        return self.finalize(p, started=started)
