"""ACOR: archive-based continuous ant colony optimization."""

from __future__ import annotations
import time, numpy as np
from .base_optimizer import BaseOptimizer


class AntColonyContinuousOptimizer(BaseOptimizer):
    name = "ACO"

    def run(self):
        started = time.perf_counter()
        n = self.config.population_size
        archive = self.random_population()
        ev = self.evaluate_population(archive)
        q = float(self.config.parameters.get("q", 0.5))
        xi = float(self.config.parameters.get("xi", 0.85))
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            order = self.order(ev)
            archive = archive[order]
            ev = [ev[i] for i in order]
            ranks = np.arange(n)
            weights = np.exp(-(ranks**2) / (2 * (q * n) ** 2))
            weights /= weights.sum()
            new = []
            for _ in range(n):
                if not self.can_evaluate(len(new) + 1):
                    break
                k = int(self.rng.choice(n, p=weights))
                sigma = xi * np.mean(np.abs(archive[k] - archive), axis=0)
                new.append(np.clip(self.rng.normal(archive[k], sigma + 1e-12), 0, 1))
            if not new:
                break
            ne = self.evaluate_population(np.asarray(new))
            combo = np.vstack([archive, np.asarray(new)])
            all_ev = ev + ne
            order = sorted(
                range(len(all_ev)),
                key=lambda i: (
                    0 if all_ev[i].feasible else 1,
                    all_ev[i].value if all_ev[i].feasible else all_ev[i].violation,
                ),
            )[:n]
            archive = combo[order]
            ev = [all_ev[i] for i in order]
            self.record()
        return self.finalize(archive, started=started)
