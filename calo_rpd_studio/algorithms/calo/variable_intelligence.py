"""Mixed-variable group intelligence for CALO v4."""
from __future__ import annotations

import numpy as np

GROUP_NAMES = ("generator_voltage", "transformer_tap", "shunt_compensation")
# statistics: attempts, successes, objective gain, feasibility gain, productive step
N_GROUP_STATS = 5


def infer_variable_groups(variables) -> np.ndarray:
    groups = []
    for variable in variables or []:
        name = str(getattr(variable, "name", "")).lower()
        if name.startswith("tap"):
            groups.append(1)
        elif "qsh" in name or "shunt" in name:
            groups.append(2)
        else:
            groups.append(0)
    return np.asarray(groups, dtype=np.int8)


class VariableGroupIntelligence:
    def __init__(self, variables, n_regimes: int = 4, decay: float = 0.90) -> None:
        self.variable_groups = infer_variable_groups(variables)
        self.n_groups = len(GROUP_NAMES)
        self.n_regimes = int(n_regimes)
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.stats = np.zeros((self.n_regimes, self.n_groups, N_GROUP_STATS), dtype=np.float64)
        self._counts = np.bincount(self.variable_groups, minlength=self.n_groups)

    def available(self) -> np.ndarray:
        return self._counts > 0

    def probabilities(self, regime: int) -> np.ndarray:
        available = self.available()
        if not np.any(available):
            return np.full(self.n_groups, 1.0 / self.n_groups)
        s = self.stats[int(regime)]
        attempts = s[:, 0]
        successes = s[:, 1]
        success_rate = np.divide(successes, attempts, out=np.zeros_like(successes), where=attempts > 0)
        objective = s[:, 2]
        feasibility = s[:, 3]
        # Priors are intentionally weak; online evidence should dominate after enough attempts.
        if int(regime) <= 1:
            score = 0.30 + 0.25 * success_rate + 0.15 * objective + 0.30 * feasibility
        else:
            score = 0.30 + 0.30 * success_rate + 0.35 * objective + 0.05 * feasibility
        score = np.where(available, np.maximum(score, 0.02), 0.0)
        total = score.sum()
        return score / total if total > 0 else available.astype(float) / np.count_nonzero(available)

    def choose(self, regime: int, rng: np.random.Generator, deterministic: bool = False) -> int:
        probabilities = self.probabilities(regime)
        return int(np.argmax(probabilities)) if deterministic else int(rng.choice(self.n_groups, p=probabilities))

    def mask(self, group: int, dimension: int) -> np.ndarray:
        if int(group) < 0 or self.variable_groups.shape != (dimension,):
            return np.ones(dimension, dtype=bool)
        mask = self.variable_groups == int(group)
        return mask if np.any(mask) else np.ones(dimension, dtype=bool)

    def batch_update(
        self,
        regime,
        groups,
        successful,
        objective_gain,
        feasibility_gain,
        step_norm,
    ) -> None:
        regimes = np.asarray(regime, dtype=int)
        groups = np.asarray(groups, dtype=int)
        successful = np.asarray(successful, dtype=bool)
        objective = np.asarray(objective_gain, dtype=float)
        feasibility = np.asarray(feasibility_gain, dtype=float)
        steps = np.asarray(step_norm, dtype=float)
        if regimes.ndim == 0:
            regimes = np.full(len(groups), int(regimes), dtype=int)
        n = min(len(regimes), len(groups), len(successful), len(objective), len(feasibility), len(steps))
        if n == 0:
            return
        regimes = np.clip(regimes[:n], 0, self.n_regimes - 1)
        objective_inf = np.isposinf(objective)
        feasibility_inf = np.isposinf(feasibility)
        objective = np.where(np.isfinite(objective) & (objective > 0), objective, 0.0)
        feasibility = np.where(np.isfinite(feasibility) & (feasibility > 0), feasibility, 0.0)
        steps = np.where(np.isfinite(steps) & (steps >= 0), steps, 0.0)
        obj_peak = max(float(np.max(objective[:n], initial=0.0)), 1e-12)
        feas_peak = max(float(np.max(feasibility[:n], initial=0.0)), 1e-12)
        objective_scaled = objective / obj_peak
        feasibility_scaled = feasibility / feas_peak
        objective_scaled[objective_inf] = 1.0
        feasibility_scaled[feasibility_inf] = 1.0
        groups = groups[:n]
        successful = successful[:n]
        for regime in range(self.n_regimes):
            mask_regime = regimes == regime
            if not np.any(mask_regime):
                continue
            for group in range(self.n_groups):
                mask = mask_regime & (groups == group)
                count = int(np.count_nonzero(mask))
                if not count:
                    continue
                success_mask = mask & successful
                target = np.asarray(
                    [
                        float(count),
                        float(np.count_nonzero(success_mask)),
                        float(np.mean(objective_scaled[:n][mask])),
                        float(np.mean(feasibility_scaled[:n][mask])),
                        float(np.mean(steps[:n][success_mask])) if np.any(success_mask) else 0.0,
                    ]
                )
                old = self.stats[regime, group]
                # Attempts/successes are cumulative evidence; gain/step statistics are EMA.
                old[0] += target[0]
                old[1] += target[1]
                old[2:] = self.decay * old[2:] + (1.0 - self.decay) * target[2:]
        self.stats = np.where(np.isfinite(self.stats), self.stats, 0.0)
