"""Recency-weighted online operator credit for CALO."""
from __future__ import annotations

import numpy as np


class OperatorCredit:
    def __init__(self, n_operators: int = 6, decay: float = 0.90, floor: float = 0.03) -> None:
        self.n_operators = int(n_operators)
        self.decay = float(decay)
        self.floor = float(floor)
        self.credit = np.full(self.n_operators, 1.0 / self.n_operators, dtype=float)
        self.attempts = np.zeros(self.n_operators, dtype=int)
        self.successes = np.zeros(self.n_operators, dtype=int)

    def update(self, operator: int, reward: float, successful: bool) -> None:
        operator = int(operator)
        self.attempts[operator] += 1
        if successful:
            self.successes[operator] += 1
        target = max(float(reward), 0.0) + (0.25 if successful else 0.0)
        self.credit *= self.decay
        self.credit[operator] += (1.0 - self.decay) * target
        self.credit = np.maximum(self.credit, self.floor)
        self.credit /= self.credit.sum()

    def probabilities(self) -> np.ndarray:
        values = np.maximum(self.credit, self.floor)
        return values / values.sum()

    def success_rates(self) -> np.ndarray:
        rates = np.zeros(self.n_operators, dtype=float)
        mask = self.attempts > 0
        rates[mask] = self.successes[mask] / self.attempts[mask]
        return rates


def blend_probabilities(ai_probabilities, credit_probabilities, alpha: float = 0.65) -> np.ndarray:
    ai = np.asarray(ai_probabilities, dtype=float)
    credit = np.asarray(credit_probabilities, dtype=float)
    if ai.shape != credit.shape:
        raise ValueError("AI and online-credit probability vectors must have the same shape")
    epsilon = 1e-8
    combined = np.power(ai + epsilon, float(alpha)) * np.power(
        credit + epsilon, 1.0 - float(alpha)
    )
    if not np.all(np.isfinite(combined)) or combined.sum() <= 0:
        return np.full(ai.shape, 1.0 / len(ai), dtype=float)
    return combined / combined.sum()
