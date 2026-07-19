"""Recency-weighted online operator credit for CALO."""
from __future__ import annotations

import numpy as np


_MAX_SAFE_REWARD = np.finfo(float).max / 1024.0


def _safe_nonnegative_reward(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if np.isnan(number) or number <= 0.0 or np.isneginf(number):
        return 0.0
    if np.isposinf(number):
        return _MAX_SAFE_REWARD
    return min(number, _MAX_SAFE_REWARD)


def _normalise_positive(values: np.ndarray, floor: float) -> np.ndarray:
    """Return a finite probability-like positive vector without overflow."""
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.maximum(values, float(floor))
    peak = float(np.max(values, initial=0.0))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.full(values.shape, 1.0 / max(values.size, 1), dtype=float)
    scaled = values / peak
    total = float(np.sum(scaled))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(values.shape, 1.0 / max(values.size, 1), dtype=float)
    return scaled / total


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
        if not 0 <= operator < self.n_operators:
            raise IndexError(f"Operator index {operator} outside [0, {self.n_operators})")
        self.attempts[operator] += 1
        if successful:
            self.successes[operator] += 1
        target = _safe_nonnegative_reward(reward) + (0.25 if successful else 0.0)
        decay = self.decay if np.isfinite(self.decay) else 0.90
        decay = min(max(decay, 0.0), 1.0)
        self.credit = np.where(np.isfinite(self.credit), self.credit, self.floor)
        self.credit *= decay
        self.credit[operator] += (1.0 - decay) * target
        self.credit = _normalise_positive(self.credit, self.floor)

    def probabilities(self) -> np.ndarray:
        return _normalise_positive(self.credit, self.floor)

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
    if ai.size == 0:
        return ai.copy()
    ai = np.where(np.isfinite(ai) & (ai >= 0.0), ai, 0.0)
    credit = np.where(np.isfinite(credit) & (credit >= 0.0), credit, 0.0)
    epsilon = 1e-8
    alpha = float(np.clip(alpha, 0.0, 1.0)) if np.isfinite(alpha) else 0.65
    combined = np.power(ai + epsilon, alpha) * np.power(credit + epsilon, 1.0 - alpha)
    if not np.all(np.isfinite(combined)) or combined.sum() <= 0:
        return np.full(ai.shape, 1.0 / len(ai), dtype=float)
    return combined / combined.sum()
