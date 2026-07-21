"""Descriptive statistics for repeated stochastic runs."""

from __future__ import annotations
import numpy as np
from scipy import stats


def descriptive_statistics(values, confidence=0.95):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    sem = std / np.sqrt(len(x)) if len(x) > 1 else 0.0
    multiplier = stats.t.ppf((1 + confidence) / 2, len(x) - 1) if len(x) > 1 else 0.0
    return {
        "count": int(len(x)),
        "best": float(np.min(x)),
        "mean": mean,
        "median": float(np.median(x)),
        "worst": float(np.max(x)),
        "std": std,
        "variance": float(std**2),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "coefficient_of_variation": float(std / abs(mean)) if mean else float("nan"),
        "confidence_low": float(mean - multiplier * sem),
        "confidence_high": float(mean + multiplier * sem),
    }
