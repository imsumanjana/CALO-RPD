"""Robust mean confidence intervals."""
import numpy as np
from scipy import stats


def mean_confidence_interval(values, confidence=0.95):
    x = np.asarray(values, float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(x))
    if x.size < 2:
        return mean, mean
    half = float(stats.t.ppf((1 + confidence) / 2, x.size - 1) * stats.sem(x, nan_policy="omit"))
    return mean - half, mean + half
