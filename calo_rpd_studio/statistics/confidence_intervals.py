"""Mean confidence intervals."""

import numpy as np
from scipy import stats


def mean_confidence_interval(values, confidence=0.95):
    x = np.asarray(values, float)
    mean = float(np.mean(x))
    if len(x) < 2:
        return mean, mean
    half = float(stats.t.ppf((1 + confidence) / 2, len(x) - 1) * stats.sem(x))
    return mean - half, mean + half
