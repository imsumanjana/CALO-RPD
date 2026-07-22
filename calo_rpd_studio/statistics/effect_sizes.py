"""Robust Cliff's delta nonparametric effect size."""
import numpy as np


def cliffs_delta(a, b):
    a = np.asarray(a, dtype=float).ravel(); b = np.asarray(b, dtype=float).ravel()
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    greater = int(np.count_nonzero(diff > 0)); less = int(np.count_nonzero(diff < 0))
    return float((greater - less) / (a.size * b.size))
