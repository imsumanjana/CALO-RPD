"""Average algorithm ranks across complete finite benchmark tasks."""
import numpy as np
from scipy.stats import rankdata


def average_ranks(matrix):
    x = np.asarray(matrix, float)
    if x.ndim != 2 or x.shape[1] == 0:
        return np.asarray([], dtype=float)
    complete = x[np.all(np.isfinite(x), axis=1)]
    if complete.size == 0:
        return np.full(x.shape[1], np.nan)
    return np.mean([rankdata(row, method="average") for row in complete], axis=0)
