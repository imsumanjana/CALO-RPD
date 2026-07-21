"""Average algorithm ranks across benchmark tasks."""

import numpy as np
from scipy.stats import rankdata


def average_ranks(matrix):
    x = np.asarray(matrix, float)
    return np.mean([rankdata(row, method="average") for row in x], axis=0)
