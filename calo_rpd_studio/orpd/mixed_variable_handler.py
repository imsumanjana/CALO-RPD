"""Scale-independent normalized mixed-variable decoding."""

from __future__ import annotations
import numpy as np


def decode_continuous(z, lower, upper):
    return float(lower + np.clip(z, 0, 1) * (upper - lower))


def decode_discrete(z, values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Discrete value set cannot be empty")
    i = min(int(np.floor(np.clip(z, 0, 1) * values.size)), values.size - 1)
    return float(values[i])


def stepped_values(lower, upper, step):
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(round((upper - lower) / step))
    return tuple(float(lower + i * step) for i in range(n + 1))
