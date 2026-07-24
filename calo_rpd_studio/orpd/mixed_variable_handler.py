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
    """Return the declared discrete lattice without ever crossing ``upper``.

    ``round((upper-lower)/step)`` can create one extra point when the interval is not an
    integer multiple of ``step`` (for example 0.9..1.1 by 0.03 produced 1.11).  The
    lattice is therefore floor-bounded with a small floating-point tolerance and every
    emitted value is explicitly clipped by the declared upper bound.
    """
    lower = float(lower)
    upper = float(upper)
    step = float(step)
    if not np.isfinite(lower) or not np.isfinite(upper) or not np.isfinite(step):
        raise ValueError("lower, upper and step must be finite")
    if step <= 0:
        raise ValueError("step must be positive")
    if upper < lower:
        raise ValueError("upper must be greater than or equal to lower")
    ratio = (upper - lower) / step
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(ratio))
    n = int(np.floor(ratio + tolerance))
    values = [float(lower + i * step) for i in range(n + 1)]
    bound_tol = 16.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
    return tuple(value for value in values if value <= upper + bound_tol)
