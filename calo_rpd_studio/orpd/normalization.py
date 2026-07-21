"""Robust finite normalization helpers."""

from __future__ import annotations
import numpy as np


def safe_normalize(values):
    x = np.asarray(values, dtype=float)
    finite = x[np.isfinite(x)]
    if not finite.size:
        return np.zeros_like(x)
    lo = float(finite.min())
    hi = float(finite.max())
    return np.where(np.isfinite(x), (x - lo) / max(hi - lo, 1e-15), 1.0)
