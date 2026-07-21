"""Elite-preserving controlled recovery."""

from __future__ import annotations
import numpy as np


def recovery_indices(evaluations, fraction, order_fn):
    order = order_fn(evaluations)
    count = max(1, min(len(order) - 1, int(round(len(order) * fraction))))
    return np.asarray(order[-count:], dtype=int), np.asarray(
        order[: max(1, len(order) // 10)], dtype=int
    )
