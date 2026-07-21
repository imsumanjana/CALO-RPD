"""Exact weighted empirical conditional value-at-risk utilities.

The upper-tail CVaR is evaluated as the quantile integral over ``[alpha, 1]``.
This matters for discrete scenario distributions because the VaR cut can pass
through an atom of probability mass.  In that case only the fractional part of
that atom that lies in the tail is included.
"""

from __future__ import annotations

import numpy as np


def _validated_numpy_inputs(values, weights, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    alpha = float(alpha)
    if v.ndim != 1 or w.ndim != 1 or v.shape != w.shape:
        raise ValueError("values and weights must be one-dimensional arrays with identical shape")
    if v.size == 0:
        raise ValueError("at least one scenario value is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights must have a positive total")
    return v, w / total, alpha


def weighted_cvar(values, weights, alpha: float = 0.95) -> float:
    """Return exact upper-tail weighted empirical CVaR.

    Example: values ``[0, 100]``, weights ``[0.96, 0.04]`` and ``alpha=0.95``
    produce ``80`` because the worst 5% tail contains all 4% mass at 100 and
    1% of the mass at 0.
    """

    v, w, alpha = _validated_numpy_inputs(values, weights, alpha)
    order = np.argsort(v, kind="stable")
    sorted_v = v[order]
    sorted_w = w[order]
    cdf = np.cumsum(sorted_w)
    previous = cdf - sorted_w
    overlap = np.maximum(0.0, cdf - np.maximum(previous, alpha))
    tail_mass = float(np.sum(overlap))
    if tail_mass <= np.finfo(float).eps:
        raise RuntimeError("CVaR tail mass is numerically zero")
    return float(np.sum(sorted_v * overlap) / tail_mass)


def weighted_cvar_torch(values, weights, alpha: float = 0.95):
    """Torch implementation supporting vectors or batches along the last axis."""

    import torch

    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    v = torch.as_tensor(values)
    w = torch.as_tensor(weights, dtype=v.dtype, device=v.device)
    if v.ndim < 1:
        raise ValueError("values must have at least one dimension")
    if w.ndim == 1 and v.ndim > 1:
        w = w.expand_as(v)
    if w.shape != v.shape:
        raise ValueError("weights must match values or be a one-dimensional last-axis vector")
    if not bool(torch.all(torch.isfinite(v))):
        raise ValueError("values must be finite")
    if not bool(torch.all(torch.isfinite(w))) or bool(torch.any(w < 0)):
        raise ValueError("weights must be finite and non-negative")
    totals = torch.sum(w, dim=-1, keepdim=True)
    if bool(torch.any(totals <= 0)):
        raise ValueError("weights must have a positive total")
    normalized = w / totals
    sorted_v, order = torch.sort(v, dim=-1, stable=True)
    sorted_w = torch.gather(normalized, -1, order)
    cdf = torch.cumsum(sorted_w, dim=-1)
    previous = cdf - sorted_w
    alpha_tensor = torch.as_tensor(alpha, dtype=v.dtype, device=v.device)
    overlap = torch.clamp(cdf - torch.maximum(previous, alpha_tensor), min=0.0)
    tail_mass = torch.sum(overlap, dim=-1)
    eps = torch.finfo(v.dtype).eps
    if bool(torch.any(tail_mass <= eps)):
        raise RuntimeError("CVaR tail mass is numerically zero")
    return torch.sum(sorted_v * overlap, dim=-1) / tail_mass
