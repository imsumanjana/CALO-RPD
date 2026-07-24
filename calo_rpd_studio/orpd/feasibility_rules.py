"""Deb-style feasibility-first comparisons with one tolerance policy."""

from __future__ import annotations


DEFAULT_FEASIBILITY_TOLERANCE = 1e-12


def _tolerance(evaluation, fallback: float = DEFAULT_FEASIBILITY_TOLERANCE) -> float:
    value = getattr(evaluation, "feasibility_tolerance", fallback)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(fallback)
    return max(0.0, value)


def _is_feasible(evaluation, tol: float) -> bool:
    violation = float(getattr(evaluation, "violation", float("inf")))
    return violation <= float(tol)


def sort_key(e, tol: float | None = None):
    """Return the canonical total ordering used by *all* generic optimizers.

    The configured feasibility tolerance is carried by each Evaluation and is used only for the
    feasibility boundary. Once classified, infeasible candidates are ordered by exact aggregate
    violation and feasible candidates by objective, giving one deterministic transitive ordering
    for both pairwise comparisons and bulk sorting.
    """
    tolerance = _tolerance(e) if tol is None else max(0.0, float(tol))
    feasible = _is_feasible(e, tolerance)
    value = float(getattr(e, "value", float("inf")))
    violation = float(getattr(e, "violation", float("inf")))
    if feasible:
        return (0, value, violation)
    return (1, violation, value)


def better(a, b, tol: float | None = None):
    if b is None:
        return True
    if tol is None:
        tolerance = max(_tolerance(a), _tolerance(b))
    else:
        tolerance = max(0.0, float(tol))
    return sort_key(a, tolerance) < sort_key(b, tolerance)
