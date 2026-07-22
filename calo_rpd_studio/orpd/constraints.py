"""Physical ORPD constraint audit with per-generator P/Q capability enforcement."""

from __future__ import annotations
import numpy as np
from calo_rpd_studio.power_system.case_model import *
from .constraint_violation import ConstraintViolation


def _normalized_below(x, lo, span, *, absolute_tolerance=0.0):
    excess = lo - x
    excess = np.where(excess > float(absolute_tolerance), excess, 0.0)
    return np.maximum(excess, 0.0) / np.maximum(span, 1e-12)


def _normalized_above(x, hi, span, *, absolute_tolerance=0.0):
    excess = x - hi
    excess = np.where(excess > float(absolute_tolerance), excess, 0.0)
    return np.maximum(excess, 0.0) / np.maximum(span, 1e-12)


def generator_limit_violation(case) -> tuple[float, float]:
    """Return normalized Q and P violations for every individual online generator.

    The solved power-flow output matrix contains the actual distributed QG for every online unit
    and the actual slack PG for the participating reference generator.  Checking each row avoids
    hiding a unit violation behind unused capability of a co-located generator.
    """
    online = np.where(case.gen[:, GEN_STATUS] > 0)[0]
    if not online.size:
        return 0.0, 0.0
    g = case.gen[online]
    qspan = np.maximum(g[:, QMAX] - g[:, QMIN], 1.0)
    pspan = np.maximum(g[:, PMAX] - g[:, PMIN], 1.0)
    # Ignore only sub-micro-unit numerical residue at an enforced capability boundary.  The
    # 1e-6 engineering-unit tolerance matches the default Q-limit switching tolerance and is far
    # below any material ORPD capability violation; it also keeps CPU/accelerator feasibility
    # classification invariant to floating-point last-bit differences.
    limit_tolerance = 1e-6
    qv = np.sum(
        _normalized_below(g[:, QG], g[:, QMIN], qspan, absolute_tolerance=limit_tolerance)
        + _normalized_above(g[:, QG], g[:, QMAX], qspan, absolute_tolerance=limit_tolerance)
    )
    pv = np.sum(
        _normalized_below(g[:, PG], g[:, PMIN], pspan, absolute_tolerance=limit_tolerance)
        + _normalized_above(g[:, PG], g[:, PMAX], pspan, absolute_tolerance=limit_tolerance)
    )
    return float(qv), float(pv)


def evaluate_constraints(pf):
    if not pf.converged:
        return ConstraintViolation(float("inf"), {"power_flow": float("inf")})
    case = pf.case
    v = pf.vm_pu
    lo = case.bus[:, VMIN]
    hi = case.bus[:, VMAX]
    span = hi - lo
    vv = float(np.sum(_normalized_below(v, lo, span) + _normalized_above(v, hi, span)))
    qv, pv = generator_limit_violation(case)
    bv = 0.0
    if pf.branch is not None:
        rated = case.branch[:, RATE_A] > 0
        bv = float(np.sum(np.maximum(pf.branch.loading_percent[rated] - 100, 0) / 100))
    comp = {
        "bus_voltage": vv,
        "generator_q": qv,
        "generator_p": pv,
        "branch_thermal": bv,
        "power_flow": 0.0,
    }
    return ConstraintViolation(float(sum(comp.values())), comp)
