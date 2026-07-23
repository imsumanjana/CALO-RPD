"""Physical ORPD constraint audit with explicit, versioned engineering tolerances."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from calo_rpd_studio.power_system.case_model import *
from .constraint_violation import ConstraintViolation


CONSTRAINT_TOLERANCE_SCHEMA = "calo_rpd_constraint_tolerance_v5.9"


@dataclass(slots=True)
class ConstraintToleranceConfig:
    """Engineering/numerical tolerances used identically by all scientific backends.

    Tolerances suppress only numerically insignificant residue; they are not soft constraints.
    Values are persisted in experiment/config fingerprints in v5.9.
    """

    voltage_pu: float = 1e-7
    generator_p_mw: float = 1e-6
    generator_q_mvar: float = 1e-6
    branch_loading_percent: float = 1e-6
    branch_angle_deg: float = 1e-6
    feasibility_total: float = 1e-12
    schema_version: str = CONSTRAINT_TOLERANCE_SCHEMA

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "voltage_pu",
            "generator_p_mw",
            "generator_q_mvar",
            "branch_loading_percent",
            "branch_angle_deg",
            "feasibility_total",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Constraint tolerance {name} must be finite and non-negative")
        if str(self.schema_version) != CONSTRAINT_TOLERANCE_SCHEMA:
            raise ValueError(
                f"Unsupported constraint tolerance schema {self.schema_version!r}; "
                f"expected {CONSTRAINT_TOLERANCE_SCHEMA!r}"
            )


def _normalized_below(x, lo, span, *, absolute_tolerance=0.0):
    excess = lo - x
    excess = np.where(excess > float(absolute_tolerance), excess, 0.0)
    return np.maximum(excess, 0.0) / np.maximum(span, 1e-12)


def _normalized_above(x, hi, span, *, absolute_tolerance=0.0):
    excess = x - hi
    excess = np.where(excess > float(absolute_tolerance), excess, 0.0)
    return np.maximum(excess, 0.0) / np.maximum(span, 1e-12)


def generator_limit_violation(
    case, tolerances: ConstraintToleranceConfig | None = None
) -> tuple[float, float]:
    """Return normalized Q and P violations for every individual online generator."""
    tolerances = tolerances or ConstraintToleranceConfig()
    online = np.where(case.gen[:, GEN_STATUS] > 0)[0]
    if not online.size:
        return 0.0, 0.0
    g = case.gen[online]
    qspan = np.maximum(g[:, QMAX] - g[:, QMIN], 1.0)
    pspan = np.maximum(g[:, PMAX] - g[:, PMIN], 1.0)
    qv = np.sum(
        _normalized_below(
            g[:, QG], g[:, QMIN], qspan, absolute_tolerance=tolerances.generator_q_mvar
        )
        + _normalized_above(
            g[:, QG], g[:, QMAX], qspan, absolute_tolerance=tolerances.generator_q_mvar
        )
    )
    pv = np.sum(
        _normalized_below(
            g[:, PG], g[:, PMIN], pspan, absolute_tolerance=tolerances.generator_p_mw
        )
        + _normalized_above(
            g[:, PG], g[:, PMAX], pspan, absolute_tolerance=tolerances.generator_p_mw
        )
    )
    return float(qv), float(pv)


def branch_angle_limit_violation(
    case, va_deg: np.ndarray, tolerances: ConstraintToleranceConfig | None = None
) -> float:
    """Return normalized MATPOWER ANGMIN/ANGMAX violation over active constrained branches.

    MATPOWER defines the constrained quantity as angle(Vf) - angle(Vt) in degrees.  A bound is
    unbounded below when ANGMIN <= -360, unbounded above when ANGMAX >= 360, and a (0, 0) pair
    means unconstrained. Transformer phase shift affects the network equations but is not subtracted
    from this bus-voltage angle-difference definition.
    """
    tolerances = tolerances or ConstraintToleranceConfig()
    if case.branch.shape[1] <= ANGMAX:
        return 0.0
    idx = case.bus_index_map()
    total = 0.0
    for row in case.branch:
        if row[BR_STATUS] <= 0:
            continue
        lo = float(row[ANGMIN])
        hi = float(row[ANGMAX])
        if lo == 0.0 and hi == 0.0:
            continue
        f = idx[int(row[F_BUS])]
        t = idx[int(row[T_BUS])]
        delta = float(va_deg[f] - va_deg[t])
        lo_active = lo > -360.0
        hi_active = hi < 360.0
        # A 360-degree reference span avoids arbitrary inflation for one-sided limits.
        span = max((hi - lo) if lo_active and hi_active else 360.0, 1.0)
        if lo_active and lo - delta > tolerances.branch_angle_deg:
            total += (lo - delta) / span
        if hi_active and delta - hi > tolerances.branch_angle_deg:
            total += (delta - hi) / span
    return float(total)


def evaluate_constraints(pf, tolerances: ConstraintToleranceConfig | None = None):
    tolerances = tolerances or ConstraintToleranceConfig()
    tolerances.validate()
    if not pf.converged:
        return ConstraintViolation(float("inf"), {"power_flow": float("inf")})
    case = pf.case
    v = pf.vm_pu
    lo = case.bus[:, VMIN]
    hi = case.bus[:, VMAX]
    span = hi - lo
    vv = float(
        np.sum(
            _normalized_below(v, lo, span, absolute_tolerance=tolerances.voltage_pu)
            + _normalized_above(v, hi, span, absolute_tolerance=tolerances.voltage_pu)
        )
    )
    qv, pv = generator_limit_violation(case, tolerances)
    bv = 0.0
    if pf.branch is not None:
        rated = (case.branch[:, BR_STATUS] > 0) & (case.branch[:, RATE_A] > 0)
        overload = pf.branch.loading_percent[rated] - 100.0
        overload = np.where(overload > tolerances.branch_loading_percent, overload, 0.0)
        bv = float(np.sum(np.maximum(overload, 0.0) / 100.0))
    av = branch_angle_limit_violation(case, pf.va_deg, tolerances)
    comp = {
        "bus_voltage": vv,
        "generator_q": qv,
        "generator_p": pv,
        "branch_thermal": bv,
        "branch_angle": av,
        "power_flow": 0.0,
    }
    return ConstraintViolation(float(sum(comp.values())), comp)
