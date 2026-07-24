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


def _safe_normalization_span(span, *, absolute_tolerance=0.0):
    """Return a stable normalization scale for bounded engineering quantities.

    A zero/near-zero declared span represents a fixed target, not a request to amplify
    numerical residue by 1e12.  Spans no wider than twice the numerical/engineering
    tolerance therefore use a 1.0 absolute engineering-unit scale; ordinary bounded
    ranges retain their declared span.
    """
    span = np.asarray(span, dtype=float)
    threshold = max(2.0 * float(absolute_tolerance), 1e-12)
    return np.where(span > threshold, span, 1.0)


def _normalized_below(x, lo, span, *, absolute_tolerance=0.0):
    excess = lo - x
    excess = np.where(excess > float(absolute_tolerance), excess, 0.0)
    return np.maximum(excess, 0.0) / _safe_normalization_span(
        span, absolute_tolerance=absolute_tolerance
    )


def _normalized_above(x, hi, span, *, absolute_tolerance=0.0):
    excess = x - hi
    excess = np.where(excess > float(absolute_tolerance), excess, 0.0)
    return np.maximum(excess, 0.0) / _safe_normalization_span(
        span, absolute_tolerance=absolute_tolerance
    )


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
    branch = np.asarray(case.branch, dtype=float)
    if branch.size == 0:
        return 0.0
    active = branch[:, BR_STATUS] > 0
    lo = branch[:, ANGMIN]
    hi = branch[:, ANGMAX]
    constrained = ~((lo == 0.0) & (hi == 0.0))
    mask = active & constrained
    if not np.any(mask):
        return 0.0
    rows = branch[mask]
    # Fully vectorized bus-number -> row-index mapping. Avoid a Python dictionary lookup loop for
    # every constrained branch in every objective evaluation.
    bus_numbers = np.asarray(case.bus[:, BUS_I], dtype=np.int64)
    order = np.argsort(bus_numbers, kind="stable")
    sorted_bus_numbers = bus_numbers[order]
    f_numbers = np.asarray(rows[:, F_BUS], dtype=np.int64)
    t_numbers = np.asarray(rows[:, T_BUS], dtype=np.int64)
    f_pos = np.searchsorted(sorted_bus_numbers, f_numbers)
    t_pos = np.searchsorted(sorted_bus_numbers, t_numbers)
    if (
        np.any(f_pos >= len(sorted_bus_numbers))
        or np.any(t_pos >= len(sorted_bus_numbers))
        or not np.array_equal(sorted_bus_numbers[f_pos], f_numbers)
        or not np.array_equal(sorted_bus_numbers[t_pos], t_numbers)
    ):
        raise ValueError("Branch angle constraints reference an unknown bus number")
    f = order[f_pos]
    t = order[t_pos]
    delta = np.asarray(va_deg, dtype=float)[f] - np.asarray(va_deg, dtype=float)[t]
    lo = rows[:, ANGMIN]
    hi = rows[:, ANGMAX]
    lo_active = lo > -360.0
    hi_active = hi < 360.0
    spans = np.where(lo_active & hi_active, hi - lo, 360.0)
    spans = np.maximum(spans, 1.0)
    low_excess = np.where(
        lo_active & ((lo - delta) > float(tolerances.branch_angle_deg)),
        lo - delta,
        0.0,
    )
    high_excess = np.where(
        hi_active & ((delta - hi) > float(tolerances.branch_angle_deg)),
        delta - hi,
        0.0,
    )
    return float(np.sum((low_excess + high_excess) / spans))


def evaluate_constraints(pf, tolerances: ConstraintToleranceConfig | None = None):
    tolerances = tolerances or ConstraintToleranceConfig()
    tolerances.validate()
    if not pf.converged:
        return ConstraintViolation(float("inf"), {"power_flow": float("inf")}, tolerances.feasibility_total)
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
    return ConstraintViolation(float(sum(comp.values())), comp, tolerances.feasibility_total)
