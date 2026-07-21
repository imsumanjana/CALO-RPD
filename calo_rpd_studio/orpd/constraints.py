"""Physical ORPD constraint audit."""

from __future__ import annotations
import numpy as np
from calo_rpd_studio.power_system.case_model import *
from .constraint_violation import ConstraintViolation


def _normalized_below(x, lo, span):
    return np.maximum(lo - x, 0) / np.maximum(span, 1e-12)


def _normalized_above(x, hi, span):
    return np.maximum(x - hi, 0) / np.maximum(span, 1e-12)


def evaluate_constraints(pf):
    if not pf.converged:
        return ConstraintViolation(float("inf"), {"power_flow": float("inf")})
    case = pf.case
    v = pf.vm_pu
    lo = case.bus[:, VMIN]
    hi = case.bus[:, VMAX]
    span = hi - lo
    vv = float(np.sum(_normalized_below(v, lo, span) + _normalized_above(v, hi, span)))
    # Recompute actual bus generation from solved network injections.
    from calo_rpd_studio.power_system.ybus import build_ybus

    inj = pf.voltage * np.conj(build_ybus(case).ybus @ pf.voltage) * case.base_mva
    actual_pg = inj.real + case.bus[:, PD]
    actual_qg = inj.imag + case.bus[:, QD]
    idx = case.bus_index_map()
    qv = 0.0
    pv = 0.0
    for busnum in case.bus[:, BUS_I].astype(int):
        gens = np.where(
            (case.gen[:, GEN_STATUS] > 0) & (case.gen[:, GEN_BUS].astype(int) == busnum)
        )[0]
        if not gens.size:
            continue
        bi = idx[busnum]
        qmin = float(case.gen[gens, QMIN].sum())
        qmax = float(case.gen[gens, QMAX].sum())
        qspan = max(qmax - qmin, 1.0)
        qv += max(qmin - actual_qg[bi], 0) / qspan + max(actual_qg[bi] - qmax, 0) / qspan
        pmin = float(case.gen[gens, PMIN].sum())
        pmax = float(case.gen[gens, PMAX].sum())
        pspan = max(pmax - pmin, 1.0)
        pv += max(pmin - actual_pg[bi], 0) / pspan + max(actual_pg[bi] - pmax, 0) / pspan
    bv = 0.0
    if pf.branch is not None:
        rated = case.branch[:, RATE_A] > 0
        bv = float(np.sum(np.maximum(pf.branch.loading_percent[rated] - 100, 0) / 100))
    comp = {
        "bus_voltage": vv,
        "generator_q": float(qv),
        "generator_p": float(pv),
        "branch_thermal": bv,
        "power_flow": 0.0,
    }
    return ConstraintViolation(float(sum(comp.values())), comp)
