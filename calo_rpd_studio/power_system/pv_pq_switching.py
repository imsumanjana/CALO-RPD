"""Generator reactive-limit aggregation and allocation."""

from __future__ import annotations
import numpy as np
from .case_model import *


def online_generators_at_bus(case, bus_number):
    return np.where(
        (case.gen[:, GEN_STATUS] > 0) & (case.gen[:, GEN_BUS].astype(int) == int(bus_number))
    )[0]


def aggregate_q_limits(case, bus_number):
    g = online_generators_at_bus(case, bus_number)
    return (
        (float(np.sum(case.gen[g, QMIN])), float(np.sum(case.gen[g, QMAX])))
        if g.size
        else (0.0, 0.0)
    )


def distribute_reactive_power(case, bus_number, required_q, *, clip_to_limits=True):
    """Allocate aggregate reactive output across online generators at one bus.

    For ordinary PV/PQ reporting and PV->PQ switching the aggregate value is clipped to
    the declared generator-Q capability.  A reference/slack bus is different: the solved
    network may require Q outside the declared capability while the bus must remain REF to
    keep the Newton system well posed.  In that case ``clip_to_limits=False`` preserves the
    *actual solved aggregate Q* in the reported generator state, allowing the ORPD constraint
    layer and independent validators to expose the violation instead of silently hiding it.
    """
    g = online_generators_at_bus(case, bus_number)
    if not g.size:
        return
    required_q = float(required_q)
    qmin = case.gen[g, QMIN]
    qmax = case.gen[g, QMAX]
    span = np.maximum(qmax - qmin, 0)
    total = float(span.sum())
    bounded = float(np.clip(required_q, float(qmin.sum()), float(qmax.sum())))
    if total > 0:
        q = qmin + (bounded - float(qmin.sum())) * span / total
    else:
        q = np.full(g.size, bounded / g.size)
    if not clip_to_limits:
        # Preserve the exact aggregate network requirement.  Any residual beyond aggregate
        # capability is placed on the first online generator (the slack participant) so the
        # violation remains explicit in QG rather than being silently clipped.
        q[0] += required_q - float(np.sum(q))
    case.gen[g, QG] = q
