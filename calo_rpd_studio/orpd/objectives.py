"""Modular ORPD objectives with a fixed, pre-solve mathematical bus partition."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np
from calo_rpd_studio.power_system.case_model import BUS_TYPE, PQ
from calo_rpd_studio.power_system.voltage_stability import kessel_glavitsch_l_index


class ObjectiveKind(str, Enum):
    ACTIVE_POWER_LOSS = "active_power_loss"
    VOLTAGE_DEVIATION = "voltage_deviation"
    L_INDEX = "l_index"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass(slots=True)
class ObjectiveConfig:
    kind: ObjectiveKind = ObjectiveKind.ACTIVE_POWER_LOSS
    weight_loss: float = 1.0
    weight_voltage_deviation: float = 0.0
    weight_l_index: float = 0.0
    loss_scale: float = 1.0
    voltage_deviation_scale: float = 1.0
    l_index_scale: float = 1.0


@dataclass(slots=True)
class ObjectiveResult:
    value: float
    components: dict[str, float]


def calculate_objective(pf, config: ObjectiveConfig, *, formulation_case=None):
    """Calculate the objective using the declared pre-solve formulation partition.

    Dynamic PV->PQ switching is a numerical feasibility mechanism and must not redefine which
    buses belong to the voltage-deviation or L-index objective from candidate to candidate.
    ``formulation_case`` is therefore the controlled/scenario case *before* Q-limit switching.
    """
    if not pf.converged:
        return ObjectiveResult(
            float("inf"),
            {
                "active_power_loss_mw": float("inf"),
                "voltage_deviation_pu": float("inf"),
                "l_index_max": float("inf"),
            },
        )
    reference = formulation_case if formulation_case is not None else pf.case
    pq = np.where(reference.bus[:, BUS_TYPE].astype(int) == PQ)[0]
    loss = float(pf.total_loss_mw)
    vd = float(np.sum(np.abs(pf.vm_pu[pq] - 1.0)))
    li = kessel_glavitsch_l_index(
        pf.case, pf.voltage, partition_case=reference
    ).maximum
    c = {"active_power_loss_mw": loss, "voltage_deviation_pu": vd, "l_index_max": float(li)}
    if config.kind is ObjectiveKind.ACTIVE_POWER_LOSS:
        v = loss
    elif config.kind is ObjectiveKind.VOLTAGE_DEVIATION:
        v = vd
    elif config.kind is ObjectiveKind.L_INDEX:
        v = li
    else:
        v = (
            config.weight_loss * loss / max(config.loss_scale, 1e-15)
            + config.weight_voltage_deviation * vd / max(config.voltage_deviation_scale, 1e-15)
            + config.weight_l_index * li / max(config.l_index_scale, 1e-15)
        )
    return ObjectiveResult(float(v), c)
