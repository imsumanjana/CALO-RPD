"""Modular ORPD objectives with fixed pre-solve partitions and fail-fast configuration."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
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
    """Frozen scientific definition of the scalar ORPD objective.

    v5.9 validates at the scientific object boundary, not only in GUI widgets.  Invalid
    weights/scales therefore cannot silently reach an experiment loaded from YAML/JSON or a script.
    """

    kind: ObjectiveKind = ObjectiveKind.ACTIVE_POWER_LOSS
    weight_loss: float = 1.0
    weight_voltage_deviation: float = 0.0
    weight_l_index: float = 0.0
    loss_scale: float = 1.0
    voltage_deviation_scale: float = 1.0
    l_index_scale: float = 1.0

    def __post_init__(self) -> None:
        self.kind = ObjectiveKind(self.kind)
        self.validate()

    def validate(self) -> None:
        weights = {
            "weight_loss": self.weight_loss,
            "weight_voltage_deviation": self.weight_voltage_deviation,
            "weight_l_index": self.weight_l_index,
        }
        scales = {
            "loss_scale": self.loss_scale,
            "voltage_deviation_scale": self.voltage_deviation_scale,
            "l_index_scale": self.l_index_scale,
        }
        for name, value in weights.items():
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in scales.items():
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if self.kind is ObjectiveKind.MULTI_OBJECTIVE and sum(float(v) for v in weights.values()) <= 0.0:
            raise ValueError("Multi-objective ORPD requires at least one strictly positive objective weight")


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
    config.validate()
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
    li = kessel_glavitsch_l_index(pf.case, pf.voltage, partition_case=reference).maximum
    c = {"active_power_loss_mw": loss, "voltage_deviation_pu": vd, "l_index_max": float(li)}
    if config.kind is ObjectiveKind.ACTIVE_POWER_LOSS:
        v = loss
    elif config.kind is ObjectiveKind.VOLTAGE_DEVIATION:
        v = vd
    elif config.kind is ObjectiveKind.L_INDEX:
        v = li
    else:
        v = (
            config.weight_loss * loss / config.loss_scale
            + config.weight_voltage_deviation * vd / config.voltage_deviation_scale
            + config.weight_l_index * li / config.l_index_scale
        )
    return ObjectiveResult(float(v), c)
