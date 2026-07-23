"""Common normalized ORPD variable encoder/decoder used by every optimizer.

Version 3.4 replaces implicit generic shunt bounds with explicit, case-specific
formulation profiles.  Fixed reactors and undeclared shunts remain untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal
import math

import numpy as np

from calo_rpd_studio.power_system.case_model import (
    BR_STATUS,
    BS,
    BUS_I,
    BUS_TYPE,
    GEN_BUS,
    GEN_STATUS,
    PV,
    REF,
    TAP,
    T_BUS,
    F_BUS,
    VG,
    VM,
    VMAX,
    VMIN,
)

from .decision_variables import DecisionVariable, VariableKind
from .mixed_variable_handler import decode_continuous, decode_discrete, stepped_values

FORMULATION_PROFILE_VERSION = "ieee-orpd-controls-v3.4.0"


@dataclass(frozen=True, slots=True)
class ShuntControlDefinition:
    """Declared controllable shunt in MVAr at 1 p.u.

    ``semantics='absolute'`` writes the decoded MVAr value to BUS.BS.
    ``semantics='delta_from_base'`` adds the decoded value to the case's original
    BUS.BS value.  The semantics are explicit in configuration and manifests.
    """

    bus_number: int
    minimum_mvar: float
    maximum_mvar: float
    step_mvar: float = 1.0
    semantics: Literal["absolute", "delta_from_base"] = "absolute"
    source: str = "user-defined"

    def validate(self) -> None:
        values = (float(self.minimum_mvar), float(self.maximum_mvar), float(self.step_mvar))
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Shunt control at bus {self.bus_number} must use finite bounds/step")
        if self.minimum_mvar > self.maximum_mvar:
            raise ValueError(f"Shunt control at bus {self.bus_number} has inverted bounds")
        if self.step_mvar <= 0:
            raise ValueError(f"Shunt control at bus {self.bus_number} requires a positive step")
        if self.semantics not in {"absolute", "delta_from_base"}:
            raise ValueError(f"Unsupported shunt semantics: {self.semantics}")


@dataclass(slots=True)
class ORPDVariableConfig:
    generator_voltages: bool = True
    transformer_taps: bool = True
    shunt_compensation: bool = True
    discrete_transformer_taps: bool = True
    discrete_shunts: bool = True
    transformer_minimum: float = 0.90
    transformer_maximum: float = 1.10
    transformer_step: float = 0.0125
    shunt_controls: tuple[ShuntControlDefinition, ...] = ()
    formulation_profile: str = FORMULATION_PROFILE_VERSION

    def __post_init__(self) -> None:
        self.shunt_controls = tuple(self.shunt_controls or ())
        self.validate()

    def validate(self) -> None:
        lo = float(self.transformer_minimum)
        hi = float(self.transformer_maximum)
        step = float(self.transformer_step)
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0.0 or hi <= lo:
            raise ValueError(
                "Transformer tap bounds must be finite, strictly positive, and maximum must exceed minimum"
            )
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("Transformer tap step must be finite and strictly positive")
        if not str(self.formulation_profile).strip():
            raise ValueError("ORPD formulation_profile must be non-empty")
        seen: set[int] = set()
        for control in self.shunt_controls:
            if not isinstance(control, ShuntControlDefinition):
                raise TypeError("shunt_controls must contain ShuntControlDefinition objects")
            control.validate()
            bus = int(control.bus_number)
            if bus in seen:
                raise ValueError(f"Duplicate shunt control declaration at bus {bus}")
            seen.add(bus)


def _case_bus_bs(case) -> dict[int, float]:
    return {int(row[BUS_I]): float(row[BS]) for row in case.bus}


def default_shunt_controls(case) -> tuple[ShuntControlDefinition, ...]:
    """Return explicit, sign-safe controls for bundled IEEE cases.

    The case57/case118 controls are bounded by their original positive
    compensation.  Negative reactors in case118 are intentionally fixed.  The
    classical case30 candidate buses retain the commonly used 0--5 MVAr lattice.
    """

    buses = set(case.bus[:, BUS_I].astype(int))
    base = _case_bus_bs(case)
    if case.name == "case30":
        candidates = (10, 12, 15, 17, 20, 21, 23, 24)
        return tuple(
            ShuntControlDefinition(
                bus,
                0.0,
                5.0,
                1.0,
                "absolute",
                "CALO-RPD explicit case30 profile; classical 0-5 MVAr control lattice",
            )
            for bus in candidates
            if bus in buses
        )
    if case.name == "case57":
        candidates = (18, 25, 53)
        return tuple(
            ShuntControlDefinition(
                bus,
                0.0,
                max(0.0, base[bus]),
                0.1,
                "absolute",
                "PYPOWER case57 positive shunt rating; zero-to-original absolute control",
            )
            for bus in candidates
            if bus in buses and base[bus] > 0.0
        )
    if case.name == "case118":
        candidates = (34, 44, 45, 46, 48, 74, 79, 82, 83, 105, 107, 110)
        return tuple(
            ShuntControlDefinition(
                bus,
                0.0,
                max(0.0, base[bus]),
                1.0,
                "absolute",
                "PYPOWER case118 positive shunt rating; fixed negative reactors excluded",
            )
            for bus in candidates
            if bus in buses and base[bus] > 0.0
        )
    return ()


class ORPDVariableDecoder:
    def __init__(self, case, config: ORPDVariableConfig):
        self.case = case
        self.config = config
        self.variables: list[DecisionVariable] = []
        self._actions: list[tuple[str, int, float, float, tuple[float, ...] | None]] = []
        self._shunt_definitions: list[ShuntControlDefinition] = []
        index = case.bus_index_map()

        if config.generator_voltages:
            online = np.where(case.gen[:, GEN_STATUS] > 0)[0]
            seen: set[int] = set()
            for generator_index in online:
                bus = int(case.gen[generator_index, GEN_BUS])
                if bus in seen:
                    continue
                bus_index = index[bus]
                # Generator voltage set-points are physical ORPD controls only on voltage-controlled
                # REF/PV buses. A generator attached to a PQ bus does not impose VG in AC PF; exposing
                # such a variable would create a dead decision dimension that changes only an initial
                # guess and can bias optimizer fairness.
                if int(case.bus[bus_index, BUS_TYPE]) not in {REF, PV}:
                    continue
                seen.add(bus)
                lower = float(case.bus[bus_index, VMIN])
                upper = float(case.bus[bus_index, VMAX])
                self.variables.append(DecisionVariable(f"Vg@{bus}", lower, upper))
                self._actions.append(("vg", bus, lower, upper, None))

        if config.transformer_taps:
            taps = np.where((case.branch[:, BR_STATUS] > 0) & (case.branch[:, TAP] != 0))[0]
            lattice = stepped_values(
                config.transformer_minimum,
                config.transformer_maximum,
                config.transformer_step,
            )
            for branch_index in taps:
                from_bus = int(case.branch[branch_index, F_BUS])
                to_bus = int(case.branch[branch_index, T_BUS])
                kind = (
                    VariableKind.DISCRETE
                    if config.discrete_transformer_taps
                    else VariableKind.CONTINUOUS
                )
                self.variables.append(
                    DecisionVariable(
                        f"Tap {from_bus}-{to_bus}",
                        config.transformer_minimum,
                        config.transformer_maximum,
                        kind,
                        lattice if kind is VariableKind.DISCRETE else (),
                    )
                )
                self._actions.append(
                    (
                        "tap",
                        int(branch_index),
                        config.transformer_minimum,
                        config.transformer_maximum,
                        lattice if kind is VariableKind.DISCRETE else None,
                    )
                )

        controls = config.shunt_controls or default_shunt_controls(case)
        if config.shunt_compensation:
            seen_buses: set[int] = set()
            case_buses = set(case.bus[:, BUS_I].astype(int))
            for control in controls:
                control.validate()
                bus = int(control.bus_number)
                if bus not in case_buses:
                    raise ValueError(f"Shunt control references unknown bus {bus}")
                if bus in seen_buses:
                    raise ValueError(f"Duplicate shunt control at bus {bus}")
                seen_buses.add(bus)
                lattice = stepped_values(
                    control.minimum_mvar,
                    control.maximum_mvar,
                    control.step_mvar,
                )
                kind = VariableKind.DISCRETE if config.discrete_shunts else VariableKind.CONTINUOUS
                label = "Qsh" if control.semantics == "absolute" else "DeltaQsh"
                self.variables.append(
                    DecisionVariable(
                        f"{label}@{bus}",
                        control.minimum_mvar,
                        control.maximum_mvar,
                        kind,
                        lattice if kind is VariableKind.DISCRETE else (),
                    )
                )
                action_kind = "shunt" if control.semantics == "absolute" else "shunt_delta"
                self._actions.append(
                    (
                        action_kind,
                        bus,
                        control.minimum_mvar,
                        control.maximum_mvar,
                        lattice if kind is VariableKind.DISCRETE else None,
                    )
                )
                self._shunt_definitions.append(control)

    @property
    def dimension(self) -> int:
        return len(self.variables)

    def decode(self, normalized):
        z = np.asarray(normalized, dtype=float)
        if z.shape != (self.dimension,):
            raise ValueError(f"Expected decision vector shape ({self.dimension},), got {z.shape}")
        output = self.case.clone()
        physical: dict[str, float] = {}
        index = output.bus_index_map()
        for value, action, variable in zip(z, self._actions, self.variables):
            action_kind, target, lower, upper, lattice = action
            decoded = (
                decode_discrete(value, lattice)
                if lattice is not None
                else decode_continuous(value, lower, upper)
            )
            physical[variable.name] = decoded
            if action_kind == "vg":
                generators = np.where(
                    (output.gen[:, GEN_STATUS] > 0) & (output.gen[:, GEN_BUS].astype(int) == target)
                )[0]
                output.gen[generators, VG] = decoded
                output.bus[index[target], VM] = decoded
            elif action_kind == "tap":
                output.branch[target, TAP] = decoded
            elif action_kind == "shunt":
                output.bus[index[target], BS] = decoded
            elif action_kind == "shunt_delta":
                output.bus[index[target], BS] = self.case.bus[index[target], BS] + decoded
        return output, physical

    def control_validity(self, normalized) -> bool:
        z = np.asarray(normalized, dtype=float)
        return bool(
            z.shape == (self.dimension,)
            and np.all(np.isfinite(z))
            and np.all((z >= 0.0) & (z <= 1.0))
        )

    def formulation_manifest(self) -> dict:
        """Return a serializable declaration of the exact control formulation."""

        generator_buses = [int(action[1]) for action in self._actions if action[0] == "vg"]
        tap_branches = []
        for action in self._actions:
            if action[0] != "tap":
                continue
            branch_index = int(action[1])
            tap_branches.append(
                {
                    "branch_index": branch_index,
                    "from_bus": int(self.case.branch[branch_index, F_BUS]),
                    "to_bus": int(self.case.branch[branch_index, T_BUS]),
                    "minimum": float(action[2]),
                    "maximum": float(action[3]),
                    "step": float(self.config.transformer_step),
                    "discrete": bool(self.config.discrete_transformer_taps),
                }
            )
        return {
            "schema_version": 1,
            "profile_version": self.config.formulation_profile,
            "case_name": self.case.name,
            "case_checksum": self.case.checksum(),
            "dimension": self.dimension,
            "generator_voltage_buses": generator_buses,
            "transformer_taps": tap_branches,
            "shunt_controls": [asdict(item) for item in self._shunt_definitions],
            "fixed_shunts": [
                {"bus_number": int(row[BUS_I]), "bs_mvar": float(row[BS])}
                for row in self.case.bus
                if abs(float(row[BS])) > 0.0
                and int(row[BUS_I]) not in {item.bus_number for item in self._shunt_definitions}
            ],
            "semantics": {
                "generator_voltage": "absolute per-unit setpoint",
                "transformer_tap": "absolute off-nominal ratio",
                "shunt": "explicit per-control absolute or delta MVAr semantics",
            },
        }
