"""Structural and engineering validation for MATPOWER/PYPOWER case data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .case_model import (
    ANGMAX,
    ANGMIN,
    BR_R,
    BR_STATUS,
    BR_X,
    BUS_I,
    BUS_TYPE,
    F_BUS,
    GEN_BUS,
    GEN_STATUS,
    NONE,
    PMAX,
    PMIN,
    QMAX,
    QMIN,
    RATE_A,
    RATE_B,
    RATE_C,
    REF,
    T_BUS,
    VMAX,
    VMIN,
)


@dataclass(slots=True)
class CaseValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def require_valid(self) -> None:
        if not self.valid:
            raise ValueError("Invalid power-system case: " + " ".join(self.errors))


def _connected_components(case) -> list[set[int]]:
    bus_numbers = [int(value) for value in case.bus[:, BUS_I]]
    adjacency = {number: set() for number in bus_numbers}
    for branch in case.branch:
        if branch[BR_STATUS] <= 0:
            continue
        from_bus = int(branch[F_BUS])
        to_bus = int(branch[T_BUS])
        if from_bus in adjacency and to_bus in adjacency:
            adjacency[from_bus].add(to_bus)
            adjacency[to_bus].add(from_bus)
    components: list[set[int]] = []
    remaining = set(bus_numbers)
    while remaining:
        root = remaining.pop()
        component = {root}
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def validate_case(case) -> CaseValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    if not np.isfinite(case.base_mva) or float(case.base_mva) <= 0:
        errors.append("baseMVA must be finite and positive.")
    for name, array, minimum_columns in (
        ("bus", case.bus, 13),
        ("gen", case.gen, 10),
        ("branch", case.branch, 13),
    ):
        if array.ndim != 2 or array.shape[1] < minimum_columns:
            errors.append(
                f"{name} matrix must be two-dimensional with at least {minimum_columns} columns."
            )
        elif not np.all(np.isfinite(array)):
            errors.append(f"{name} matrix contains NaN or infinite values.")
    if errors:
        return CaseValidationReport(False, errors, warnings)

    numbers = case.bus[:, BUS_I].astype(int)
    if len(set(numbers)) != len(numbers):
        errors.append("Bus numbers must be unique.")
    bus_types = case.bus[:, BUS_TYPE].astype(int)
    if np.any(~np.isin(bus_types, (1, 2, 3, 4))):
        errors.append("Bus types must use MATPOWER codes 1, 2, 3, or 4.")
    if np.sum(bus_types == REF) != 1:
        errors.append("Exactly one reference bus is required.")
    known = set(numbers)

    generator_status = case.gen[:, GEN_STATUS]
    if np.any(~np.isin(generator_status, (0, 1))):
        errors.append("Generator status values must be 0 or 1.")
    for bus in case.gen[:, GEN_BUS].astype(int):
        if bus not in known:
            errors.append(f"Generator references unknown bus {bus}.")

    branch_status = case.branch[:, BR_STATUS]
    if np.any(~np.isin(branch_status, (0, 1))):
        errors.append("Branch status values must be 0 or 1.")
    for from_bus, to_bus in case.branch[:, [F_BUS, T_BUS]].astype(int):
        if from_bus not in known or to_bus not in known:
            errors.append(f"Branch references unknown buses {from_bus}-{to_bus}.")
        if from_bus == to_bus:
            errors.append(f"In-service topology contains a self-loop branch at bus {from_bus}.")

    if np.any(case.bus[:, VMAX] <= case.bus[:, VMIN]):
        errors.append("Every bus voltage maximum must exceed its minimum.")
    if np.any(case.bus[:, VMIN] <= 0):
        errors.append("Every bus voltage minimum must be positive.")
    if np.any(case.gen[:, QMAX] < case.gen[:, QMIN]):
        errors.append("Generator reactive-power limits are inconsistent.")
    if np.any(case.gen[:, PMAX] < case.gen[:, PMIN]):
        errors.append("Generator active-power limits are inconsistent.")
    if np.any(case.branch[:, [RATE_A, RATE_B, RATE_C]] < 0):
        errors.append("Branch thermal ratings cannot be negative.")
    if np.any(case.branch[:, ANGMAX] < case.branch[:, ANGMIN]):
        errors.append("Branch angle-difference limits are inconsistent.")

    in_service = case.branch[:, BR_STATUS] > 0
    impedance = np.hypot(case.branch[:, BR_R], case.branch[:, BR_X])
    zero_impedance = np.where(in_service & (impedance <= 1e-12))[0]
    if zero_impedance.size:
        errors.append(
            "In-service zero-impedance branches are unsupported because silently treating them as open circuits changes topology: "
            + ", ".join(str(int(index)) for index in zero_impedance)
            + "."
        )

    active_buses = {int(row[BUS_I]) for row in case.bus if int(row[BUS_TYPE]) != NONE}
    components = _connected_components(case)
    active_components = [
        component & active_buses for component in components if component & active_buses
    ]
    if len(active_components) > 1:
        errors.append(
            "The in-service network has disconnected active islands: "
            + "; ".join(",".join(map(str, sorted(component))) for component in active_components)
            + "."
        )

    online_generator_buses = {int(row[GEN_BUS]) for row in case.gen if row[GEN_STATUS] > 0}
    reference_bus = (
        int(case.bus[np.where(bus_types == REF)[0][0], BUS_I]) if np.any(bus_types == REF) else None
    )
    if reference_bus is not None and reference_bus not in online_generator_buses:
        errors.append("The reference bus must have at least one online generator.")

    unrated = int(np.sum(in_service & (case.branch[:, RATE_A] <= 0)))
    if unrated:
        warnings.append(
            f"{unrated} in-service branch(es) have RATE_A <= 0 and will be excluded from thermal-limit penalties."
        )
    return CaseValidationReport(not errors, errors, warnings)
