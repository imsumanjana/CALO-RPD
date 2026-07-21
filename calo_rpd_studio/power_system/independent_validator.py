"""Independent PYPOWER cross-validation using matched PV-only Q-limit semantics.

CALO-RPD keeps the single reference/slack bus as REF and treats any reactive-power capability
excess at that bus as an explicit constraint violation.  Upstream PYPOWER's
``ENFORCE_Q_LIMS=1`` may instead try to demote the REF bus and select a replacement reference;
for IEEE-300 this can leave no PV candidate and fail inside ``bustypes``.

To compare like with like without reusing CALO-RPD's Newton implementation, this module runs
PYPOWER's own AC Newton solver with ``ENFORCE_Q_LIMS=0`` and independently applies aggregate
Q-limit switching to *PV buses only* between solves.  The starting point is always the original
controlled case, so the validation remains independent of the internal solver's switched state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pypower.idx_brch import PF, PT

from .case_model import (
    BUS_I,
    BUS_TYPE,
    GEN_BUS,
    GEN_STATUS,
    PV,
    QG,
    QMAX,
    QMIN,
    VA,
    VM,
)


@dataclass(slots=True)
class CrossValidationResult:
    available: bool
    passed: bool
    max_vm_difference: float
    max_va_difference_deg: float
    loss_difference_mw: float
    message: str
    bus_type_mismatches: int = 0
    q_limit_mismatches: int = 0
    max_q_difference_mvar: float = 0.0


def _angle_difference_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return wrapped angular differences in degrees."""
    return np.abs((np.asarray(left) - np.asarray(right) + 180.0) % 360.0 - 180.0)


def _online_generator_rows(gen: np.ndarray, bus_number: int) -> np.ndarray:
    return np.where((gen[:, GEN_STATUS] > 0) & (gen[:, GEN_BUS].astype(int) == int(bus_number)))[0]


def _aggregate_q_by_bus(gen: np.ndarray) -> dict[int, float]:
    totals: dict[int, float] = {}
    for row in gen:
        if row[GEN_STATUS] <= 0:
            continue
        bus = int(row[GEN_BUS])
        totals[bus] = totals.get(bus, 0.0) + float(row[QG])
    return totals


def _allocate_aggregate_q_at_limit(gen: np.ndarray, bus_number: int, target_q: float) -> None:
    """Independently distribute an aggregate PV limit across same-bus generators.

    The allocation uses each unit's available Q range.  It intentionally lives in the
    independent validator rather than importing the internal PV/PQ switching implementation.
    """
    rows = _online_generator_rows(gen, bus_number)
    if not rows.size:
        return
    qmin = np.asarray(gen[rows, QMIN], dtype=float)
    qmax = np.asarray(gen[rows, QMAX], dtype=float)
    span = np.maximum(qmax - qmin, 0.0)
    total_span = float(np.sum(span))
    target = float(np.clip(target_q, float(np.sum(qmin)), float(np.sum(qmax))))
    if total_span > 0.0:
        values = qmin + (target - float(np.sum(qmin))) * span / total_span
    else:
        values = np.full(rows.size, target / rows.size, dtype=float)
    gen[rows, QG] = values


def _run_pypower_with_pv_q_switching(case, *, max_rounds: int = 20, tolerance_mvar: float = 1e-6):
    """Solve with PYPOWER while matching CALO-RPD's PV-only aggregate Q switching."""
    from pypower.ppoption import ppoption
    from pypower.runpf import runpf

    working = case.clone()
    solved = None
    for _round in range(max_rounds + 1):
        ppc = {
            "version": "2",
            "baseMVA": working.base_mva,
            "bus": working.bus.copy(),
            "gen": working.gen.copy(),
            "branch": working.branch.copy(),
        }
        if working.gencost is not None:
            ppc["gencost"] = working.gencost.copy()
        solved, success = runpf(ppc, ppoption(VERBOSE=0, OUT_ALL=0, ENFORCE_Q_LIMS=0))
        if not success:
            return None, False, "PYPOWER did not converge."

        # Detect aggregate violations only at buses that are currently PV. REF is deliberately
        # retained as REF; any REF Q excess remains visible as a constraint violation.
        solved_bus_map = {int(number): i for i, number in enumerate(solved["bus"][:, BUS_I])}
        violations: list[tuple[int, float]] = []
        for row in working.bus:
            if int(row[BUS_TYPE]) != PV:
                continue
            bus_number = int(row[BUS_I])
            gen_rows = _online_generator_rows(working.gen, bus_number)
            if not gen_rows.size:
                continue
            solved_gen_rows = _online_generator_rows(solved["gen"], bus_number)
            required_q = float(np.sum(solved["gen"][solved_gen_rows, QG]))
            qmin = float(np.sum(working.gen[gen_rows, QMIN]))
            qmax = float(np.sum(working.gen[gen_rows, QMAX]))
            if required_q > qmax + tolerance_mvar:
                violations.append((bus_number, qmax))
            elif required_q < qmin - tolerance_mvar:
                violations.append((bus_number, qmin))

        if not violations:
            return solved, True, ""
        if _round >= max_rounds:
            return (
                solved,
                False,
                "Independent PYPOWER PV Q-limit switching reached its round limit.",
            )

        # Carry the converged state forward only as the next independent PYPOWER initial state.
        # Keep original matrix widths because PYPOWER appends solved branch-flow columns.
        for bus_number, solved_index in solved_bus_map.items():
            working_index = np.where(working.bus[:, BUS_I].astype(int) == bus_number)[0]
            if working_index.size:
                wi = int(working_index[0])
                working.bus[wi, VM] = solved["bus"][solved_index, VM]
                working.bus[wi, VA] = solved["bus"][solved_index, VA]
        solved_q = _aggregate_q_by_bus(solved["gen"])
        for bus_number, total_q in solved_q.items():
            rows = _online_generator_rows(working.gen, bus_number)
            solved_rows = _online_generator_rows(solved["gen"], bus_number)
            if rows.size == solved_rows.size:
                working.gen[rows, QG] = solved["gen"][solved_rows, QG]

        bus_index = {int(number): i for i, number in enumerate(working.bus[:, BUS_I])}
        for bus_number, limit in violations:
            _allocate_aggregate_q_at_limit(working.gen, bus_number, limit)
            working.bus[bus_index[bus_number], BUS_TYPE] = 1  # PQ

    return solved, False, "Independent PYPOWER validation reached an unreachable state."


def validate_against_pypower(
    case,
    internal,
    vm_tolerance: float = 1e-5,
    va_tolerance_deg: float = 1e-3,
    loss_tolerance_mw: float = 1e-3,
    q_tolerance_mvar: float = 1e-3,
) -> CrossValidationResult:
    """Cross-check voltages, losses, final bus types, and aggregate generator Q against PYPOWER."""
    try:
        import pypower  # noqa: F401
    except (ModuleNotFoundError, ImportError) as exc:
        return CrossValidationResult(
            False,
            False,
            np.nan,
            np.nan,
            np.nan,
            f"PYPOWER cross-validation is unavailable: {exc}",
        )

    try:
        solved, success, failure_message = _run_pypower_with_pv_q_switching(case)
    except Exception as exc:  # third-party cross-check must never terminate the GUI
        return CrossValidationResult(
            True,
            False,
            np.inf,
            np.inf,
            np.inf,
            f"PYPOWER cross-validation failed: {exc}",
        )
    if not success or solved is None:
        return CrossValidationResult(
            True,
            False,
            np.inf,
            np.inf,
            np.inf,
            failure_message or "PYPOWER did not converge.",
        )

    # Align buses by external number rather than assuming row order.
    internal_bus_map = {
        int(number): index for index, number in enumerate(internal.case.bus[:, BUS_I])
    }
    pp_bus_map = {int(number): index for index, number in enumerate(solved["bus"][:, BUS_I])}
    common_buses = sorted(set(internal_bus_map) & set(pp_bus_map))
    if len(common_buses) != internal.case.n_bus:
        return CrossValidationResult(
            True, False, np.inf, np.inf, np.inf, "PYPOWER returned a different bus set."
        )
    ii = np.asarray([internal_bus_map[number] for number in common_buses], dtype=int)
    pi = np.asarray([pp_bus_map[number] for number in common_buses], dtype=int)
    vm_difference = float(np.max(np.abs(solved["bus"][pi, VM] - internal.vm_pu[ii])))
    va_difference = float(np.max(_angle_difference_deg(solved["bus"][pi, VA], internal.va_deg[ii])))
    bus_type_mismatches = int(
        np.count_nonzero(
            solved["bus"][pi, BUS_TYPE].astype(int) != internal.case.bus[ii, BUS_TYPE].astype(int)
        )
    )

    pypower_loss = float(np.sum(solved["branch"][:, PF] + solved["branch"][:, PT]))
    loss_difference = abs(pypower_loss - internal.total_loss_mw)

    # Aggregate Q by bus. This avoids a false mismatch when multiple same-bus generators use
    # different but electrically equivalent sharing conventions while preserving exact bus-level
    # reactive injection, which is what enters the network equations and aggregate Q limits.
    internal_q = _aggregate_q_by_bus(internal.case.gen)
    pypower_q = _aggregate_q_by_bus(solved["gen"])
    common_q_buses = sorted(set(internal_q) & set(pypower_q))
    missing_q_buses = len(set(internal_q) ^ set(pypower_q))
    q_differences = [abs(internal_q[key] - pypower_q[key]) for key in common_q_buses]
    max_q_difference = float(max(q_differences, default=0.0))
    q_limit_mismatches = int(
        missing_q_buses + sum(value > q_tolerance_mvar for value in q_differences)
    )

    passed = (
        vm_difference <= vm_tolerance
        and va_difference <= va_tolerance_deg
        and loss_difference <= loss_tolerance_mw
        and bus_type_mismatches == 0
        and q_limit_mismatches == 0
    )
    details = (
        f"Vm={vm_difference:.3g}, Va={va_difference:.3g} deg, loss={loss_difference:.3g} MW, "
        f"bus-type mismatches={bus_type_mismatches}, aggregate-Q mismatches={q_limit_mismatches}, "
        f"max aggregate dQ={max_q_difference:.3g} MVAr."
    )
    return CrossValidationResult(
        True,
        passed,
        vm_difference,
        va_difference,
        loss_difference,
        (
            "Cross-validation passed. "
            if passed
            else "Cross-validation exceeded at least one tolerance. "
        )
        + details,
        bus_type_mismatches,
        q_limit_mismatches,
        max_q_difference,
    )
