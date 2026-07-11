"""Independent PYPOWER cross-validation of solved electrical state.

PYPOWER's top-level ``pypower.api`` imports optional helper modules that still
reference ``numpy.in1d``. NumPy 2.4 removed that deprecated alias. The validator
therefore imports only the two PYPOWER modules it actually needs. This keeps the
cross-check compatible with current NumPy while avoiding unrelated PYPOWER API
imports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .case_model import VA, VM


@dataclass(slots=True)
class CrossValidationResult:
    available: bool
    passed: bool
    max_vm_difference: float
    max_va_difference_deg: float
    loss_difference_mw: float
    message: str


def validate_against_pypower(
    case,
    internal,
    vm_tolerance: float = 1e-5,
    va_tolerance_deg: float = 1e-3,
    loss_tolerance_mw: float = 1e-3,
) -> CrossValidationResult:
    """Cross-check an internal solution against PYPOWER without importing its full API."""
    try:
        from pypower.ppoption import ppoption
        from pypower.runpf import runpf
    except (ModuleNotFoundError, ImportError) as exc:
        return CrossValidationResult(
            False,
            False,
            np.nan,
            np.nan,
            np.nan,
            f"PYPOWER cross-validation is unavailable: {exc}",
        )

    vc = internal.case if internal.q_limit_rounds > 0 else case
    ppc = {
        "version": "2",
        "baseMVA": vc.base_mva,
        "bus": vc.bus.copy(),
        "gen": vc.gen.copy(),
        "branch": vc.branch.copy(),
    }
    if vc.gencost is not None:
        ppc["gencost"] = vc.gencost.copy()

    try:
        solved, success = runpf(
            ppc,
            ppoption(VERBOSE=0, OUT_ALL=0, ENFORCE_Q_LIMS=0),
        )
    except Exception as exc:  # third-party cross-check must never terminate the GUI
        return CrossValidationResult(
            True,
            False,
            np.inf,
            np.inf,
            np.inf,
            f"PYPOWER cross-validation failed: {exc}",
        )

    if not success:
        return CrossValidationResult(
            True,
            False,
            np.inf,
            np.inf,
            np.inf,
            "PYPOWER did not converge.",
        )

    vm_difference = float(np.max(np.abs(solved["bus"][:, VM] - internal.vm_pu)))
    va_difference = float(np.max(np.abs(solved["bus"][:, VA] - internal.va_deg)))
    pypower_loss = float(np.sum(solved["branch"][:, 13] + solved["branch"][:, 15]))
    loss_difference = abs(pypower_loss - internal.total_loss_mw)
    passed = (
        vm_difference <= vm_tolerance
        and va_difference <= va_tolerance_deg
        and loss_difference <= loss_tolerance_mw
    )
    return CrossValidationResult(
        True,
        passed,
        vm_difference,
        va_difference,
        loss_difference,
        "Cross-validation passed."
        if passed
        else "Cross-validation exceeded at least one tolerance.",
    )
